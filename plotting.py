"""
Plotting Module
Uses matplotlib to draw and save the following charts:
  1. Equity curve     - account equity over time
  2. Return curve     - cumulative and daily returns
  3. Position chart   - long/short position size over time
  4. Price & signals  - close price with buy/sell signal markers
"""

import datetime
import logging
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Use a non-interactive backend (works without a display)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)


# Global plotting style
plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans', 'SimHei', 'Arial Unicode MS'],
    'axes.unicode_minus': False,
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'grid.color':       '#21262d',
    'text.color':       '#c9d1d9',
    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',
})

COLOR_UP    = '#3fb950'   # green (profit / long)
COLOR_DOWN  = '#f85149'   # red   (loss / short)
COLOR_FLAT  = '#8b949e'   # gray  (flat)
COLOR_PRICE = '#58a6ff'   # blue  (price line)
COLOR_EXEC  = '#ffa657'   # orange (calendar contract close)
COLOR_EQ    = '#d2a8ff'   # purple (equity line)


class BacktestPlotter:
    """
    Backtest chart generator.

    Parameters
    ----------
    equity_records : list of dict
        Fields: date, equity, position, daily_return. ``position`` is a
        ``{symbol: net_lots}`` dict covering every loaded product.
    trade_logs : list of dict
        Ledger rows. Fields: open_date, close_date, direction, ...
    price_dfs : dict[str, pd.DataFrame]
        OI-weighted close per product (index=date, contains a `close` column);
        used for the price + signals chart. Keyed by product symbol.
    signal_log : list of dict
        Engine's full signal_log (all products). Fields: date, price,
        direction ('buy' | 'sell' | 'close'), symbol.
    metrics : dict
        Dictionary of backtest metrics.
    config : dict
        Backtest configuration (contains results_dir, strategy_name, etc.).
    exec_price_dfs : dict[str, pd.DataFrame], optional
        Calendar-contract close per product (and optional `contract` column)
        for overlay. Keyed by product symbol.
    symbols : list of str, optional
        Products to plot, in display order. Defaults to ``price_dfs.keys()``.
    """

    def __init__(
        self,
        equity_records: list,
        trade_logs:     list,
        price_dfs:      dict,
        signal_log:     list,
        metrics:        dict,
        config:         dict,
        exec_price_dfs: dict = None,
        symbols:        list = None,
    ):
        self.equity_records = equity_records
        self.trade_logs     = trade_logs
        self.price_dfs      = {sym: df.copy() for sym, df in (price_dfs or {}).items()}
        self.signal_log     = signal_log or []
        self.metrics        = metrics
        self.config         = config
        self.exec_price_dfs = (
            {sym: df.copy() for sym, df in exec_price_dfs.items()}
            if exec_price_dfs else {}
        )
        self.symbols = list(symbols) if symbols else list(self.price_dfs.keys())

        self._results_dir   = config.get('results_dir', 'results')
        self._strategy_name = config.get('strategy_name', 'strategy')
        self._ts            = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        os.makedirs(self._results_dir, exist_ok=True)

        # Signals bucketed by product for the per-symbol signal chart.
        self._signals_by_symbol = {}
        default_symbol = self.symbols[0] if self.symbols else None
        for sig in self.signal_log:
            sym = sig.get('symbol') or default_symbol
            self._signals_by_symbol.setdefault(sym, []).append(sig)

        # Build DataFrame
        self._equity_df  = pd.DataFrame(equity_records)
        if not self._equity_df.empty:
            self._equity_df['date'] = pd.to_datetime(self._equity_df['date'])
            self._equity_df.set_index('date', inplace=True)
            self._equity_df['cum_return'] = (
                self._equity_df['equity'] / self._equity_df['equity'].iloc[0] - 1
            ) * 100

        # Per-product position series, extracted from the {symbol: lots} dict
        # recorded for every bar.
        self._position_by_symbol = {}
        if not self._equity_df.empty:
            for sym in self.symbols:
                self._position_by_symbol[sym] = self._equity_df['position'].apply(
                    lambda pos, s=sym: (pos or {}).get(s, 0)
                )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def plot_all(self) -> dict:
        """
        Draw all charts and return a dict of file paths:
        {
          'equity':   str,
          'returns':  str,
          'position': str,
          'signals':  str,
          'summary':  str,   # four-in-one summary
        }
        """
        paths = {}
        paths['equity']   = self._plot_equity_curve()
        paths['returns']  = self._plot_return_curve()
        paths['position'] = self._plot_position()
        paths['signals']  = self._plot_price_signals()
        paths['summary']  = self._plot_summary()
        return paths

    # ------------------------------------------------------------------
    # Chart 1: equity curve
    # ------------------------------------------------------------------

    def _plot_equity_curve(self) -> str:
        fig, ax = plt.subplots(figsize=(12, 5))
        df = self._equity_df

        ax.plot(df.index, df['equity'], color=COLOR_EQ, linewidth=1.5, label='Equity')

        # Fill the drawdown area
        running_max = df['equity'].cummax()
        ax.fill_between(df.index, df['equity'], running_max,
                        where=(df['equity'] < running_max),
                        alpha=0.3, color=COLOR_DOWN, label='Drawdown')

        ax.set_title(f'{self._strategy_name} - Equity Curve', fontsize=13)
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity (CNY)')
        ax.legend(loc='upper left')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate()
        ax.grid(True, alpha=0.3)

        # Annotate final metrics below the plot, without clobbering the x-axis label.
        m = self.metrics
        info = (
            f"Total Return: {m.get('total_return', 0):.2f}%  "
            f"Max Drawdown: {m.get('max_drawdown', 0):.2f}%  "
            f"Sharpe: {m.get('sharpe_ratio', 0):.3f}"
        )
        fig.text(0.5, 0.01, info, ha='center', fontsize=9, color='#8b949e')

        path = os.path.join(
            self._results_dir,
            f"{self._strategy_name}_equity_{self._ts}.png"
        )
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Equity curve saved: %s", path)
        return path

    # ------------------------------------------------------------------
    # Chart 2: return curve
    # ------------------------------------------------------------------

    def _plot_return_curve(self) -> str:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        df = self._equity_df

        # Top: cumulative return
        ax1 = axes[0]
        ax1.plot(df.index, df['cum_return'], color=COLOR_EQ, linewidth=1.5)
        ax1.fill_between(df.index, df['cum_return'], 0,
                         where=(df['cum_return'] >= 0), alpha=0.2, color=COLOR_UP)
        ax1.fill_between(df.index, df['cum_return'], 0,
                         where=(df['cum_return'] < 0), alpha=0.2, color=COLOR_DOWN)
        ax1.axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')
        ax1.set_title(f'{self._strategy_name} - Return Curve', fontsize=13)
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.grid(True, alpha=0.3)

        # Bottom: daily return (bar chart)
        ax2 = axes[1]
        colors = [COLOR_UP if v >= 0 else COLOR_DOWN for v in df['daily_return'] * 100]
        ax2.bar(df.index, df['daily_return'] * 100, color=colors, alpha=0.7, width=1)
        ax2.axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')
        ax2.set_ylabel('Daily Return (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate()

        path = os.path.join(
            self._results_dir,
            f"{self._strategy_name}_returns_{self._ts}.png"
        )
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Return curve saved: %s", path)
        return path

    # ------------------------------------------------------------------
    # Chart 3: position state
    # ------------------------------------------------------------------

    def _plot_position(self) -> str:
        symbols = self.symbols or ['position']
        n = len(symbols)
        fig, axes = plt.subplots(n, 1, figsize=(12, min(3 * n, 18)), sharex=True, squeeze=False)
        axes = axes[:, 0]

        for ax, symbol in zip(axes, symbols):
            pos = self._position_by_symbol.get(symbol)
            if pos is None or pos.empty:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', color='#8b949e')
                ax.set_ylabel(symbol)
                continue

            long_mask  = pos > 0
            short_mask = pos < 0

            ax.fill_between(pos.index, pos, 0, where=long_mask,
                            alpha=0.5, color=COLOR_UP, label='Long')
            ax.fill_between(pos.index, pos, 0, where=short_mask,
                            alpha=0.5, color=COLOR_DOWN, label='Short')
            ax.step(pos.index, pos, color='#c9d1d9', linewidth=0.8, where='post')
            ax.axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')

            ax.set_title(symbol, fontsize=10, loc='left')
            ax.set_ylabel('Lots')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'{self._strategy_name} - Position by Product', fontsize=13)
        axes[-1].set_xlabel('Date')
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate()

        path = os.path.join(
            self._results_dir,
            f"{self._strategy_name}_position_{self._ts}.png"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Position chart saved: %s", path)
        return path

    # ------------------------------------------------------------------
    # Chart 4: price + signals
    # ------------------------------------------------------------------

    def _plot_price_signals(self) -> str:
        symbols = self.symbols or list(self.price_dfs.keys())
        n = len(symbols) or 1
        fig, axes = plt.subplots(n, 1, figsize=(14, min(4 * n, 24)), sharex=True, squeeze=False)
        axes = axes[:, 0]

        has_exec = any(
            self.exec_price_dfs.get(sym) is not None
            and 'close' in self.exec_price_dfs[sym].columns
            for sym in symbols
        )

        for ax, symbol in zip(axes, symbols):
            price_df = self.price_dfs.get(symbol)
            if price_df is None or 'close' not in price_df.columns:
                ax.text(0.5, 0.5, 'No price data', transform=ax.transAxes,
                        ha='center', va='center', color='#8b949e')
                ax.set_ylabel(symbol)
                continue

            price_df = price_df.copy()
            price_df.index = pd.to_datetime(price_df.index)
            ax.plot(price_df.index, price_df['close'],
                    color=COLOR_PRICE, linewidth=1.2, label='Weighted Close', zorder=2)

            exec_df = self.exec_price_dfs.get(symbol)
            if exec_df is not None and 'close' in exec_df.columns:
                exec_df = exec_df.copy()
                exec_df.index = pd.to_datetime(exec_df.index)
                ax.plot(exec_df.index, exec_df['close'],
                        color=COLOR_EXEC, linewidth=1.0, linestyle='--',
                        label='Contract Close', zorder=3, alpha=0.9)

            for sig in self._signals_by_symbol.get(symbol, []):
                sig_date = pd.to_datetime(sig['date'])
                sig_price = sig['price']
                direction = sig['direction']
                if direction == 'buy':
                    ax.scatter(sig_date, sig_price, marker='^', color=COLOR_UP,
                               s=80, zorder=5, label='_nolegend_')
                elif direction == 'sell':
                    ax.scatter(sig_date, sig_price, marker='v', color=COLOR_DOWN,
                               s=80, zorder=5, label='_nolegend_')

            ax.set_title(symbol, fontsize=10, loc='left')
            ax.set_ylabel('Price')
            ax.grid(True, alpha=0.3)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=COLOR_PRICE, linewidth=1.5, label='Weighted Close'),
        ]
        if has_exec:
            legend_elements.append(
                Line2D([0], [0], color=COLOR_EXEC, linewidth=1.2,
                       linestyle='--', label='Contract Close')
            )
        legend_elements.extend([
            plt.scatter([], [], marker='^', color=COLOR_UP,  s=60, label='Long Signal'),
            plt.scatter([], [], marker='v', color=COLOR_DOWN, s=60, label='Short Signal'),
        ])
        axes[0].legend(handles=legend_elements, loc='upper left', fontsize=8)

        fig.suptitle(f'{self._strategy_name} - Price and Trading Signals', fontsize=13)
        axes[-1].set_xlabel('Date')
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate()

        path = os.path.join(
            self._results_dir,
            f"{self._strategy_name}_signals_{self._ts}.png"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Price & signals chart saved: %s", path)
        return path

    # ------------------------------------------------------------------
    # Chart 5: four-in-one summary
    # ------------------------------------------------------------------

    def _plot_summary(self) -> str:
        fig = plt.figure(figsize=(16, 8))
        recovery_days = self.metrics.get('max_drawdown_recovery_days')
        recovery_text = (
            f'{recovery_days} days'
            if recovery_days is not None else 'Not recovered'
        )
        fig.suptitle(
            f'{self._strategy_name}  Backtest Summary\n'
            f'Return: {self.metrics.get("total_return", 0):.2f}%  '
            f'Sharpe: {self.metrics.get("sharpe_ratio", 0):.3f}  '
            f'MaxDD: {self.metrics.get("max_drawdown", 0):.2f}%  '
            f'Recovery: {recovery_text}  '
            f'WinRate: {self.metrics.get("win_rate", 0):.2f}%  '
            f'P/L Ratio: {self.metrics.get("profit_loss_ratio", 0):.3f}  '
            f'Trades: {self.metrics.get("n_trades", 0)}',
            fontsize=11, color='#c9d1d9', y=0.98
        )

        gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)

        df = self._equity_df

        # 1. Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df.index, df['equity'], color=COLOR_EQ, linewidth=1.5, label='Equity')
        running_max = df['equity'].cummax()
        ax1.fill_between(df.index, df['equity'], running_max,
                         where=(df['equity'] < running_max),
                         alpha=0.3, color=COLOR_DOWN, label='Drawdown')
        ax1.set_title('Equity Curve', fontsize=10)
        ax1.set_ylabel('Equity (CNY)')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')

        # 2. Cumulative return
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(df.index, df['cum_return'], color=COLOR_EQ, linewidth=1.2)
        ax2.fill_between(df.index, df['cum_return'], 0,
                         where=(df['cum_return'] >= 0), alpha=0.2, color=COLOR_UP)
        ax2.fill_between(df.index, df['cum_return'], 0,
                         where=(df['cum_return'] < 0), alpha=0.2, color=COLOR_DOWN)
        ax2.axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')
        ax2.set_title('Cum. Return', fontsize=10)
        ax2.set_ylabel('%')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')

        # 3. Daily return
        ax3 = fig.add_subplot(gs[1, 1])
        colors = [COLOR_UP if v >= 0 else COLOR_DOWN for v in df['daily_return'] * 100]
        ax3.bar(df.index, df['daily_return'] * 100, color=colors, alpha=0.7, width=1)
        ax3.axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')
        ax3.set_title('Daily Return', fontsize=10)
        ax3.set_ylabel('%')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax3.get_xticklabels(), rotation=30, ha='right')

        # Position and price/signals are dropped from the summary: with one
        # subplot per product they no longer fit a single panel each, and
        # both charts are already available on their own (see _plot_position
        # and _plot_price_signals).

        path = os.path.join(
            self._results_dir,
            f"{self._strategy_name}_summary_{self._ts}.png"
        )
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Summary chart saved: %s", path)
        return path
