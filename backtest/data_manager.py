"""
Data Management Module
Responsible for loading, processing, and updating futures data.
"""

import os
import pandas as pd
import backtrader as bt

BACKTEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKTEST_DIR)

from .data_update import DataUpdate
from .products import require_products
from .roll_calendar import (
    annotate_expiries,
    build_date_contract_map,
    colliding_codes,
    contract_feed_name,
    mapping_as_dates,
    normalize_contract_code,
)

_PRICE_COLS = ['open', 'high', 'low', 'close', 'settle']
_ALIGN_COLS = ['open', 'high', 'low', 'close', 'settle', 'oi', 'volume']


class FuturesDailyData(bt.feeds.PandasData):
    """
    Daily futures feed based on PandasData.

    Columns: date, open, high, low, close, settle, oi, volume, session

    Extra custom lines:
      data.settle   - settlement price
      data.session  - 1 if this date had a real exchange print, else 0
    Built-in line mapping:
      openinterest -> oi column
    """
    lines = ('settle', 'session')

    params = (
        ('datetime',     None),
        ('open',         'open'),
        ('high',         'high'),
        ('low',          'low'),
        ('close',        'close'),
        ('volume',       'volume'),
        ('openinterest', 'oi'),
        ('settle',       'settle'),
        ('session',      'session'),
    )


# Backward-compatible alias
SAWeightedData = FuturesDailyData


class DataManager:
    """
    Data manager.
    Loads, filters, and updates futures weighted data and contract-level bars
    for one or more registered products.
    """

    DATA_DIR = os.path.join(ROOT_DIR, 'data')

    def __init__(self, symbols=None, symbol: str = None, update: bool = False):
        """
        Parameters
        ----------
        symbols : list[str], optional
            Product codes to load, e.g. ``['SA', 'FG', 'CF']``.
        symbol : str, optional
            Single-product alias used when ``symbols`` is omitted.
        update : bool
            Whether to refresh data from the exchange (incremental: current year).
        """
        if symbols is None:
            symbols = [symbol] if symbol else ['SA']
        self.symbols = require_products(symbols)
        self.symbol = self.symbols[0]

        if update:
            self._update_data()

    def weighted_path(self, symbol: str = None) -> str:
        return os.path.join(self.DATA_DIR, f'{self._sym(symbol)}_weighted.csv')

    def raw_path(self, symbol: str = None) -> str:
        return os.path.join(self.DATA_DIR, f'{self._sym(symbol)}.csv')

    def _sym(self, symbol: str = None) -> str:
        return (symbol or self.symbol).upper()

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _update_data(self):
        """Refresh exchange history incrementally and regenerate weighted data."""
        for symbol in self.symbols:
            try:
                path = self.weighted_path(symbol)
                print(f"[DataManager] Updating {symbol} data ...")
                DataUpdate(symbol).update()
                print(f"[DataManager] {symbol} data update done. Path: {path}")
            except Exception as e:
                print(f"[DataManager] Data update failed for {symbol}: {e}")
                raise

    def _feed_from_df(self, df: pd.DataFrame) -> FuturesDailyData:
        frame = df
        if 'session' not in frame.columns:
            frame = frame.copy()
            frame['session'] = 1.0
        return FuturesDailyData(dataname=frame)

    @staticmethod
    def _align_contract_ohlc(cdf: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Reindex one series onto the shared calendar without looking ahead.

        Real prints keep their OHLC. Days with no print are *not* tradable
        (``session=0``). After the first print, close/settle/oi are ffilled so
        existing positions can mark to the last session; open/high/low are
        flattened to that close so a dark bar cannot print a fake open.
        ``bfill`` is never used.
        """
        frame = cdf.copy()
        for col in _ALIGN_COLS:
            if col not in frame.columns:
                frame[col] = float('nan')
        out = frame[_ALIGN_COLS].reindex(index)
        session = out['close'].notna()
        out[_PRICE_COLS] = out[_PRICE_COLS].ffill()
        out['oi'] = out['oi'].ffill().fillna(0)
        out['volume'] = out['volume'].where(session, 0).fillna(0)
        out['session'] = session.astype(float)
        dark = out['close'].notna() & ~session
        for col in ('open', 'high', 'low', 'settle'):
            out.loc[dark, col] = out.loc[dark, 'close']
        return out

    def _read_weighted(self, symbol: str) -> pd.DataFrame:
        path = self.weighted_path(symbol)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Weighted data file not found: {path}\n"
                "Run python -m backtest.data_update first, or pass update=True "
                "when constructing DataManager."
            )

        df = pd.read_csv(path, parse_dates=['date'])
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index).normalize()
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)
        return df

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def load_dataframe(
        self,
        start_date: str = None,
        end_date: str = None,
        symbol: str = None,
    ) -> pd.DataFrame:
        """
        Load the weighted CSV data and return a DataFrame filtered by date.

        Parameters
        ----------
        start_date : str, optional
            Start date, formatted as 'YYYY-MM-DD'.
        end_date : str, optional
            End date, formatted as 'YYYY-MM-DD'.
        symbol : str, optional
            Product code. Defaults to the first loaded symbol.

        Returns
        -------
        pd.DataFrame
            Columns: date(index), open, high, low, close, settle, oi, volume.
        """
        symbol = self._sym(symbol)
        df = self._read_weighted(symbol)

        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        if df.empty:
            raise ValueError(
                f"Filtered data is empty for {symbol}. Check that the date range "
                f"[{start_date}, {end_date}] falls within the available data."
            )

        return df

    def load_contracts_dataframe(self, symbol: str = None) -> pd.DataFrame:
        """Load contract-level CZCE history from ``data/{symbol}.csv``."""
        symbol = self._sym(symbol)
        path = self.raw_path(symbol)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Contract data file not found: {path}\n"
                "Run python -m backtest.data_update first, or pass update=True "
                "when constructing DataManager."
            )

        df = pd.read_csv(path)
        if 'date' not in df.columns or 'contract' not in df.columns:
            raise ValueError(
                f"{path} must contain 'date' and 'contract' columns"
            )
        df['date'] = pd.to_datetime(df['date'].astype(str), errors='coerce')
        df = df.dropna(subset=['date'])
        df['contract'] = df['contract'].map(normalize_contract_code)
        df = df[df['contract'] != '']
        df.sort_values(['date', 'contract'], inplace=True)
        return df.reset_index(drop=True)

    def get_bt_feed(
        self,
        start_date: str = None,
        end_date: str = None,
        symbol: str = None,
    ) -> FuturesDailyData:
        """
        Return a data feed that can be passed directly to backtrader's Cerebro.

        Parameters
        ----------
        start_date : str, optional
        end_date : str, optional
        symbol : str, optional

        Returns
        -------
        FuturesDailyData
        """
        df = self.load_dataframe(start_date, end_date, symbol=symbol)
        return FuturesDailyData(dataname=df)

    @staticmethod
    def _codes_in_window(raw: pd.DataFrame, index: pd.DatetimeIndex) -> list:
        """Return feed names for contract segments that print inside ``index``."""
        if raw.empty or index.empty:
            return []
        start, end = index.min(), index.max()
        df = raw if 'expiry' in raw.columns else annotate_expiries(raw)
        window = df[(df['date'] >= start) & (df['date'] <= end)]
        colliding = colliding_codes(df, start, end)
        names = []
        seen = set()
        for code, expiry in zip(window['contract'], window['expiry']):
            name = contract_feed_name(code, expiry, colliding)
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names

    def _contract_segments(
        self, raw: pd.DataFrame, index: pd.DatetimeIndex
    ) -> list:
        """Split ``raw`` into (feed_name, frame) pairs for one backtest window.

        Same 3-digit code in two decades becomes two segments so 2015 FG501
        prices are not ffilled into the 2025 FG501 feed.
        """
        if raw.empty or index.empty:
            return []
        start, end = index.min(), index.max()
        df = raw if 'expiry' in raw.columns else annotate_expiries(raw)
        window = df[(df['date'] >= start) & (df['date'] <= end)]
        colliding = colliding_codes(df, start, end)
        segments = []
        seen = set()
        for (code, expiry), grp in window.groupby(['contract', 'expiry'], sort=False):
            name = contract_feed_name(code, expiry, colliding)
            if name in seen:
                continue
            seen.add(name)
            segments.append((name, grp))
        return segments

    def _align_contracts(self, segments: list, index: pd.DatetimeIndex,
                         symbol: str) -> tuple:
        """Align listed contract segments onto ``index``. Returns (feeds, aligned_frames)."""
        contract_feeds = {}
        aligned = {}
        for name, cdf in segments:
            if cdf.empty:
                print(f"[DataManager] Skipping {name}: no rows in {symbol}.csv")
                continue
            frame = cdf.copy()
            frame = frame.set_index('date').sort_index()
            frame.index = pd.to_datetime(frame.index).normalize()
            frame = frame[~frame.index.duplicated(keep='last')]
            aligned_df = self._align_contract_ohlc(frame, index)
            if aligned_df['close'].notna().sum() == 0:
                print(f"[DataManager] Skipping {name}: no usable OHLC after align")
                continue
            aligned[name] = aligned_df
            contract_feeds[name] = self._feed_from_df(aligned_df)
        return contract_feeds, aligned

    def _bundle_symbol(
        self,
        symbol: str,
        weighted_src: pd.DataFrame,
        calendar: pd.DatetimeIndex,
        first_print,
    ) -> dict:
        """Build weighted + contract feeds for one product on ``calendar``."""
        raw = annotate_expiries(self.load_contracts_dataframe(symbol))
        weighted_df = self._align_contract_ohlc(weighted_src, calendar)

        mapping = build_date_contract_map(
            calendar, raw, listed_from=first_print
        )
        calendar_codes = []
        seen_cal = set()
        for code in mapping.values():
            if code not in seen_cal:
                seen_cal.add(code)
                calendar_codes.append(code)

        if not calendar_codes:
            raise ValueError(
                f"Roll calendar produced no contracts for {symbol}. "
                f"Check data/{symbol}.csv coverage."
            )

        segments = self._contract_segments(raw, calendar)
        names = [name for name, _ in segments]
        seen = set(names)
        extra = []
        for code in calendar_codes:
            if code not in seen:
                extra.append(code)
                seen.add(code)
        if extra:
            # Calendar name missing from window segments: include matching rows
            colliding = colliding_codes(raw, calendar.min(), calendar.max())
            by_name = {}
            for (code, expiry), grp in raw.groupby(['contract', 'expiry'], sort=False):
                by_name[contract_feed_name(code, expiry, colliding)] = grp
            for name in extra:
                if name in by_name:
                    segments.append((name, by_name[name]))

        contract_feeds, aligned = self._align_contracts(
            segments, calendar, symbol
        )

        missing = [c for c in calendar_codes if c not in contract_feeds]
        if missing:
            raise ValueError(
                f"{symbol} calendar contracts have no aligned OHLC: {missing}"
            )

        exec_close = []
        exec_code = []
        for dt in calendar:
            key = pd.Timestamp(dt).normalize()
            code = mapping.get(key)
            if code is None or code not in aligned:
                exec_close.append(float('nan'))
                exec_code.append('')
                continue
            row = aligned[code].loc[dt]
            if float(row.get('session', 0) or 0) <= 0:
                exec_close.append(float('nan'))
                exec_code.append('')
                continue
            exec_close.append(float(row['close']))
            exec_code.append(code)
        exec_price_df = pd.DataFrame(
            {'close': exec_close, 'contract': exec_code},
            index=calendar,
        )

        n_cal = len(calendar_codes)
        n_all = len(contract_feeds)
        print(
            f"[DataManager] {symbol} feeds: weighted + {n_all} contracts "
            f"({n_cal} calendar: {', '.join(calendar_codes)})"
        )

        return {
            'weighted_df': weighted_df,
            'weighted_feed': self._feed_from_df(weighted_df),
            'contract_feeds': contract_feeds,
            'contract_by_date': mapping_as_dates(mapping),
            'calendar_codes': calendar_codes,
            'exec_price_df': exec_price_df,
            'first_print': first_print,
        }

    def get_universe_bundle(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> dict:
        """Build weighted + real-contract feeds for every loaded product.

        All products are aligned onto the union of their weighted calendars
        inside the backtest window so Backtrader can step them together.
        Missing days are *not* backfilled; ``session=0`` bars are not tradable.

        Returns
        -------
        dict
            symbols, calendar, products[symbol] -> per-product bundle
        """
        frames = {}
        first_prints = {}
        for symbol in self.symbols:
            full = self._read_weighted(symbol)
            if full.empty:
                raise ValueError(f'{symbol} weighted data is empty')
            first_prints[symbol] = pd.Timestamp(full.index.min()).date()
            df = full
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if df.empty:
                raise ValueError(
                    f"Filtered data is empty for {symbol}. Check that the date range "
                    f"[{start_date}, {end_date}] falls within the available data."
                )
            frames[symbol] = df

        calendar = frames[self.symbols[0]].index
        for symbol in self.symbols[1:]:
            calendar = calendar.union(frames[symbol].index)
        calendar = calendar.sort_values()

        products = {}
        for symbol in self.symbols:
            products[symbol] = self._bundle_symbol(
                symbol, frames[symbol], calendar, first_prints[symbol]
            )

        return {
            'symbols': list(self.symbols),
            'calendar': calendar,
            'products': products,
        }

    def get_contract_bundle(
        self,
        start_date: str = None,
        end_date: str = None,
        symbol: str = None,
    ) -> dict:
        """Build weighted + all real-contract feeds for one product.

        Prefer ``get_universe_bundle`` when loading more than one symbol.
        """
        symbol = self._sym(symbol)
        if self.symbols == [symbol]:
            universe = self.get_universe_bundle(start_date, end_date)
            return universe['products'][symbol]
        dm = DataManager(symbols=[symbol], update=False)
        universe = dm.get_universe_bundle(start_date, end_date)
        return universe['products'][symbol]

    def get_raw_dataframe(self, symbol: str = None) -> pd.DataFrame:
        """Return the full unfiltered weighted DataFrame (useful for plotting, etc.)."""
        return self.load_dataframe(symbol=symbol)
