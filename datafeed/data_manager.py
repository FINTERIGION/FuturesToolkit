"""
Data Management Module
Responsible for loading, processing, and updating futures data.

Produces plain pandas DataFrames (no backtrader feed objects). The weighted
series is aligned/ffilled onto the shared calendar for indicator use; each
real contract keeps only the bars where it actually printed -- see
``core.market.build_market_data``, which turns this bundle into a
``MarketData`` instance.
"""

import logging
import os

import pandas as pd

DATAFEED_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(DATAFEED_DIR)

logger = logging.getLogger(__name__)

from .data_update import DataUpdate
from .products import require_products, roll_rule
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
            Product codes to load, e.g. ``['SA', 'FG', 'CF', 'MA']``.
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
                logger.info("Updating %s data ...", symbol)
                DataUpdate(symbol).update()
                logger.info("%s data update done. Path: %s", symbol, path)
            except Exception as e:
                logger.error("Data update failed for %s: %s", symbol, e)
                raise

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
                "Run python -m datafeed.data_update first, or pass update=True "
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
                "Run python -m datafeed.data_update first, or pass update=True "
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
        prices are not carried into the 2025 FG501 series.
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

    def _raw_contract_frames(self, segments: list, symbol: str) -> dict:
        """Only-real-prints frames per contract: no calendar reindex, no ffill.

        This is what ``core.market.ContractSeries`` is built from -- a
        63-contract x 1610-bar union calendar would be ~85% padding; keeping
        just the real rows is the whole point of the rewrite (see
        docs/rewrite-plan.md §6).
        """
        frames = {}
        for name, cdf in segments:
            if cdf.empty:
                logger.warning("Skipping %s: no rows in %s.csv", name, symbol)
                continue
            frame = cdf.copy().set_index('date').sort_index()
            frame.index = pd.to_datetime(frame.index).normalize()
            frame = frame[~frame.index.duplicated(keep='last')]
            for col in _ALIGN_COLS:
                if col not in frame.columns:
                    frame[col] = float('nan')
            frame = frame[_ALIGN_COLS].dropna(subset=['close'])
            if frame.empty:
                logger.warning("Skipping %s: no usable OHLC", name)
                continue
            frames[name] = frame
        return frames

    def _bundle_symbol(
        self,
        symbol: str,
        weighted_src: pd.DataFrame,
        calendar: pd.DatetimeIndex,
        first_print,
    ) -> dict:
        """Build the weighted series + real-contract bundle for one product."""
        raw = annotate_expiries(self.load_contracts_dataframe(symbol))
        weighted_df = self._align_contract_ohlc(weighted_src, calendar)

        rule = roll_rule(symbol)
        mapping = build_date_contract_map(
            calendar,
            raw,
            listed_from=first_print,
            main_months=rule['main_months'],
            lead_months=rule['lead_months'],
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

        contract_frames = self._raw_contract_frames(segments, symbol)

        missing = [c for c in calendar_codes if c not in contract_frames]
        if missing:
            raise ValueError(
                f"{symbol} calendar contracts have no aligned OHLC: {missing}"
            )

        exec_close = []
        exec_code = []
        for dt in calendar:
            key = pd.Timestamp(dt).normalize()
            code = mapping.get(key)
            frame = contract_frames.get(code) if code else None
            if frame is None or key not in frame.index:
                exec_close.append(float('nan'))
                exec_code.append('')
                continue
            exec_close.append(float(frame.loc[key, 'close']))
            exec_code.append(code)
        exec_price_df = pd.DataFrame(
            {'close': exec_close, 'contract': exec_code},
            index=calendar,
        )

        n_cal = len(calendar_codes)
        n_all = len(contract_frames)
        logger.info(
            "%s feeds: weighted + %d contracts (%d calendar: %s); "
            "main months %s, rolled %d month(s) before delivery",
            symbol, n_all, n_cal, ', '.join(calendar_codes),
            '/'.join(f'{m:02d}' for m in rule['main_months']),
            rule['lead_months'],
        )

        return {
            'weighted_df': weighted_df,
            'contract_frames': contract_frames,
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
        """Build weighted + real-contract bundles for every loaded product.

        All products are aligned onto the union of their weighted calendars
        inside the backtest window so the engine can step them together.
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
        """Build weighted + all real-contract frames for one product.

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
