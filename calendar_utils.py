"""
P2-ETF-COPULA-ENGINE  ·  calendar_utils.py
NYSE calendar utilities — compute the next US market trading day.
"""

from __future__ import annotations
from datetime import date, timedelta
import pandas as pd

try:
    import pandas_market_calendars as mcal
    _NYSE = mcal.get_calendar("NYSE")
    _USE_MCal = True
except ImportError:
    _USE_MCal = False


def next_trading_day(from_date: date | str | None = None) -> str:
    """
    Return the next NYSE trading day after from_date (default: today).

    Returns
    -------
    str  ISO date string "YYYY-MM-DD"
    """
    if from_date is None:
        from_date = date.today()
    elif isinstance(from_date, str):
        from_date = pd.Timestamp(from_date).date()

    if _USE_MCal:
        start  = pd.Timestamp(from_date) + pd.Timedelta(days=1)
        end    = start + pd.Timedelta(days=14)
        sched  = _NYSE.schedule(start_date=start, end_date=end)
        if len(sched) > 0:
            return sched.index[0].strftime("%Y-%m-%d")

    # Fallback: skip weekends (no holiday awareness)
    d = from_date + timedelta(days=1)
    while d.weekday() >= 5:        # 5=Saturday, 6=Sunday
        d += timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def last_trading_day(df_index: pd.DatetimeIndex) -> str:
    """Return the last date present in a DataFrame index as ISO string."""
    return df_index.max().strftime("%Y-%m-%d")
