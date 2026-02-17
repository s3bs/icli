"""Technical analysis calculations — TWEMA, ATR, and supporting math."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

# 23_400 seconds == 6.5 hours (390 minutes) — the length of a Regular Trading Hour session
RTH_EMA_VWAP: Final = 23_400


def rmsnorm(x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Implements Root Mean Square (RMS) Normalization

    Args:
        x: Input array/tensor to normalize
        epsilon: Small constant for numerical stability (default: 1e-6)

    Returns:
        Normalized array/tensor
    """
    # Calculate the root mean square
    rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True))

    # Normalize the input
    normalized = x / (rms + epsilon)

    return normalized


@dataclass(slots=True)
class ATR:
    # decay/lookback length
    length: int = 20
    prevClose: float = 0.0
    current: float = 0
    updated: int = 0

    def __post_init__(self) -> None:
        assert self.length >= 1

    def update(self, high: float, low: float, close: float) -> float:
        """Update rolling ATR using O(1) storage"""

        self.updated += 1

        if not self.current:
            # if we have no current history, we start from the local price data,
            # otherwise, if we didn't have this case, we would generate a too#
            # wide range and converge it from the top (where as here, we basically
            # use a too low range and converge it from the lower bound instead).

            # NOTE: if the first update has high==low then
            #       the ATR is 0 because the "range" of two equal prices is 0.
            self.current = high - low  # / self.length
        else:
            # else, if we DO have history, then we use a modified update length
            # to improve the bootstrap process (so we don't have to wait for tiny
            # numbers to converge into the full length, we just mock a dynamic length
            # until we reach  max capacity)
            useUpdateLen = min(self.updated, self.length)

            currentTR = max(
                max(high - low, abs(high - self.prevClose)),
                abs(low - self.prevClose),
            )

            # this is an embedded RMA of the TR to create an ATR
            self.current = (
                currentTR + (useUpdateLen - 1) * self.current
            ) / useUpdateLen

        self.prevClose = close

        return self.current


@dataclass(slots=True)
class ATRLive:
    """An ATR but with live trades instead of bars, so it keeps a tiny local history."""

    length: int = 20
    bufferLength: int = 55
    buffer: deque[float] = field(init=False)
    atr: ATR = field(init=False)

    def __post_init__(self) -> None:
        self.buffer = deque(maxlen=self.bufferLength)
        self.atr = ATR(self.length)

    @property
    def current(self) -> float:
        # passthrough...
        return self.atr.current

    def update(self, price: float) -> float:
        self.buffer.append(price)
        high = max(self.buffer)
        low = min(self.buffer)

        return self.atr.update(high, low, price)


@dataclass(slots=True)
class TWEMA:
    """Time-Weighted EMA for when we have un-equal event arrival, but we want to still collect events-over-time.

    (e.g. we can't just have an EMA of "last N data points" because datapoints could be arriving in 250ms or 3 s or 15 s or 300s...
    """

    # EMA durations in seconds
    # 3,900 seconds is 65 minutes; 23_400 seconds is 6.5 hours (390 minutes)
    durations: tuple[int, ...] = (
        0,  # we use '0' to mean "last value seen"
        15,
        30,
        60,
        120,
        180,
        300,
        900,
        1800,
        3_900,
        RTH_EMA_VWAP,
    )

    # actual EMA values
    # Dict is format [EMA duration in seconds, EMA value]
    emas: dict[int, float] = field(default_factory=dict)

    # metadata EMAs
    diffVWAP: dict[int, float] = field(default_factory=dict)
    diffVWAPLog: dict[int, float] = field(default_factory=dict)
    diffPrevLog: dict[int, float] = field(default_factory=dict)

    # metadata scores
    diffVWAPLogScore: float = 0.0
    diffPrevLogScore: float = 0.0

    # i put emas in ur emas
    diffVWAPLogScoreEMA: dict[int, float] = field(default_factory=dict)
    diffPrevLogScoreEMA: dict[int, float] = field(default_factory=dict)

    last_update: float = 0

    def __post_init__(self) -> None:
        # Just verify durations are ALWAYS sorted from smallest to largest
        self.durations = tuple(sorted(self.durations))

    def update(self, new_value: float | None, timestamp: float) -> None:
        if new_value is None:
            return

        if self.last_update == 0:
            self.last_update = timestamp

            for duration in self.durations:
                self.emas[duration] = new_value

            return

        time_diff = timestamp - self.last_update
        self.last_update = timestamp

        # update all EMAs
        # Use position 0 to store the current "live" input value without any adjustments.
        self.emas[0] = new_value

        # skip the 0th entry because we manully write into it

        for period in self.durations[1:]:
            value = self.emas[period]
            alpha = 1 - math.exp(-time_diff / period)
            last = alpha * new_value + (1 - alpha) * value
            self.emas[period] = last

        # now update difference EMAs:
        #  - price differences (from previous)
        #  - difference from VWAP (longest duration)
        #  - difference from previous

        # can't update logs of negative prices if we init weird
        # (things like NYSE-TICK have negative "price" ranges)
        # if last <= 0:
        #    return

        # VWAP vs. Current comparisons
        # loglast = math.log(last)
        self.diffVWAPLogScore = 0.0
        for k in self.durations:
            v = self.emas[k]

            # same check against negative prices here too...
            # if v <= 0:
            #     return

            # price difference VWAP
            self.diffVWAP[k] = v - last

            # log difference VWAP
            # dvl = 100 * (math.log(v) - loglast)
            dvl = 100 * ((v - last) / (last or 1))
            self.diffVWAPLog[k] = dvl

            if k > 0:
                self.diffVWAPLogScore += (1 / k) * dvl

        # Previous vs. Current comparisons
        # process differences from high to low, so reverse, because
        # we know the dict keys are _already_ in sorted order from lowest to highest.
        prev = 0.0
        self.diffPrevLogScore = 0.0
        for k in reversed(self.durations):
            # here = math.log(self.emas[k])
            here = self.emas[k]
            if not prev:
                prev = here
                continue

            # log difference vs previous EMA price
            dpl = 100 * ((here - prev) / (prev or 1))
            prev = here

            self.diffPrevLog[k] = dpl

            if k > 0:
                self.diffPrevLogScore += (1 / k) * dpl

        self.updateDiffEMAs(time_diff)

    def updateDiffEMAs(self, time_diff: float) -> None:
        prevLog = self.diffPrevLogScore
        vwapLog = self.diffVWAPLogScore

        # if first run, just set both of them then wait for more updates
        if not self.diffPrevLogScoreEMA:
            for duration in self.durations:
                self.diffPrevLogScoreEMA[duration] = prevLog
                self.diffVWAPLogScoreEMA[duration] = vwapLog

            return

        durationsWithoutZero = self.durations[1:]

        # update all EMAs
        for period in durationsWithoutZero:
            value = self.diffPrevLogScoreEMA[period]
            alpha = 1 - math.exp(-time_diff / period)
            self.diffPrevLogScoreEMA[period] = last = (
                alpha * prevLog + (1 - alpha) * value
            )

        for period in durationsWithoutZero:
            value = self.diffVWAPLogScoreEMA[period]
            alpha = 1 - math.exp(-time_diff / period)
            self.diffVWAPLogScoreEMA[period] = last = (
                alpha * vwapLog + (1 - alpha) * value
            )

        # set current values as position zero...
        self.diffPrevLogScoreEMA[0] = prevLog
        self.diffVWAPLogScoreEMA[0] = vwapLog

    def __getitem__(self, idx: int) -> float:
        return self.emas.get(idx, 0)

    def get(self, idx: int, default: float | None = None) -> float | None:
        return self.emas.get(idx, default)

    def rms(self) -> dict[int, float]:
        """Calculate RMS for each slice of the EMAs going higher and higher"""

        # verify order is correct for our math to work
        # This is just walking a [(lookback, ema)] list of stuff in depth-adjusted inputs all the way down.
        sema = sorted(self.emas.items(), reverse=True)

        def genscore(threshold):
            """Generate an adaptive end-start RMS score for each step of the EMA lookback"""
            use = list(filter(lambda x: x[0] <= threshold, sema))

            # if only one element matched, we can't compare it against itself, so the result is always zero.
            if len(use) == 1:
                return 0

            _idxs, emas = zip(*use)
            scores = rmsnorm(emas)
            return scores[-1] - scores[0]

        scores = {}
        for k, v in sema:
            scores[k] = genscore(k)

        return scores

    def logScoreFrame(self, digits: int = 2) -> pd.DataFrame:
        rms = self.rms()
        return pd.DataFrame(
            dict(
                prevlog={k: round(v, 4) for k, v in reversed(self.diffPrevLog.items())},
                prevscore={k: v * 1000 for k, v in self.diffPrevLogScoreEMA.items()},
                vwaplog={k: round(v, 4) for k, v in self.diffVWAPLog.items()},
                vwapscore={k: v * 1000 for k, v in self.diffVWAPLogScoreEMA.items()},
                ema={k: round(v, digits) for k, v in self.emas.items()},
                vwapdiff={k: round(v, digits) for k, v in self.diffVWAP.items()},
                rms={k: round(v, 6) for k, v in rms.items()},
            )
        )


def analyze_trend_strength(df, ema_col="ema", periods=None):
    """
    Analyzes the directional strength of a time series using multiple timeframes.

    Parameters:
    df: DataFrame with time series data
    ema_col: name of the EMA column to analyze
    periods: list of periods to compare (if None, uses all available rows)

    Returns:
    dict containing trend analysis and strength metrics
    """
    if periods is None:
        # Use all available periods except the last row (current)
        periods = df.index[1].tolist()

    # Calculate changes from current value
    current_value = df[ema_col].iloc[0]
    changes = {
        period: {
            "change": current_value - df[ema_col].loc[period],
            "pct_change": (
                (current_value - df[ema_col].loc[period]) / df[ema_col].loc[period]
            )
            * 100,
        }
        for period in periods
        if not pd.isna(df[ema_col].loc[period])
    }

    # Calculate trend metrics
    changes_array = np.array([v["change"] for v in changes.values()])

    # Overall trend strength metrics
    trend_metrics = {
        "direction": "UP" if np.mean(changes_array) > 0 else "DOWN",
        "strength": abs(np.mean(changes_array)),
        "consistency": np.mean(
            np.sign(changes_array) == np.sign(np.mean(changes_array))
        )
        * 100,
        # for some reason, mypy is reporting np.polyfit doesn't exist when it clearly does
        "acceleration": np.polyfit(range(len(changes_array)), changes_array, 1)[0],  # type: ignore
    }

    # Determine trend phase
    recent_direction = math.copysign(1, changes_array[:3].mean())  # Nearest 3 periods
    overall_direction = math.copysign(1, changes_array.mean())

    if recent_direction > 0 and overall_direction > 0:
        trend_phase = "STRONGLY UP"
    elif recent_direction < 0 and overall_direction < 0:
        trend_phase = "STRONGLY DOWN"
    elif recent_direction > 0 and overall_direction < 0:
        trend_phase = "TURNING UP"
    elif recent_direction < 0 and overall_direction > 0:
        trend_phase = "TURNING DOWN"
    else:
        trend_phase = "NEUTRAL"

    # Calculate trend components
    trend_components = {
        "short_term": math.copysign(1, changes_array[:3].mean()),  # Nearest 3 periods
        "medium_term": math.copysign(
            1, changes_array[: len(changes_array) // 2].mean()
        ),  # First half
        "long_term": math.copysign(1, changes_array.mean()),  # All periods
    }

    return {
        "trend_phase": trend_phase,
        "metrics": trend_metrics,
        "components": trend_components,
        "changes": changes,
    }


def generate_trend_summary(df, ema_col="ema"):
    """
    Generates a human-readable summary of the trend analysis.
    """
    analysis = analyze_trend_strength(df, ema_col)

    strength_desc = (
        "strong"
        if analysis["metrics"]["strength"] > 1
        else "moderate"
        if analysis["metrics"]["strength"] > 0.5
        else "weak"
    )
    consistency_desc = (
        "consistent"
        if analysis["metrics"]["consistency"] > 80
        else "moderately consistent"
        if analysis["metrics"]["consistency"] > 60
        else "inconsistent"
    )

    acceleration_desc = (
        "accelerating"
        if analysis["metrics"]["acceleration"] > 0.1
        else "decelerating"
        if analysis["metrics"]["acceleration"] < -0.1
        else "steady"
    )

    summary = f"The trend is currently in a {analysis['trend_phase']} phase with {strength_desc} momentum. "
    summary += f"The movement is {consistency_desc} across timeframes and is {acceleration_desc}. "

    # Add component analysis
    components = []
    if analysis["components"]["short_term"] > 0:
        components.append("short-term upward")
    if analysis["components"]["medium_term"] > 0:
        components.append("medium-term upward")
    if analysis["components"]["long_term"] > 0:
        components.append("long-term upward")
    if analysis["components"]["short_term"] < 0:
        components.append("short-term downward")
    if analysis["components"]["medium_term"] < 0:
        components.append("medium-term downward")
    if analysis["components"]["long_term"] < 0:
        components.append("long-term downward")

    summary += f"The trend shows {', '.join(components)} movement."

    return summary
