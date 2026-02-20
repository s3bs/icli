# ATR: Convert from Count-Based to Time-Based

## Problem

`ATRLive` (in `icli/engine/technicals.py`) is purely count-based. It maintains a
fixed-size `deque` of price samples and computes high/low over the entire buffer.
The "time" labels (e.g. `atrs[180]` = "3 minute ATR") assume ticks arrive at a
steady 4 Hz (250 ms), but actual tick rates vary widely:

- **Active futures (ES, NQ) during RTH:** ~4 Hz — assumption holds
- **Indexes (SPX, VIX):** updates every few seconds to minutes — a "3 min" ATR
  can span 30+ minutes of wall time
- **Pre/post-market:** tick rate drops significantly
- **Missed ticks (network lag, API throttling):** buffer just has fewer entries,
  no awareness of the time gap

Meanwhile, the EMA system (`TWEMA` in the same file) is already time-weighted via
`alpha = 1 - exp(-time_diff / period)` and handles irregular arrivals correctly.

## Current Implementation

```python
# helpers.py — ATR init (line ~210)
for lookback in (90, 120, 180, 300, 420, 600, 840, 900, 1260, 1800):
    self.atrs[lookback] = ATRLive(
        int(lookback / 0.25),        # buffer length: assumes 4 Hz
        int(lookback / 2 / 0.25)     # ATR decay length: half the buffer
    )

# technicals.py — ATRLive.update()
def update(self, price: float) -> float:
    self.buffer.append(price)        # fixed-size deque, no timestamp
    high = max(self.buffer)
    low = min(self.buffer)
    return self.atr.update(high, low, price)
```

The call site (`processTickerUpdate` in `helpers.py`) has access to
`self.ticker.timestamp` — the IBKR-provided quote timestamp — but does not pass
it to the ATR.

## Proposed Fix

Make `ATRLive` time-aware:

1. **Change buffer type:** `deque[float]` → `deque[tuple[float, float]]` storing
   `(price, timestamp)` pairs.

2. **Accept timestamp in update:** `update(self, price, timestamp)` instead of
   just `update(self, price)`.

3. **Time-based pruning:** On each update, drop entries older than the configured
   time window (e.g. 180 seconds) rather than relying on `maxlen`.

4. **Compute high/low from time window:** `max`/`min` over only the entries
   within the actual time range.

5. **Scale ATR decay:** The inner `ATR.length` currently uses a fixed sample
   count for its RMA decay. Options:
   - Keep count-based decay but now the count reflects actual samples within the
     time window (self-adjusting).
   - Or switch to a time-weighted RMA similar to TWEMA's approach.

## Call Site Change

```python
# helpers.py processTickerUpdate() — pass timestamp
for atr in self.atrs.values():
    atr.update(current, self.ticker.timestamp)
```

## Key Files

- `icli/engine/technicals.py` — `ATR` and `ATRLive` classes
- `icli/helpers.py` — `ITicker.__init_iticker__()` (ATR init), `processTickerUpdate()` (ATR update)
- `icli/engine/toolbar.py` — ATR display (`atrs[180].atr.current`)
- `icli/cmds/utilities/info.py` — prints all ATR values
- `tests/baseline/test_tinyalgo.py` — `TestATR` and `TestATRLive` test classes

## Considerations

- This is a behavioral change to a core indicator — all ATR values will shift.
- Low-volume instruments will see the biggest difference (shorter effective
  lookback currently inflated by slow tick rate).
- The buffer `max()`/`min()` over tuples needs a key or list comprehension, but
  buffer sizes are small (hundreds of entries) — negligible on the hot path.
- Tests in `test_tinyalgo.py` will need updating to pass timestamps.
