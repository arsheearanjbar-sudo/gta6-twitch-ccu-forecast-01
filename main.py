"""Forecast a big game launch's Twitch viewers hour by hour, from its first hour.

Model (original work, standard library only, deterministic):
  The forecast anchors on the case's hour-1 print p1.  From the case's
  comparables -- past big launches with full release-day hourly curves --
  it learns, for each hour h = 2..24, how the log ratio ln(viewers_h / p1)
  depends on the launch's hour-1 size ln(p1):

      ln(viewers_h / p1)  ~  a_h + b_h * (ln(p1) - mean ln(p1))

  fitted by ordinary least squares per hour over the comparable launches
  that look like this one.  The spread s_h is the residual standard
  deviation of that regression, so the five quantiles are

      q_tau(h) = p1 * exp(mu_h + z_tau * s_h)

  with z the standard normal's quantiles at the scored levels.

Why the regression, and not a flat median ratio: launches whose hour 1 is
small are the ones whose audience is still asleep or not yet unlocked
(the US unlocks hours after the UTC day starts), so they ramp the hardest
-- across the comparables, log(h1) correlates about -0.8 with the log
peak-to-hour-1 ratio.  A flat median ratio splits the difference and is
wrong for both small and large hour-1 prints; the regression conditions
the shape on what hour 1 actually says.

Which comparables train the fit: only launches whose hour 1 is a genuine
early hour of a rolling-midnight launch -- tier "comparable" in the unlock
notes (no early-access contamination, no free-to-play weak analogues, and
not the target's own pre-release category history), with hour-1 viewers
between 5% and 80% of that launch's own release-day peak.  That drops
simultaneous-global launches (hour 1 is pre-launch noise, ratios of
hundreds) and launches whose hour 1 is already the peak (flat curves).
The target launch itself is never in its own training set: a practice
case's comparables already leave the practice launches out, and the fit
only ever reads the case file it is given.

Everything is computed from the one case file: no network, no files
outside it, no randomness, so two runs write identical bytes.
"""

import json
import math
import os
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Spread of a log-ratio forecast: residual std, kept inside these bounds.
SIGMA_FLOOR = 0.25
SIGMA_CAP = 1.6
# A training launch must have hour-1 viewers in this band of its own peak.
PEAK_LO = 0.05
PEAK_HI = 0.8
# Fewer usable training launches than this: fall back to a pooled median.
MIN_TRAIN = 8
# Fallback spread when a per-hour fit has too little data.
FALLBACK_SIGMA = 0.5


def parse_time(text):
    """ISO 8601 timestamp (with a trailing Z) as an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def quantile(values, p):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return float("nan")
    pos = p * (n - 1)
    lo = int(pos)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[min(lo + 1, n - 1)] * frac


def round2(value):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    rounded = float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    return int(rounded) if rounded.is_integer() else rounded


def tier_of(comparable):
    """The research tier in a comparable's unlock notes, or 'ok' when unnamed."""
    notes = comparable.get("unlock") or ""
    if "early-access-contaminated" in notes:
        return "ea"
    if "Tier: weak" in notes:
        return "weak"
    if "Not a launch" in notes:
        return "target-history"
    return "ok"


def release_day_curve(comparable):
    """The comparable's 24 release-day hourly viewer counts (None when absent)."""
    day_start = parse_time(comparable["day_start"])
    hours = [None] * 24
    for point in comparable.get("hourly") or []:
        try:
            offset = int((parse_time(point["from"]) - day_start).total_seconds() // 3600)
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= offset < 24:
            viewers = point.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                hours[offset] = float(viewers)
    return hours


def training_points(comparables):
    """Usable training launches: (log hour-1 viewers, {hour: log ratio})."""
    points = []
    for comparable in comparables or []:
        if tier_of(comparable) != "ok":
            continue
        curve = release_day_curve(comparable)
        first = curve[0]
        if not first:
            continue
        peak = max((v for v in curve if v), default=None)
        if not peak:
            continue
        share = first / peak
        if share < PEAK_LO or share > PEAK_HI:
            continue
        ratios = {}
        for hour in range(2, 25):
            value = curve[hour - 1]
            if value:
                ratios[hour] = math.log(value / first)
        if ratios:
            points.append((math.log(first), ratios))
    return points


def fit_hour(points, hour, centre):
    """Least-squares fit of one hour's log ratio on log hour-1 size.

    Returns (median log ratio at the target, log spread).  Falls back to a
    pooled median with an interquartile spread when the hour has too few
    training points, and to a flat wide spread when it has almost none.
    """
    rows = [(x - centre, ratios[hour]) for x, ratios in points if hour in ratios]
    if len(rows) >= MIN_TRAIN:
        n = len(rows)
        mean_x = sum(a for a, _ in rows) / n
        mean_y = sum(b for _, b in rows) / n
        sxx = sum((a - mean_x) ** 2 for a, _ in rows)
        slope = (
            sum((a - mean_x) * (b - mean_y) for a, b in rows) / sxx if sxx > 0 else 0.0
        )
        intercept = mean_y - slope * mean_x
        residuals = [b - (intercept + slope * a) for a, b in rows]
        spread = math.sqrt(sum(r * r for r in residuals) / len(residuals))
        return intercept, slope, spread
    if len(rows) >= 5:
        values = [b for _, b in rows]
        spread = (quantile(values, 0.75) - quantile(values, 0.25)) / 1.349
        return median(values), 0.0, spread
    if rows:
        return median([b for _, b in rows]), 0.0, FALLBACK_SIGMA
    return 0.0, 0.0, FALLBACK_SIGMA


def hour_bounds(points, hour):
    """The observed range of one hour's log ratios across training launches."""
    values = [ratios[hour] for _, ratios in points if hour in ratios]
    if len(values) < 5:
        return None
    return min(values), max(values)


def forecast_rows(p1, comparables):
    """The 23 forecast rows for one case, from its hour-1 print."""
    p1 = max(1.0, float(p1))
    points = training_points(comparables)
    centre = sum(x for x, _ in points) / len(points) if points else 0.0
    # Never extrapolate the regression past the training launches' own
    # hour-1 sizes: beyond them the line keeps falling (or rising) to
    # ratios no launch ever showed.  Clamp to the observed range, so a
    # record-breaking first hour forecasts like the biggest launches
    # seen, not like a line drawn past them.
    if points:
        low = min(x for x, _ in points)
        high = max(x for x, _ in points)
        target_x = min(high, max(low, math.log(p1))) - centre
    else:
        target_x = math.log(p1) - centre
    rows = []
    for hour in range(2, 25):
        intercept, slope, spread = fit_hour(points, hour, centre)
        middle = intercept + slope * target_x
        # And never predict a median ratio outside the range the training
        # launches actually showed at that hour.
        bounds = hour_bounds(points, hour)
        if bounds is not None:
            middle = min(bounds[1], max(bounds[0], middle))
    rows = []
    for hour in range(2, 25):
        intercept, slope, spread = fit_hour(points, hour, centre)
        middle = intercept + slope * target_x
        spread = min(SIGMA_CAP, max(SIGMA_FLOOR, spread))
        quantiles = []
        for z in Z:
            value = p1 * math.exp(middle + z * spread)
            value = min(1e9, max(1.0, value))
            quantiles.append(round2(value))
        for i in range(1, 5):
            quantiles[i] = max(quantiles[i], quantiles[i - 1])
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def hour_one_print(case):
    """The case's hour-1 print, with fallbacks if it is missing or unusable."""
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            viewers = row.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                return float(viewers)
    hourly = case.get("history_hourly") or []
    for row in reversed(hourly):
        viewers = row.get("viewers")
        if isinstance(viewers, (int, float)) and viewers > 0:
            return float(viewers)
    return 1.0


def forecast_case(case):
    return forecast_rows(hour_one_print(case), case.get("comparables"))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
