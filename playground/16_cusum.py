"""CUSUM: detecting a shift that no single prediction error shows.

    uv run python playground/16_cusum.py

Terms, with an everyday equivalent for each:
  prediction error     what was observed minus what the model predicted (a surprise). Each one here is
                       standardized: divided by its standard deviation while nothing has changed, so 1 is a
                       typical error (a z-score).
  shift                a lasting change in the errors' mean (a lean): from the change point on, every error leans
                       by SHIFT.
  CUSUM statistics     two cumulative sums of the errors (running totals): the upper one, S+, collects errors
                       leaning positive, and the lower one, S-, errors leaning negative.
  reference value k    subtracted from every error before it is added (also called the allowance or slack), so
                       errors from noise alone drain away; about half the smallest shift worth detecting.
  decision threshold h an alarm is raised when S+ or S- exceeds it.
  false alarm          an alarm when nothing has changed; the false-alarm rate is how often that happens.
  detection delay      samples from the change point to the alarm (how long it takes to notice).

From the change point, every error leans by SHIFT until the first alarm. The alarm makes the controller become
unsure and relearn its parameter estimates, so after it the errors centre on zero again. In the adaptive controller,
each sample would be one 50 ms learning point, and the prediction error the push it applied minus the push its
parameter estimates predicted for the acceleration it measured.

Prints when this stream's alarms went off, then the trade-off over many streams (false-alarm rate against detection
delay for several thresholds), saves the picture to outputs/cusum.png and opens it in a window (add --no-show to
skip that). A few seconds.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---- Things to change and re-run ---------------------------------------------
SHIFT = 1.0       # how far every prediction error leans after the change point, in standard deviations
ALLOWANCE = 0.5   # the reference value k: about half the smallest shift worth detecting
THRESHOLD = 5.0   # the decision threshold h
CHANGE_AT = 100   # the change point: the sample at which the shift begins
SAMPLES = 160     # samples in the stream that is drawn
SEED = 24         # which noise; change it to see another stream (this one has a typical detection delay)
# -------------------------------------------------------------------------------

THRESHOLDS = (2.0, 3.0, 4.0, 5.0, 8.0)  # the decision thresholds compared in the table, plus THRESHOLD
STREAMS = 300                            # streams per threshold, for the table
IN_CONTROL_SAMPLES = 1000                # samples per stream with no shift ("in control"), for the false-alarm rate
DELAY_SAMPLES = 2000                     # samples per stream starting at the change point; longer than any delay shown
OUTPUT = Path(__file__).resolve().parent.parent / "outputs" / "cusum.png"

# Colours: the two CUSUM statistics in the project's validated blue and orange; everything else in neutral greys.
SURFACE, INK, INK_SOFT, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
UPPER, LOWER = "#2a78d6", "#eb6834"


def main():
    parser = argparse.ArgumentParser(description="CUSUM on standardized prediction errors with one shift in them.")
    parser.add_argument("--no-show", action="store_true", help="save the picture without opening a window")
    args = parser.parse_args()

    # 1. One stream: standardized prediction errors that are pure noise until CHANGE_AT, then lean by SHIFT until the
    #    first alarm. That alarm makes the controller become unsure and relearn, so from then on its errors centre on
    #    zero again (in the controller relearning takes a few samples; here it is immediate).
    rng = np.random.default_rng(SEED)
    errors = rng.normal(0.0, 1.0, SAMPLES)
    errors[CHANGE_AT:] += SHIFT
    detected = next((a for a in cusum(errors, ALLOWANCE, THRESHOLD)[2] if a >= CHANGE_AT), None)
    if detected is not None:
        errors[detected + 1:] -= SHIFT

    # 2. Run CUSUM along it and say when the alarms went off.
    upper, lower, alarms = cusum(errors, ALLOWANCE, THRESHOLD)
    print_stream(alarms, detected)

    # 3. The trade-off over many streams: the false-alarm rate with no shift, against the detection delay.
    print_trade_off(np.random.default_rng(SEED + 1))

    draw(errors, upper, lower, alarms, detected)
    print(f"\npicture saved to {OUTPUT}")
    if not args.no_show:
        plt.show()


def cusum(errors, allowance, threshold):
    """The upper and lower CUSUM statistics along the errors, and the samples at which an alarm was raised."""
    upper = lower = 0.0
    uppers, lowers, alarms = [], [], []
    for i, error in enumerate(errors):
        upper = max(0.0, upper + error - allowance)  # S+: evidence that the errors lean positive
        lower = max(0.0, lower - error - allowance)  # S-: evidence that they lean negative
        uppers.append(upper)
        lowers.append(lower)
        if upper > threshold or lower > threshold:
            alarms.append(i)
            upper = lower = 0.0  # restart both, as the controller would after becoming unsure
    return np.array(uppers), np.array(lowers), alarms


def false_alarm_rate(threshold, rng):
    """Alarms per 1000 samples in streams with no shift, so every alarm is a false one (1000 / ARL0)."""
    count = sum(len(cusum(rng.normal(0.0, 1.0, IN_CONTROL_SAMPLES), ALLOWANCE, threshold)[2])
                for _ in range(STREAMS))
    return 1000 * count / (STREAMS * IN_CONTROL_SAMPLES)


def detection_delay(threshold, rng):
    """The average number of samples from the change point to the alarm, with both statistics starting at zero
    (ARL1)."""
    delays = []
    for _ in range(STREAMS):
        alarms = cusum(rng.normal(SHIFT, 1.0, DELAY_SAMPLES), ALLOWANCE, threshold)[2]
        delays.append(alarms[0] + 1 if alarms else DELAY_SAMPLES)
    return float(np.mean(delays))


def print_stream(alarms, detected):
    def listed(samples):
        return f"{len(samples)}" + (f" (at samples {', '.join(str(a) for a in samples)})" if samples else "")
    print(f"One stream of {SAMPLES} samples; from sample {CHANGE_AT} (the change point), every prediction error "
          f"leans by {SHIFT:+g} (in standard deviations).")
    print(f"Reference value k = {ALLOWANCE:g}, decision threshold h = {THRESHOLD:g}.")
    print(f"  false alarms before the change point: {listed([a for a in alarms if a < CHANGE_AT])}")
    if detected is None:
        print("  the shift was never detected")
        return
    print(f"  shift detected at sample {detected}: a detection delay of {detected - CHANGE_AT + 1} samples")
    print(f"  false alarms after relearning: {listed([a for a in alarms if a > detected])}")


def print_trade_off(rng):
    print(f"\nThe trade-off, over {STREAMS} streams per threshold (k = {ALLOWANCE:g}, shift {SHIFT:+g}):")
    print(f"{'threshold h':>13s}{'false alarms per 1000 samples':>33s}{'detection delay (samples)':>29s}")
    for threshold in sorted(set(THRESHOLDS) | {THRESHOLD}):
        mark = "   <- THRESHOLD" if threshold == THRESHOLD else ""
        print(f"{threshold:>13g}{false_alarm_rate(threshold, rng):>33.1f}{detection_delay(threshold, rng):>29.1f}{mark}")


def draw(errors, upper, lower, alarms, detected):
    samples = np.arange(len(errors))
    noticed = len(errors) if detected is None else detected
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True, facecolor=SURFACE,
                                      gridspec_kw={"height_ratios": (1, 1.2)})

    # Top: the prediction errors themselves. No single one tells noise from a shift.
    top.scatter(samples, errors, s=11, color=MUTED, linewidths=0)
    top.axhline(0, color=AXIS, linewidth=0.8)
    top.plot([CHANGE_AT, noticed], [SHIFT, SHIFT], color=INK_SOFT, linewidth=1.4, linestyle=(0, (4, 2)))
    top.set_ylabel("prediction error\n(standardized)")

    # Bottom: the two CUSUM statistics, the decision threshold, and the alarms.
    bottom.plot(samples, upper, color=UPPER, linewidth=1.8, label="upper CUSUM statistic $S^+$ (errors leaning positive)")
    bottom.plot(samples, lower, color=LOWER, linewidth=1.8, linestyle=(0, (5, 2)),
                label="lower CUSUM statistic $S^-$ (errors leaning negative)")
    bottom.axhline(THRESHOLD, color=INK_SOFT, linewidth=1, linestyle=":")
    bottom.annotate(f"decision threshold h = {THRESHOLD:g}", xy=(0, THRESHOLD), xytext=(2, 4),
                    textcoords="offset points", ha="left", va="bottom", color=INK_SOFT, fontsize=8.5)
    bottom.scatter(alarms, [max(upper[a], lower[a]) for a in alarms], marker="v", s=46, color=INK, zorder=3,
                   label="alarm")
    if detected is not None:
        bottom.annotate(f"alarm after a detection delay of {detected - CHANGE_AT + 1} samples:\nthe controller becomes "
                        "unsure and relearns", xy=(detected, max(upper[detected], lower[detected])), xytext=(10, -2),
                        textcoords="offset points", ha="left", va="top", color=INK, fontsize=8.5)
    bottom.set_ylim(0, 1.45 * max(THRESHOLD, upper.max(), lower.max()))
    bottom.set_ylabel("CUSUM statistic")
    bottom.set_xlabel("sample")
    bottom.legend(loc="upper left", frameon=False, labelcolor=INK_SOFT, fontsize=8.5)

    # Both: the detection delay, when the shift is real but not yet detected.
    for ax in (top, bottom):
        ax.axvspan(CHANGE_AT, noticed, color=AXIS, alpha=0.35, linewidth=0)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=INK_SOFT, labelsize=9)
        ax.yaxis.label.set_color(INK_SOFT)
        ax.xaxis.label.set_color(INK_SOFT)
    top.annotate("detection delay", xy=(CHANGE_AT, 1), xycoords=("data", "axes fraction"), xytext=(-4, -2),
                 textcoords="offset points", ha="right", va="top", color=INK_SOFT, fontsize=8.5)
    bottom.set_xlim(0, len(errors))

    fig.suptitle("CUSUM detects a shift that no single prediction error shows", x=0.08, ha="left", color=INK,
                 fontsize=12.5)
    fig.text(0.08, 0.93, f"Reference value k = {ALLOWANCE:g}, decision threshold h = {THRESHOLD:g}. From the change "
             f"point (sample {CHANGE_AT}), every error leans by {SHIFT:+g} (in standard deviations; dashed)\nuntil the "
             "alarm; the controller then relearns, so its errors centre on zero again.", ha="left", va="top",
             color=INK_SOFT, fontsize=9, linespacing=1.4)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.85, bottom=0.09, hspace=0.12)
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
