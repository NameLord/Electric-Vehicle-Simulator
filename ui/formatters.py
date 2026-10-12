"""Number formatting for the UI layer.

The dashboard shows raw SI values (volts, amps, watts) that span six orders of
magnitude, plus derived rates that are undefined until the car has actually
moved. Everything funnels through :func:`human` so the panel never shows a
17-digit float or a ``ZeroDivisionError``.

Running this module directly executes its unit tests::

    python -m ui.formatters
"""

from __future__ import annotations

import math
from typing import Optional

__all__ = [
    "human",
    "human_units",
    "signed",
    "fixed",
    "percent",
    "ratio",
    "DASH",
]

#: Placeholder shown when a quantity has no meaningful value yet.
DASH = "\u2014"

#: Magnitude suffixes, largest first. Each entry is ``(threshold, divisor, suffix)``.
_SUFFIXES = (
    (1_000_000_000.0, 1e9, "G"),
    (1_000_000.0, 1e6, "M"),
    (1_000.0, 1e3, "K"),
)


def human(n: float) -> str:
    """Format a number compactly, with an SI-style magnitude suffix.

    Rules (exact):

    * ``abs(n) < 1_000`` - plain integer-ish text, no decimals, e.g. ``999``.
    * ``abs(n) >= 1_000`` - one decimal plus ``K``, e.g. ``1.5K``.
    * ``abs(n) >= 1_000_000`` - one decimal plus ``M``, e.g. ``2.4M``.
    * ``abs(n) >= 1_000_000_000`` - one decimal plus ``G``, e.g. ``1.2G``.
    * The sign is preserved, so regen power reads as ``-1.2K``.
    * Values below 1000 never carry decimals; ``K``/``M``/``G`` always carry
      exactly one.

    Non-finite inputs (``inf``/``nan``, which the controller ratings can be)
    render as an em dash rather than raising or printing ``inf``.

    Args:
        n: Value to format. Anything convertible to ``float``.

    Returns:
        Compact string, never empty.
    """
    try:
        value = float(n)
    except (TypeError, ValueError):
        return DASH
    if not math.isfinite(value):
        return DASH

    magnitude = abs(value)
    for threshold, divisor, suffix in _SUFFIXES:
        if magnitude >= threshold:
            return f"{value / divisor:.1f}{suffix}"
    return str(int(value))


def human_units(n: float, unit: str) -> str:
    """``human(n)`` with a unit appended, e.g. ``-1.2K W``."""
    text = human(n)
    return text if text == DASH else f"{text} {unit}"


def signed(n: float, digits: int = 1, unit: str = "") -> str:
    """Always show an explicit sign (useful for regen current/power).

    Args:
        n: Value to format.
        digits: Decimal places.
        unit: Optional unit appended after a space.

    Returns:
        ``"+1.2 A"`` / ``"-1.2 A"`` / ``"—"`` for non-finite input.
    """
    try:
        value = float(n)
    except (TypeError, ValueError):
        return DASH
    if not math.isfinite(value):
        return DASH
    text = f"{value:+.{max(int(digits), 0)}f}"
    return f"{text} {unit}" if unit else text


def fixed(n: float, digits: int = 1, unit: str = "") -> str:
    """Fixed-point rendering with the same non-finite guard as :func:`human`."""
    try:
        value = float(n)
    except (TypeError, ValueError):
        return DASH
    if not math.isfinite(value):
        return DASH
    text = f"{value:.{max(int(digits), 0)}f}"
    return f"{text} {unit}" if unit else text


def percent(fraction: float, digits: int = 1) -> str:
    """Render a ``0..1`` fraction as a percentage string."""
    try:
        value = float(fraction)
    except (TypeError, ValueError):
        return DASH
    if not math.isfinite(value):
        return DASH
    return f"{value * 100.0:.{max(int(digits), 0)}f}%"


def ratio(numerator: float, denominator: float, digits: int = 1) -> str:
    """Safe division for display, returning :data:`DASH` when undefined.

    Used for ``Wh/km``, which is meaningless before the car has moved, and for
    regression-free cost per km figures.

    Args:
        numerator: Value on top.
        denominator: Value on the bottom.
        digits: Decimal places.

    Returns:
        Formatted quotient, or :data:`DASH` when the denominator is zero,
        negative or the quotient is not finite.
    """
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return DASH
    if not math.isfinite(num) or not math.isfinite(den) or den <= 0.0:
        return DASH
    return f"{num / den:.{max(int(digits), 0)}f}"


def _run_tests() -> int:
    """Run the formatter unit tests. Returns a process exit code."""
    failures = []

    def check(name: str, got: object, want: object) -> None:
        ok = got == want
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got!r}, want {want!r}")
        if not ok:
            failures.append(name)

    # --- below 1000: no decimals, truncating, sign preserved -----------
    check("zero", human(0), "0")
    check("small positive", human(7.8), "7")
    check("truncates toward zero", human(999.9), "999")
    check("negative truncates toward zero", human(-999.9), "-999")
    check("one below the K boundary", human(999), "999")

    # --- K / M / G boundaries ------------------------------------------
    check("K boundary is inclusive", human(1_000), "1.0K")
    check("K rounds to one decimal", human(1_234), "1.2K")
    check("K shows 1 decimal", human(1_250), "1.2K")
    check("K rounds up", human(1_299), "1.3K")
    check("K just below M", human(999_949), "999.9K")
    check("M boundary is inclusive", human(1_000_000), "1.0M")
    check("M value", human(2_450_000), "2.5M")
    check("G boundary is inclusive", human(1_000_000_000), "1.0G")
    check("G value", human(2_500_000_000), "2.5G")

    # --- negative values keep the sign (regen) -------------------------
    check("negative K", human(-1_200), "-1.2K")
    check("negative M", human(-3_400_000), "-3.4M")
    check("negative G", human(-1_100_000_000), "-1.1G")
    check("negative zero-ish", human(-0.4), "0")

    # --- non-finite guards ---------------------------------------------
    check("inf is a dash", human(float("inf")), DASH)
    check("nan is a dash", human(float("nan")), DASH)
    check("non-numeric is a dash", human("nope"), DASH)  # type: ignore[arg-type]

    # --- helpers --------------------------------------------------------
    check("human_units", human_units(-1_200, "W"), "-1.2K W")
    check("human_units dash", human_units(float("nan"), "W"), DASH)
    check("signed positive", signed(1.2, 1, "A"), "+1.2 A")
    check("signed negative", signed(-1.2, 1, "A"), "-1.2 A")
    check("signed no unit", signed(-1.2), "-1.2")
    check("fixed", fixed(12.345, 2), "12.35")
    check("percent", percent(0.735, 1), "73.5%")
    check("ratio normal", ratio(1_500.0, 10.0, 1), "150.0")
    check("ratio undefined at zero distance", ratio(5.0, 0.0), DASH)
    check("ratio undefined when negative", ratio(5.0, -1.0), DASH)

    print()
    if failures:
        print(f"RESULT: {len(failures)} formatter check(s) failed -> {failures}")
        return 1
    print("RESULT: all formatter checks passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_run_tests())
