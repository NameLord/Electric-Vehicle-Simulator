"""UI layer for the EV simulator.

Modules:
    * :mod:`ui.formatters` - compact number formatting shared by every widget.
    * :mod:`ui.hud` - the live bottom-right dashboard.
    * :mod:`ui.garage` - the Tab menu that edits the vehicle build.

The UI layer sits *on top of* ``core`` and ``world`` and never modifies them: it
reads telemetry from :class:`core.vehicle.VehicleState` and rebuilds vehicles
through the public constructors of :class:`core.battery_pack.BatteryPack`,
:class:`core.motor.Motor` and :class:`core.vehicle.Vehicle`.

Names are re-exported lazily so that importing :mod:`ui` does not pull in pygame
or the core (which keeps ``python -m ui.garage`` and lightweight tooling quiet).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

__all__ = [
    "DASH",
    "fixed",
    "human",
    "human_units",
    "percent",
    "ratio",
    "signed",
    "DriveMode",
    "DRIVE_MODES",
    "REGEN_LEVELS",
    "HudModel",
    "Warning",
    "Garage",
    "GarageConfig",
    "GARAGE_SAVE_PATH",
    "SHARE_PREFIX",
    "SECTIONS",
    "build_vehicle",
    "estimate_build",
    "encode_share_code",
    "decode_share_code",
    "load_presets",
    "save_presets",
]

#: ``public name -> (module suffix, attribute name)``.
_LAZY: Dict[str, Tuple[str, str]] = {
    "DASH": ("formatters", "DASH"),
    "fixed": ("formatters", "fixed"),
    "human": ("formatters", "human"),
    "human_units": ("formatters", "human_units"),
    "percent": ("formatters", "percent"),
    "ratio": ("formatters", "ratio"),
    "signed": ("formatters", "signed"),
    "DriveMode": ("hud", "DriveMode"),
    "DRIVE_MODES": ("hud", "DRIVE_MODES"),
    "REGEN_LEVELS": ("hud", "REGEN_LEVELS"),
    "HudModel": ("hud", "HudModel"),
    "Warning": ("hud", "Warning"),
    "Garage": ("garage", "Garage"),
    "GarageConfig": ("garage", "GarageConfig"),
    "GARAGE_SAVE_PATH": ("garage", "GARAGE_SAVE_PATH"),
    "SHARE_PREFIX": ("garage", "SHARE_PREFIX"),
    "SECTIONS": ("garage", "SECTIONS"),
    "build_vehicle": ("garage", "build_vehicle"),
    "estimate_build": ("garage", "estimate_build"),
    "encode_share_code": ("garage", "encode_share_code"),
    "decode_share_code": ("garage", "decode_share_code"),
    "load_presets": ("garage", "load_presets"),
    "save_presets": ("garage", "save_presets"),
}


def __getattr__(name: str):
    """Import a public name on first use (PEP 562)."""
    try:
        module_name, attribute = _LAZY[name]
    except KeyError:  # pragma: no cover - standard attribute error path
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    """Include the lazy names in ``dir()``."""
    return sorted(set(globals()) | set(_LAZY))
