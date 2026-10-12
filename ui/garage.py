"""In-game garage: edit the vehicle build, preview it, save and share it.

The garage is a *builder*, not a physics engine. Every editable physics field is
written into the same JSON schema that :class:`core.vehicle.Vehicle` already
reads, and every derived number is read back out of
:class:`core.battery_pack.BatteryPack` / :class:`core.motor.Motor` rather than
recomputed:

===============  =====================================================
UI field         Core schema key
===============  =====================================================
Mass             ``chassis.mass_kg``
Wheelbase        ``chassis.wheelbase_m``
Track            ``chassis.track_width_front_m`` / ``..._rear_m``
Weight split     ``chassis.cg_to_front_axle_m`` / ``cg_to_rear_axle_m``
Drag Cd          ``chassis.drag_coefficient``
Frontal area     ``chassis.frontal_area_m2``
CG height        ``chassis.cg_height_m``
Chemistry        ``batteries.json`` key
S / P            :class:`BatteryPack` ``S`` / ``P``
Motor            ``motors.json`` id
Gear ratio       ``drivetrain.gear_ratio``
Drive layout     ``wheels.driven_axle``
Grip level       ``tires.mu_base``
Cornering stiff. ``tires.cornering_stiffness_N_per_rad``
Long. stiffness  ``tires.longitudinal_stiffness_N_per_slip``
Rolling res.     ``wheels.rolling_resistance_coeff``
===============  =====================================================

Fields that the core schema has *no* key for (body length/width, sprite colour,
tire width) are kept in ``build`` / ``style`` sections that the core ignores.
They are cosmetic or informational only and are never passed to the physics -
see the README table. Two derived conversions the garage does perform are pure
algebra on the core's own formulas, both documented at the point of use:

* wheelbase + weight split -> ``cg_to_front_axle_m`` / ``cg_to_rear_axle_m``,
  which is the core's static axle-load relation inverted;
* nothing else. Torque, power, efficiency, energy and pack scalars all come
  from the core objects.

Estimates (:func:`estimate_build`) use the core's torque/power/efficiency/energy
APIs inside a documented single-track point-mass model. They are labelled
estimates everywhere they appear and are never fed into the simulation.

Running this module directly executes its self-test::

    python -m ui.garage
"""

from __future__ import annotations

import base64
import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pygame

from core.battery_pack import BatteryPack, available_chemistries
from core.motor import Motor, available_motors
from core.vehicle import GRAVITY, Vehicle

from .formatters import fixed, human, human_units, percent, ratio

__all__ = [
    "GARAGE_SAVE_PATH",
    "SHARE_PREFIX",
    "FieldSpec",
    "Section",
    "SECTIONS",
    "GarageConfig",
    "BuildEstimate",
    "estimate_build",
    "build_vehicle",
    "load_presets",
    "save_presets",
    "encode_share_code",
    "decode_share_code",
    "Garage",
]

#: Where named presets live, relative to the repository root.
GARAGE_SAVE_PATH: Path = (
    Path(__file__).resolve().parents[1] / "save" / "garage_save.json"
)

#: Fallback location used to paste a share code when the clipboard is
#: unavailable (which is common on headless or DRM/KMS setups).
IMPORT_CODE_PATH: Path = (
    Path(__file__).resolve().parents[1] / "save" / "import_code.txt"
)

#: Prefix that identifies a share code and pins its schema version.
SHARE_PREFIX = "EVSG1."

#: Gravity used by the estimates, taken from the core so they cannot disagree.
_G = GRAVITY

#: Body colours offered by the CHASSIS section (cosmetic).
STYLE_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "Azure": (52, 120, 200),
    "Crimson": (196, 84, 72),
    "Racing Green": (46, 140, 96),
    "Graphite": (124, 128, 138),
    "Sunburst": (232, 176, 72),
    "Violet": (142, 112, 220),
}

#: Drive layouts, mapped onto the core's ``wheels.driven_axle`` values.
DRIVE_LAYOUTS: Tuple[Tuple[str, str], ...] = (
    ("RWD", "rear"),
    ("FWD", "front"),
    ("AWD (dual motor)", "all"),
)


# ======================================================================
# Field schema
# ======================================================================
@dataclass(frozen=True)
class FieldSpec:
    """One editable row in the garage.

    Attributes:
        key: Dotted path into the flat value map, e.g. ``"chassis.mass_kg"``.
            The path is also the destination inside the vehicle config, so the
            UI cannot invent a key the core does not read.
        label: Row caption.
        kind: ``"float"``, ``"int"`` or ``"choice"``.
        minimum: Lower clamp for numeric kinds.
        maximum: Upper clamp for numeric kinds.
        step: Increment for numeric kinds (``SHIFT`` multiplies it by 10).
        unit: Suffix shown after the value.
        choices: Allowed values for ``"choice"`` kinds.
        note: Extra explanation shown next to the description line.
    """

    key: str
    label: str
    kind: str = "float"
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 1.0
    unit: str = ""
    choices: Tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class Section:
    """A tab of the garage with its own rows."""

    title: str
    description: str
    fields: Tuple[FieldSpec, ...]


SECTIONS: Tuple[Section, ...] = (
    Section(
        "CHASSIS",
        "Chassis geometry and aerodynamics (core chassis.* keys).",
        (
            FieldSpec("chassis.mass_kg", "Mass", "float", 500.0, 3200.0, 25.0, "kg",
                      note="Vehicle mass the physics integrates (chassis.mass_kg)."),
            FieldSpec("chassis.wheelbase_m", "Wheelbase", "float", 1.8, 3.8, 0.05, "m",
                      note="cg_to_front_axle_m + cg_to_rear_axle_m must equal this."),
            FieldSpec("chassis.weight_distribution_front_pct", "Weight front",
                      "float", 30.0, 70.0, 0.5, "%",
                      note="Inverts the core's static axle-load relation into "
                           "cg_to_front_axle_m / cg_to_rear_axle_m."),
            FieldSpec("chassis.track_width_front_m", "Track front", "float",
                      1.1, 2.0, 0.01, "m"),
            FieldSpec("chassis.track_width_rear_m", "Track rear", "float",
                      1.1, 2.0, 0.01, "m"),
            FieldSpec("chassis.drag_coefficient", "Drag Cd", "float", 0.15, 0.80,
                      0.01, ""),
            FieldSpec("chassis.frontal_area_m2", "Frontal area", "float", 1.0, 3.6,
                      0.05, "m2"),
            FieldSpec("chassis.cg_height_m", "CG height", "float", 0.25, 0.80,
                      0.01, "m",
                      note="Drives longitudinal and lateral load transfer."),
            FieldSpec("chassis.yaw_inertia_kg_m2", "Yaw inertia (est)", "float",
                      800.0, 4500.0, 50.0, "kgm2",
                      note="Chassis yaw moment of inertia; no body-dimension "
                           "input exists in the schema so it is its own row."),
            FieldSpec("style.body_length_m", "Body length", "float", 2.0, 6.0,
                      0.05, "m", note="Cosmetic: the sprite outline in main.py only."),
            FieldSpec("style.body_width_m", "Body width", "float", 1.0, 2.6,
                      0.05, "m", note="Cosmetic: the sprite outline in main.py only."),
            FieldSpec("style.color", "Colour", "choice", choices=tuple(STYLE_COLOURS),
                      note="Cosmetic body colour."),
        ),
    ),
    Section(
        "BATTERY",
        "Cell chemistry and S/P. Every number below is read back from the pack.",
        (
            FieldSpec("build.chemistry", "Chemistry", "choice",
                      choices=tuple(available_chemistries()),
                      note="Key from config/batteries.json."),
            FieldSpec("build.series", "Series (S)", "int", 1, 220, 1, "cells"),
            FieldSpec("build.parallel", "Parallel (P)", "int", 1, 60, 1, "strings"),
        ),
    ),
    Section(
        "MOTOR",
        "Traction motor. Envelope and efficiency come from config/motors.json.",
        (
            FieldSpec("build.motor_id", "Motor", "choice",
                      choices=tuple(available_motors()),
                      note="Key from config/motors.json."),
        ),
    ),
    Section(
        "DRIVETRAIN",
        "Gear ratio and which axle the motor drives (core drivetrain.*/wheels.*).",
        (
            FieldSpec("drivetrain.gear_ratio", "Gear ratio", "float", 2.0, 16.0,
                      0.1, ":1"),
            FieldSpec("drivetrain.final_drive_efficiency", "Final drive eff",
                      "float", 0.70, 1.0, 0.01, ""),
            FieldSpec("wheels.driven_axle", "Drive layout", "choice",
                      choices=tuple(label for label, _ in DRIVE_LAYOUTS),
                      note="This model has one motor, so 'AWD (dual motor)' maps "
                           "onto wheels.driven_axle = all."),
            FieldSpec("drivetrain.max_friction_brake_torque_Nm_per_wheel",
                      "Brake torque/wheel", "float", 200.0, 4000.0, 50.0, "Nm"),
            FieldSpec(
                "drivetrain.traction_control", "Traction control", "choice",
                choices=("on", "off"),
                note="Core slip regulator; off makes wheelspin much easier to trigger.",
            ),
        ),
    ),
    Section(
        "TIRES",
        "Grip and stiffness (core tires.* and wheels.rolling_resistance_coeff).",
        (
            FieldSpec("tires.mu_base", "Grip level (mu)", "float", 0.40, 1.60,
                      0.01, ""),
            FieldSpec("tires.cornering_stiffness_N_per_rad", "Cornering stiff",
                      "float", 12_000.0, 80_000.0, 500.0, "N/rad"),
            FieldSpec("tires.longitudinal_stiffness_N_per_slip", "Long. stiff",
                      "float", 15_000.0, 120_000.0, 1_000.0, "N/slip"),
            FieldSpec("wheels.rolling_resistance_coeff", "Rolling res.", "float",
                      0.004, 0.040, 0.001, ""),
            FieldSpec("style.tire_width_mm", "Tire width", "int", 135, 355, 5, "mm",
                      note="Info only: the core tire model is parameterised by "
                           "stiffness and mu, not by section width."),
        ),
    ),
    Section(
        "SAVE / LOAD",
        "Named presets in save/garage_save.json, plus base64 share codes.",
        (),
    ),
)


# ======================================================================
# Estimates
# ======================================================================
@dataclass(frozen=True)
class BuildEstimate:
    """Predicted performance for a build.

    Produced by :func:`estimate_build`. All of it is a *prediction*, not a
    measurement: it uses the core's torque envelope, efficiency map and pack
    energy, but replaces the two-track transient model with a point mass.
    """

    top_speed_kmh: float
    zero_to_100_s: float
    range_km: float
    consumption_Wh_per_km: float
    cruise_kmh: float
    tractive_force_N: float
    power_limit_kW: float
    assumptions: Tuple[str, ...] = ()


def _nest(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Expand ``{"a.b": 1}`` into ``{"a": {"b": 1}}``."""
    nested: Dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split(".")
        cursor = nested
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return nested


def _flatten(prefix: str, source: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a nested mapping into dotted keys."""
    flat: Dict[str, Any] = {}
    for key, value in source.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(path, value))
        else:
            flat[path] = value
    return flat


def build_vehicle(
    build: "GarageConfig",
    *,
    initial_soc: Optional[float] = 1.0,
) -> Vehicle:
    """Rebuild a :class:`core.vehicle.Vehicle` from a garage configuration.

    The pack and motor are constructed through their public constructors, and
    the chassis/tire/drivetrain sections are handed straight to
    ``Vehicle(vehicle_config=...)``. No physics is re-implemented here.

    Args:
        build: The configuration to instantiate.
        initial_soc: SOC for the fresh pack (``None`` keeps the pack default).

    Returns:
        A ready-to-drive vehicle.

    Raises:
        ValueError: If the core rejects the combination (an unknown chemistry,
            an S/P mismatch, or a wheelbase that does not match the axle
            distances).
    """
    config = build.to_vehicle_config()
    pack = BatteryPack(
        int(config["build"]["series"]),
        int(config["build"]["parallel"]),
        str(config["build"]["chemistry"]),
        initial_soc=1.0 if initial_soc is None else float(initial_soc),
    )
    motor = Motor(str(config["build"]["motor_id"]))
    return Vehicle(
        pack,
        motor,
        initial_soc=initial_soc,
        ambient_temp_C=float(config["physics"].get("ambient_temp_C", 25.0)),
        vehicle_config=config,
    )


def estimate_build(
    config: Dict[str, Any],
    motor: Motor,
    pack: BatteryPack,
    *,
    cruise_kmh: float = 90.0,
    soc: float = 1.0,
) -> BuildEstimate:
    """Predict top speed, 0-100 km/h and range for a build.

    Model (documented so the numbers can be checked by hand):

    * Point mass ``m`` = ``chassis.mass_kg`` - the *same* mass the core
      integrates. Note the core does **not** add pack or motor mass to the
      vehicle, so the estimate must not either or it would disagree with the
      simulation.
    * Road load ``F = 0.5*rho*Cd*A*v^2 + m*g*crr``.
    * Tractive force ``F_max(v) = T_motor(rpm)*gear*eff / R`` with ``rpm`` from
      the wheel speed, the torque from :meth:`core.motor.Motor.torque_at`, and
      an electrical cap ``T <= P_pack_available * eff / omega`` using
      :meth:`core.motor.Motor.efficiency_at_torque` and
      :meth:`core.battery_pack.BatteryPack.available_discharge_power`.
    * Traction cap ``F <= mu * m * g * driven_load_fraction``, with the driven
      fraction taken from the same static axle-load relation the core uses.
    * Top speed: first speed where ``F_max == road load``.
    * 0-100: explicit Euler on ``dv/dt = (F - road load)/m``.
    * Range: cruise at ``cruise_kmh``, wheel power from the road load, divided
      by :meth:`core.motor.Motor.efficiency_at_torque`, then
      ``range = pack_usable_energy_Wh / Wh_per_km``.

    Deliberately ignored: tire slip and weight transfer, drivetrain inertia,
    thermal derating over time, wind, HVAC and any accessory load.

    Args:
        config: Full vehicle config (the same mapping the core is given).
        motor: Motor to read the envelope from.
        pack: Pack to read the power and energy limits from.
        cruise_kmh: Reference cruise speed for the consumption figure.
        soc: SOC assumed for the pack power limit.

    Returns:
        A :class:`BuildEstimate`.
    """
    chassis = config["chassis"]
    wheels = config["wheels"]
    drivetrain = config["drivetrain"]
    tires = config["tires"]

    mass = float(chassis["mass_kg"])
    radius = float(wheels["radius_m"])
    gear = float(drivetrain["gear_ratio"])
    final_drive_eff = float(drivetrain["final_drive_efficiency"])
    gain = gear * final_drive_eff
    rho = float(chassis["air_density_kg_m3"])
    cd = float(chassis["drag_coefficient"])
    area = float(chassis["frontal_area_m2"])
    crr = float(wheels["rolling_resistance_coeff"])
    mu = float(tires["mu_base"])

    axle = str(wheels.get("driven_axle", "rear")).lower()
    wheelbase = float(chassis["wheelbase_m"])
    rear_fraction = float(chassis["cg_to_rear_axle_m"]) / max(wheelbase, 1e-6)
    if axle in ("all", "awd", "4wd"):
        driven_fraction = 1.0
    elif axle in ("front", "fwd"):
        driven_fraction = 1.0 - rear_fraction
    else:
        driven_fraction = rear_fraction

    pack_power_W = float(pack.available_discharge_power(soc=soc))
    traction_limit_N = mu * mass * _G * driven_fraction

    def road_load(v: float) -> float:
        return 0.5 * rho * cd * area * v * v + mass * _G * crr

    def tractive(v: float) -> float:
        """Available tractive force at speed ``v`` (m/s)."""
        speed = max(v, 0.05)
        rpm = speed / radius * gear * 60.0 / (2.0 * math.pi)
        torque = float(motor.torque_at(rpm))
        omega = max(rpm * 2.0 * math.pi / 60.0, 1e-6)
        if pack_power_W > 0.0:
            eff = float(motor.efficiency_at_torque(rpm, torque))
            torque = min(torque, pack_power_W * eff / omega)
        force = torque * gain / radius
        return min(force, traction_limit_N)

    # -- top speed: first crossing of available force and road load ----
    top_speed = 0.0
    step = 0.1
    speed = step
    limit = 400.0
    while speed < limit:
        if tractive(speed) <= road_load(speed):
            lo, hi = speed - step, speed
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if tractive(mid) > road_load(mid):
                    lo = mid
                else:
                    hi = mid
            top_speed = 0.5 * (lo + hi)
            break
        speed += step
    else:
        top_speed = limit

    # -- 0-100 km/h ----------------------------------------------------
    target = 100.0 / 3.6
    dt = 0.005
    v = 0.0
    elapsed = 0.0
    zero_to_100 = float("inf")
    while elapsed < 60.0:
        net = tractive(v) - road_load(v)
        if net <= 0.0 and v > 0.1:
            break
        v += net / mass * dt
        elapsed += dt
        if v >= target:
            zero_to_100 = elapsed
            break

    # -- cruise consumption and range ----------------------------------
    v_ref = max(float(cruise_kmh) / 3.6, 1.0)
    force = road_load(v_ref)
    rpm = v_ref / radius * gear * 60.0 / (2.0 * math.pi)
    torque = force * radius / max(gain, 1e-9)
    eff = max(float(motor.efficiency_at_torque(rpm, torque)), 0.05)
    electrical_W = force * v_ref / eff
    wh_per_km = electrical_W / (3.6 * v_ref)
    range_km = pack.pack_usable_energy_Wh / wh_per_km if wh_per_km > 0 else 0.0

    assumptions = (
        f"point mass {mass:.0f} kg (core integrates chassis mass only; "
        f"pack {pack.pack_mass_kg:.0f} kg + motor {motor.mass_kg:.0f} kg are not "
        "added to it)",
        f"driven axle load fraction {driven_fraction * 100:.0f}% "
        f"(traction cap {traction_limit_N / 1000.0:.1f} kN)",
        f"pack power limit {pack_power_W / 1000.0:.0f} kW at SOC {soc * 100:.0f}%",
        "no tire slip, no load transfer, no thermal derating",
    )
    return BuildEstimate(
        top_speed_kmh=top_speed * 3.6,
        zero_to_100_s=zero_to_100,
        range_km=range_km,
        consumption_Wh_per_km=wh_per_km,
        cruise_kmh=float(cruise_kmh),
        tractive_force_N=tractive(0.0),
        power_limit_kW=pack_power_W / 1000.0,
        assumptions=assumptions,
    )


# ======================================================================
# Configuration model
# ======================================================================
#: ``wheels.driven_axle`` value -> UI label.
_AXLE_TO_LABEL = {value: label for label, value in DRIVE_LAYOUTS}
_LABEL_TO_AXLE = dict(DRIVE_LAYOUTS)

#: Default build, matching :func:`core.vehicle.create_demo_vehicle`.
DEFAULT_BUILD: Dict[str, Any] = {
    "chemistry": "Li-Ion (NMC)",
    "series": 92,
    "parallel": 3,
    "motor_id": "pmsm-150kw",
}

#: The only editable row whose *key* does not exist in the core schema: it is a
#: UI presentation of the weight split, converted into ``cg_to_front_axle_m`` /
#: ``cg_to_rear_axle_m``. It must therefore never appear in a config handed to
#: the core.
UI_ONLY_KEYS = frozenset({"chassis.weight_distribution_front_pct"})

#: Rows that are excluded from the blind value merge because they need an
#: explicit conversion first. Unlike :data:`UI_ONLY_KEYS`, the other two *are*
#: core keys - only their JSON representation differs:
#:
#: * ``weight_distribution_front_pct`` -> ``cg_to_front_axle_m`` / ``cg_to_rear_axle_m``
#: * ``wheels.driven_axle``             -> ``rear`` / ``front`` / ``all``
#: * ``drivetrain.traction_control``    -> a JSON boolean
DERIVED_INPUT_KEYS = UI_ONLY_KEYS | {
    "wheels.driven_axle",
    "drivetrain.traction_control",
}


def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``source`` into ``target`` in place."""
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


@dataclass
class GarageConfig:
    """A full vehicle configuration plus its editable flat-value view.

    ``base`` is the untouched nested config (every key the core reads, including
    the ones the UI does not expose). ``values`` is a flat dotted-key overlay
    holding exactly the rows in :data:`SECTIONS`, so a field can only ever write
    to a key that already exists in the schema.

    Attributes:
        base: Nested config, normally loaded from ``config/vehicle.json``.
        values: Flat editable overlay.
        name: Preset name this config was loaded from or saved as.
    """

    base: Dict[str, Any] = field(default_factory=dict)
    values: Dict[str, Any] = field(default_factory=dict)
    name: str = "current"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def load_defaults(cls, config_path: Optional[Path] = None) -> "GarageConfig":
        """Build a config from ``config/vehicle.json`` plus the demo build."""
        path = (
            Path(config_path)
            if config_path is not None
            else Path(__file__).resolve().parents[1] / "config" / "vehicle.json"
        )
        with open(path, "r", encoding="utf-8") as handle:
            base = json.load(handle)
        return cls.from_vehicle_config(
            {**base, "build": copy.deepcopy(DEFAULT_BUILD), "style": {}}
        )

    @classmethod
    def from_vehicle(
        cls, vehicle: Vehicle, *, name: str = "current"
    ) -> "GarageConfig":
        """Capture the build of a running vehicle.

        Reading the build off the live objects is what keeps the garage and the
        physics in sync: the chemistry/S/P come from the actual
        :class:`BatteryPack` and the motor id from the actual
        :class:`core.motor.Motor`, not from a guess.
        """
        config = copy.deepcopy(vehicle.config)
        config["build"] = {
            "chemistry": str(vehicle.pack.chem.name),
            "series": int(vehicle.pack.S),
            "parallel": int(vehicle.pack.P),
            "motor_id": str(vehicle.motor.id),
        }
        config.setdefault("style", {})
        garage = cls.from_vehicle_config(config)
        garage.name = name
        return garage

    @classmethod
    def from_vehicle_config(cls, config: Dict[str, Any]) -> "GarageConfig":
        """Rebuild the editable view from a nested vehicle config."""
        base = copy.deepcopy(config)
        base.setdefault("build", copy.deepcopy(DEFAULT_BUILD))
        base.setdefault("style", {})
        chassis = base["chassis"]
        wheels = base["wheels"]
        drivetrain = base["drivetrain"]

        wheelbase = max(float(chassis["wheelbase_m"]), 1e-6)
        front_pct = 100.0 * float(chassis["cg_to_rear_axle_m"]) / wheelbase
        style = base["style"]

        values: Dict[str, Any] = {
            "chassis.mass_kg": float(chassis["mass_kg"]),
            "chassis.wheelbase_m": float(chassis["wheelbase_m"]),
            "chassis.weight_distribution_front_pct": front_pct,
            "chassis.track_width_front_m": float(chassis["track_width_front_m"]),
            "chassis.track_width_rear_m": float(chassis["track_width_rear_m"]),
            "chassis.drag_coefficient": float(chassis["drag_coefficient"]),
            "chassis.frontal_area_m2": float(chassis["frontal_area_m2"]),
            "chassis.cg_height_m": float(chassis["cg_height_m"]),
            "chassis.yaw_inertia_kg_m2": float(chassis["yaw_inertia_kg_m2"]),
            "style.body_length_m": float(
                style.get(
                    "body_length_m",
                    (float(chassis["cg_to_front_axle_m"])
                     + float(chassis["cg_to_rear_axle_m"])) + 0.7,
                )
            ),
            "style.body_width_m": float(
                style.get(
                    "body_width_m",
                    max(float(chassis["track_width_front_m"]),
                        float(chassis["track_width_rear_m"])) + 0.36,
                )
            ),
            "style.color": str(style.get("color", "Azure")),
            "build.chemistry": str(base["build"].get("chemistry", "Li-Ion (NMC)")),
            "build.series": int(base["build"].get("series", 92)),
            "build.parallel": int(base["build"].get("parallel", 3)),
            "build.motor_id": str(base["build"].get("motor_id", "pmsm-150kw")),
            "drivetrain.gear_ratio": float(drivetrain["gear_ratio"]),
            "drivetrain.final_drive_efficiency": float(
                drivetrain["final_drive_efficiency"]
            ),
            "wheels.driven_axle": _AXLE_TO_LABEL.get(
                str(wheels.get("driven_axle", "rear")).lower(), "RWD"
            ),
            "drivetrain.max_friction_brake_torque_Nm_per_wheel": float(
                drivetrain["max_friction_brake_torque_Nm_per_wheel"]
            ),
            "drivetrain.traction_control": (
                "on" if bool(drivetrain.get("traction_control", True)) else "off"
            ),
            "tires.mu_base": float(base["tires"]["mu_base"]),
            "tires.cornering_stiffness_N_per_rad": float(
                base["tires"]["cornering_stiffness_N_per_rad"]
            ),
            "tires.longitudinal_stiffness_N_per_slip": float(
                base["tires"]["longitudinal_stiffness_N_per_slip"]
            ),
            "wheels.rolling_resistance_coeff": float(
                wheels["rolling_resistance_coeff"]
            ),
            "style.tire_width_mm": int(style.get("tire_width_mm", 225)),
        }
        garage = cls(base=base, values=values)
        garage.clamp()
        return garage

    def copy(self) -> "GarageConfig":
        """Deep copy, so an edit can be abandoned without touching the original."""
        return GarageConfig(
            base=copy.deepcopy(self.base),
            values=dict(self.values),
            name=self.name,
        )

    # ------------------------------------------------------------------
    # Field access
    # ------------------------------------------------------------------
    @staticmethod
    def spec_for(key: str) -> Optional[FieldSpec]:
        """Look up the :class:`FieldSpec` that owns ``key``."""
        for section in SECTIONS:
            for spec in section.fields:
                if spec.key == key:
                    return spec
        return None

    def get(self, key: str, default: Any = None) -> Any:
        """Read a flat value."""
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Write a flat value (clamped against the field's own spec)."""
        spec = self.spec_for(key)
        if spec is not None:
            value = _coerce(spec, value)
        self.values[key] = value

    def clamp(self) -> None:
        """Clamp every value against its :class:`FieldSpec`."""
        for key, value in list(self.values.items()):
            spec = self.spec_for(key)
            if spec is not None:
                self.values[key] = _coerce(spec, value)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def to_vehicle_config(self) -> Dict[str, Any]:
        """Return the nested config to hand to the core.

        Two derived conversions happen here and nowhere else:

        1. ``wheels.driven_axle`` - the UI shows RWD/FWD/AWD labels, the core
           wants ``rear``/``front``/``all``.
        2. Weight split -> axle distances. The core's static axle load is
           ``front_share = cg_to_rear_axle_m / wheelbase_m``, so the UI split is
           inverted from that same relation to keep
           ``cg_to_front_axle_m + cg_to_rear_axle_m == wheelbase_m``, which the
           core asserts.

        Returns:
            A complete, core-readable config with ``build`` and ``style``
            sections appended (the core ignores keys it does not read).
        """
        config = copy.deepcopy(self.base)
        # UI-only input rows are converted below instead of being merged, so the
        # config the core receives contains only keys the core reads.
        overlay = {
            key: value
            for key, value in self.values.items()
            if key not in DERIVED_INPUT_KEYS
        }
        _deep_update(config, _nest(overlay))

        chassis = config["chassis"]
        wheelbase = max(float(chassis["wheelbase_m"]), 1e-6)
        front_pct = float(
            self.values.get("chassis.weight_distribution_front_pct", 50.0)
        )
        rear_axle_m = wheelbase * min(max(front_pct, 1.0), 99.0) / 100.0
        chassis["wheelbase_m"] = wheelbase
        chassis["cg_to_front_axle_m"] = wheelbase - rear_axle_m
        chassis["cg_to_rear_axle_m"] = rear_axle_m

        layout = self.values.get("wheels.driven_axle")
        config["wheels"]["driven_axle"] = _LABEL_TO_AXLE.get(str(layout), "rear")

        traction = self.values.get("drivetrain.traction_control")
        if traction is not None:
            config["drivetrain"]["traction_control"] = str(traction) == "on"

        config["build"] = {
            "chemistry": str(self.values.get("build.chemistry", "Li-Ion (NMC)")),
            "series": int(self.values.get("build.series", 92)),
            "parallel": int(self.values.get("build.parallel", 3)),
            "motor_id": str(self.values.get("build.motor_id", "pmsm-150kw")),
        }
        config["style"] = {
            key.split(".", 1)[1]: value
            for key, value in self.values.items()
            if key.startswith("style.")
        }
        return config

    def preset_dict(self) -> Dict[str, Any]:
        """Self-contained JSON payload for a preset or share code."""
        return self.to_vehicle_config()

    # ------------------------------------------------------------------
    # Live read-outs (everything below comes from the core objects)
    # ------------------------------------------------------------------
    def make_pack(self, *, initial_soc: float = 1.0) -> BatteryPack:
        """Instantiate the configured pack."""
        return BatteryPack(
            int(self.values.get("build.series", 92)),
            int(self.values.get("build.parallel", 3)),
            str(self.values.get("build.chemistry", "Li-Ion (NMC)")),
            initial_soc=initial_soc,
        )

    def make_motor(self) -> Motor:
        """Instantiate the configured motor."""
        return Motor(str(self.values.get("build.motor_id", "pmsm-150kw")))

    def body_colour(self) -> Tuple[int, int, int]:
        """RGB body colour for the sprite (cosmetic)."""
        return STYLE_COLOURS.get(
            str(self.values.get("style.color", "Azure")), (52, 120, 200)
        )


def _coerce(spec: FieldSpec, value: Any) -> Any:
    """Clamp/convert one value to its spec."""
    if spec.kind == "choice":
        if value in spec.choices:
            return value
        return spec.choices[0] if spec.choices else value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return spec.minimum
    if not math.isfinite(number):
        return spec.minimum
    number = min(max(number, spec.minimum), spec.maximum)
    if spec.kind == "int":
        return int(round(number))
    return float(number)


# ======================================================================
# Presets and share codes
# ======================================================================
def load_presets(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Read the preset file.

    A missing or corrupt file yields an empty mapping rather than raising, so a
    bad save can never stop the game starting.

    Args:
        path: Override for ``save/garage_save.json``.

    Returns:
        Mapping of preset name to vehicle config.
    """
    target = Path(path) if path is not None else GARAGE_SAVE_PATH
    try:
        with open(target, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    presets = raw.get("presets") if isinstance(raw, dict) else None
    if not isinstance(presets, dict):
        return {}
    return {
        str(name): data
        for name, data in presets.items()
        if isinstance(data, dict) and "chassis" in data
    }


def save_presets(
    presets: Dict[str, Dict[str, Any]], path: Optional[Path] = None
) -> Path:
    """Write the preset file atomically, creating ``save/`` if needed.

    Args:
        presets: Mapping of preset name to vehicle config.
        path: Override for ``save/garage_save.json``.

    Returns:
        The path written.
    """
    target = Path(path) if path is not None else GARAGE_SAVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_about": (
            "EV-Simulator garage presets. Each entry is a complete vehicle "
            "config in the same schema as config/vehicle.json, plus a 'build' "
            "section (chemistry/S/P/motor_id) and a cosmetic 'style' section. "
            "Add a preset by copying an existing entry and renaming the key."
        ),
        "version": 1,
        "presets": presets,
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(target)
    return target


def _strip_comments(source: Any) -> Any:
    """Recursively drop documentation keys (those starting with ``_``).

    ``config/vehicle.json`` carries ``_about`` / ``_todo`` / ``_field_notes``
    strings. They are helpful in the file and are kept in saved presets, but they
    are not part of the build: including them tripled the size of a share code
    (3.5 kB -> ~1.5 kB without). The core only ever reads named keys, so a config
    without them is still perfectly valid.

    Args:
        source: Mapping, list or scalar from a config.

    Returns:
        The same structure with comment keys removed.
    """
    if isinstance(source, dict):
        return {
            key: _strip_comments(value)
            for key, value in source.items()
            if not key.startswith("_")
        }
    if isinstance(source, list):
        return [_strip_comments(item) for item in source]
    return source


def encode_share_code(config: Dict[str, Any]) -> str:
    """Encode a vehicle config as a share code.

    Format: ``EVSG1.`` followed by the URL-safe base64 of the minified JSON of
    the config, with documentation-only keys (``_about``, ``_todo``, ...)
    stripped. The prefix pins the schema version so a future format can be
    rejected rather than silently misread.

    Args:
        config: Nested vehicle config (as produced by
            :meth:`GarageConfig.preset_dict`).

    Returns:
        A single-line ASCII share code.
    """
    payload = json.dumps(
        _strip_comments(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return SHARE_PREFIX + base64.urlsafe_b64encode(payload).decode("ascii")


def decode_share_code(code: str) -> Dict[str, Any]:
    """Decode a share code back into a vehicle config.

    Args:
        code: Share code, with or without surrounding whitespace/newlines.

    Returns:
        The decoded config.

    Raises:
        ValueError: If the prefix is wrong, the base64 is corrupt, the payload
            is not JSON, or required sections are missing.
    """
    text = "".join(str(code).split())
    if not text.startswith(SHARE_PREFIX):
        raise ValueError(f"share code must start with {SHARE_PREFIX!r}")
    body = text[len(SHARE_PREFIX) :]
    if not body:
        raise ValueError("share code has no payload")
    try:
        raw = base64.urlsafe_b64decode(body.encode("ascii"))
    except Exception as exc:  # noqa: BLE001 - any decode failure is the same error
        raise ValueError(f"share code is not valid base64: {exc}") from exc
    try:
        config = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"share code payload is not JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("share code payload is not an object")
    for key in ("chassis", "wheels", "tires", "drivetrain"):
        if key not in config:
            raise ValueError(f"share code is missing the {key!r} section")
    return config


# ======================================================================
# Colours
# ======================================================================
BG = (12, 15, 20)
PANEL = (20, 24, 31)
PANEL_EDGE = (58, 66, 78)
TEXT = (228, 234, 242)
LABEL = (140, 150, 165)
DIM = (100, 110, 124)
SELECT = (38, 58, 84)
ACCENT = (110, 190, 230)
GOOD = (104, 208, 136)
WARN = (238, 176, 84)
BAD = (232, 104, 96)


def _fit(text: str, font: "pygame.font.Font", width: int) -> str:
    """Truncate ``text`` with an ellipsis so it cannot overlap the next column."""
    if width <= 0 or font.size(text)[0] <= width:
        return text
    clipped = text
    while clipped and font.size(clipped + "...")[0] > width:
        clipped = clipped[:-1]
    return clipped + "..."


def _format_number(value: float, unit: str = "") -> str:
    """Render a numeric field without scientific notation.

    The spec ranges span 0.004 to 120000, so a fixed format either loses small
    values or prints ``1.25e+03`` for a mass. Precision is chosen from the
    magnitude instead.

    Args:
        value: Number to render.
        unit: Optional unit appended after a space.

    Returns:
        Display string.
    """
    magnitude = abs(float(value))
    if magnitude >= 1000.0:
        text = f"{value:,.0f}"
    elif magnitude >= 100.0:
        text = f"{value:.1f}"
    elif magnitude >= 10.0:
        text = f"{value:.2f}"
    elif magnitude >= 1.0:
        text = f"{value:.3f}".rstrip("0").rstrip(".")
    else:
        text = f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{text} {unit}" if unit else text


def _draw_curve(
    surface: pygame.Surface,
    rect: pygame.Rect,
    xs: Sequence[float],
    ys: Sequence[float],
    colour: Tuple[int, int, int],
    title: str,
    *,
    font: "pygame.font.Font",
    small: "pygame.font.Font",
    x_label: str,
    y_label: str,
    markers: Sequence[Tuple[float, float, str]] = (),
) -> None:
    """Plot one series with pygame primitives.

    Drawn by hand rather than with matplotlib: the simulator already depends on
    pygame, and this keeps the UI layer dependency-free and fast enough to
    redraw every frame.

    Args:
        surface: Target surface.
        rect: Plot area including axis labels.
        xs: Sample x values.
        ys: Sample y values (same length as ``xs``).
        colour: Line colour.
        title: Plot caption.
        font: Caption font.
        small: Axis label font.
        x_label: Unit shown on the x axis.
        y_label: Unit shown on the y axis.
        markers: Optional ``(x, y, label)`` points to highlight.
    """
    pygame.draw.rect(surface, PANEL, rect, border_radius=4)
    pygame.draw.rect(surface, PANEL_EDGE, rect, width=1, border_radius=4)
    # The title is fitted so a long caption (a chemistry name, say) cannot run
    # off the panel at narrow window sizes; the unit keeps the right-hand end.
    unit = small.render(y_label, True, DIM)
    title_text = _fit(title, font, rect.width - 40 - unit.get_width())
    surface.blit(font.render(title_text, True, LABEL), (rect.x + 8, rect.y + 5))
    surface.blit(unit, (rect.right - 10 - unit.get_width(), rect.y + 6))

    left = rect.x + 52
    right = rect.right - 10
    top = rect.y + 30
    bottom = rect.bottom - 20
    plot = pygame.Rect(left, top, max(right - left, 8), max(bottom - top, 8))
    if len(xs) < 2 or len(ys) < 2:
        return

    x_lo, x_hi = float(min(xs)), float(max(xs))
    y_lo, y_hi = float(min(ys)), float(max(ys))
    if y_hi - y_lo < 1e-9:
        y_hi = y_lo + 1.0
    x_span = max(x_hi - x_lo, 1e-9)

    def project(x: float, y: float) -> Tuple[int, int]:
        px = plot.x + int((x - x_lo) / x_span * plot.width)
        py = plot.bottom - int((y - y_lo) / (y_hi - y_lo) * plot.height)
        return px, py

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = plot.bottom - int(fraction * plot.height)
        pygame.draw.line(surface, (34, 40, 49), (plot.x, yy), (plot.right, yy), 1)
        value = y_lo + fraction * (y_hi - y_lo)
        rendered = small.render(f"{value:.0f}", True, DIM)
        surface.blit(rendered, (plot.x - 6 - rendered.get_width(), yy - 6))

    points = [project(float(x), float(y)) for x, y in zip(xs, ys)]
    if len(points) >= 2:
        pygame.draw.lines(surface, colour, False, points, 2)

    for mx, my, label in markers:
        if x_lo <= mx <= x_hi:
            px, py = project(mx, my)
            pygame.draw.circle(surface, GOOD, (px, py), 3)
            surface.blit(small.render(label, True, GOOD), (px + 5, py - 14))

    surface.blit(small.render(f"{x_lo:.0f}", True, DIM), (plot.x, plot.bottom + 2))
    end = small.render(f"{x_hi:.0f} {x_label}", True, DIM)
    surface.blit(end, (plot.right - end.get_width(), plot.bottom + 2))


# ======================================================================
# The menu
# ======================================================================
class Garage:
    """The Tab menu: edit a build, preview it, apply, save, share.

    The garage never mutates the running vehicle. :meth:`handle_event` returns an
    action string (``"apply"`` / ``"close"``) and the host decides what to do, so
    the physics can simply be paused while the menu is open.

    Args:
        config: The working configuration. Edits land on a copy.
        save_path: Override for ``save/garage_save.json``.
        cruise_kmh: Reference speed for the range prediction.
    """

    def __init__(
        self,
        config: GarageConfig,
        *,
        save_path: Optional[Path] = None,
        cruise_kmh: float = 90.0,
    ) -> None:
        self.save_path = Path(save_path) if save_path is not None else GARAGE_SAVE_PATH
        self.cruise_kmh = float(cruise_kmh)
        self.config = config.copy()
        self.applied = config.copy()
        self.section_index = 0
        self.field_index = 0
        self.preset_index = 0
        self.message = ""
        self.open = False
        self.presets: Dict[str, Dict[str, Any]] = load_presets(self.save_path)

        # -- mouse-driven UI state -------------------------------------
        # The menu is operated entirely by clicking; no key press is ever
        # required. Keyboard events are accepted as a convenience only.
        self.mouse: Tuple[int, int] = (0, 0)
        self.hover: Optional[str] = None
        self.dragging: Optional[str] = None
        self.modal: Optional[Dict[str, Any]] = None
        self.scroll_fields = 0.0
        self.scroll_presets = 0.0
        self.scroll_modal = 0.0
        self._widgets: List[Tuple[str, pygame.Rect]] = []
        self._layout_cache: Dict[str, Any] = {}
        self._fonts: Optional[Dict[str, pygame.font.Font]] = None
        self._anim: Dict[str, float] = {}
        self._last_click: Tuple[str, int] = ("", -10_000)
        self._message_until = 0
        #: Layout-dependent limits, refreshed every frame by :meth:`_layout`.
        self._max_field_scroll = 0
        self._max_preset_scroll = 0
        self._modal_visible = 8
        self._refresh_derived()

    # ------------------------------------------------------------------
    # Live read-outs
    # ------------------------------------------------------------------
    def _refresh_derived(self) -> None:
        """Rebuild the preview objects and the prediction.

        Called whenever the build changes. Everything here goes through the core
        constructors, so the panel can never disagree with what the physics
        would do with the same config.
        """
        self.vehicle_config = self.config.to_vehicle_config()
        try:
            self.preview_pack: Optional[BatteryPack] = self.config.make_pack()
            self.preview_motor: Optional[Motor] = self.config.make_motor()
        except (ValueError, OSError, json.JSONDecodeError):
            self.preview_pack = None
            self.preview_motor = None
            self.findings = ["invalid battery/motor selection"]
            self.estimate: Optional[BuildEstimate] = None
            return

        self.estimate = estimate_build(
            self.vehicle_config,
            self.preview_motor,
            self.preview_pack,
            cruise_kmh=self.cruise_kmh,
        )
        # Same aggregation core.vehicle.Vehicle.validate performs, including the
        # motor's controller rating and continuous power. Passing them is what
        # makes this stop nagging about a perfectly matched pack, while still
        # flagging a pack whose max voltage exceeds the controller (or one too
        # weak to feed the motor continuously).
        self.findings = list(
            self.preview_pack.validate(
                controller_max_V=self.preview_motor.controller_max_V,
                required_cont_power_W=self.preview_motor.cont_power_W,
                emit_warnings=False,
            )
        )
        self.findings.extend(self.preview_motor.validate(emit_warnings=False))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def pending_changes(self) -> bool:
        """``True`` when the working config differs from the applied one."""
        return self.config.values != self.applied.values

    def mark_applied(self) -> None:
        """Record the working config as the one now driving the car."""
        self.applied = self.config.copy()

    def current_section(self) -> Section:
        """The visible section."""
        return SECTIONS[int(self.section_index) % len(SECTIONS)]

    def current_field(self) -> Optional[FieldSpec]:
        """The selected row, or ``None`` for the SAVE/LOAD tab."""
        fields = self.current_section().fields
        if not fields:
            return None
        return fields[int(self.field_index) % len(fields)]

    @property
    def preset_names(self) -> List[str]:
        """Preset names, sorted for stable navigation."""
        return sorted(self.presets)

    def selected_preset(self) -> Optional[str]:
        """Name of the highlighted preset, if any."""
        names = self.preset_names
        if not names:
            return None
        return names[int(self.preset_index) % len(names)]

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------
    def change_section(self, delta: int) -> None:
        """Step the section tab and reset the row selection."""
        self.section_index = (int(self.section_index) + int(delta)) % len(SECTIONS)
        self.field_index = 0

    def move_field(self, delta: int) -> None:
        """Step the row selection, or the preset selection on SAVE/LOAD."""
        fields = self.current_section().fields
        if not fields:
            names = self.preset_names
            if names:
                self.preset_index = (int(self.preset_index) + int(delta)) % len(names)
            return
        self.field_index = (int(self.field_index) + int(delta)) % len(fields)

    def adjust_value(self, delta: int, *, coarse: bool = False) -> None:
        """Change the selected row by ``delta`` steps.

        Choice rows cycle through their options. Numeric rows step by
        :attr:`FieldSpec.step`, multiplied by ten when ``coarse`` is set.
        Out-of-range values are clamped rather than rejected.

        Args:
            delta: ``+1`` / ``-1`` (any integer works as a multiplier).
            coarse: Use the ten-times step.
        """
        spec = self.current_field()
        if spec is None:
            return
        if spec.kind == "choice":
            if not spec.choices:
                return
            current = str(self.config.get(spec.key, spec.choices[0]))
            try:
                index = list(spec.choices).index(current)
            except ValueError:
                index = 0
            index = (index + int(delta)) % len(spec.choices)
            self.config.set(spec.key, spec.choices[index])
        else:
            step = spec.step * (10.0 if coarse else 1.0) * float(delta)
            self.config.set(spec.key, float(self.config.get(spec.key, spec.minimum)) + step)
        self._refresh_derived()

    def revert_section(self) -> None:
        """Restore the visible section's rows to the applied build."""
        for spec in self.current_section().fields:
            if spec.key in self.applied.values:
                self.config.values[spec.key] = self.applied.values[spec.key]
        self._refresh_derived()
        self._flash("section reverted to the applied build")

    def revert_all(self) -> None:
        """Restore every row to the applied build."""
        self.config = self.applied.copy()
        self._refresh_derived()
        self._flash("all fields reverted to the applied build")

    # ------------------------------------------------------------------
    # Presets and sharing
    # ------------------------------------------------------------------
    def reload_presets(self) -> int:
        """Re-read the preset file and return how many presets were found."""
        self.presets = load_presets(self.save_path)
        self.preset_index = 0
        return len(self.presets)

    def save_preset(self, name: str) -> str:
        """Save the working config under ``name``.

        Args:
            name: Preset name; whitespace is trimmed and empty names rejected.

        Returns:
            A status message for the menu.
        """
        clean = " ".join(str(name).split())
        if not clean:
            return "preset name cannot be empty"
        self.presets[clean] = self.config.preset_dict()
        try:
            path = save_presets(self.presets, self.save_path)
        except OSError as exc:
            return f"could not write {self.save_path}: {exc}"
        self.config.name = clean
        self.preset_index = max(self.preset_names.index(clean), 0)
        return f"saved preset {clean!r} to {path.name} ({len(self.presets)} stored)"

    def load_preset(self, name: Optional[str] = None) -> str:
        """Load a preset into the working config.

        Args:
            name: Preset to load; defaults to the highlighted one.

        Returns:
            A status message for the menu.
        """
        target = name or self.selected_preset()
        if not target or target not in self.presets:
            return "no preset selected"
        try:
            self.config = GarageConfig.from_vehicle_config(self.presets[target])
        except (KeyError, TypeError, ValueError) as exc:
            return f"preset {target!r} is not usable: {exc}"
        self.config.name = target
        self._refresh_derived()
        return f"loaded preset {target!r} (press ENTER to apply)"

    def delete_preset(self, name: Optional[str] = None) -> str:
        """Remove a preset from the file."""
        target = name or self.selected_preset()
        if not target or target not in self.presets:
            return "no preset selected"
        del self.presets[target]
        try:
            save_presets(self.presets, self.save_path)
        except OSError as exc:
            return f"could not write {self.save_path}: {exc}"
        self.preset_index = 0
        return f"deleted preset {target!r}"

    def share_code(self) -> str:
        """Share code for the working config, ready to paste or print."""
        return encode_share_code(self.config.preset_dict())

    def import_share_code(self, code: str) -> str:
        """Load a share code into the working config.

        Args:
            code: The share code text.

        Returns:
            A status message for the menu.
        """
        try:
            config = decode_share_code(code)
            self.config = GarageConfig.from_vehicle_config(config)
        except (ValueError, KeyError, TypeError) as exc:
            return f"import failed: {exc}"
        self.config.name = "imported"
        self._refresh_derived()
        return "share code imported (press ENTER to apply)"

    def _flash(self, message: str) -> None:
        """Show a transient status message for a few seconds."""
        self.message = message
        self._message_until = pygame.time.get_ticks() + 6000

    # ------------------------------------------------------------------
    # Mouse-driven interaction
    # ------------------------------------------------------------------
    #: Background used for the widget under the cursor.
    HOVER_BG = (32, 44, 61)
    #: Background used for the pressed / selected widget.
    ACTIVE_BG = (46, 68, 95)
    #: Slider track and its filled portion.
    TRACK_BG = (34, 40, 49)
    TRACK_FILL = (58, 100, 148)
    #: Background for a stepper button or a value pill.
    CONTROL_BG = (28, 34, 44)
    #: Panel and shadow for floating elements.
    PANEL_BG = (22, 27, 35)
    PANEL_SHADOW = (0, 0, 0, 120)

    #: ``(action, label, tooltip)`` for the bottom button bar, in draw order.
    BUTTONS: Tuple[Tuple[str, str, str], ...] = (
        ("apply", "APPLY BUILD", "Rebuild the car from these settings - no restart"),
        ("save", "SAVE PRESET", "Save this build as a new preset (auto-named)"),
        ("load", "LOAD PRESET", "Load the highlighted preset into the editor"),
        ("delete", "DELETE PRESET", "Delete the highlighted preset from disk"),
        ("close", "CLOSE", "Leave the garage and unpause"),
        ("revert_section", "REVERT TAB", "Undo the changes on this tab only"),
        ("revert_all", "REVERT ALL", "Undo every change since the last apply"),
        ("export", "EXPORT CODE", "Print/copy a share code for this build"),
        ("import", "IMPORT CODE", "Import a share code from the clipboard or file"),
    )

    def _fonts_for(self) -> Dict[str, pygame.font.Font]:
        """Build the fonts once, on first use."""
        if self._fonts is None:
            self._fonts = {
                "title": pygame.font.SysFont("monospace", 17, bold=True),
                "body": pygame.font.SysFont("monospace", 15),
                "small": pygame.font.SysFont("monospace", 13),
                "tiny": pygame.font.SysFont("monospace", 11),
            }
        return self._fonts

    @staticmethod
    def _panel(
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        fill: Tuple[int, int, int] = (22, 27, 35),
        edge: Tuple[int, int, int] = PANEL_EDGE,
        radius: int = 6,
        shadow: bool = True,
    ) -> None:
        """Draw a rounded panel, optionally with a soft drop shadow."""
        if shadow:
            pad = 5
            shadow_surface = pygame.Surface(
                (rect.width + pad * 2, rect.height + pad * 2), pygame.SRCALPHA
            )
            for offset, alpha in ((pad, 26), (3, 38), (1, 52)):
                pygame.draw.rect(
                    shadow_surface,
                    (0, 0, 0, alpha),
                    pygame.Rect(pad - offset, pad - offset, rect.width + 2 * offset,
                                rect.height + 2 * offset),
                    border_radius=radius + offset,
                )
            surface.blit(shadow_surface, (rect.x - pad, rect.y - pad))
        pygame.draw.rect(surface, fill, rect, border_radius=radius)
        pygame.draw.rect(surface, edge, rect, width=1, border_radius=radius)

    def _layout(self, size: Tuple[int, int]) -> Dict[str, Any]:
        """Compute every widget rectangle for a window size.

        Called once per frame from :meth:`draw` and reused by
        :meth:`handle_event` for hit-testing, so geometry is defined in exactly
        one place. All rectangles are in screen coordinates.

        Args:
            size: Window size in pixels.

        Returns:
            Mapping of layout keys to rectangles / widget lists.
        """
        fonts = self._fonts_for()
        tiny = fonts["tiny"]
        small = fonts["small"]
        width, height = size
        pad = 16
        tabs_w = 172
        right_w = max(min(432, width // 3), 252)

        button_h = 26
        bar_h = button_h * 2 + 14
        bar = pygame.Rect(pad, height - bar_h - pad, width - 2 * pad, bar_h)
        columns = 5
        button_w = (bar.width - (columns - 1) * 8) // columns
        buttons = []
        for index, (action, label, tip) in enumerate(self.BUTTONS):
            row, column = divmod(index, columns)
            rect = pygame.Rect(
                bar.x + column * (button_w + 8),
                bar.y + 6 + row * (button_h + 6),
                button_w,
                button_h,
            )
            buttons.append((rect, action, label, tip))

        top = pad + 30
        tabs = [
            (pygame.Rect(pad, top + index * 26, tabs_w, 24), index, section.title)
            for index, section in enumerate(SECTIONS)
        ]
        tabs_bottom = top + len(SECTIONS) * 26 + 10

        presets_area = pygame.Rect(
            pad, tabs_bottom + 18, tabs_w, max(bar.y - tabs_bottom - 28, 40)
        )
        preset_row_h = 20
        preset_rows = []
        names = self.preset_names
        visible = max(int(presets_area.height // preset_row_h), 1)
        max_preset_scroll = max(len(names) - visible, 0)
        self._preset_visible = visible
        for slot in range(visible):
            index = int(self.scroll_presets) + slot
            if index >= len(names):
                break
            rect = pygame.Rect(
                presets_area.x,
                presets_area.y + slot * preset_row_h,
                presets_area.width,
                preset_row_h - 2,
            )
            preset_rows.append((rect, index, names[index]))
        self._max_preset_scroll = max_preset_scroll

        fields_x = pad + tabs_w + 16
        fields_w = max(width - fields_x - right_w - pad - 16, 200)
        section = self.current_section()
        # Reserve a two-line footer below the list: one line for the selected
        # row's note/range, one for the status message. Without this the note is
        # pushed up on top of the last row at small window sizes.
        footer_h = 44
        rows_area = pygame.Rect(
            fields_x, top + 48, fields_w, max(bar.y - top - 48 - footer_h, 80)
        )
        row_h = 30
        content_h = len(section.fields) * row_h
        self._max_field_scroll = max(content_h - rows_area.height, 0)
        self.scroll_fields = min(max(self.scroll_fields, 0.0), float(self._max_field_scroll))
        rows = []
        sliders = []
        for index, spec in enumerate(section.fields):
            y = rows_area.y + index * row_h - int(self.scroll_fields)
            rect = pygame.Rect(rows_area.x, y, fields_w, row_h - 5)
            rows.append((rect, index, spec))
            label_w = int(rect.width * 0.34)
            if spec.kind == "choice":
                # ``<`` and ``>`` cycle the list one step at a time; the value
                # itself opens the full picker (12 motors do not fit in a row).
                minus = pygame.Rect(rect.x + label_w, rect.y + 2, 20, rect.height - 4)
                value = pygame.Rect(
                    minus.right + 3, rect.y + 2,
                    max(rect.width - label_w - 52, 40), rect.height - 4,
                )
                plus = pygame.Rect(value.right + 3, rect.y + 2, 20, rect.height - 4)
                sliders.append((None, minus, value, plus, index, spec))
            else:
                minus = pygame.Rect(rect.x + label_w, rect.y + 2, 20, rect.height - 4)
                available = rect.right - minus.right - 46 - 8
                track = pygame.Rect(
                    minus.right + 4,
                    rect.centery - 3,
                    max(int(available * 0.60), 40),
                    6,
                )
                plus = pygame.Rect(track.right + 4, rect.y + 2, 20, rect.height - 4)
                sliders.append((track, minus, None, plus, index, spec))

        right_x = width - right_w - pad + 12
        right = pygame.Rect(right_x, top, right_w, 0)

        # -- floating picker (choice rows with many options) ---------------
        modal_panel: Optional[pygame.Rect] = None
        modal_close: Optional[pygame.Rect] = None
        modal_visible = 1
        modal_rect = None
        if self.modal is not None:
            options = list(self.modal.get("options", ()))
            panel_w = min(440, max(width - 80, 260))
            panel_h = min(96 + len(options) * 24, max(height - 140, 200))
            modal_panel = pygame.Rect(
                (width - panel_w) // 2, (height - panel_h) // 2, panel_w, panel_h
            )
            modal_visible = max((panel_h - 66) // 24, 1)
            modal_close = pygame.Rect(
                modal_panel.right - 82, modal_panel.bottom - 34, 72, 24
            )
            self._modal_visible = modal_visible

        self._rows_area = rows_area
        self._presets_area = presets_area
        self._modal_panel = modal_panel
        return {
            "size": (width, height),
            "pad": pad,
            "bar": bar,
            "buttons": buttons,
            "tabs": tabs,
            "presets_area": presets_area,
            "preset_rows": preset_rows,
            "fields_area": rows_area,
            "rows": rows,
            "sliders": sliders,
            "right_x": right_x,
            "right": right,
            "top": top,
            "fields_x": fields_x,
            "fields_w": fields_w,
            "fonts": fonts,
            "tiny": tiny,
            "small": small,
            "modal_panel": modal_panel,
            "modal_close": modal_close,
            "modal_visible": modal_visible,
            "modal_rect": modal_rect,
        }

    def _widget_under(self, pos: Tuple[int, int]) -> Optional[str]:
        """Id of the topmost widget containing ``pos``, if any.

        Widgets are registered back-to-front (rows first, then the sliders and
        steppers that sit on top of them), so the search runs in reverse: a
        click on a slider must not be swallowed by the row rectangle behind it.
        """
        if self.modal is not None:
            for widget_id, rect in reversed(self._it_widgets()):
                if rect.collidepoint(pos):
                    return widget_id
            return None
        for widget_id, rect in reversed(self._widgets):
            if rect.collidepoint(pos):
                return widget_id
        return None

    def _it_widgets(self) -> List[Tuple[str, pygame.Rect]]:
        """Widgets inside the open modal (drawn last, so hit-tested first)."""
        return [entry for entry in self._widgets if entry[0].startswith(("opt:", "mclose"))]

    def _set_from_track(self, index: int, x: int) -> None:
        """Set row ``index`` from a horizontal position along its slider track."""
        fields = self.current_section().fields
        if not 0 <= index < len(fields):
            return
        spec = fields[index]
        sliders = self._layout_cache.get("sliders", [])
        track = next(
            (entry[0] for entry in sliders if entry[4] == index and entry[0] is not None),
            None,
        )
        if track is None or track.width <= 0:
            return
        fraction = (x - track.x) / track.width
        fraction = min(max(fraction, 0.0), 1.0)
        raw = spec.minimum + fraction * (spec.maximum - spec.minimum)
        if spec.step > 0.0:
            raw = round(raw / spec.step) * spec.step
        self.config.set(spec.key, raw)
        self.field_index = index
        self._refresh_derived()

    def _activate(self, widget_id: str, pos: Tuple[int, int]) -> Optional[str]:
        """Run the action bound to a widget.

        Returns:
            ``"apply"`` / ``"close"`` for the host, otherwise ``None``.
        """
        kind, _, argument = widget_id.partition(":")
        if kind == "tab":
            self.section_index = int(argument)
            self.field_index = 0
            self.scroll_fields = 0.0
        elif kind == "row":
            self.field_index = int(argument)
        elif kind == "minus":
            self.field_index = int(argument)
            self.adjust_value(-1)
        elif kind == "plus":
            self.field_index = int(argument)
            self.adjust_value(1)
        elif kind == "slider":
            self.dragging = widget_id
            self._set_from_track(int(argument), pos[0])
        elif kind == "choice":
            spec = self.current_section().fields[int(argument)]
            self.field_index = int(argument)
            self.modal = {
                "title": spec.label,
                "key": spec.key,
                "options": list(spec.choices),
                "note": spec.note,
            }
            self.scroll_modal = 0.0
        elif kind == "opt":
            if self.modal is not None:
                option = self.modal["options"][int(argument)]
                self.config.set(str(self.modal["key"]), option)
                self._refresh_derived()
                self.modal = None
        elif kind == "mclose":
            self.modal = None
        elif kind == "preset":
            index = int(argument)
            double = (
                widget_id == self._last_click[0]
                and pygame.time.get_ticks() - self._last_click[1] < 350
            )
            self.preset_index = index
            if double:
                self._flash(self.load_preset())
                self._last_click = (widget_id, -10_000)
                return None
        elif kind == "btn":
            return self._do_action(argument)
        return None

    def _do_action(self, action: str) -> Optional[str]:
        """Run a named button action.

        Returns:
            ``"apply"`` / ``"close"`` for the host, otherwise ``None``.
        """
        if action == "apply":
            return "apply"
        if action == "close":
            return "close"
        if action == "revert_section":
            self.revert_section()
        elif action == "revert_all":
            self.revert_all()
        elif action == "save":
            self._flash(self.save_preset(self.auto_preset_name()))
        elif action == "load":
            self._flash(self.load_preset())
        elif action == "delete":
            self._flash(self.delete_preset())
        elif action == "export":
            code = self.share_code()
            copied = _copy_to_clipboard(code)
            print(f"SHARE_CODE {code}")
            self._flash(
                f"share code ({len(code)} chars) printed to the console"
                + (" and copied to the clipboard" if copied else
                   f"; paste-ready text also available in {IMPORT_CODE_PATH.name}")
            )
        elif action == "import":
            code = _read_clipboard() or _read_import_file()
            if not code:
                self._flash(
                    f"nothing in the clipboard - paste a share code into "
                    f"{IMPORT_CODE_PATH.name} and click IMPORT again"
                )
            else:
                self._flash(self.import_share_code(code))
        return None

    def auto_preset_name(self) -> str:
        """Derive a unique, informative preset name from the current build.

        Names are generated rather than typed so that saving is a single click;
        rename an entry in ``save/garage_save.json`` if you want something else.

        Returns:
            A name such as ``"pmsm-150kw 92S3P RWD"``, suffixed with a counter
            if that build is already stored.
        """
        base = (
            f"{self.config.get('build.motor_id')} "
            f"{int(self.config.get('build.series'))}S"
            f"{int(self.config.get('build.parallel'))}P "
            f"{self.config.get('wheels.driven_axle')}"
        )
        if base not in self.presets:
            return base
        for counter in range(2, 100):
            candidate = f"{base} ({counter})"
            if candidate not in self.presets:
                return candidate
        return f"{base} ({pygame.time.get_ticks()})"

    def _wheel(self, steps: int) -> None:
        """Scroll whatever the cursor is over."""
        delta = -float(steps) * 36.0
        if self.modal is not None:
            options = self.modal.get("options", [])
            visible = self._modal_visible
            self.scroll_modal = min(
                max(self.scroll_modal + delta, 0.0),
                float(max(len(options) - visible, 0)),
            )
            return
        layout = self._layout_cache
        if layout.get("fields_area", pygame.Rect(0, 0, 0, 0)).collidepoint(self.mouse):
            self.scroll_fields = min(
                max(self.scroll_fields + delta, 0.0), float(self._max_field_scroll)
            )
        elif layout.get("presets_area", pygame.Rect(0, 0, 0, 0)).collidepoint(self.mouse):
            self.scroll_presets = min(
                max(self.scroll_presets + delta, 0.0), float(self._max_preset_scroll)
            )

    def _scrollbar(
        self,
        surface: pygame.Surface,
        area: pygame.Rect,
        offset: float,
        maximum: int,
        total: int,
    ) -> None:
        """Draw a slim scroll indicator on the right edge of a scrolling area."""
        bar = pygame.Rect(area.right - 5, area.y + 2, 5, max(area.height - 4, 12))
        pygame.draw.rect(surface, self.TRACK_BG, bar, border_radius=2)
        visible = max(total - maximum, 1)
        height = max(int(bar.height * visible / max(total, 1)), 14)
        travel = max(bar.height - height, 0)
        position = int(travel * (offset / maximum)) if maximum > 0 else 0
        pygame.draw.rect(
            surface, ACCENT,
            pygame.Rect(bar.x, bar.y + position, bar.width, height),
            border_radius=2,
        )

    def _draw_field_row(
        self,
        surface: pygame.Surface,
        row: pygame.Rect,
        index: int,
        spec: FieldSpec,
        slider: Tuple[Any, Any, Any, Any, int, FieldSpec],
        fonts: Dict[str, pygame.font.Font],
    ) -> None:
        """Draw one editable row: label, control, live value, change markers.

        Numeric rows get a draggable slider plus ``-``/``+`` steppers; choice
        rows get steppers for single steps and a clickable value that opens the
        picker. A dot appears when the row differs from the applied build, and
        numeric rows also show the signed delta.

        Args:
            surface: Target surface.
            row: Row rectangle.
            index: Row index within the current section.
            spec: The field this row edits.
            slider: ``(track, minus, value_pill, plus, index, spec)`` from the
                layout, any of which may be ``None``.
            fonts: Cached fonts.
        """
        tiny = fonts["tiny"]
        small = fonts["small"]
        selected = index == self.field_index
        hovered = self.hover is not None and self.hover.endswith(f":{index}")
        if selected or hovered:
            self._panel(
                surface,
                row,
                fill=self.ACTIVE_BG if selected else self.HOVER_BG,
                edge=ACCENT if selected else PANEL_EDGE,
                radius=5,
                shadow=False,
            )
        current = self.config.get(spec.key)
        applied = self.applied.values.get(spec.key)
        modified = current != applied
        if modified:
            pygame.draw.circle(surface, WARN, (row.x + 5, row.centery), 3)

        label_w = int(row.width * 0.34)
        surface.blit(
            small.render(
                _fit(spec.label, small, label_w - 14), True,
                TEXT if (selected or hovered) else LABEL,
            ),
            (row.x + 12, row.y + 4),
        )

        track, minus, pill, plus, _row_index, _row_spec = slider
        for rect, glyph, widget in (
            (minus, "-" if spec.kind != "choice" else "<", f"minus:{index}"),
            (plus, "+" if spec.kind != "choice" else ">", f"plus:{index}"),
        ):
            if rect is None:
                continue
            hovered_button = self.hover == widget
            self._panel(
                surface, rect,
                fill=self.ACTIVE_BG if hovered_button else self.CONTROL_BG,
                edge=PANEL_EDGE, radius=4, shadow=False,
            )
            glyph_surface = small.render(glyph, True, ACCENT if hovered_button else TEXT)
            surface.blit(
                glyph_surface,
                (rect.centerx - glyph_surface.get_width() // 2, rect.y + 3),
            )

        if spec.kind == "choice":
            hovered_value = self.hover == f"choice:{index}"
            self._panel(
                surface, pill,
                fill=self.ACTIVE_BG if hovered_value else self.CONTROL_BG,
                edge=ACCENT, radius=4, shadow=False,
            )
            surface.blit(
                small.render(_fit(str(current), small, pill.width - 14), True, ACCENT),
                (pill.x + 7, pill.y + 3),
            )
            surface.blit(tiny.render("click to choose", True, DIM), (pill.right + 30, row.y + 6))
            return

        # -- numeric row: slider -----------------------------------------
        low, high = float(spec.minimum), float(spec.maximum)
        span = max(high - low, 1e-9)
        target = min(max((float(current) - low) / span, 0.0), 1.0)
        animated = self._anim.get(spec.key)
        animated = target if animated is None else animated + (target - animated) * 0.4
        self._anim[spec.key] = animated
        pygame.draw.rect(surface, self.TRACK_BG, track, border_radius=3)
        filled = pygame.Rect(track.x, track.y, int(track.width * animated), track.height)
        if filled.width > 0:
            pygame.draw.rect(surface, self.TRACK_FILL, filled, border_radius=3)
        handle_x = track.x + int(track.width * animated)
        pygame.draw.circle(surface, ACCENT if (selected or hovered) else TEXT,
                           (handle_x, track.centery), 7)
        pygame.draw.circle(surface, (16, 20, 26), (handle_x, track.centery), 3)

        value_x = plus.right + 8
        delta_w = 52 if modified else 0
        value_w = max(row.right - value_x - delta_w - 4, 30)
        value_text = (
            f"{int(current)} {spec.unit}".strip()
            if spec.kind == "int"
            else _format_number(float(current), spec.unit)
        )
        surface.blit(
            small.render(_fit(value_text, small, value_w), True, TEXT),
            (value_x, row.y + 4),
        )
        if modified:
            delta = float(current) - float(applied)
            text = f"{delta:+,.0f}" if abs(delta) >= 1000 else f"{delta:+.3g}"
            rendered = tiny.render(text, True, WARN)
            surface.blit(rendered, (row.right - rendered.get_width() - 2, row.y + 6))

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        """Handle one pygame event.

        The menu is click-only: mouse motion, clicks, drags and the wheel all
        work, and a click is never required to be preceded by a key press. Tab
        and Enter are accepted as optional shortcuts.

        Args:
            event: The event to process.

        Returns:
            ``"close"`` to leave the menu, ``"apply"`` to rebuild the vehicle,
            otherwise ``None``.
        """
        if event.type == pygame.MOUSEMOTION:
            self.mouse = event.pos
            if self.dragging is not None:
                kind, _, argument = self.dragging.partition(":")
                if kind == "slider":
                    self._set_from_track(int(argument), event.pos[0])
            return None

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.mouse = event.pos
            if self.modal is not None:
                widget_id = self._widget_under(event.pos)
                if widget_id is None:
                    self.modal = None  # clicking outside dismisses the picker
                    self._last_click = ("", -10_000)
                    return None
            else:
                widget_id = self._widget_under(event.pos)
            if widget_id is None:
                return None
            action = self._activate(widget_id, event.pos)
            self._last_click = (widget_id, pygame.time.get_ticks())
            return action

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = None
            return None

        if event.type == pygame.MOUSEWHEEL:
            self._wheel(event.y)
            return None

        if event.type == pygame.KEYDOWN:
            # Convenience only: everything here is also reachable by clicking.
            if event.key in (pygame.K_TAB, pygame.K_ESCAPE):
                if self.modal is not None:
                    self.modal = None
                    return None
                return "close"
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                if self.modal is not None:
                    return None
                return "apply"
        return None

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw(
        self,
        surface: pygame.Surface,
        *,
        mouse: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Draw the whole menu and register its widget rectangles.

        Geometry comes from :meth:`_layout`, so the rectangles hit-tested by
        :meth:`handle_event` are exactly the ones drawn here.

        Args:
            surface: Target surface.
            mouse: Cursor position. Defaults to the live mouse position; passed
                explicitly by tests so clicks can be simulated headlessly.
        """
        fonts = self._fonts_for()
        self.mouse = tuple(mouse) if mouse is not None else pygame.mouse.get_pos()
        layout = self._layout(surface.get_size())
        self._layout_cache = layout
        width, height = layout["size"]
        pad = int(layout["pad"])
        tiny = layout["tiny"]
        small = layout["small"]
        body = fonts["body"]
        heading = fonts["title"]

        # -- backdrop: soft vertical gradient over the paused world --------
        backdrop = pygame.Surface((width, height))
        for band in range(0, height, 4):
            shade = 8 + int(16 * band / max(height, 1))
            pygame.draw.rect(
                backdrop, (shade, shade + 2, shade + 7), (0, band, width, 4)
            )
        surface.blit(backdrop, (0, 0))
        veil = pygame.Surface((width, height), pygame.SRCALPHA)
        veil.fill((6, 8, 12, 218))
        surface.blit(veil, (0, 0))

        # -- register widget rectangles (hit-testing for the next event) ---
        widgets: List[Tuple[str, pygame.Rect]] = []
        for rect, index, _label in layout["tabs"]:
            widgets.append((f"tab:{index}", rect))
        for rect, index, _spec in layout["rows"]:
            widgets.append((f"row:{index}", rect))
        for track, minus, pill, plus, index, _spec in layout["sliders"]:
            if track is not None:
                widgets.append((f"slider:{index}", track))
            if minus is not None:
                widgets.append((f"minus:{index}", minus))
            if pill is not None:
                widgets.append((f"choice:{index}", pill))
            if plus is not None:
                widgets.append((f"plus:{index}", plus))
        for rect, index, _name in layout["preset_rows"]:
            widgets.append((f"preset:{index}", rect))
        for rect, action, _label, _tip in layout["buttons"]:
            # A disabled button is not registered, so it cannot be clicked.
            if self._button_enabled(action):
                widgets.append((f"btn:{action}", rect))
        if self.modal is not None:
            widgets.extend(self._modal_widgets())
        self._widgets = widgets
        self.hover = self._widget_under(self.mouse)

        # -- header --------------------------------------------------------
        surface.blit(heading.render("GARAGE", True, TEXT), (pad, pad - 3))
        subtitle = (
            "click a tab, drag a slider, click a value to pick from a list"
        )
        surface.blit(tiny.render(subtitle, True, DIM), (pad + 96, pad + 2))
        if self.pending_changes():
            badge = pygame.Rect(width - pad - 150, pad - 2, 150, 22)
            self._panel(surface, badge, fill=(60, 44, 18), edge=WARN, radius=5, shadow=False)
            surface.blit(
                small.render("unapplied changes", True, WARN),
                (badge.x + 8, badge.y + 3),
            )
        if self.config.name != "current":
            surface.blit(
                tiny.render(_fit(f"editing '{self.config.name}'", tiny, 200), True, DIM),
                (pad + 96, pad + 16),
            )

        # -- section tabs --------------------------------------------------
        for rect, index, label in layout["tabs"]:
            active = index == self.section_index
            hovered = self.hover == f"tab:{index}"
            if active:
                self._panel(surface, rect, fill=self.ACTIVE_BG, edge=ACCENT, radius=5,
                            shadow=False)
                pygame.draw.rect(
                    surface, ACCENT,
                    pygame.Rect(rect.x - 3, rect.y + 4, 3, rect.height - 8),
                    border_radius=2,
                )
            elif hovered:
                self._panel(surface, rect, fill=self.HOVER_BG, edge=PANEL_EDGE, radius=5,
                            shadow=False)
            surface.blit(
                small.render(label, True, TEXT if active else LABEL),
                (rect.x + 12, rect.y + 4),
            )

        # -- preset list (always visible, click to select, double-click to load)
        presets_area = layout["presets_area"]
        surface.blit(
            tiny.render(
                f"PRESETS  {self.preset_index + 1 if self.preset_names else 0}"
                f"/{len(self.preset_names)}  (hover + wheel to scroll)",
                True,
                LABEL,
            ),
            (presets_area.x, presets_area.y - 16),
        )
        if not self.preset_names:
            surface.blit(
                tiny.render("none yet - click SAVE", True, DIM),
                (presets_area.x + 4, presets_area.y + 2),
            )
        for rect, index, name in layout["preset_rows"]:
            selected = index == self.preset_index
            hovered = self.hover == f"preset:{index}"
            if selected:
                self._panel(surface, rect, fill=self.ACTIVE_BG, edge=ACCENT, radius=4,
                            shadow=False)
            elif hovered:
                self._panel(surface, rect, fill=self.HOVER_BG, edge=PANEL_EDGE, radius=4,
                            shadow=False)
            surface.blit(
                tiny.render(
                    _fit(("> " if selected else "  ") + name, tiny, rect.width - 8),
                    True,
                    TEXT if selected else LABEL,
                ),
                (rect.x + 4, rect.y + 3),
            )
        if self._max_preset_scroll > 0:
            self._scrollbar(surface, presets_area, self.scroll_presets,
                            self._max_preset_scroll, len(self.preset_names))

        # -- field column ----------------------------------------------
        fields_x = layout["fields_x"]
        fields_w = layout["fields_w"]
        fields_area = layout["fields_area"]
        section = self.current_section()
        surface.blit(small.render(section.title, True, ACCENT), (fields_x, layout["top"] + 2))
        surface.blit(
            tiny.render(_fit(section.description, tiny, fields_w), True, DIM),
            (fields_x, layout["top"] + 20),
        )

        if not section.fields:
            surface.blit(
                small.render("Pick a preset on the left, then click LOAD PRESET.",
                             True, TEXT),
                (fields_x, fields_area.y),
            )
            for offset, line in enumerate(
                (
                    "SAVE PRESET stores this build as a new, auto-named entry",
                    "   in save/garage_save.json (rename it there if you like)",
                    "LOAD PRESET loads the highlighted entry into the editor",
                    "DELETE PRESET removes the highlighted entry from the file",
                    "EXPORT CODE prints a base64 'EVSG1.' share code",
                    "IMPORT CODE reads the clipboard, else save/import_code.txt",
                    "REVERT ALL undoes every change since the last apply",
                    "APPLY BUILD rebuilds the car - no restart needed",
                )
            ):
                surface.blit(
                    tiny.render(line, True, LABEL),
                    (fields_x, fields_area.y + 30 + offset * 17),
                )
        else:
            # The list scrolls with the wheel, so clip it to its own area.
            previous_clip = surface.get_clip()
            surface.set_clip(fields_area)
            slider_map = {entry[4]: entry for entry in layout["sliders"]}
            for row, index, spec in layout["rows"]:
                self._draw_field_row(
                    surface, row, index, spec, slider_map[index], layout["fonts"]
                )
            surface.set_clip(previous_clip)
            if self._max_field_scroll > 0:
                self._scrollbar(
                    surface, fields_area, self.scroll_fields,
                    self._max_field_scroll, len(section.fields),
                )

        spec = self.current_field()
        note_y = layout["bar"].y - 40
        if spec is not None and spec.note:
            surface.blit(
                tiny.render(_fit("note: " + spec.note, tiny, fields_w * 1.7), True, DIM),
                (fields_x, note_y),
            )
        elif spec is not None:
            surface.blit(
                tiny.render(
                    f"range {_format_number(float(spec.minimum))} .. "
                    f"{_format_number(float(spec.maximum))}   step "
                    f"{_format_number(float(spec.step))}",
                    True, DIM,
                ),
                (fields_x, note_y),
            )

        # -- live read-outs ------------------------------------------------
        right_x = layout["right_x"]
        right_w = max(width - right_x - pad + 12, 200)
        right_y = layout["top"] + 2
        surface.blit(small.render("LIVE BUILD", True, ACCENT), (right_x, right_y))
        right_y += 20

        lines: List[Tuple[str, str, Tuple[int, int, int]]] = []
        pack = self.preview_pack
        motor = self.preview_motor
        if pack is not None:
            lines.extend(
                (
                    ("Pack V", f"{pack.pack_nominal_v:.0f} nom / {pack.pack_max_V:.0f} max",
                     TEXT),
                    ("Capacity", f"{pack.pack_capacity_Ah:.1f} Ah", TEXT),
                    ("Energy", human_units(pack.pack_energy_Wh, "Wh"), TEXT),
                    ("Usable", human_units(pack.pack_usable_energy_Wh, "Wh"), TEXT),
                    ("Max current", f"{pack.pack_max_cont_A:.0f} A cont / "
                                    f"{pack.pack_max_burst_A:.0f} A burst", TEXT),
                    ("Pack mass", f"{pack.pack_mass_kg:.1f} kg", TEXT),
                    ("Cell mass", f"{pack.chem.cell_mass_kg:.3f} kg x "
                                  f"{pack.S * pack.P} cells", DIM),
                )
            )
        if motor is not None:
            lines.extend(
                (
                    ("Motor", f"{motor.display_name}", TEXT),
                    ("Peak power", f"{motor.peak_power_kW:.0f} kW "
                                   f"(cont {motor.cont_power_kW:.0f} kW)", TEXT),
                    ("Peak torque", f"{motor.peak_torque_Nm:.0f} Nm to "
                                    f"{motor.torque_end_rpm:.0f} rpm", TEXT),
                    ("Max speed", f"{motor.max_rpm:.0f} rpm", TEXT),
                    ("Motor mass", f"{motor.mass_kg:.1f} kg", DIM),
                )
            )
        estimate = self.estimate
        if estimate is not None:
            lines.extend(
                (
                    ("Top speed", f"{estimate.top_speed_kmh:.0f} km/h", GOOD),
                    ("0-100", f"{estimate.zero_to_100_s:.1f} s", GOOD),
                    ("Range", f"{estimate.range_km:.0f} km at "
                              f"{estimate.cruise_kmh:.0f} km/h", GOOD),
                    ("Uses", f"{estimate.consumption_Wh_per_km:.0f} Wh/km", DIM),
                )
            )
        for label, value, colour in lines[:13]:
            surface.blit(tiny.render(_fit(label, tiny, 92), True, LABEL), (right_x, right_y))
            surface.blit(
                tiny.render(_fit(value, tiny, right_w - 100), True, colour),
                (right_x + 94, right_y),
            )
            right_y += 14

        # -- plots ---------------------------------------------------------
        plot_w = right_w - 12
        plot_h = 126
        bottom_limit = layout["bar"].y - 10
        plots_y = max(min(right_y + 8, bottom_limit - plot_h * 2 - 8), right_y + 8)
        if motor is not None:
            rpms, torques = motor.torque_curve(96)
            _draw_curve(
                surface,
                pygame.Rect(right_x, plots_y, plot_w, plot_h),
                list(rpms),
                [float(t) for t in torques],
                ACCENT,
                "Motor torque envelope",
                font=small,
                small=tiny,
                x_label="rpm",
                y_label="Nm",
                markers=((motor.torque_end_rpm,
                          float(motor.torque_at(motor.torque_end_rpm)),
                          f"{motor.peak_torque_Nm:.0f} Nm"),),
            )
        if pack is not None:
            curve = pack.chem.ocv_curve
            _draw_curve(
                surface,
                pygame.Rect(right_x, plots_y + plot_h + 6, plot_w, plot_h),
                [float(point[0]) * 100.0 for point in curve],
                [float(point[1]) * pack.S for point in curve],
                GOOD,
                f"Pack OCV curve ({pack.S}S, {pack.chem.display_name})",
                font=small,
                small=tiny,
                x_label="% SOC",
                y_label="V",
                markers=((pack.soc * 100.0, pack.ocv(), f"{pack.ocv():.0f} V"),),
            )

        # -- button bar ----------------------------------------------------
        tooltip: Optional[str] = None
        for rect, action, label, tip in layout["buttons"]:
            enabled = self._button_enabled(action)
            hovered = self.hover == f"btn:{action}" and enabled
            if not enabled:
                fill, edge, colour = (21, 25, 32), (44, 50, 60), (76, 84, 94)
            else:
                primary = action == "apply" and self.pending_changes()
                fill = self.ACTIVE_BG if hovered else self.CONTROL_BG
                edge = ACCENT if (hovered or primary) else PANEL_EDGE
                colour = TEXT
            self._panel(surface, rect, fill=fill, edge=edge, radius=5, shadow=False)
            rendered = small.render(label, True, colour)
            surface.blit(
                rendered,
                (rect.centerx - rendered.get_width() // 2, rect.y + 5),
            )
            if hovered:
                tooltip = tip

        # -- hover tooltip (buttons, then rows) ----------------------------
        if tooltip is None and self.hover is not None:
            kind, _, argument = self.hover.partition(":")
            if kind in ("row", "slider", "minus", "plus", "choice"):
                try:
                    spec = self.current_section().fields[int(argument)]
                    tooltip = spec.note or (
                        f"{spec.label}: {_format_number(float(spec.minimum))} .. "
                        f"{_format_number(float(spec.maximum))}, step "
                        f"{_format_number(float(spec.step))}"
                    )
                except (IndexError, ValueError):
                    tooltip = None
            elif kind == "preset":
                tooltip = "click to select, double-click to load"
            elif kind == "tab":
                tooltip = SECTIONS[int(argument)].description
        if tooltip:
            self._draw_tooltip(surface, tooltip)

        # -- status line ---------------------------------------------------
        status = pygame.Rect(pad, layout["bar"].y - 24, width - 2 * pad, 20)
        if pygame.time.get_ticks() < self._message_until and self.message:
            failed = any(word in self.message for word in ("fail", "could not", "empty"))
            self._panel(
                surface, status,
                fill=(56, 30, 30) if failed else (26, 44, 32),
                edge=BAD if failed else GOOD, radius=5, shadow=False,
            )
            surface.blit(
                small.render(_fit(self.message, small, status.width - 16), True, TEXT),
                (status.x + 8, status.y + 2),
            )
        elif self.findings:
            extra = f"  (+{len(self.findings) - 1} more)" if len(self.findings) > 1 else ""
            surface.blit(
                tiny.render(
                    _fit(f"[core] {self.findings[0]}{extra}", tiny, status.width), True, WARN
                ),
                (status.x + 4, status.y + 4),
            )
        else:
            surface.blit(
                tiny.render(
                    "Hover a row to see its range and notes. Changes are previewed "
                    "live; nothing reaches the car until APPLY BUILD.",
                    True, DIM,
                ),
                (status.x + 4, status.y + 4),
            )

        # -- the picker, drawn last so it sits on top ----------------------
        if self.modal is not None:
            self._draw_modal(surface, layout)

    def _button_enabled(self, action: str) -> bool:
        """Whether a button can do anything right now."""
        if action in ("load", "delete"):
            return bool(self.preset_names)
        return True

    def _draw_tooltip(self, surface: pygame.Surface, text: str) -> None:
        """Draw a small floating hint just under the cursor.

        Offset far enough down to clear the row being described, so the hint
        never hides the thing it is explaining.
        """
        tiny = self._fonts_for()["tiny"]
        rendered = tiny.render(text, True, TEXT)
        box = pygame.Rect(
            self.mouse[0] + 16, self.mouse[1] + 28, rendered.get_width() + 16, 20
        )
        box.clamp_ip(pygame.Rect(0, 0, surface.get_width(), surface.get_height()))
        self._panel(surface, box, fill=(40, 50, 64), edge=ACCENT, radius=4)
        surface.blit(rendered, (box.x + 8, box.y + 4))

    def _modal_widgets(self) -> List[Tuple[str, pygame.Rect]]:
        """Hit-test rectangles for the open picker."""
        layout = self._layout_cache
        panel = layout.get("modal_panel")
        if panel is None or self.modal is None:
            return []
        widgets: List[Tuple[str, pygame.Rect]] = []
        options = list(self.modal.get("options", ()))
        start = int(self.scroll_modal)
        visible = int(layout.get("modal_visible", 1))
        for slot in range(visible):
            index = start + slot
            if index >= len(options):
                break
            widgets.append(
                (
                    f"opt:{index}",
                    pygame.Rect(panel.x + 10, panel.y + 40 + slot * 24, panel.width - 20, 22),
                )
            )
        close = layout.get("modal_close")
        if close is not None:
            widgets.append(("mclose", close))
        return widgets

    def _draw_modal(self, surface: pygame.Surface, layout: Dict[str, Any]) -> None:
        """Draw the option picker for a choice row."""
        if self.modal is None:
            return
        panel = layout.get("modal_panel")
        if panel is None:
            return
        fonts = layout["fonts"]
        tiny = fonts["tiny"]
        small = fonts["small"]
        body = fonts["body"]

        dim = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        dim.fill((4, 5, 8, 150))
        surface.blit(dim, (0, 0))

        self._panel(surface, panel, fill=self.PANEL_BG, edge=ACCENT, radius=8)
        surface.blit(body.render(str(self.modal.get("title", "")), True, TEXT),
                     (panel.x + 12, panel.y + 10))
        current = str(self.config.get(str(self.modal.get("key")), ""))
        surface.blit(
            tiny.render(f"current: {current}", True, DIM),
            (panel.x + 12 + body.size(str(self.modal.get("title", "")))[0] + 14,
             panel.y + 15),
        )

        options = list(self.modal.get("options", ()))
        start = int(self.scroll_modal)
        visible = int(layout.get("modal_visible", 1))
        list_top = panel.y + 40
        previous_clip = surface.get_clip()
        surface.set_clip(pygame.Rect(panel.x + 4, list_top, panel.width - 8,
                                     visible * 24))
        for slot in range(visible):
            index = start + slot
            if index >= len(options):
                break
            option = options[index]
            rect = pygame.Rect(panel.x + 10, list_top + slot * 24, panel.width - 20, 22)
            is_current = str(option) == current
            hovered = self.hover == f"opt:{index}"
            if is_current or hovered:
                self._panel(
                    surface, rect,
                    fill=self.ACTIVE_BG if is_current else self.HOVER_BG,
                    edge=ACCENT if is_current else PANEL_EDGE, radius=4, shadow=False,
                )
            surface.blit(
                small.render(_fit(str(option), small, rect.width - 26), True,
                             ACCENT if is_current else TEXT),
                (rect.x + 8, rect.y + 3),
            )
            if is_current:
                surface.blit(tiny.render("selected", True, GOOD),
                             (rect.right - 60, rect.y + 5))
        surface.set_clip(previous_clip)

        if len(options) > visible:
            self._scrollbar(
                surface,
                pygame.Rect(panel.x + 4, list_top, panel.width - 8, visible * 24),
                self.scroll_modal,
                max(len(options) - visible, 0),
                len(options),
            )

        close = layout.get("modal_close")
        if close is not None:
            hovered = self.hover == "mclose"
            self._panel(surface, close,
                        fill=self.ACTIVE_BG if hovered else self.CONTROL_BG,
                        edge=PANEL_EDGE, radius=4, shadow=False)
            rendered = small.render("CLOSE", True, TEXT)
            surface.blit(rendered, (close.centerx - rendered.get_width() // 2, close.y + 3))
        surface.blit(
            tiny.render(
                _fit("click an option to pick it  -  wheel to scroll  -  "
                     "click outside to cancel", tiny, panel.width - 102),
                True, DIM,
            ),
            (panel.x + 12, panel.bottom - 16),
        )



# ======================================================================
# Clipboard helpers (best effort, never fatal)
# ======================================================================
def _copy_to_clipboard(text: str) -> bool:
    """Try to put ``text`` on the system clipboard. Returns success."""
    try:
        if not pygame.scrap.get_init():
            pygame.scrap.init()
        pygame.scrap.put(pygame.SCRAP_TEXT, text.encode("utf-8"))
        return True
    except Exception:  # noqa: BLE001 - clipboard support is optional
        return False


def _read_clipboard() -> str:
    """Try to read the system clipboard. Returns ``""`` on any failure."""
    try:
        if not pygame.scrap.get_init():
            pygame.scrap.init()
        raw = pygame.scrap.get(pygame.SCRAP_TEXT)
        if not raw:
            return ""
        return raw.decode("utf-8", errors="ignore").replace("\x00", "").strip()
    except Exception:  # noqa: BLE001 - clipboard support is optional
        return ""


def _read_import_file() -> str:
    """Read the paste-here fallback file. Returns ``""`` if absent.

    Lines starting with ``#`` are ignored so the shipped file can explain itself
    without its own instructions being fed to the share-code decoder.
    """
    try:
        with open(IMPORT_CODE_PATH, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return ""
    payload = "\n".join(
        line for line in lines if line.strip() and not line.lstrip().startswith("#")
    ).strip()
    return payload


# ======================================================================
# Self-test
# ======================================================================
def _run_tests() -> int:
    """Exercise the garage model, persistence, sharing and drawing."""
    import os
    import tempfile

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    pygame.display.set_mode((640, 400))

    failures: List[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    from core.vehicle import create_demo_vehicle

    vehicle = create_demo_vehicle(initial_soc=0.9)
    config = GarageConfig.from_vehicle(vehicle)

    # -- schema fidelity -------------------------------------------------
    check("captures the live build from the core objects",
          config.get("build.chemistry") == vehicle.pack.chem.name
          and config.get("build.series") == vehicle.pack.S
          and config.get("build.motor_id") == vehicle.motor.id,
          f"{config.get('build.chemistry')} {config.get('build.series')}S"
          f"{config.get('build.parallel')}P {config.get('build.motor_id')}")

    unexplained: List[str] = []
    nested = config.to_vehicle_config()
    flat_base = _flatten("", nested)
    for section in SECTIONS:
        for spec in section.fields:
            # A row is legitimate if it lands on a config key, or if it is the
            # one declared UI-only presentation row (the weight split).
            if spec.key not in flat_base and spec.key not in UI_ONLY_KEYS:
                unexplained.append(spec.key)
    check("every editable row maps to a config key or a declared UI-only row",
          not unexplained, f"unexplained={unexplained}")

    check("core ignores the UI-only sections",
          GarageConfig.from_vehicle_config(nested).values == config.values,
          "round-trip through the nested config is lossless")

    # UI-only input rows must not leak into the config the core is handed: the
    # core ignores unknown keys today, but they would silently bloat presets and
    # share codes, and a future strict schema would reject them.
    leaked = [key for key in UI_ONLY_KEYS if key in flat_base]
    check("UI-only input rows never reach the core config", not leaked,
          f"leaked={leaked}")
    check("converted rows are still present in their core form",
          float(nested["chassis"]["cg_to_front_axle_m"]) > 0.0
          and nested["wheels"]["driven_axle"] in ("rear", "front", "all")
          and isinstance(nested["drivetrain"]["traction_control"], bool),
          f"driven_axle={nested['wheels']['driven_axle']!r}, "
          f"traction_control={nested['drivetrain']['traction_control']!r}")
    check("the core config has no keys the base config lacked",
          set(nested["chassis"]) == set(config.base["chassis"])
          and set(nested["wheels"]) == set(config.base["wheels"]),
          f"chassis extra={set(nested['chassis']) - set(config.base['chassis'])}")

    # -- derived axle geometry ------------------------------------------
    geometry_ok = True
    detail = ""
    for split in (30.0, 42.5, 50.0, 61.0, 70.0):
        probe = config.copy()
        probe.set("chassis.weight_distribution_front_pct", split)
        probe.set("chassis.wheelbase_m", 2.85)
        built = probe.to_vehicle_config()["chassis"]
        total = built["cg_to_front_axle_m"] + built["cg_to_rear_axle_m"]
        share = built["cg_to_rear_axle_m"] / built["wheelbase_m"] * 100.0
        if abs(total - 2.85) > 1e-9 or abs(share - split) > 1e-6:
            geometry_ok = False
            detail = f"split {split}: total {total}, share {share}"
            break
    check("weight split and wheelbase stay consistent with the core relation",
          geometry_ok, detail or "cg_to_front + cg_to_rear == wheelbase for every split")

    # -- clamping --------------------------------------------------------
    clamp_probe = config.copy()
    clamp_probe.set("chassis.mass_kg", 999_999.0)
    clamp_probe.set("wheels.rolling_resistance_coeff", -5.0)
    clamp_probe.set("build.series", 1000)
    check("out-of-range values are clamped to the spec",
          clamp_probe.get("chassis.mass_kg") == 3200.0
          and clamp_probe.get("wheels.rolling_resistance_coeff") == 0.004
          and clamp_probe.get("build.series") == 220,
          f"{clamp_probe.get('chassis.mass_kg')}, "
          f"{clamp_probe.get('wheels.rolling_resistance_coeff')}, "
          f"{clamp_probe.get('build.series')}")
    check("choice rows reject unknown options",
          clamp_probe.set("build.chemistry", "Unobtainium") is None
          and clamp_probe.get("build.chemistry") in available_chemistries(),
          str(clamp_probe.get("build.chemistry")))

    # -- the core accepts what the garage produces -----------------------
    built = build_vehicle(config)
    check("core accepts the garage config",
          abs(built.mass_kg - config.get("chassis.mass_kg")) < 1e-9
          and built.pack.S == config.get("build.series")
          and built.motor.id == config.get("build.motor_id"),
          f"{built.mass_kg:.0f} kg, {built.pack.S}S{built.pack.P}P, {built.motor.id}")
    check("core validation is clean for the stock build",
          not built.validate(emit_warnings=False),
          str(built.validate(emit_warnings=False)))

    # The garage surfaces core's own validate() findings. It must stay quiet on
    # a matched pack/motor and shout when the pack would over-volt the
    # controller, which is the whole point of showing them.
    def live_findings(garage_config: "GarageConfig") -> List[str]:
        """Findings a Garage would display, via the real code path."""
        return Garage(
            garage_config,
            save_path=Path(tempfile.gettempdir()) / "ev_sim_no_presets.json",
        ).findings

    check("garage reports no findings for the stock build",
          not live_findings(config), str(live_findings(config)))
    mismatch = config.copy()
    mismatch.set("build.series", 200)
    mismatch_findings = live_findings(mismatch)
    check("garage flags a pack that would over-volt the controller",
          any("controller" in finding.lower() for finding in mismatch_findings),
          str(mismatch_findings[:1]))

    check("changing the axle layout changes the driven wheels",
          GarageConfig.from_vehicle_config(
              {**config.to_vehicle_config(),
               "wheels": {**config.to_vehicle_config()["wheels"],
                          "driven_axle": "all"}}
          ).get("wheels.driven_axle") == "AWD (dual motor)",
          str(config.get("wheels.driven_axle")))

    # -- estimates -------------------------------------------------------
    estimate = estimate_build(config.to_vehicle_config(), config.make_motor(),
                              config.make_pack())
    check("top speed estimate is plausible",
          120.0 < estimate.top_speed_kmh < 400.0, f"{estimate.top_speed_kmh:.0f} km/h")
    check("0-100 estimate is plausible",
          1.5 < estimate.zero_to_100_s < 25.0, f"{estimate.zero_to_100_s:.1f} s")
    check("range estimate is plausible",
          80.0 < estimate.range_km < 1200.0, f"{estimate.range_km:.0f} km")

    heavy = config.copy()
    heavy.set("chassis.mass_kg", config.get("chassis.mass_kg") + 900.0)
    heavy_estimate = estimate_build(heavy.to_vehicle_config(), heavy.make_motor(),
                                    heavy.make_pack())
    check("a heavier build predicts a slower 0-100",
          heavy_estimate.zero_to_100_s >= estimate.zero_to_100_s,
          f"{estimate.zero_to_100_s:.1f} -> {heavy_estimate.zero_to_100_s:.1f} s")

    brick = config.copy()
    brick.set("chassis.drag_coefficient", 0.70)
    brick_estimate = estimate_build(brick.to_vehicle_config(), brick.make_motor(),
                                    brick.make_pack())
    check("a draggier build predicts a lower top speed",
          brick_estimate.top_speed_kmh < estimate.top_speed_kmh,
          f"{estimate.top_speed_kmh:.0f} -> {brick_estimate.top_speed_kmh:.0f} km/h")

    slippery = config.copy()
    slippery.set("tires.mu_base", 0.40)
    slippery_estimate = estimate_build(
        slippery.to_vehicle_config(), slippery.make_motor(), slippery.make_pack()
    )
    check("low grip predicts a slower 0-100",
          slippery_estimate.zero_to_100_s >= estimate.zero_to_100_s,
          f"{estimate.zero_to_100_s:.1f} -> {slippery_estimate.zero_to_100_s:.1f} s")

    # -- the build really does reach the physics -------------------------
    from core.vehicle import Controls

    def launch(vehicle_config: Dict[str, Any], motor_id: str, seconds: float = 2.0) -> float:
        probe = GarageConfig.from_vehicle_config(
            {**vehicle_config, "build": {**vehicle_config["build"], "motor_id": motor_id}}
        )
        car = build_vehicle(probe, initial_soc=1.0)
        dt = 1.0 / 60.0
        for _ in range(int(seconds / dt)):
            car.step(dt, Controls(throttle=1.0))
        return float(car.get_state().speed_kmh)

    weak = launch(config.to_vehicle_config(), "pmsm-50kw")
    strong = launch(config.to_vehicle_config(), "pmsm-335kw")
    check("a bigger motor really does accelerate harder in the simulation",
          strong > weak + 5.0,
          f"50 kW {weak:.1f} km/h vs 335 kW {strong:.1f} km/h after 2 s")

    # -- presets ---------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "garage_save.json"
        garage = Garage(config, save_path=path)
        check("a missing save file yields no presets instead of raising",
              garage.preset_names == [], str(garage.preset_names))

        message = garage.save_preset("track day")
        check("preset saves to disk", path.exists() and "saved" in message, message)
        check("preset round-trips through the file",
              load_presets(path).get("track day", {}).get("chassis")
              == config.to_vehicle_config()["chassis"],
              "chassis section survives")

        garage.adjust_value(1)  # nudge a value in the CHASSIS section
        check("editing marks the build as unapplied", garage.pending_changes())
        message = garage.load_preset("track day")
        check("loading a preset restores the saved value",
              not garage.pending_changes() and "loaded" in message, message)
        message = garage.delete_preset("track day")
        check("preset delete removes it from disk",
              load_presets(path) == {} and "deleted" in message, message)

        # -- share codes -------------------------------------------------
        garage = Garage(config, save_path=path)
        code = garage.share_code()
        check("share code has the documented prefix",
              code.startswith(SHARE_PREFIX) and " " not in code, code[:24] + "...")
        decoded = decode_share_code(code)
        check("share code round-trips",
              GarageConfig.from_vehicle_config(decoded).values == config.values,
              "decoded values match")

        wrapped = "  " + code[:20] + "\n" + code[20:] + "\n"
        check("share code tolerates whitespace and newlines",
              decode_share_code(wrapped)["chassis"] == decoded["chassis"])

        for bad, why in (
            ("hello world", "wrong prefix"),
            (SHARE_PREFIX + "!!!!", "bad base64"),
            (SHARE_PREFIX + base64.urlsafe_b64encode(b"not json").decode(), "bad JSON"),
            (SHARE_PREFIX + base64.urlsafe_b64encode(b'{"chassis":{}}').decode(),
             "missing sections"),
            (SHARE_PREFIX, "empty payload"),
        ):
            try:
                decode_share_code(bad)
                check(f"share code rejects {why}", False, "no error raised")
            except ValueError:
                check(f"share code rejects {why}", True)

        garage.import_share_code(code)
        check("import leaves the editor usable",
              garage.preview_pack is not None and garage.estimate is not None,
              f"{garage.estimate.top_speed_kmh:.0f} km/h predicted")

        # -- interaction: everything must be reachable with the mouse ----
        surface = pygame.Surface((1280, 720))

        def redraw() -> None:
            """Redraw so the widget rectangles exist before a simulated click."""
            garage.draw(surface, mouse=garage.mouse)

        def hit(widget_id: str) -> Tuple[int, int]:
            for candidate, rect in garage._widgets:
                if candidate == widget_id:
                    return rect.center
            raise AssertionError(f"no widget {widget_id!r} in the current layout")

        def click(widget_id: str) -> Optional[str]:
            """Simulate a full hover + left click on a widget's centre."""
            pos = hit(widget_id)
            garage.handle_event(
                pygame.event.Event(
                    pygame.MOUSEMOTION, pos=pos, rel=(0, 0), buttons=(0, 0, 0)
                )
            )
            result = garage.handle_event(
                pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1)
            )
            garage.handle_event(
                pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=1)
            )
            return result

        def click_at(pos: Tuple[int, int]) -> Optional[str]:
            garage.handle_event(
                pygame.event.Event(
                    pygame.MOUSEMOTION, pos=pos, rel=(0, 0), buttons=(0, 0, 0)
                )
            )
            result = garage.handle_event(
                pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1)
            )
            garage.handle_event(
                pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=1)
            )
            return result

        garage.section_index = 0
        garage.field_index = 0
        redraw()

        click("tab:1")
        check("clicking a tab switches section",
              garage.current_section().title == "BATTERY",
              garage.current_section().title)
        redraw()

        click("tab:0")
        redraw()
        click("row:3")
        check("clicking a row selects it", garage.field_index == 3, str(garage.field_index))

        # Slider: clicking near the right end must push the value up.
        track_rect = next(
            rect for widget_id, rect in garage._widgets if widget_id == "slider:3"
        )
        before = float(garage.config.get(SECTIONS[0].fields[3].key))
        click_at((track_rect.right - 2, track_rect.centery))
        after = float(garage.config.get(SECTIONS[0].fields[3].key))
        check("clicking a slider track sets the value",
              after > before and after <= float(SECTIONS[0].fields[3].maximum),
              f"{before} -> {after} "
              f"(range {SECTIONS[0].fields[3].minimum}..{SECTIONS[0].fields[3].maximum})")

        # Dragging keeps updating while the button is held down: press on the
        # track, move, and only then release (a release ends the drag).
        mid = track_rect.x + track_rect.width // 4
        dragged = float(garage.config.get(SECTIONS[0].fields[3].key))
        garage.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                pos=(track_rect.x + track_rect.width - 4, track_rect.centery),
                button=1,
            )
        )
        held = float(garage.config.get(SECTIONS[0].fields[3].key))
        garage.handle_event(
            pygame.event.Event(
                pygame.MOUSEMOTION, pos=(mid, track_rect.centery), rel=(-5, 0),
                buttons=(1, 0, 0),
            )
        )
        check("dragging the held slider keeps updating the value",
              garage.dragging is not None
              and float(garage.config.get(SECTIONS[0].fields[3].key)) < held,
              f"{held} -> {garage.config.get(SECTIONS[0].fields[3].key)}")
        garage.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONUP, pos=(mid, track_rect.centery), button=1
            )
        )
        check("releasing the button ends the drag", garage.dragging is None,
              str(garage.dragging))

        # Steppers.
        redraw()
        stepped = float(garage.config.get(SECTIONS[0].fields[3].key))
        click("plus:3")
        check("the + stepper raises the value",
              float(garage.config.get(SECTIONS[0].fields[3].key)) > stepped,
              f"{stepped} -> {garage.config.get(SECTIONS[0].fields[3].key)}")
        redraw()
        click("minus:3")
        check("the - stepper lowers it again",
              abs(float(garage.config.get(SECTIONS[0].fields[3].key)) - stepped) < 1e-9,
              f"{garage.config.get(SECTIONS[0].fields[3].key)}")

        # Choice row: the value opens a picker, an option picks it, and a click
        # outside cancels.
        choice_index = next(
            index for index, spec in enumerate(SECTIONS[0].fields)
            if spec.kind == "choice"
        )
        choice_key = SECTIONS[0].fields[choice_index].key
        redraw()
        click(f"choice:{choice_index}")
        check("clicking a value opens the option picker", garage.modal is not None,
              str(garage.modal and garage.modal.get("title")))

        redraw()
        before_choice = garage.config.get(choice_key)
        option_index = next(
            index for index, name in enumerate(garage.modal["options"])
            if name != before_choice
        )
        click(f"opt:{option_index}")
        check("clicking an option sets the value and closes the picker",
              garage.modal is None
              and garage.config.get(choice_key) != before_choice,
              f"{before_choice} -> {garage.config.get(choice_key)}")

        redraw()
        click(f"choice:{choice_index}")
        redraw()
        kept = garage.config.get(choice_key)
        click_at((surface.get_width() // 2, 8))
        check("clicking outside the picker cancels it",
              garage.modal is None and garage.config.get(choice_key) == kept,
              f"modal={garage.modal}, value unchanged={kept}")

        # Scrolling. The CHASSIS tab fits on a big window, so this is checked on
        # a small one where the list genuinely overflows - and the clamp is
        # checked on the big window where it genuinely does not.
        redraw()
        big_area = garage._layout_cache["fields_area"]
        garage.mouse = big_area.center
        garage.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-3,
                                               flipped=False))
        check("the wheel does not scroll a list that already fits",
              garage.scroll_fields == 0.0 and garage._max_field_scroll == 0,
              f"scroll={garage.scroll_fields}, max={garage._max_field_scroll}")

        compact = pygame.Surface((640, 480))
        garage.section_index = 0
        garage.draw(compact)
        compact_area = garage._layout_cache["fields_area"]
        garage.mouse = compact_area.center
        garage.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-3,
                                               flipped=False))
        check("the wheel scrolls the field list when it overflows",
              garage.scroll_fields > 0.0,
              f"scroll={garage.scroll_fields} of {garage._max_field_scroll} rows")
        garage.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=9,
                                               flipped=False))
        check("scrolling is clamped at the top", garage.scroll_fields == 0.0,
              str(garage.scroll_fields))
        garage.draw(surface)

        # Hover is recomputed on the next frame's draw (that is what real
        # frames do), so move the mouse and then redraw. At a row's centre the
        # widget found is the slider that sits on top of the row - either way it
        # belongs to row 2, which is what the tooltip keys off.
        row_two = next(r for wid, r in garage._widgets if wid == "row:2")
        garage.handle_event(
            pygame.event.Event(
                pygame.MOUSEMOTION, pos=row_two.center, rel=(0, 0), buttons=(0, 0, 0)
            )
        )
        redraw()
        check("hovering a row is tracked for tooltips",
              garage.hover is not None and garage.hover.endswith(":2"),
              f"hover={garage.hover!r} over row 2")

        # SAVE with no typing at all: the name is derived from the build.
        garage.section_index = 0
        redraw()
        names_before = len(garage.preset_names)
        click("btn:save")
        check("SAVE PRESET works with a click and no typing",
              len(garage.preset_names) == names_before + 1,
              str(garage.preset_names))
        saved_name = garage.config.name
        check("the preset is auto-named from the build",
              garage.config.get("build.motor_id") in saved_name
              and "S" in saved_name and "P" in saved_name,
              repr(saved_name))
        redraw()
        click("btn:delete")
        check("DELETE PRESET removes it again",
              len(garage.preset_names) == names_before, str(garage.preset_names))

        # APPLY / CLOSE come back as actions for the host.
        click("btn:save")  # leave a preset so LOAD is enabled
        redraw()
        garage.preset_index = 0
        before_motor = garage.config.get("build.motor_id")
        other = next(m for m in available_motors() if m != before_motor)
        garage.config.set("build.motor_id", other)
        garage._refresh_derived()
        redraw()
        click("btn:load")
        check("LOAD PRESET restores the saved build",
              garage.config.get("build.motor_id") == before_motor,
              f"{other} -> {garage.config.get('build.motor_id')}")

        redraw()
        check("APPLY BUILD asks the host to apply",
              click("btn:apply") == "apply", "")
        redraw()
        check("CLOSE asks the host to close",
              click("btn:close") == "close", "")

        # The headline requirement: a complete edit -> save -> apply flow using
        # mouse events only. No KEYDOWN is ever delivered.
        mouse_only = Garage(config, save_path=path)
        mouse_only.draw(surface)
        mouse_only.section_index = 1
        mouse_only.draw(surface)
        series_key = "build.series"
        series_before = int(mouse_only.config.get(series_key))
        for widget_id in ("plus:1", "plus:1", "plus:1"):
            rect = next(r for wid, r in mouse_only._widgets if wid == widget_id)
            mouse_only.handle_event(
                pygame.event.Event(
                    pygame.MOUSEMOTION, pos=rect.center, rel=(0, 0), buttons=(0, 0, 0)
                )
            )
            mouse_only.handle_event(
                pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=rect.center, button=1)
            )
            mouse_only.handle_event(
                pygame.event.Event(pygame.MOUSEBUTTONUP, pos=rect.center, button=1)
            )
            mouse_only.draw(surface)
        changed = int(mouse_only.config.get(series_key)) != series_before
        for widget_id in ("btn:save", "btn:load", "btn:apply"):
            rect = next(r for wid, r in mouse_only._widgets if wid == widget_id)
            mouse_only.handle_event(
                pygame.event.Event(
                    pygame.MOUSEMOTION, pos=rect.center, rel=(0, 0), buttons=(0, 0, 0)
                )
            )
            mouse_only.handle_event(
                pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=rect.center, button=1)
            )
            mouse_only.handle_event(
                pygame.event.Event(pygame.MOUSEBUTTONUP, pos=rect.center, button=1)
            )
            mouse_only.draw(surface)
        check("a whole edit/save/load/apply flow needs no key press at all",
              changed and mouse_only.preset_names, f"series changed={changed}")

        # Keyboard remains available as a convenience, never a requirement.
        redraw()
        close_action = garage.handle_event(
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB, mod=0, unicode="")
        )
        check("TAB still closes as a shortcut", close_action == "close", str(close_action))

        # -- drawing -----------------------------------------------------
        try:
            garage.section_index = 0
            garage.draw(surface)
            check("draws on a full-size window", True)
        except Exception as exc:  # noqa: BLE001 - any failure is a test failure
            check("draws on a full-size window", False, repr(exc))

        garage.section_index = len(SECTIONS) - 1
        small_surface = pygame.Surface((520, 380))
        try:
            garage.draw(small_surface)
            check("draws on a small window", True)
        except Exception as exc:  # noqa: BLE001
            check("draws on a small window", False, repr(exc))
        garage.section_index = 0
        for size in ((1280, 720), (1024, 640), (900, 560), (640, 480)):
            try:
                garage.draw(pygame.Surface(size))
            except Exception as exc:  # noqa: BLE001
                check(f"draws at {size[0]}x{size[1]}", False, repr(exc))
                break
        else:
            check("draws at every tested window size", True,
                  "640x480 .. 1280x720")

        # Every registered widget must be inside the window, or a click can
        # never reach it.
        garage.draw(surface)
        outside = [
            widget_id for widget_id, rect in garage._widgets
            if not pygame.Rect(0, 0, 1280, 720).contains(rect)
        ]
        check("every widget rectangle is on screen", not outside, f"outside={outside}")

        # -- a broken preset file must not be fatal ----------------------
        path.write_text("{not json", encoding="utf-8")
        check("a corrupt preset file is ignored",
              load_presets(path) == {}, "returned no presets")

        # -- every chemistry / motor combination must build --------------
        broken: List[str] = []
        for chemistry in available_chemistries():
            for motor_id in ("mid-bldc-20kw", "pmsm-150kw", "hub-8000w"):
                probe = config.copy()
                probe.set("build.chemistry", chemistry)
                probe.set("build.motor_id", motor_id)
                if chemistry == "Pb-Acid":
                    probe.set("build.series", 6)
                    probe.set("build.parallel", 1)
                try:
                    build_vehicle(probe, initial_soc=1.0)
                except Exception as exc:  # noqa: BLE001
                    broken.append(f"{chemistry}/{motor_id}: {exc}")
        check("every chemistry builds against every sampled motor", not broken,
              "; ".join(broken[:3]))

    pygame.quit()
    print()
    if failures:
        print(f"RESULT: {len(failures)} garage check(s) failed -> {failures}")
        return 1
    print("RESULT: all garage checks passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_run_tests())
