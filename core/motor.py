"""Data-driven traction motor model.

Every motor in ``config/motors.json`` describes its own torque curve and a 2D
efficiency table.  This module turns those numbers into four behaviours:

* :meth:`Motor.torque_at` - the classic constant-torque -> constant-power ->
  field-weakening envelope.
* :meth:`Motor.efficiency_at` - bilinear interpolation into the efficiency map.
* :meth:`Motor.power_required` - mechanical torque converted to electrical
  power (with motoring / regen asymmetry).
* :meth:`Motor.max_regen_torque` - regenerating torque, tapering at high RPM.

All public methods accept either a Python scalar or a NumPy array and return
the matching shape (a ``float`` for scalar input), so callers can vectorise
across the four wheels.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

__all__ = ["Motor", "load_motors", "available_motors"]

#: Default location of the motor database.
DEFAULT_CONFIG_PATH: Path = (
    Path(__file__).resolve().parent.parent / "config" / "motors.json"
)

#: Minimum efficiency ever returned (prevents division blow-ups at 0 torque).
MIN_EFFICIENCY: float = 0.05

#: Torque magnitude below which electrical power is reported as zero (Nm).
ZERO_TORQUE_EPS: float = 1e-6


def _result(scalar: bool, array: np.ndarray) -> Union[float, np.ndarray]:
    """Return a float for scalar inputs, otherwise the array itself."""
    return float(array[0]) if scalar else array


def _bilinear(
    x: np.ndarray,
    y: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    table: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation of ``table[row= xs, col= ys]``.

    Args:
        x: Query values along the first (row) axis.
        y: Query values along the second (column) axis.
        xs: Ascending axis values for the rows of ``table``.
        ys: Ascending axis values for the columns of ``table``.
        table: 2D array of shape ``(len(xs), len(ys))``.

    Returns:
        Interpolated values, flattened to the broadcast shape of ``x``/``y``.
        Queries outside the axes are clamped to the edge cells.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    i = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
    j = np.clip(np.searchsorted(ys, y, side="right") - 1, 0, len(ys) - 2)

    x0, x1 = xs[i], xs[i + 1]
    y0, y1 = ys[j], ys[j + 1]

    tx = np.clip((x - x0) / np.maximum(x1 - x0, 1e-12), 0.0, 1.0)
    ty = np.clip((y - y0) / np.maximum(y1 - y0, 1e-12), 0.0, 1.0)

    z00 = table[i, j]
    z01 = table[i, np.minimum(j + 1, len(ys) - 1)]
    z10 = table[np.minimum(i + 1, len(xs) - 1), j]
    z11 = table[np.minimum(i + 1, len(xs) - 1), np.minimum(j + 1, len(ys) - 1)]

    return (
        (1.0 - tx) * (1.0 - ty) * z00
        + (1.0 - tx) * ty * z01
        + tx * (1.0 - ty) * z10
        + tx * ty * z11
    )


@dataclass(frozen=True)
class MotorSpec:
    """Raw motor description exactly as it appears in ``motors.json``."""

    id: str
    display_name: str
    type: str
    peak_power_kW: float
    cont_power_kW: float
    peak_torque_Nm: float
    torque_end_rpm: float
    field_weaken_rpm: float
    fw_torque_exponent: float
    max_rpm: float
    regen_max_Nm: float
    regen_taper_rpm: float
    controller_max_A: float
    controller_max_V: float
    mass_kg: float
    pole_pairs: int
    efficiency_map: Dict[str, Any]

    @classmethod
    def from_dict(cls, motor_id: str, data: Dict[str, Any]) -> "MotorSpec":
        """Build a spec from one entry of ``motors.json``.

        Args:
            motor_id: Key of the motor in the JSON file.
            data: Raw mapping for the motor.

        Returns:
            A validated :class:`MotorSpec`.

        Raises:
            ValueError: If a required field is missing.
        """
        required = (
            "peak_power_kW",
            "cont_power_kW",
            "peak_torque_Nm",
            "torque_end_rpm",
            "max_rpm",
            "efficiency_map",
        )
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"motor {motor_id!r} is missing keys: {missing}")

        max_rpm = float(data["max_rpm"])
        return cls(
            id=motor_id,
            display_name=str(data.get("display_name", motor_id)),
            type=str(data.get("type", "pmsm")),
            peak_power_kW=float(data["peak_power_kW"]),
            cont_power_kW=float(data["cont_power_kW"]),
            peak_torque_Nm=float(data["peak_torque_Nm"]),
            torque_end_rpm=float(data["torque_end_rpm"]),
            field_weaken_rpm=float(data.get("field_weaken_rpm", max_rpm)),
            fw_torque_exponent=float(data.get("fw_torque_exponent", 1.3)),
            max_rpm=max_rpm,
            regen_max_Nm=float(data.get("regen_max_Nm", 0.0)),
            regen_taper_rpm=float(data.get("regen_taper_rpm", max_rpm * 0.8)),
            controller_max_A=float(data.get("controller_max_A", math.inf)),
            controller_max_V=float(data.get("controller_max_V", math.inf)),
            mass_kg=float(data.get("mass_kg", 0.0)),
            pole_pairs=int(data.get("pole_pairs", 1)),
            efficiency_map=dict(data["efficiency_map"]),
        )


def load_motors(
    path: Union[str, Path, None] = None,
) -> Dict[str, MotorSpec]:
    """Load every motor spec from the JSON database.

    Args:
        path: Optional override for the ``motors.json`` location.

    Returns:
        Mapping of motor id to :class:`MotorSpec`.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        mid: MotorSpec.from_dict(mid, data)
        for mid, data in raw["motors"].items()
    }


def available_motors(path: Union[str, Path, None] = None) -> List[str]:
    """Return the ids of every motor in the database."""
    return list(load_motors(path).keys())


class Motor:
    """A traction motor with a torque envelope and efficiency map.

    Args:
        motor: Motor id from ``motors.json`` (e.g. ``"pmsm-150kw"``) or a
            ready-made :class:`MotorSpec`.
        config_path: Optional override for the motor database.

    Raises:
        ValueError: If the id is not present in the database or the efficiency
            map is malformed.
    """

    def __init__(
        self,
        motor: Union[str, MotorSpec],
        config_path: Union[str, Path, None] = None,
    ) -> None:
        if isinstance(motor, MotorSpec):
            self.spec: MotorSpec = motor
        else:
            db = load_motors(config_path)
            if motor not in db:
                raise ValueError(
                    f"unknown motor {motor!r}; available: {sorted(db)}"
                )
            self.spec = db[motor]

        spec = self.spec
        self.id: str = spec.id
        self.display_name: str = spec.display_name
        self.type: str = spec.type
        self.peak_power_W: float = spec.peak_power_kW * 1000.0
        self.cont_power_W: float = spec.cont_power_kW * 1000.0
        self.peak_torque_Nm: float = spec.peak_torque_Nm
        self.torque_end_rpm: float = spec.torque_end_rpm
        self.field_weaken_rpm: float = spec.field_weaken_rpm
        self.fw_torque_exponent: float = spec.fw_torque_exponent
        self.max_rpm: float = spec.max_rpm
        self.regen_max_Nm: float = spec.regen_max_Nm
        self.regen_taper_rpm: float = spec.regen_taper_rpm
        self.controller_max_A: float = spec.controller_max_A
        self.controller_max_V: float = spec.controller_max_V
        self.mass_kg: float = spec.mass_kg
        self.pole_pairs: int = spec.pole_pairs

        # -- efficiency table ---------------------------------------------
        emap = spec.efficiency_map
        try:
            rpm_axis = np.asarray(emap["rpm_axis"], dtype=float)
            load_axis = np.asarray(emap["load_axis"], dtype=float)
            table = np.asarray(emap["values"], dtype=float)
        except (KeyError, TypeError) as exc:  # pragma: no cover - guard
            raise ValueError(
                f"motor {self.id!r}: malformed efficiency_map ({exc})"
            ) from exc
        if table.shape != (rpm_axis.size, load_axis.size):
            raise ValueError(
                f"motor {self.id!r}: efficiency_map values shape {table.shape} "
                f"does not match rpm_axis x load_axis "
                f"({rpm_axis.size} x {load_axis.size})"
            )
        self.rpm_axis: np.ndarray = rpm_axis
        self.load_axis: np.ndarray = load_axis
        self.efficiency_table: np.ndarray = table

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Motor({self.id}, {self.type}, "
            f"{self.peak_power_kW:.1f} kW peak, "
            f"{self.peak_torque_Nm:.0f} Nm, {self.max_rpm:.0f} rpm)"
        )

    @property
    def peak_power_kW(self) -> float:
        """Peak electrical power (kW)."""
        return self.peak_power_W / 1000.0

    @property
    def cont_power_kW(self) -> float:
        """Continuous electrical power (kW)."""
        return self.cont_power_W / 1000.0

    @property
    def base_speed_omega(self) -> float:
        """Angular speed at the end of the constant-torque region (rad/s)."""
        return self.torque_end_rpm * 2.0 * math.pi / 60.0

    # ------------------------------------------------------------------
    # Torque envelope
    # ------------------------------------------------------------------
    def torque_at(self, rpm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Maximum motor torque available at a given shaft speed.

        Three regions are modelled:

        1. **Constant torque** (``rpm <= torque_end_rpm``): the full
           :attr:`peak_torque_Nm`.
        2. **Constant power** (``torque_end_rpm < rpm <= field_weaken_rpm``):
           ``T = P_peak / omega``, capped at the peak torque.
        3. **Field weakening** (``rpm > field_weaken_rpm``): torque decays as
           ``T = T_fw * (fw_rpm / rpm) ** fw_torque_exponent``.

        Torque is zero at and above :attr:`max_rpm`, and full peak torque at
        standstill (needed for a standing start).

        Args:
            rpm: Shaft speed in rpm (scalar or array).

        Returns:
            Available torque in Nm (positive = motoring).
        """
        r = np.asarray(rpm, dtype=float)
        scalar = r.ndim == 0
        r = np.atleast_1d(r)

        omega = np.maximum(r, 1e-6) * 2.0 * math.pi / 60.0
        t_const = self.peak_torque_Nm
        t_const_power = np.minimum(self.peak_power_W / omega, self.peak_torque_Nm)

        omega_fw = max(self.field_weaken_rpm, 1e-6) * 2.0 * math.pi / 60.0
        t_fw_base = min(self.peak_power_W / omega_fw, self.peak_torque_Nm)
        t_field_weaken = t_fw_base * (
            max(self.field_weaken_rpm, 1e-6) / np.maximum(r, 1e-6)
        ) ** self.fw_torque_exponent

        t = np.where(
            r <= self.torque_end_rpm,
            t_const,
            np.where(r <= self.field_weaken_rpm, t_const_power, t_field_weaken),
        )
        t = np.where(r >= self.max_rpm, 0.0, t)
        t = np.where(r <= 0.0, t_const, t)
        return _result(scalar, t)

    # ------------------------------------------------------------------
    # Efficiency
    # ------------------------------------------------------------------
    def efficiency_at(
        self,
        rpm: Union[float, np.ndarray],
        load: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Bilinear interpolation of the efficiency map.

        Args:
            rpm: Shaft speed in rpm.
            load: Load fraction in ``[0, 1]``, defined as
                ``|torque| / peak_torque_Nm``. Values outside the axis range
                are clamped.

        Returns:
            Efficiency in ``[MIN_EFFICIENCY, 1.0]``.
        """
        r = np.asarray(rpm, dtype=float)
        l = np.asarray(load, dtype=float)
        scalar = r.ndim == 0 and l.ndim == 0

        r = np.atleast_1d(r)
        l = np.atleast_1d(np.broadcast_to(l, r.shape))

        eff = _bilinear(r, l, self.rpm_axis, self.load_axis, self.efficiency_table)
        eff = np.clip(eff, MIN_EFFICIENCY, 1.0)
        return _result(scalar, eff)

    def efficiency_at_torque(
        self, rpm: Union[float, np.ndarray], torque_Nm: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Convenience wrapper around :meth:`efficiency_at` taking torque (Nm)."""
        torque = np.asarray(torque_Nm, dtype=float)
        load = np.abs(torque) / max(self.peak_torque_Nm, 1e-9)
        return self.efficiency_at(rpm, load)

    # ------------------------------------------------------------------
    # Power
    # ------------------------------------------------------------------
    def mechanical_power(
        self, rpm: Union[float, np.ndarray], torque_Nm: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Mechanical shaft power (W). Positive = motoring, negative = regen."""
        r = np.asarray(rpm, dtype=float)
        t = np.asarray(torque_Nm, dtype=float)
        scalar = r.ndim == 0 and t.ndim == 0
        omega = np.atleast_1d(r) * 2.0 * math.pi / 60.0
        mech = np.atleast_1d(t) * omega
        return _result(scalar, mech)

    def power_required(
        self, rpm: Union[float, np.ndarray], torque_Nm: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Electrical power drawn (or returned) to produce a shaft torque.

        Motoring divides the mechanical power by the efficiency; regeneration
        multiplies it, so regen always recovers less than the mechanical
        energy taken out of the vehicle.

        Args:
            rpm: Shaft speed in rpm.
            torque_Nm: Shaft torque in Nm (positive = motoring).

        Returns:
            Electrical power at the motor terminals in watts.
            Positive = battery discharge, negative = regen into the battery.
        """
        r = np.asarray(rpm, dtype=float)
        t = np.asarray(torque_Nm, dtype=float)
        scalar = r.ndim == 0 and t.ndim == 0
        r = np.atleast_1d(r)
        t = np.atleast_1d(np.broadcast_to(t, r.shape))

        omega = r * 2.0 * math.pi / 60.0
        mech = t * omega

        load = np.abs(t) / max(self.peak_torque_Nm, 1e-9)
        eff = np.clip(
            np.atleast_1d(_bilinear(r, load, self.rpm_axis, self.load_axis, self.efficiency_table)),
            MIN_EFFICIENCY,
            1.0,
        )

        motoring = mech / eff
        regen = mech * eff
        elec = np.where(mech >= 0.0, motoring, regen)
        elec = np.where(np.abs(t) < ZERO_TORQUE_EPS, 0.0, elec)
        return _result(scalar, elec)

    # ------------------------------------------------------------------
    # Regeneration
    # ------------------------------------------------------------------
    def max_regen_torque(
        self, rpm: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Maximum regenerative braking torque at a given speed.

        The regen envelope tapers linearly from :attr:`regen_taper_rpm` to
        zero at :attr:`max_rpm` (back-EMF runs out of headroom), and is also
        capped by the controller power ``controller_max_A * controller_max_V``.
        There is no regen at standstill.

        Args:
            rpm: Shaft speed in rpm.

        Returns:
            Regenerative torque in Nm (``<= 0``).
        """
        r = np.asarray(rpm, dtype=float)
        scalar = r.ndim == 0
        r = np.atleast_1d(r)

        span = max(self.max_rpm - self.regen_taper_rpm, 1e-9)
        taper = np.clip((self.max_rpm - r) / span, 0.0, 1.0)
        magnitude = self.regen_max_Nm * taper

        omega = np.maximum(r, 1e-6) * 2.0 * math.pi / 60.0
        p_cap = self.controller_max_A * self.controller_max_V
        if math.isfinite(p_cap):
            magnitude = np.minimum(magnitude, p_cap / omega)

        magnitude = np.where(r >= self.max_rpm, 0.0, magnitude)
        magnitude = np.where(r <= 0.0, 0.0, magnitude)
        return _result(scalar, -magnitude)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def torque_curve(
        self, points: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample the torque envelope for plotting/debugging.

        Args:
            points: Number of samples between 0 and ``max_rpm``.

        Returns:
            ``(rpm, torque_Nm)`` arrays.
        """
        rpm = np.linspace(0.0, self.max_rpm, max(int(points), 2))
        return rpm, np.asarray(self.torque_at(rpm), dtype=float)

    def power_curve(
        self, points: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample the electrical power at the torque limit, for plotting.

        Args:
            points: Number of samples between 0 and ``max_rpm``.

        Returns:
            ``(rpm, electrical_power_W)`` arrays.
        """
        rpm, torque = self.torque_curve(points)
        power = np.asarray(self.power_required(rpm, torque), dtype=float)
        return rpm, power

    def validate(self, *, emit_warnings: bool = True) -> List[str]:
        """Check the motor's internal consistency.

        Verifies that the torque and power ratings agree at the base speed and
        that the efficiency map is well formed.

        Args:
            emit_warnings: Also emit findings through :mod:`warnings`.

        Returns:
            A list of human readable findings.
        """
        import warnings

        msgs: List[str] = []

        implied_power = (
            self.peak_torque_Nm * self.torque_end_rpm * 2.0 * math.pi / 60.0
        )
        if implied_power > 0.0:
            ratio = implied_power / max(self.peak_power_W, 1e-9)
            if ratio > 1.25 or ratio < 0.75:
                msgs.append(
                    f"peak_torque x torque_end_rpm implies {implied_power / 1000.0:.1f} kW "
                    f"but peak_power_kW is {self.peak_power_kW:.1f} kW "
                    f"({ratio * 100:.0f}%)."
                )

        if not np.all(np.diff(self.rpm_axis) > 0):
            msgs.append("efficiency_map rpm_axis is not strictly increasing.")
        if not np.all(np.diff(self.load_axis) > 0):
            msgs.append("efficiency_map load_axis is not strictly increasing.")
        if self.efficiency_table.max() > 1.0:
            msgs.append("efficiency_map contains values greater than 1.0.")
        if self.efficiency_table.max() < 0.5:
            msgs.append("efficiency_map peak is below 50%; check the table units.")

        if emit_warnings:
            for msg in msgs:
                warnings.warn(f"[Motor:{self.id}] {msg}", stacklevel=2)
        return msgs
