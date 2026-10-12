"""Data-driven battery pack model.

The model is split in two layers:

* :class:`Chemistry` - an immutable description of a single cell (voltage
  limits, resistance, capacity, mass, Peukert exponent, OCV curve, thermal
  properties).  Everything is loaded from ``config/batteries.json``.
* :class:`BatteryPack` - a pack built from a cell chemistry plus a series
  count ``S`` and a parallel count ``P``.  It exposes the electro-thermal
  behaviour the vehicle needs: open-circuit voltage, exact current solution
  for a power demand, SOC integration (with Peukert effect for lead-acid),
  cell heating/cooling and charge/discharge power limits.

Series/parallel math (all per-cell values come from the JSON):

.. math::

    V_{nom,pack} = V_{nom,cell} \\cdot S

    C_{pack} = C_{cell} \\cdot P

    R_{pack} = \\frac{R_{cell} \\cdot S}{P}

    I_{cont,pack} = I_{cont,cell} \\cdot P

    E_{pack} = V_{nom,pack} \\cdot C_{pack}
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "Chemistry",
    "BatteryPack",
    "load_chemistries",
    "available_chemistries",
]

#: Default location of the chemistry database.
DEFAULT_CONFIG_PATH: Path = (
    Path(__file__).resolve().parent.parent / "config" / "batteries.json"
)

#: Keys that every chemistry entry must define.
_REQUIRED_KEYS: Tuple[str, ...] = (
    "nominal_v",
    "max_v",
    "min_v",
    "cell_resistance_mohm",
    "max_cont_discharge_A",
    "max_burst_A",
    "cell_capacity_Ah",
    "cell_mass_kg",
    "peukert_exponent",
    "ocv_curve",
)

#: Optional keys and the default value used when they are missing.
_OPTIONAL_KEYS: Dict[str, Any] = {
    "display_name": "",
    "max_cont_charge_A": None,  # falls back to 0.5C
    "coulombic_efficiency": 1.0,
    "charge_efficiency": 1.0,
    "cell_thermal_mass_J_K": None,  # falls back to mass * 1000 J/kgK
    "cell_cooling_W_K": 0.2,
    "derate_start_C": 45.0,
    "max_cell_temp_C": 60.0,
    "soc_taper_start": 0.1,
    "cycle_life": 0,
    "notes": "",
}


@dataclass(frozen=True)
class Chemistry:
    """Immutable per-cell description of a battery chemistry.

    Attributes:
        name: Key used in ``batteries.json`` (e.g. ``"LiFePO4"``).
        display_name: Human readable name.
        nominal_v: Nominal cell voltage (V).
        max_v: Maximum charge voltage per cell (V).
        min_v: Minimum discharge voltage per cell (V).
        cell_resistance_mohm: Cell DC internal resistance (milliohm).
        max_cont_discharge_A: Continuous discharge current per cell (A).
        max_burst_A: Burst discharge current per cell (A).
        max_cont_charge_A: Continuous charge (regen) current per cell (A).
        cell_capacity_Ah: Rated capacity per cell (Ah).
        cell_mass_kg: Mass of a single cell (kg).
        peukert_exponent: Peukert exponent ``k``.
        coulombic_efficiency: Discharge efficiency (0..1).
        charge_efficiency: Charge (regen) efficiency (0..1).
        cell_thermal_mass_J_K: Lumped thermal capacity per cell (J/K).
        cell_cooling_W_K: Lumped heat transfer to ambient per cell (W/K).
        derate_start_C: Temperature where discharge power starts derating (C).
        max_cell_temp_C: Temperature where discharge power reaches zero (C).
        soc_taper_start: SOC below which available power tapers (0..1).
        cycle_life: Indicative full-depth cycle life.
        notes: Extra free-form notes from the JSON.
    """

    name: str
    display_name: str
    nominal_v: float
    max_v: float
    min_v: float
    cell_resistance_mohm: float
    max_cont_discharge_A: float
    max_burst_A: float
    max_cont_charge_A: float
    cell_capacity_Ah: float
    cell_mass_kg: float
    peukert_exponent: float
    coulombic_efficiency: float
    charge_efficiency: float
    cell_thermal_mass_J_K: float
    cell_cooling_W_K: float
    derate_start_C: float
    max_cell_temp_C: float
    soc_taper_start: float
    cycle_life: int
    notes: str
    ocv_curve: Tuple[Tuple[float, float], ...]

    # -- derived helpers ---------------------------------------------------

    @property
    def cell_resistance_ohm(self) -> float:
        """Cell internal resistance in ohm."""
        return self.cell_resistance_mohm / 1000.0

    @property
    def cell_energy_Wh(self) -> float:
        """Nominal energy of a single cell (Wh)."""
        return self.nominal_v * self.cell_capacity_Ah

    @property
    def cell_specific_energy_Wh_per_kg(self) -> float:
        """Gravimetric energy density of the cell (Wh/kg)."""
        return self.cell_energy_Wh / max(self.cell_mass_kg, 1e-9)

    def ocv(self, soc: float) -> float:
        """Return the open-circuit voltage (V) for a state of charge.

        The ``ocv_curve`` is a monotonic ``[[soc, volts], ...]`` table; values
        are linearly interpolated and clamped outside the tabulated range.

        Args:
            soc: State of charge in ``[0, 1]``.

        Returns:
            Open-circuit voltage in volts.
        """
        s = float(np.clip(soc, 0.0, 1.0))
        xs = np.fromiter((p[0] for p in self.ocv_curve), dtype=float)
        ys = np.fromiter((p[1] for p in self.ocv_curve), dtype=float)
        return float(np.interp(s, xs, ys))

    def ocv_derivative(self, soc: float) -> float:
        """Return d(OCV)/d(SOC) in V per unit SOC (numerical)."""
        h = 1e-3
        lo = max(soc - h, 0.0)
        hi = min(soc + h, 1.0)
        span = max(hi - lo, 1e-9)
        return (self.ocv(hi) - self.ocv(lo)) / span

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the chemistry."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "nominal_v": self.nominal_v,
            "max_v": self.max_v,
            "min_v": self.min_v,
            "cell_resistance_mohm": self.cell_resistance_mohm,
            "max_cont_discharge_A": self.max_cont_discharge_A,
            "max_burst_A": self.max_burst_A,
            "max_cont_charge_A": self.max_cont_charge_A,
            "cell_capacity_Ah": self.cell_capacity_Ah,
            "cell_mass_kg": self.cell_mass_kg,
            "peukert_exponent": self.peukert_exponent,
            "cell_thermal_mass_J_K": self.cell_thermal_mass_J_K,
            "cell_cooling_W_K": self.cell_cooling_W_K,
            "ocv_curve": [list(p) for p in self.ocv_curve],
        }

    # -- construction ------------------------------------------------------

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "Chemistry":
        """Build a :class:`Chemistry` from one JSON entry.

        Args:
            name: Chemistry key from the JSON file.
            data: Raw mapping for the chemistry.

        Returns:
            A validated :class:`Chemistry`.

        Raises:
            ValueError: If a required field is missing or the OCV curve is
                malformed.
        """
        missing = [k for k in _REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(f"chemistry {name!r} is missing keys: {missing}")

        curve_raw = data["ocv_curve"]
        if not isinstance(curve_raw, (list, tuple)) or len(curve_raw) < 2:
            raise ValueError(
                f"chemistry {name!r}: ocv_curve needs at least 2 [soc, v] points"
            )
        curve: List[Tuple[float, float]] = []
        for point in curve_raw:
            if len(point) != 2:
                raise ValueError(
                    f"chemistry {name!r}: ocv_curve points must be [soc, volts]"
                )
            curve.append((float(point[0]), float(point[1])))
        curve.sort(key=lambda p: p[0])
        if curve[0][0] > 1e-6 or abs(curve[-1][0] - 1.0) > 1e-6:
            raise ValueError(
                f"chemistry {name!r}: ocv_curve must span soc 0.0 .. 1.0"
            )

        cap = float(data["cell_capacity_Ah"])
        mass = float(data["cell_mass_kg"])
        charge_a = data.get("max_cont_charge_A")
        if charge_a is None:
            charge_a = 0.5 * cap  # 0.5C fallback
        thermal_mass = data.get("cell_thermal_mass_J_K")
        if thermal_mass is None:
            # 1000 J/kgK is a reasonable lumped specific heat for a cell.
            thermal_mass = mass * 1000.0

        return cls(
            name=name,
            display_name=str(data.get("display_name", name)),
            nominal_v=float(data["nominal_v"]),
            max_v=float(data["max_v"]),
            min_v=float(data["min_v"]),
            cell_resistance_mohm=float(data["cell_resistance_mohm"]),
            max_cont_discharge_A=float(data["max_cont_discharge_A"]),
            max_burst_A=float(data["max_burst_A"]),
            max_cont_charge_A=float(charge_a),
            cell_capacity_Ah=cap,
            cell_mass_kg=mass,
            peukert_exponent=float(data["peukert_exponent"]),
            coulombic_efficiency=float(data.get("coulombic_efficiency", 1.0)),
            charge_efficiency=float(data.get("charge_efficiency", 1.0)),
            cell_thermal_mass_J_K=float(thermal_mass),
            cell_cooling_W_K=float(data.get("cell_cooling_W_K", 0.2)),
            derate_start_C=float(data.get("derate_start_C", 45.0)),
            max_cell_temp_C=float(data.get("max_cell_temp_C", 60.0)),
            soc_taper_start=float(data.get("soc_taper_start", 0.1)),
            cycle_life=int(data.get("cycle_life", 0)),
            notes=str(data.get("notes", data.get("_todo", ""))),
            ocv_curve=tuple(curve),
        )


def load_chemistries(
    path: Union[str, Path, None] = None,
) -> Dict[str, Chemistry]:
    """Load every chemistry from the JSON database.

    Args:
        path: Optional override for the ``batteries.json`` location.

    Returns:
        Mapping of chemistry name to :class:`Chemistry`.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    entries = raw["chemistries"]
    return {name: Chemistry.from_dict(name, data) for name, data in entries.items()}


def available_chemistries(path: Union[str, Path, None] = None) -> List[str]:
    """Return the names of all chemistries in the database."""
    return list(load_chemistries(path).keys())


class BatteryPack:
    """A series/parallel battery pack built from a cell chemistry.

    The pack solves its own terminal behaviour so callers never have to
    approximate ``P = V * I``: :meth:`solve_current` returns the exact root of
    the quadratic ``R*I^2 - OCV*I + P = 0``.

    Sign convention:
        * Positive current = discharge (power flowing out of the pack).
        * Negative current = charge / regen (power flowing into the pack).
    """

    #: SOC above which regen current starts to taper.
    REGEN_TAPER_SOC: float = 0.90
    #: SOC at (and above) which regen is refused entirely.
    REGEN_CUTOFF_SOC: float = 0.995

    def __init__(
        self,
        S: int,
        P: int,
        chemistry: Union[str, Chemistry],
        *,
        ambient_temp_C: float = 25.0,
        initial_soc: float = 1.0,
        initial_cell_temp_C: Optional[float] = None,
        soc_floor: float = 0.05,
        config_path: Union[str, Path, None] = None,
    ) -> None:
        """Create a pack.

        Args:
            S: Number of cells in series (>= 1).
            P: Number of parallel cell strings (>= 1).
            chemistry: Chemistry name (looked up in the JSON) or a
                :class:`Chemistry` instance.
            ambient_temp_C: Ambient temperature used by the thermal model (C).
            initial_soc: Starting state of charge in ``[0, 1]``.
            initial_cell_temp_C: Starting cell temperature (C). Defaults to
                the ambient temperature.
            soc_floor: SOC below which no discharge power is available.
            config_path: Optional override for the chemistry database.

        Raises:
            ValueError: If ``S`` or ``P`` is not a positive integer, or the
                chemistry name is unknown.
        """
        if int(S) < 1 or int(P) < 1:
            raise ValueError(f"S and P must be >= 1 (got S={S}, P={P})")
        self.S: int = int(S)
        self.P: int = int(P)

        if isinstance(chemistry, Chemistry):
            self.chem: Chemistry = chemistry
        else:
            db = load_chemistries(config_path)
            if chemistry not in db:
                raise ValueError(
                    f"unknown chemistry {chemistry!r}; "
                    f"available: {sorted(db)}"
                )
            self.chem = db[chemistry]

        self.ambient_temp_C: float = float(ambient_temp_C)
        self.soc_floor: float = float(soc_floor)
        self.soc: float = float(np.clip(initial_soc, 0.0, 1.0))
        self.cell_temp_C: float = (
            float(initial_cell_temp_C)
            if initial_cell_temp_C is not None
            else float(ambient_temp_C)
        )

        # -- pack scalars derived from per-cell data -----------------------
        cell = self.chem
        self.pack_nominal_v: float = cell.nominal_v * self.S
        self.pack_capacity_Ah: float = cell.cell_capacity_Ah * self.P
        self.pack_internal_R: float = (
            cell.cell_resistance_ohm * self.S / self.P
        )
        self.pack_max_cont_A: float = cell.max_cont_discharge_A * self.P
        self.pack_max_burst_A: float = cell.max_burst_A * self.P
        self.pack_max_charge_A: float = cell.max_cont_charge_A * self.P
        self.pack_energy_Wh: float = self.pack_nominal_v * self.pack_capacity_Ah
        self.pack_mass_kg: float = cell.cell_mass_kg * self.S * self.P
        self.pack_max_V: float = cell.max_v * self.S
        self.pack_min_V: float = cell.min_v * self.S

        # -- runtime telemetry --------------------------------------------
        self.last_current_A: float = 0.0
        self.last_voltage_V: float = self.pack_nominal_v
        self.last_power_W: float = 0.0
        self.delivered_Wh: float = 0.0
        self.recovered_Wh: float = 0.0
        self.heat_generated_J: float = 0.0
        self.elapsed_s: float = 0.0

        self._peukert_ref_A: float = max(cell.cell_capacity_Ah, 1e-9)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"BatteryPack({self.chem.name}, S={self.S}, P={self.P}, "
            f"{self.pack_nominal_v:.1f} V, {self.pack_capacity_Ah:.1f} Ah, "
            f"{self.pack_energy_Wh / 1000.0:.2f} kWh, "
            f"soc={self.soc * 100:.1f}%)"
        )

    @property
    def pack_usable_energy_Wh(self) -> float:
        """Energy available between ``soc_floor`` and 100% SOC (Wh)."""
        return self.pack_energy_Wh * (1.0 - self.soc_floor)

    @property
    def pack_specific_energy_Wh_per_kg(self) -> float:
        """Pack-level gravimetric energy density (Wh/kg)."""
        return self.pack_energy_Wh / max(self.pack_mass_kg, 1e-9)

    # ------------------------------------------------------------------
    # Electrical behaviour
    # ------------------------------------------------------------------
    def ocv(self, soc: Optional[float] = None) -> float:
        """Pack open-circuit voltage (V) at the given (or current) SOC."""
        s = self.soc if soc is None else soc
        return float(self.chem.ocv(float(s)) * self.S)

    def terminal_voltage(
        self, current_A: float, soc: Optional[float] = None
    ) -> float:
        """Pack terminal voltage (V) under a given current.

        Implements ``V = OCV - I * R`` with the sign convention that positive
        current discharges the pack (and therefore lowers the terminal voltage).
        """
        return float(self.ocv(soc) - float(current_A) * self.pack_internal_R)

    def max_power_point_W(self, soc: Optional[float] = None) -> float:
        """Maximum power the pack could deliver into a matched load (W).

        Occurs at ``I = OCV / (2R)`` and equals ``OCV^2 / (4R)``.  Used as a
        hard physical ceiling by :meth:`solve_current`.
        """
        ocv = self.ocv(soc)
        if self.pack_internal_R <= 1e-9:
            return math.inf
        return ocv * ocv / (4.0 * self.pack_internal_R)

    def solve_current(
        self, power_demand_W: float, soc: Optional[float] = None
    ) -> float:
        """Solve for the exact pack current that meets a power demand.

        Given ``P = V * I`` and ``V = OCV - I * R`` we get the quadratic
        ``R*I^2 - OCV*I + P = 0`` whose physical root is

        .. math:: I = \\frac{OCV - \\sqrt{OCV^2 - 4 R P}}{2 R}

        No linearisation is used.  If the discriminant goes negative (the
        demand exceeds the battery's maximum power point) the current is
        clamped to ``OCV / (2R)``.

        Args:
            power_demand_W: Electrical power at the terminals (W). Positive
                values discharge the pack, negative values charge it.
            soc: Optional SOC override (defaults to the current SOC).

        Returns:
            Pack current in amperes (positive = discharge).
        """
        ocv = self.ocv(soc)
        R = self.pack_internal_R
        p = float(power_demand_W)
        if R <= 1e-9:
            return p / max(ocv, 1e-9)
        disc = ocv * ocv - 4.0 * R * p
        if disc < 0.0:
            # Beyond the maximum-power point: clamp to I = OCV / (2R).
            return ocv / (2.0 * R)
        return (ocv - math.sqrt(disc)) / (2.0 * R)

    def _temp_derate_factor(self) -> float:
        """Fraction of rated power still available given the cell temperature."""
        start = self.chem.derate_start_C
        limit = self.chem.max_cell_temp_C
        t = self.cell_temp_C
        if t <= start:
            return 1.0
        if t >= limit:
            return 0.0
        return float((limit - t) / max(limit - start, 1e-9))

    def _soc_derate_factor(self, soc: float) -> float:
        """Fraction of rated power available as SOC approaches the floor."""
        taper = self.chem.soc_taper_start
        if soc >= taper:
            return 1.0
        if soc <= self.soc_floor:
            return 0.0
        return float((soc - self.soc_floor) / max(taper - self.soc_floor, 1e-9))

    def available_discharge_power(
        self, soc: Optional[float] = None, *, burst: bool = False
    ) -> float:
        """Maximum discharge power the pack will currently supply (W).

        The result is the *tightest* of four constraints:

        1. The continuous (or burst) current rating scaled by ``P``.
        2. Cell temperature derating above ``derate_start_C``.
        3. SOC tapering toward ``soc_floor`` (zero at/below the floor).
        4. The minimum cell voltage and the battery's maximum power point.

        Args:
            soc: Optional SOC override.
            burst: Use the burst current rating instead of the continuous one.

        Returns:
            Available power at the terminals in watts (>= 0).
        """
        s = self.soc if soc is None else soc
        if s <= self.soc_floor:
            return 0.0

        ocv = self.ocv(s)
        R = self.pack_internal_R
        i_limit = self.pack_max_burst_A if burst else self.pack_max_cont_A
        i_limit *= self._temp_derate_factor()
        i_limit *= self._soc_derate_factor(s)

        if R > 1e-9:
            # Do not pull the pack below its minimum terminal voltage.
            i_voltage_floor = max((ocv - self.pack_min_V) / R, 0.0)
            i_limit = min(i_limit, i_voltage_floor)
            # Cannot exceed the physical maximum-power point.
            i_limit = min(i_limit, ocv / (2.0 * R))

        if i_limit <= 0.0:
            return 0.0
        return float(max((ocv - i_limit * R) * i_limit, 0.0))

    def regen_accept(
        self, current_A: float, soc: Optional[float] = None
    ) -> float:
        """Return the charge current the pack will actually accept.

        The requested regen current is capped by:

        * the maximum continuous charge current (temperature derated),
        * a linear taper from 90% SOC to zero at
          :attr:`REGEN_CUTOFF_SOC`,
        * the maximum cell voltage (so the pack cannot be over-charged).

        Args:
            current_A: Requested pack current. Negative values are charge
                (regen) requests; zero/positive requests yield ``0.0``.
            soc: Optional SOC override.

        Returns:
            The accepted charge current in amperes (``<= 0``).
        """
        s = self.soc if soc is None else soc
        req = min(float(current_A), 0.0)
        if req == 0.0 or s >= self.REGEN_CUTOFF_SOC:
            return 0.0

        limit = self.pack_max_charge_A * self._temp_derate_factor()
        if s > self.REGEN_TAPER_SOC:
            span = self.REGEN_CUTOFF_SOC - self.REGEN_TAPER_SOC
            limit *= max((self.REGEN_CUTOFF_SOC - s) / span, 0.0)

        bound = -max(limit, 0.0)

        R = self.pack_internal_R
        if R > 1e-9:
            ocv = self.ocv(s)
            i_voltage_ceiling = (ocv - self.pack_max_V) / R  # negative
            bound = max(bound, i_voltage_ceiling)

        bound = min(bound, 0.0)
        # ``req`` and ``bound`` are both <= 0; the larger (less negative) wins.
        return float(max(req, bound))

    def max_regen_power_W(self, soc: Optional[float] = None) -> float:
        """Maximum regen power the pack can absorb right now (W, >= 0)."""
        accepted = self.regen_accept(-math.inf, soc=soc)
        if accepted >= 0.0:
            return 0.0
        return float(abs(self.terminal_voltage(accepted, soc) * accepted))

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------
    def update(self, dt: float, current_A: float) -> None:
        """Advance the pack state by ``dt`` seconds under a constant current.

        Performs coulomb counting for SOC (with Peukert scaling for
        high-exponent chemistries such as lead-acid) and a lumped thermal
        step where ``I^2 * R`` heating is balanced against cooling toward the
        ambient temperature.

        Args:
            dt: Timestep in seconds (no-op if ``<= 0``).
            current_A: Pack current in amperes (positive = discharge).
        """
        dt = float(dt)
        if dt <= 0.0:
            return
        current_A = float(current_A)
        chem = self.chem

        # --- SOC (coulomb counting) --------------------------------------
        if current_A >= 0.0:
            # Peukert: at currents above the 1C reference the cell delivers
            # less than its rated Ah, so we remove charge faster.
            k = chem.peukert_exponent
            if k > 1.0:
                factor = (max(current_A, 1e-9) / self._peukert_ref_A) ** (k - 1.0)
            else:
                factor = 1.0
            delta_charge_As = current_A * factor * chem.coulombic_efficiency * dt
        else:
            delta_charge_As = current_A * chem.charge_efficiency * dt

        self.soc = float(
            np.clip(
                self.soc - delta_charge_As / (self.pack_capacity_Ah * 3600.0),
                0.0,
                1.0,
            )
        )

        # --- thermal ------------------------------------------------------
        q_gen = current_A * current_A * chem.cell_resistance_ohm  # per cell (W)
        q_cool = chem.cell_cooling_W_K * (self.cell_temp_C - self.ambient_temp_C)
        self.cell_temp_C += (q_gen - q_cool) / chem.cell_thermal_mass_J_K * dt
        self.heat_generated_J += q_gen * self.S * dt

        # --- telemetry ----------------------------------------------------
        voltage = self.terminal_voltage(current_A)
        power = voltage * current_A
        self.last_current_A = current_A
        self.last_voltage_V = voltage
        self.last_power_W = power
        self.elapsed_s += dt
        if power >= 0.0:
            self.delivered_Wh += power * dt / 3600.0
        else:
            self.recovered_Wh += -power * dt / 3600.0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, float]:
        """Return a snapshot of the pack's dynamic state."""
        return {
            "soc": self.soc,
            "soc_pct": self.soc * 100.0,
            "cell_temp_C": self.cell_temp_C,
            "voltage_V": self.last_voltage_V,
            "current_A": self.last_current_A,
            "power_W": self.last_power_W,
            "delivered_Wh": self.delivered_Wh,
            "recovered_Wh": self.recovered_Wh,
            "elapsed_s": self.elapsed_s,
        }

    def validate(
        self,
        *,
        required_cont_power_W: Optional[float] = None,
        controller_max_V: Optional[float] = None,
        emit_warnings: bool = True,
    ) -> List[str]:
        """Sanity-check the pack configuration.

        Checks for bad S/P combinations, unrealistic voltage for common EV
        controllers, weak specific energy and insufficient continuous power.

        Args:
            required_cont_power_W: If given, verify the pack can sustain this
                continuous power.
            controller_max_V: If given, verify the pack's maximum voltage fits
                under the controller rating.
            emit_warnings: Emit each finding through :mod:`warnings`.

        Returns:
            A list of human readable findings (empty means no problems).
        """
        msgs: List[str] = []
        chem = self.chem

        # --- S/P sanity ---------------------------------------------------
        if self.S < 1 or self.P < 1:
            msgs.append(f"Invalid S/P: S={self.S}, P={self.P}.")
        if self.pack_capacity_Ah < 1.0:
            msgs.append(
                f"Pack capacity is only {self.pack_capacity_Ah:.2f} Ah "
                f"(P={self.P} of {chem.cell_capacity_Ah:g} Ah cells)."
            )
        if self.pack_nominal_v < 12.0:
            msgs.append(
                f"Pack nominal voltage {self.pack_nominal_v:.1f} V is very low "
                f"for a traction pack (S={self.S})."
            )
        if self.pack_internal_R > 0.5:
            msgs.append(
                f"Pack internal resistance {self.pack_internal_R * 1000:.0f} mOhm "
                f"is high; expect heavy voltage sag (S={self.S}, P={self.P})."
            )

        # --- controller voltage compatibility -----------------------------
        typical_classes = (12, 24, 36, 48, 60, 72, 96, 144, 400, 800)
        if controller_max_V is not None and self.pack_max_V > controller_max_V:
            msgs.append(
                f"Pack max voltage {self.pack_max_V:.1f} V exceeds the controller "
                f"rating {controller_max_V:.1f} V."
            )
        if controller_max_V is None and self.pack_max_V > 100.0:
            msgs.append(
                f"Pack max voltage {self.pack_max_V:.1f} V requires a high-voltage "
                f"controller; the common low-voltage classes top out near 96-100 V "
                f"(classes: {typical_classes})."
            )

        # --- energy density ----------------------------------------------
        spec = self.pack_specific_energy_Wh_per_kg
        if spec < 20.0:
            msgs.append(
                f"Pack specific energy {spec:.1f} Wh/kg is low; the vehicle will be "
                f"heavy for its range."
            )

        # --- continuous power --------------------------------------------
        avail = self.available_discharge_power(soc=1.0)
        if required_cont_power_W is not None and avail < required_cont_power_W:
            msgs.append(
                f"Pack can only supply {avail / 1000.0:.1f} kW continuously at 100% "
                f"SOC but {required_cont_power_W / 1000.0:.1f} kW is required."
            )

        # --- sag estimate -------------------------------------------------
        if self.pack_max_cont_A > 0 and self.pack_internal_R > 0:
            ocv = self.ocv(0.9)
            sag = self.pack_max_cont_A * self.pack_internal_R
            frac = sag / max(ocv, 1e-9)
            if frac > 0.10:
                msgs.append(
                    f"Voltage sag at continuous current is {frac * 100:.1f}% of OCV "
                    f"({sag:.1f} V); consider increasing P."
                )

        if emit_warnings:
            for msg in msgs:
                warnings.warn(f"[BatteryPack] {msg}", stacklevel=2)
        return msgs
