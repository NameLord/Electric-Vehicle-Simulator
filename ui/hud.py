"""Live dashboard for the EV simulator.

A fixed panel in the bottom-right corner showing pack, motor and driver-mode
telemetry. It reads state from :class:`core.vehicle.VehicleState` (which already
aggregates :class:`core.battery_pack.BatteryPack.state_dict`) and re-derives
nothing that the core already computes:

* terminal voltage, current, power and energy all come from the pack's own
  telemetry, so the sag shown is the sag the physics actually produced;
* OCV comes from :meth:`core.battery_pack.BatteryPack.ocv`;
* the torque/power envelopes behind the warnings come from
  :class:`core.motor.Motor`.

The panel also owns the two *driver* concepts the core has no opinion about:
**drive mode** (Eco/Normal/Sport) and **regen level** (0-3). Both are implemented
as input shaping on :class:`core.vehicle.Controls` - a pedal map and an extra
lift-off brake request - so no physics is re-implemented here.

One display-only estimate lives here: :attr:`HudModel.motor_temp_C`. The core has
no motor thermal model (see the ``_todo`` in ``config/motors.json``), so the HUD
integrates a lumped first-order monitor from the motor's own measured loss
``|P_electrical| - |P_mechanical|``. It is never fed back into the simulation; it
exists so the driver can see the motor being worked. The constants are documented
at :data:`MOTOR_SPECIFIC_HEAT_J_per_kgK`.

Running this module directly executes its self-test::

    python -m ui.hud
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pygame

from core.battery_pack import BatteryPack
from core.motor import Motor
from core.vehicle import Controls, VehicleState

from .formatters import human_units, percent, ratio, signed

__all__ = [
    "DriveMode",
    "DRIVE_MODES",
    "REGEN_LEVELS",
    "HudModel",
    "Warning",
]


# ======================================================================
# Driver-side concepts (input shaping only)
# ======================================================================
@dataclass(frozen=True)
class DriveMode:
    """A pedal map plus the display colour for the mode badge.

    Attributes:
        name: Badge text.
        throttle_scale: Multiplier applied to pedal travel.
        throttle_curve: Exponent applied to pedal travel. ``> 1`` makes the
            pedal gentler near the top (easier to modulate), ``< 1`` makes it
            sharper (more torque for a given travel).
        steer_scale: Multiplier on steering input.
        regen_scale: Multiplier on the lift-off regen request.
        colour: Badge colour.
    """

    name: str
    throttle_scale: float
    throttle_curve: float
    steer_scale: float
    regen_scale: float
    colour: Tuple[int, int, int]


#: The three selectable modes. Eco trades response for economy, Sport sharpens
#: the pedal map and reduces lift-off regen so the car coasts more freely.
DRIVE_MODES: Tuple[DriveMode, ...] = (
    DriveMode("ECO", 0.55, 1.25, 0.80, 0.85, (110, 205, 130)),
    DriveMode("NORMAL", 1.00, 1.00, 1.00, 1.00, (226, 232, 240)),
    DriveMode("SPORT", 1.00, 0.72, 1.00, 0.55, (250, 150, 95)),
)

#: Brake-pedal equivalent requested when the driver lifts off, indexed by regen
#: level 0-3. Level 0 is a true coast; the core applies regen before friction
#: brake torque, so these small requests are recovered as energy rather than
#: burned in the discs.
REGEN_LEVELS: Tuple[float, ...] = (0.0, 0.05, 0.11, 0.20)

#: Speed (m/s) below which lift-off regen is suppressed, so the car does not
#: creep to a halt with the motor still generating.
REGEN_MIN_SPEED_M_S: float = 1.0


# ======================================================================
# Display-only motor thermal monitor
# ======================================================================
#: Lumped specific heat for a motor's active mass (stator iron + copper
#: windings). Published figures for traction machines land around 450-500
#: J/(kg*K) when referred to the whole housing mass, which is what
#: ``MotorSpec.mass_kg`` measures.
MOTOR_SPECIFIC_HEAT_J_per_kgK: float = 480.0
#: Lumped cooling conductance per kg of motor mass (W/K/kg). A 55 kg traction
#: motor ends up near 50 W/K, which is the right order for a finned, fan-cooled
#: housing.
MOTOR_COOLING_W_per_K_per_kg: float = 0.9
#: Winding temperature where sustained operation should be reduced.
MOTOR_WARN_C: float = 130.0
#: Winding temperature used for the red / "cut" warning.
MOTOR_MAX_C: float = 180.0


@dataclass(frozen=True)
class Warning:
    """One active driver warning.

    Attributes:
        code: Short machine-readable identifier.
        text: Badge text.
        critical: ``True`` for the red row (versus the amber row).
    """

    code: str
    text: str
    critical: bool = False


# ======================================================================
# Panel geometry / palette
# ======================================================================
PANEL_W = 344
PANEL_H = 386
PANEL_MARGIN = 12
ROW_H = 19

BG = (16, 19, 24)
BG_EDGE = (58, 66, 78)
TEXT = (226, 232, 240)
LABEL = (138, 148, 162)
DIM = (104, 114, 128)
GREEN = (86, 200, 120)
YELLOW = (232, 196, 90)
RED = (232, 96, 88)
AMBER = (238, 170, 72)
CYAN = (110, 200, 230)

#: ``(minimum SOC, colour, label)`` bands, most severe last.
SOC_BANDS: Tuple[Tuple[float, Tuple[int, int, int], str], ...] = (
    (0.50, GREEN, "OK"),
    (0.20, YELLOW, "LOW"),
    (0.00, RED, "CRITICAL"),
)


def _soc_band(soc: float) -> Tuple[Tuple[int, int, int], str]:
    """Return the ``(colour, label)`` band for a state of charge."""
    for minimum, colour, label in SOC_BANDS:
        if soc >= minimum:
            return colour, label
    return RED, "CRITICAL"


def _fit(text: str, font: "pygame.font.Font", width: int) -> str:
    """Truncate text with an ellipsis so it cannot overflow the panel."""
    if width <= 0 or font.size(text)[0] <= width:
        return text
    clipped = text
    while clipped and font.size(clipped + "...")[0] > width:
        clipped = clipped[:-1]
    return clipped + "..."


class HudModel:
    """Telemetry, driver settings and drawing for the dashboard.

    Args:
        pack: The battery pack being displayed.
        motor: The traction motor being displayed.
    """

    def __init__(self, pack: BatteryPack, motor: Motor) -> None:
        self.pack = pack
        self.motor = motor
        self.units: str = "kmh"  # "kmh" or "mph"; toggled with M
        self.mode_index: int = 1
        self.regen_level: int = 2

        # Trip computer (measured, not modelled).
        self.odometer_m: float = 0.0
        self.trip_consumed_Wh: float = 0.0
        self.trip_recovered_Wh: float = 0.0

        # Display-only motor thermal monitor.
        self.motor_temp_C: float = float(getattr(pack, "ambient_temp_C", 25.0))

        self._last_delivered_Wh = float(getattr(pack, "delivered_Wh", 0.0))
        self._last_recovered_Wh = float(getattr(pack, "recovered_Wh", 0.0))
        self._last_position: Optional[Tuple[float, float]] = None
        self._last_power_W = 0.0
        self._last_velocity: Tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    @property
    def drive_mode(self) -> DriveMode:
        """The currently selected drive mode."""
        return DRIVE_MODES[int(self.mode_index) % len(DRIVE_MODES)]

    def cycle_drive_mode(self, delta: int = 1) -> DriveMode:
        """Step the drive mode and return the new selection."""
        self.mode_index = (int(self.mode_index) + int(delta)) % len(DRIVE_MODES)
        return self.drive_mode

    def set_regen_level(self, level: int) -> int:
        """Clamp and store the regen level (0-3)."""
        self.regen_level = int(min(max(int(level), 0), len(REGEN_LEVELS) - 1))
        return self.regen_level

    def toggle_units(self) -> str:
        """Swap between km/h and mph, returning the new primary unit."""
        self.units = "mph" if self.units == "kmh" else "kmh"
        return self.units

    def reset_trip(self) -> None:
        """Zero the trip computer against the pack's current counters."""
        self.odometer_m = 0.0
        self.trip_consumed_Wh = 0.0
        self.trip_recovered_Wh = 0.0
        self._last_delivered_Wh = float(self.pack.delivered_Wh)
        self._last_recovered_Wh = float(self.pack.recovered_Wh)
        self._last_position = None

    def rebind(self, pack: BatteryPack, motor: Motor) -> None:
        """Point the dashboard at a newly built pack/motor.

        Used when the garage applies a build. The odometer and trip energy are
        kept, because the *car* did travel that distance; only the cumulative
        pack counters are re-based, since a fresh pack starts at zero.

        Args:
            pack: The new battery pack.
            motor: The new traction motor.
        """
        consumed = self.trip_consumed_Wh
        recovered = self.trip_recovered_Wh
        self.pack = pack
        self.motor = motor
        self._last_delivered_Wh = float(pack.delivered_Wh)
        self._last_recovered_Wh = float(pack.recovered_Wh)
        self._last_position = None
        self.motor_temp_C = min(
            self.motor_temp_C, float(getattr(pack, "ambient_temp_C", 25.0)) + 120.0
        )
        self.trip_consumed_Wh = consumed
        self.trip_recovered_Wh = recovered

    # ------------------------------------------------------------------
    # Input shaping
    # ------------------------------------------------------------------
    def shape(self, controls: Controls) -> Controls:
        """Apply the drive-mode pedal map and lift-off regen to raw inputs.

        This is the only place the HUD touches the input path, and it maps
        pedals to pedals: the core still owns the torque blending, the traction
        control and the regen/friction split.

        Args:
            controls: Raw driver inputs.

        Returns:
            A new :class:`Controls` with the mode applied.
        """
        mode = self.drive_mode
        throttle = min(max(float(controls.throttle), 0.0), 1.0)
        if throttle > 0.0:
            throttle = mode.throttle_scale * (throttle**mode.throttle_curve)
        steer = min(max(float(controls.steer), -1.0), 1.0) * mode.steer_scale

        brake = min(max(float(controls.brake), 0.0), 1.0)
        regen_request = REGEN_LEVELS[self.regen_level] * mode.regen_scale
        if regen_request > 0.0 and brake == 0.0 and controls.throttle <= 0.0:
            # Lift-off regen coasts the car down. Suppressed below walking pace
            # so the car does not motor-generate its way to a permanent crawl.
            speed = math.hypot(*self._last_velocity)
            if speed >= REGEN_MIN_SPEED_M_S:
                brake = regen_request

        return Controls(
            throttle=throttle,
            brake=brake,
            steer=steer,
            handbrake=bool(controls.handbrake),
        )

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------
    def update(self, state: VehicleState, dt: float) -> None:
        """Fold one frame of state into the trip computer and motor monitor.

        Args:
            state: Snapshot straight from :meth:`core.vehicle.Vehicle.get_state`.
            dt: Frame time in seconds.
        """
        dt = max(float(dt), 0.0)
        self._last_velocity = (
            float(state.velocity_body_m_s[0]),
            float(state.velocity_body_m_s[1]),
        )

        # -- distance: integrate the actual path, not the speed ----------
        position = (float(state.position_m[0]), float(state.position_m[1]))
        if self._last_position is not None and dt > 0.0:
            step = math.hypot(
                position[0] - self._last_position[0],
                position[1] - self._last_position[1],
            )
            # A collision push-out or a teleport would otherwise show up as
            # travelled distance; the car cannot exceed its own speed by much.
            if step <= max(state.speed_m_s, 1.0) * dt * 4.0 + 0.5:
                self.odometer_m += step
        self._last_position = position

        # -- energy: differences of the pack's own cumulative counters ---
        delivered = float(state.battery.get("delivered_Wh", self._last_delivered_Wh))
        recovered = float(state.battery.get("recovered_Wh", self._last_recovered_Wh))
        self.trip_consumed_Wh += max(delivered - self._last_delivered_Wh, 0.0)
        self.trip_recovered_Wh += max(recovered - self._last_recovered_Wh, 0.0)
        self._last_delivered_Wh = delivered
        self._last_recovered_Wh = recovered

        self._last_power_W = float(state.battery.get("power_W", 0.0))

        # -- motor thermal monitor (display only) ------------------------
        mechanical = float(
            self.motor.mechanical_power(state.motor_rpm, state.motor_torque_Nm)
        )
        electrical = float(
            self.motor.power_required(state.motor_rpm, state.motor_torque_Nm)
        )
        loss_W = max(abs(electrical) - abs(mechanical), 0.0)
        mass = max(float(self.motor.mass_kg), 1.0)
        heat_capacity = mass * MOTOR_SPECIFIC_HEAT_J_per_kgK
        cooling_W = (
            MOTOR_COOLING_W_per_K_per_kg
            * mass
            * (self.motor_temp_C - float(self.pack.ambient_temp_C))
        )
        self.motor_temp_C += (loss_W - cooling_W) / heat_capacity * dt

    # ------------------------------------------------------------------
    # Derived readings
    # ------------------------------------------------------------------
    @property
    def consumption_Wh_per_km(self) -> float:
        """Trip consumption in Wh/km, derived rather than modelled."""
        km = self.odometer_m / 1000.0
        if km <= 1e-3:
            return float("nan")
        return self.trip_consumed_Wh / km

    def warnings(self, state: VehicleState) -> List[Warning]:
        """Collect the active warnings for the current state.

        Every threshold is read from the pack and motor objects themselves, so
        the panel cannot drift out of step with the physics constants.

        Args:
            state: Current vehicle snapshot.

        Returns:
            Active warnings, most severe first.
        """
        pack = self.pack
        chemistry = pack.chem
        battery = state.battery
        found: List[Warning] = []

        # -- over-temperature ------------------------------------------
        cell_temp = float(battery.get("cell_temp_C", pack.cell_temp_C))
        if cell_temp >= chemistry.max_cell_temp_C:
            found.append(
                Warning("batt_temp", f"BATT OVER-TEMP {cell_temp:.0f}C", True)
            )
        elif cell_temp >= chemistry.derate_start_C:
            found.append(
                Warning(
                    "batt_temp",
                    f"BATT HOT {cell_temp:.0f}C (derate {chemistry.derate_start_C:.0f}C)",
                    False,
                )
            )
        if self.motor_temp_C >= MOTOR_MAX_C:
            found.append(
                Warning("motor_temp", f"MOTOR OVER-TEMP {self.motor_temp_C:.0f}C", True)
            )
        elif self.motor_temp_C >= MOTOR_WARN_C:
            found.append(
                Warning("motor_temp", f"MOTOR HOT {self.motor_temp_C:.0f}C", False)
            )

        # -- low voltage -----------------------------------------------
        voltage = float(battery.get("voltage_V", pack.pack_nominal_v))
        if voltage < pack.pack_min_V:
            found.append(
                Warning(
                    "voltage",
                    f"LOW VOLTAGE {voltage:.0f}V < {pack.pack_min_V:.0f}V",
                    True,
                )
            )
        elif voltage < 0.92 * pack.pack_min_V:
            found.append(
                Warning("voltage", f"VOLTAGE SAG {voltage:.0f}V", False)
            )

        # -- current limits --------------------------------------------
        current = float(battery.get("current_A", 0.0))
        if current >= 0.98 * pack.pack_max_cont_A:
            found.append(
                Warning(
                    "current",
                    f"DISCHARGE LIMIT {current:.0f}A / {pack.pack_max_cont_A:.0f}A",
                    False,
                )
            )
        elif current <= -0.98 * pack.pack_max_charge_A:
            found.append(
                Warning(
                    "regen",
                    f"REGEN LIMIT {abs(current):.0f}A / {pack.pack_max_charge_A:.0f}A",
                    False,
                )
            )

        # -- BMS state --------------------------------------------------
        if pack.soc <= pack.soc_floor + 1e-9:
            found.append(
                Warning("bms", f"BMS CUT - EMPTY (floor {pack.soc_floor * 100:.0f}%)", True)
            )
        if pack.soc >= pack.REGEN_CUTOFF_SOC:
            found.append(
                Warning(
                    "regen_off",
                    f"REGEN BLOCKED (SOC {pack.soc * 100:.0f}% >= "
                    f"{pack.REGEN_CUTOFF_SOC * 100:.0f}%)",
                    False,
                )
            )
        found.sort(key=lambda warning: not warning.critical)
        return found

    def snapshot(self, state: VehicleState) -> Dict[str, Any]:
        """Pure-data view of everything the panel shows.

        Exists so the dashboard can be asserted in tests without a display.

        Args:
            state: Current vehicle snapshot.

        Returns:
            Mapping of row name to display string plus raw values for the
            colour-coded elements.
        """
        pack = self.pack
        battery = state.battery
        soc = float(pack.soc)
        colour, band = _soc_band(soc)
        voltage = float(battery.get("voltage_V", pack.pack_nominal_v))
        ocv = float(pack.ocv())
        current = float(battery.get("current_A", 0.0))
        power = float(battery.get("power_W", 0.0))
        kmh = float(state.speed_kmh)
        mph = kmh * 0.621371

        speed_primary = f"{kmh:.1f} km/h" if self.units == "kmh" else f"{mph:.1f} mph"
        speed_secondary = f"{mph:.1f} mph" if self.units == "kmh" else f"{kmh:.1f} km/h"

        return {
            "soc_fraction": soc,
            "soc_text": percent(soc, 1),
            "soc_colour": colour,
            "soc_band": band,
            "soc_bar_fraction": max(min(soc, 1.0), 0.0),
            "voltage_text": f"{voltage:.1f} V",
            "ocv_text": f"{ocv:.1f} V",
            "sag_text": f"-{max(ocv - voltage, 0.0):.1f} V",
            "current_text": signed(current, 0, "A"),
            "current_A": current,
            "power_text": human_units(power, "W"),
            "power_W": power,
            "energy_text": human_units(self.trip_consumed_Wh, "Wh"),
            "recovered_text": human_units(self.trip_recovered_Wh, "Wh"),
            "consumption_text": ratio(self.consumption_Wh_per_km, 1.0, 0),
            "distance_text": human_units(self.odometer_m, "m"),
            "rpm_text": f"{state.motor_rpm:.0f} rpm",
            "torque_text": f"{state.motor_torque_Nm:.0f} Nm",
            "motor_temp_text": f"{self.motor_temp_C:.0f} C",
            "motor_load_text": percent(
                abs(float(state.motor_torque_Nm)) / max(self.motor.peak_torque_Nm, 1e-9),
                0,
            ),
            "cell_temp_text": f"{float(battery.get('cell_temp_C', pack.cell_temp_C)):.0f} C",
            "speed_primary": speed_primary,
            "speed_secondary": speed_secondary,
            "mode": self.drive_mode.name,
            "mode_colour": self.drive_mode.colour,
            "regen_level": self.regen_level,
            "regen_text": f"{self.regen_level} / {len(REGEN_LEVELS) - 1}",
            "warnings": self.warnings(state),
        }

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw(
        self,
        surface: pygame.Surface,
        state: VehicleState,
        *,
        font: Optional[pygame.font.Font] = None,
        small: Optional[pygame.font.Font] = None,
        fps: Optional[float] = None,
        extra_lines: Sequence[str] = (),
        helper: Optional[str] = None,
    ) -> pygame.Rect:
        """Draw the panel in the bottom-right corner.

        Args:
            surface: Target surface.
            state: Current vehicle snapshot.
            font: Row font. Built locally when omitted.
            small: Smaller font for hints. Built locally when omitted.
            fps: Optional frame rate to show in the panel header.
            extra_lines: World-layer lines (chunks, props) appended below the
                vehicle rows, so one panel owns the whole dashboard.
            helper: Optional key-hint line drawn at the very bottom.

        Returns:
            The rectangle the panel occupied.
        """
        if font is None:
            font = pygame.font.SysFont("monospace", 14)
        if small is None:
            small = pygame.font.SysFont("monospace", 12)

        data = self.snapshot(state)
        rows: List[Tuple[str, str, Tuple[int, int, int]]] = [
            ("SOC", f"{data['soc_text']}  {data['soc_band']}", data["soc_colour"]),
            ("Pack V", data["voltage_text"], TEXT),
            ("OCV", data["ocv_text"], DIM),
            ("Sag", data["sag_text"], CYAN if data["sag_text"] != "-0.0 V" else DIM),
            ("Current", data["current_text"], CYAN),
            ("Power", data["power_text"], TEXT),
            ("Energy", data["energy_text"], TEXT),
            ("Regen", data["recovered_text"], GREEN if self.trip_recovered_Wh else DIM),
            ("Wh/km", data["consumption_text"], TEXT),
            ("Motor", data["rpm_text"], TEXT),
            ("Torque", data["torque_text"], TEXT),
            ("Motor temp", f"{data['motor_temp_text']}  (load {data['motor_load_text']})",
             AMBER if self.motor_temp_C >= MOTOR_WARN_C else TEXT),
            ("Cell temp", data["cell_temp_text"],
             AMBER if float(self.pack.cell_temp_C) >= self.pack.chem.derate_start_C else TEXT),
            ("Mode", data["mode"], data["mode_colour"]),
            ("Regen lvl", data["regen_text"], TEXT),
        ]

        speed_rows = 2
        warning_rows = max(len(data["warnings"]), 1)
        height = (
            PANEL_MARGIN * 2
            + 22  # header
            + 16  # SOC bar
            + ROW_H * (len(rows) + speed_rows + len(extra_lines))
            + 20  # warning heading
            + 18 * warning_rows
            + (18 if helper else 0)
        )
        panel = pygame.Rect(
            max(surface.get_width() - PANEL_W - PANEL_MARGIN, 0),
            max(surface.get_height() - height - PANEL_MARGIN, 0),
            PANEL_W,
            height,
        )

        pygame.draw.rect(surface, BG, panel, border_radius=6)
        pygame.draw.rect(surface, BG_EDGE, panel, width=1, border_radius=6)

        x = panel.x + PANEL_MARGIN
        y = panel.y + PANEL_MARGIN

        def blit(text: str, colour: Tuple[int, int, int], use_small: bool = False,
                 offset: int = 0) -> None:
            renderer = small if use_small else font
            surface.blit(renderer.render(text, True, colour), (x + offset, y))

        def right(text: str, colour: Tuple[int, int, int], use_small: bool = False) -> None:
            renderer = small if use_small else font
            rendered = renderer.render(text, True, colour)
            surface.blit(
                rendered,
                (panel.right - PANEL_MARGIN - rendered.get_width(), y),
            )

        header = "EV SIMULATOR"
        if fps:
            header = f"{header}   {fps:5.1f} FPS"
        blit(header, LABEL)
        y += 22

        # -- SOC bar ---------------------------------------------------
        bar = pygame.Rect(x, y + 3, PANEL_W - 2 * PANEL_MARGIN, 8)
        pygame.draw.rect(surface, (34, 40, 48), bar, border_radius=4)
        filled = int(bar.width * data["soc_bar_fraction"])
        if filled > 0:
            pygame.draw.rect(
                surface,
                data["soc_colour"],
                pygame.Rect(bar.x, bar.y, filled, bar.height),
                border_radius=4,
            )
        y += 16

        # -- fixed rows -------------------------------------------------
        for label, value, colour in rows:
            blit(label, LABEL)
            right(value, colour)
            y += ROW_H

        # -- speed, both units, the primary one highlighted -------------
        blit("Speed", LABEL)
        right(data["speed_primary"], CYAN if self.units == "kmh" else TEXT)
        y += ROW_H
        blit(data["speed_secondary"], DIM, use_small=True, offset=PANEL_W - 150)
        y += ROW_H

        for line in extra_lines:
            blit(line, DIM, use_small=True)
            y += ROW_H

        # -- warnings ---------------------------------------------------
        y += 4
        active = data["warnings"]
        blit("WARNINGS", LABEL, use_small=True)
        y += 18
        if not active:
            blit("none", DIM, use_small=True)
            y += 18
        else:
            for warning in active:
                blit(
                    ("!! " if warning.critical else "!  ") + warning.text,
                    RED if warning.critical else AMBER,
                    use_small=True,
                )
                y += 18

        if helper:
            blit(_fit(helper, small, panel.width - 2 * PANEL_MARGIN), DIM, use_small=True)

        return panel


# ======================================================================
# Self-test (headless)
# ======================================================================
def _run_tests() -> int:
    """Exercise the HUD logic without needing a human at the keyboard."""
    import os

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

    vehicle = create_demo_vehicle(initial_soc=0.95)
    hud = HudModel(vehicle.pack, vehicle.motor)

    # Drive the car for a few seconds so the trip computer has data.
    dt = 1.0 / 120.0
    for _ in range(int(6.0 / dt)):
        raw = Controls(throttle=0.6, steer=0.1)
        state = vehicle.step(dt, hud.shape(raw))
        hud.update(state, dt)

    data = hud.snapshot(state)
    check("speed shows both units", "km/h" in data["speed_primary"]
          and "mph" in data["speed_secondary"],
          f"{data['speed_primary']} / {data['speed_secondary']}")
    check("odometer advanced", hud.odometer_m > 10.0, f"{hud.odometer_m:.1f} m")
    check("energy consumed is positive", hud.trip_consumed_Wh > 0.0,
          f"{hud.trip_consumed_Wh:.1f} Wh")
    check("Wh/km is a finite number", math.isfinite(hud.consumption_Wh_per_km),
          f"{hud.consumption_Wh_per_km:.1f} Wh/km")
    check("SOC fell under load", vehicle.pack.soc < 0.95,
          f"{vehicle.pack.soc * 100:.2f}%")

    # Units toggle.
    hud.toggle_units()
    swapped = hud.snapshot(state)
    check("M swaps the primary speed unit", "mph" in swapped["speed_primary"],
          swapped["speed_primary"])
    hud.toggle_units()

    # Warnings: low SOC must trip the BMS.
    original_soc = vehicle.pack.soc
    vehicle.pack.soc = vehicle.pack.soc_floor
    codes = {warning.code for warning in hud.warnings(state)}
    check("empty pack raises the BMS cut", "bms" in codes, f"codes={sorted(codes)}")

    # Warnings: hot cells must trip the temperature row.
    vehicle.pack.soc = original_soc
    hot = vehicle.pack.chem.max_cell_temp_C + 5.0
    vehicle.pack.cell_temp_C = hot
    state = vehicle.get_state()
    hot_codes = {warning.code: warning.critical for warning in hud.warnings(state)}
    check("hot pack raises a critical over-temp", hot_codes.get("batt_temp") is True,
          f"codes={hot_codes}")
    vehicle.pack.cell_temp_C = 25.0

    # Drive-mode shaping must never exceed the raw pedal.
    hud.mode_index = 0  # ECO
    eco = hud.shape(Controls(throttle=1.0))
    check("ECO scales the pedal down", eco.throttle < 1.0, f"{eco.throttle:.3f}")
    hud.mode_index = 2  # SPORT
    sport = hud.shape(Controls(throttle=0.5))
    check("SPORT sharpens part-throttle", sport.throttle > 0.5, f"{sport.throttle:.3f}")
    shaped = hud.shape(Controls(throttle=1.0))
    check("shaped throttle never exceeds 1.0", shaped.throttle <= 1.0,
          f"{shaped.throttle:.3f}")

    # Lift-off regen must be a brake request only, and only when coasting.
    hud.mode_index = 1
    hud.set_regen_level(3)
    coasting = hud.shape(Controls())
    check("lift-off regen adds brake", coasting.brake > 0.0, f"{coasting.brake:.3f}")
    check("lift-off regen leaves throttle alone", coasting.throttle == 0.0)
    hud.set_regen_level(0)
    check("regen level 0 is a true coast", hud.shape(Controls()).brake == 0.0)
    hud.set_regen_level(3)
    check("throttle overrides lift-off regen",
          hud.shape(Controls(throttle=0.4)).brake == 0.0)
    check("regen level clamps", hud.set_regen_level(99) == len(REGEN_LEVELS) - 1)

    # Drawing must not raise, and must land in the corner.
    surface = pygame.Surface((800, 500))
    rect = hud.draw(surface, state, fps=60.0, extra_lines=["Chunks 49 loaded"],
                    helper="TAB garage")
    check("panel sits in the bottom-right",
          rect.right <= 800 and rect.bottom <= 500 and rect.x > 400,
          f"rect={tuple(rect)}")
    check("panel is inside the window",
          rect.x >= 0 and rect.y >= 0 and rect.width > 0 and rect.height > 0,
          f"rect={tuple(rect)}")

    # A tiny window must not produce a negative origin.
    small_surface = pygame.Surface((320, 240))
    small_rect = hud.draw(small_surface, state)
    check("panel degrades gracefully in a small window",
          small_rect.x >= 0 and small_rect.y >= 0, f"rect={tuple(small_rect)}")

    pygame.quit()
    print()
    if failures:
        print(f"RESULT: {len(failures)} HUD check(s) failed -> {failures}")
        return 1
    print("RESULT: all HUD checks passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_run_tests())
