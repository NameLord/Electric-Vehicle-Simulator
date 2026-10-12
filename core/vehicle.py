"""Two-track electric vehicle chassis model with a fixed-timestep integrator.

This is the top of the physics stack: it owns the position/velocity state, the
per-wheel vertical loads, the wheel spin dynamics and the yaw degree of freedom,
and it wires together the battery, motor, drivetrain and tire models.

Model summary
-------------

Body frame: ``x`` forward, ``y`` to the left, yaw ``r`` counter-clockwise
positive.  Velocity is integrated in the rotating body frame:

.. math::

    m(\\dot u - r v) = \\sum F_x - F_{drag}

    m(\\dot v + r u) = \\sum F_y

    I_z \\dot r = \\sum M_z

Per-wheel vertical load uses static load plus longitudinal and lateral weight
transfer (a two-track "bicycle-hybrid"), and every wheel has its own rotational
inertia so wheelspin and wheel lock emerge from the torque balance rather than
being scripted.  Tyre forces come from :mod:`core.tire_model`.

The integrator always advances in fixed ``1 / substep_hz`` slices (default
120 Hz) with a further sub-division of the wheel spin dynamics for stability,
regardless of the ``dt`` the caller passes to :meth:`Vehicle.step`.

Running this file directly opens a pygame debug view (or runs the headless
verification suite with ``--headless``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # normal package import
    from .battery_pack import BatteryPack
    from .drivetrain import Drivetrain, DrivetrainCommand, DrivetrainConfig
    from .motor import Motor
    from .tire_model import TireConfig, TireModel
except ImportError:  # pragma: no cover - allows `python core/vehicle.py`
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from core.battery_pack import BatteryPack
    from core.drivetrain import Drivetrain, DrivetrainCommand, DrivetrainConfig
    from core.motor import Motor
    from core.tire_model import TireConfig, TireModel

__all__ = [
    "Controls",
    "VehicleState",
    "Vehicle",
    "create_demo_vehicle",
    "GRAVITY",
    "WHEEL_FL",
    "WHEEL_FR",
    "WHEEL_RL",
    "WHEEL_RR",
]

#: Standard gravity (m/s^2).
GRAVITY: float = 9.81

#: Wheel index convention used everywhere in this module.
WHEEL_FL, WHEEL_FR, WHEEL_RL, WHEEL_RR = 0, 1, 2, 3
FRONT_WHEELS: Tuple[int, int] = (WHEEL_FL, WHEEL_FR)
REAR_WHEELS: Tuple[int, int] = (WHEEL_RL, WHEEL_RR)

#: Default location of the vehicle configuration.
DEFAULT_CONFIG_PATH: Path = (
    Path(__file__).resolve().parent.parent / "config" / "vehicle.json"
)

#: Wheel-omega integration scheme in use. Printed once at start-up so the
#: active scheme can be verified from the console.
WHEEL_INTEGRATOR: str = "semiimplicit"
_wheel_integrator_announced: bool = False


@dataclass
class Controls:
    """Driver inputs for one step.

    Attributes:
        throttle: Throttle pedal position, ``0..1``.
        brake: Brake pedal position, ``0..1`` (regen first, then friction).
        steer: Steering input, ``-1..1``. Positive steers left.
        handbrake: When ``True`` the rear lateral grip collapses, which is what
            lets the car rotate into a drift.
    """

    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    handbrake: bool = False

    def clamped(self) -> "Controls":
        """Return a copy with the pedals clamped to their valid ranges."""
        return Controls(
            throttle=float(min(max(self.throttle, 0.0), 1.0)),
            brake=float(min(max(self.brake, 0.0), 1.0)),
            steer=float(min(max(self.steer, -1.0), 1.0)),
            handbrake=bool(self.handbrake),
        )


@dataclass
class VehicleState:
    """Immutable snapshot of the vehicle, produced by :meth:`Vehicle.get_state`."""

    time_s: float
    position_m: np.ndarray
    velocity_body_m_s: np.ndarray
    heading_rad: float
    yaw_rate_rad_s: float
    speed_m_s: float
    speed_kmh: float
    steer_angle_rad: float
    wheel_speeds_rad_s: np.ndarray
    wheel_fz_N: np.ndarray
    tire_utilisation: np.ndarray
    wheel_sliding: np.ndarray
    tire_long_force_N: np.ndarray
    tire_lat_force_N: np.ndarray
    motor_rpm: float
    motor_torque_Nm: float
    drive_power_W: float
    battery: Dict[str, float]
    slip_ratios: np.ndarray
    slip_angles_rad: np.ndarray
    is_sliding: bool


class Vehicle:
    """Two-track electric vehicle with a fixed-timestep physics core.

    Args:
        pack: The battery pack feeding the drivetrain.
        motor: The traction motor.
        config_path: Optional override for ``config/vehicle.json``.
        initial_soc: Starting pack state of charge.
        ambient_temp_C: Ambient temperature for the pack thermal model.
        vehicle_config: Optional pre-loaded config dict (overrides the file).

    Raises:
        ValueError: If the configuration is inconsistent.
    """

    def __init__(
        self,
        pack: BatteryPack,
        motor: Motor,
        *,
        config_path: Optional[Path] = None,
        initial_soc: Optional[float] = None,
        ambient_temp_C: Optional[float] = None,
        vehicle_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.pack = pack
        self.motor = motor

        cfg = vehicle_config
        if cfg is None:
            path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
            with open(path, "r", encoding="utf-8") as handle:
                cfg = json.load(handle)
        self.config = cfg

        chassis = cfg["chassis"]
        wheels = cfg["wheels"]
        physics = cfg["physics"]

        # --- chassis ------------------------------------------------------
        self.mass_kg: float = float(chassis["mass_kg"])
        self.yaw_inertia_kg_m2: float = float(chassis["yaw_inertia_kg_m2"])
        self.cg_height_m: float = float(chassis["cg_height_m"])
        self.wheelbase_m: float = float(chassis["wheelbase_m"])
        self.lf: float = float(chassis["cg_to_front_axle_m"])
        self.lr: float = float(chassis["cg_to_rear_axle_m"])
        self.track_front_m: float = float(chassis["track_width_front_m"])
        self.track_rear_m: float = float(chassis["track_width_rear_m"])
        self.frontal_area_m2: float = float(chassis["frontal_area_m2"])
        self.drag_coefficient: float = float(chassis["drag_coefficient"])
        self.air_density: float = float(chassis["air_density_kg_m3"])

        if abs((self.lf + self.lr) - self.wheelbase_m) > 1e-6:
            raise ValueError(
                "wheelbase_m must equal cg_to_front_axle_m + cg_to_rear_axle_m"
            )

        # --- wheels -------------------------------------------------------
        self.wheel_radius_m: float = float(wheels["radius_m"])
        self.wheel_inertia_kg_m2: float = float(wheels["inertia_kg_m2"])
        self.rolling_resistance_coeff: float = float(
            wheels["rolling_resistance_coeff"]
        )
        self.handbrake_torque_Nm: float = float(
            wheels.get("handbrake_torque_Nm_per_wheel", 1600.0)
        )
        self.brake_distribution: str = str(
            wheels.get("brake_distribution", "fixed")
        ).lower()
        self.max_brake_grip_utilisation: float = float(
            wheels.get("max_brake_grip_utilisation", 1.0)
        )
        self.min_front_brake_bias: float = float(
            wheels.get("min_front_brake_bias", 0.45)
        )
        self.max_front_brake_bias: float = float(
            wheels.get("max_front_brake_bias", 0.85)
        )
        axle = str(wheels.get("driven_axle", "rear")).lower()
        if axle in ("front", "fwd"):
            self.driven_wheels: Tuple[int, ...] = FRONT_WHEELS
        elif axle in ("all", "awd", "4wd"):
            self.driven_wheels = (WHEEL_FL, WHEEL_FR, WHEEL_RL, WHEEL_RR)
        else:
            self.driven_wheels = REAR_WHEELS

        # --- tires --------------------------------------------------------
        self.tire_model = TireModel(TireConfig.from_dict(cfg["tires"]))

        # --- drivetrain ---------------------------------------------------
        self.drivetrain = Drivetrain(
            DrivetrainConfig.from_dict(cfg["drivetrain"], wheels), motor
        )

        # --- steering -----------------------------------------------------
        steering = cfg["steering"]
        self.max_steer_angle_rad: float = float(steering["max_steer_angle_rad"])
        self.steer_ref_speed: float = float(steering["speed_sensitive_ref_m_s"])
        self.steer_min_factor: float = float(steering["speed_sensitive_min_factor"])
        self.steer_rate_rad_s: float = float(steering["steer_rate_rad_s"])

        # --- integration --------------------------------------------------
        self.substep_hz: float = float(physics["substep_hz"])
        self.wheel_substeps: int = int(physics["wheel_substeps"])
        self.soc_floor: float = float(physics["soc_floor"])
        self.stick_speed_m_s: float = float(physics.get("stick_speed_m_s", 0.15))
        self.wheel_reverse_lock_speed_m_s: float = float(
            physics.get("wheel_reverse_lock_speed_m_s", 1.5)
        )
        self.ambient_temp_C: float = (
            float(ambient_temp_C)
            if ambient_temp_C is not None
            else float(physics.get("ambient_temp_C", 25.0))
        )

        # --- geometry helpers --------------------------------------------
        self.wheel_lx = np.array([self.lf, self.lf, -self.lr, -self.lr], dtype=float)
        self.wheel_ly = np.array(
            [self.track_front_m / 2, -self.track_front_m / 2,
             self.track_rear_m / 2, -self.track_rear_m / 2],
            dtype=float,
        )
        self.wheel_radius = np.full(4, self.wheel_radius_m, dtype=float)

        self.reset(initial_soc=initial_soc, ambient_temp_C=ambient_temp_C)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def reset(
        self,
        initial_soc: Optional[float] = None,
        ambient_temp_C: Optional[float] = None,
    ) -> None:
        """Reset the vehicle to rest at the origin with a fresh pack state.

        Args:
            initial_soc: Optional SOC to set on the pack.
            ambient_temp_C: Optional ambient temperature override.
        """
        if initial_soc is not None:
            self.pack.soc = float(np.clip(initial_soc, 0.0, 1.0))
        if ambient_temp_C is not None:
            self.pack.ambient_temp_C = float(ambient_temp_C)

        self.time_s: float = 0.0
        self.position_m: np.ndarray = np.zeros(2, dtype=float)
        self.velocity_body_m_s: np.ndarray = np.zeros(2, dtype=float)
        self.heading_rad: float = 0.0
        self.yaw_rate_rad_s: float = 0.0
        self.steer_angle_rad: float = 0.0
        self.wheel_speeds_rad_s: np.ndarray = np.zeros(4, dtype=float)

        self.accel_body_m_s2: np.ndarray = np.zeros(2, dtype=float)
        self.wheel_fz_N: np.ndarray = np.zeros(4, dtype=float)
        self.tire_long_force_N: np.ndarray = np.zeros(4, dtype=float)
        self.tire_lat_force_N: np.ndarray = np.zeros(4, dtype=float)
        self.tire_utilisation: np.ndarray = np.zeros(4, dtype=float)
        self.wheel_sliding: np.ndarray = np.zeros(4, dtype=bool)
        self.slip_ratios: np.ndarray = np.zeros(4, dtype=float)
        self.slip_angles_rad: np.ndarray = np.zeros(4, dtype=float)
        self.is_sliding: bool = False

        self.motor_rpm: float = 0.0
        self.motor_torque_Nm: float = 0.0
        self.last_command: Optional[DrivetrainCommand] = None

        self._static_fz = self._compute_static_fz()
        self.wheel_fz_N = self._static_fz.copy()

        # Announce the active wheel integration scheme once per process.
        global _wheel_integrator_announced
        if not _wheel_integrator_announced:
            print(f"INTEGRATOR={WHEEL_INTEGRATOR}")
            _wheel_integrator_announced = True

    def _compute_static_fz(self) -> np.ndarray:
        """Static vertical load per wheel at rest (N)."""
        w_f = self.mass_kg * GRAVITY * self.lr / self.wheelbase_m
        w_r = self.mass_kg * GRAVITY * self.lf / self.wheelbase_m
        return np.array(
            [w_f / 2.0, w_f / 2.0, w_r / 2.0, w_r / 2.0], dtype=float
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def step(self, dt: float, controls: Controls) -> VehicleState:
        """Advance the simulation by ``dt`` seconds.

        The step is internally divided into fixed slices of
        ``1 / substep_hz`` seconds so the result is independent of the caller's
        frame rate.

        Args:
            dt: Elapsed wall-clock time to simulate (s).
            controls: Driver inputs for this interval.

        Returns:
            The post-step :class:`VehicleState`.
        """
        if dt <= 0.0:
            return self.get_state()

        controls = controls.clamped()
        h_nominal = 1.0 / self.substep_hz
        n = max(1, int(round(dt / h_nominal)))
        h = dt / n
        for _ in range(n):
            self._substep(h, controls)
        return self.get_state()

    def get_state(self) -> VehicleState:
        """Return an immutable snapshot of the vehicle's current state."""
        speed = float(np.hypot(*self.velocity_body_m_s))
        return VehicleState(
            time_s=self.time_s,
            position_m=self.position_m.copy(),
            velocity_body_m_s=self.velocity_body_m_s.copy(),
            heading_rad=self.heading_rad,
            yaw_rate_rad_s=self.yaw_rate_rad_s,
            speed_m_s=speed,
            speed_kmh=speed * 3.6,
            steer_angle_rad=self.steer_angle_rad,
            wheel_speeds_rad_s=self.wheel_speeds_rad_s.copy(),
            wheel_fz_N=self.wheel_fz_N.copy(),
            tire_utilisation=self.tire_utilisation.copy(),
            wheel_sliding=self.wheel_sliding.copy(),
            tire_long_force_N=self.tire_long_force_N.copy(),
            tire_lat_force_N=self.tire_lat_force_N.copy(),
            motor_rpm=self.motor_rpm,
            motor_torque_Nm=self.motor_torque_Nm,
            drive_power_W=float(self.pack.last_power_W),
            battery=self.pack.state_dict(),
            slip_ratios=self.slip_ratios.copy(),
            slip_angles_rad=self.slip_angles_rad.copy(),
            is_sliding=self.is_sliding,
        )

    # ------------------------------------------------------------------
    # Steering
    # ------------------------------------------------------------------
    def _update_steering(self, h: float, controls: Controls) -> None:
        """Rate-limit and speed-sensitively scale the steering angle."""
        speed = float(np.hypot(*self.velocity_body_m_s))
        # Full authority at rest, tapering toward `steer_min_factor` at speed.
        factor = self.steer_min_factor + (1.0 - self.steer_min_factor) / (
            1.0 + speed / max(self.steer_ref_speed, 1e-6)
        )
        target = controls.steer * self.max_steer_angle_rad * factor
        max_change = self.steer_rate_rad_s * h
        delta = target - self.steer_angle_rad
        self.steer_angle_rad += float(np.clip(delta, -max_change, max_change))

    # ------------------------------------------------------------------
    # Vertical load
    # ------------------------------------------------------------------
    def _update_fz(self) -> None:
        """Update per-wheel vertical load with longitudinal + lateral transfer."""
        ax, ay = self.accel_body_m_s2
        g = GRAVITY

        w_f = self.mass_kg * g * self.lr / self.wheelbase_m
        w_r = self.mass_kg * g * self.lf / self.wheelbase_m

        # Longitudinal transfer: accelerating moves load rearward.
        dw_long = self.mass_kg * ax * self.cg_height_m / self.wheelbase_m
        w_f -= dw_long
        w_r += dw_long

        # Lateral transfer: accelerating to the left loads the right wheels.
        dw_lat_f = self.mass_kg * ay * self.cg_height_m / self.track_front_m
        dw_lat_r = self.mass_kg * ay * self.cg_height_m / self.track_rear_m

        fz = np.array(
            [
                w_f / 2.0 - dw_lat_f / 2.0,
                w_f / 2.0 + dw_lat_f / 2.0,
                w_r / 2.0 - dw_lat_r / 2.0,
                w_r / 2.0 + dw_lat_r / 2.0,
            ],
            dtype=float,
        )
        self.wheel_fz_N = np.maximum(fz, 0.0)

    # ------------------------------------------------------------------
    # Motor speed
    # ------------------------------------------------------------------
    def _driven_motor_rpm(self) -> float:
        """Motor shaft speed (rpm) implied by the driven wheel speeds."""
        omega = float(np.mean(self.wheel_speeds_rad_s[list(self.driven_wheels)]))
        return omega * self.drivetrain.config.gear_ratio * 60.0 / (2.0 * math.pi)

    # ------------------------------------------------------------------
    # Core substep
    # ------------------------------------------------------------------
    def _substep(self, h: float, controls: Controls) -> None:
        """Advance the physics by one fixed slice ``h`` (s)."""
        self._update_steering(h, controls)
        self._update_fz()

        vx, vy = self.velocity_body_m_s
        r = self.yaw_rate_rad_s

        # --- per-wheel steering angle ------------------------------------
        delta = np.zeros(4, dtype=float)
        delta[WHEEL_FL] = self.steer_angle_rad
        delta[WHEEL_FR] = self.steer_angle_rad

        cos_d = np.cos(delta)
        sin_d = np.sin(delta)

        # --- lateral grip scaling (handbrake collapses rear lateral grip) --
        lateral_mu_scale = np.ones(4, dtype=float)
        if controls.handbrake:
            lateral_mu_scale[WHEEL_RL] = (
                self.tire_model.config.handbrake_lat_mu_factor
            )
            lateral_mu_scale[WHEEL_RR] = (
                self.tire_model.config.handbrake_lat_mu_factor
            )

        long_grip_bias = np.ones(4, dtype=float)
        long_grip_bias[WHEEL_RL] = self.tire_model.config.rear_long_grip_bias
        long_grip_bias[WHEEL_RR] = self.tire_model.config.rear_long_grip_bias

        # --- contact-patch velocities (fixed for the sub-substeps) --------
        v_wx = vx - r * self.wheel_ly
        v_wy = vy + r * self.wheel_lx
        v_long = v_wx * cos_d + v_wy * sin_d
        v_lat = -v_wx * sin_d + v_wy * cos_d

        # --- drivetrain command ------------------------------------------
        self.motor_rpm = self._driven_motor_rpm()

        drive_power_limit = self.pack.available_discharge_power(soc=self.pack.soc)
        regen_power_avail = self.pack.max_regen_power_W(soc=self.pack.soc)
        regen_torque_limit = self.drivetrain.max_regen_torque_from_battery(
            regen_power_avail, self.motor_rpm
        )

        # Mean slip of the driven wheels, fed to traction control.
        driven_idx = list(self.driven_wheels)
        drive_slip = float(
            np.mean(
                self.tire_model.slip_ratio(
                    v_long[driven_idx],
                    self.wheel_speeds_rad_s[driven_idx] * self.wheel_radius_m,
                )
            )
        )

        # --- brake distribution + grip budget ----------------------------
        # A brake system strong enough to over-power the tires would lock every
        # wheel at full pedal, which zeroes the wheel speed (and therefore the
        # motor rpm and the regen current) while the car is still moving.  So
        # the driver's pedal is distributed in proportion to axle load (EBD)
        # and capped at the torque the tires can actually react.
        wheel_mu = self.tire_model.effective_mu(self.wheel_fz_N)
        grip_torque = float(np.sum(wheel_mu * self.wheel_fz_N)) * self.wheel_radius_m
        brake_torque_cap = self.max_brake_grip_utilisation * grip_torque

        if self.brake_distribution == "dynamic":
            front_axle_load = float(self.wheel_fz_N[WHEEL_FL] + self.wheel_fz_N[WHEEL_FR])
            total_load = float(np.sum(self.wheel_fz_N))
            brake_bias = (
                front_axle_load / total_load
                if total_load > 1e-9
                else self.drivetrain.config.front_brake_bias
            )
            brake_bias = float(
                np.clip(brake_bias, self.min_front_brake_bias, self.max_front_brake_bias)
            )
        else:
            brake_bias = self.drivetrain.config.front_brake_bias

        command = self.drivetrain.command(
            controls.throttle,
            controls.brake,
            self.motor_rpm,
            drive_power_limit_W=drive_power_limit,
            regen_torque_limit_Nm=regen_torque_limit,
            front_brake_bias=brake_bias,
            max_total_brake_wheel_torque_Nm=brake_torque_cap,
            drive_slip=drive_slip,
        )
        self.last_command = command
        self.motor_torque_Nm = command.motor_torque_Nm

        # --- wheel torque allocation -------------------------------------
        # Drive torque and braking torque are kept separate: drive torque may
        # spin a wheel either way, but friction brakes and regen only ever
        # oppose rotation and must never reverse it.
        n_driven = max(int(self.drivetrain.config.driven_wheels), 1)
        throttle_per_wheel = (
            command.throttle_torque_Nm
            * self.drivetrain.motor_to_wheel_torque_gain
            / n_driven
        )
        regen_per_wheel = command.regen_wheel_torque_Nm / n_driven

        drive_torque = np.zeros(4, dtype=float)
        brake_torque = np.zeros(4, dtype=float)
        for idx in self.driven_wheels:
            drive_torque[idx] += throttle_per_wheel
            brake_torque[idx] += regen_per_wheel

        brake_torque[WHEEL_FL] += command.friction_brake_front_Nm_per_wheel
        brake_torque[WHEEL_FR] += command.friction_brake_front_Nm_per_wheel
        brake_torque[WHEEL_RL] += command.friction_brake_rear_Nm_per_wheel
        brake_torque[WHEEL_RR] += command.friction_brake_rear_Nm_per_wheel
        if controls.handbrake:
            # The handbrake is deliberately *not* grip-capped: breaking rear
            # traction is the entire point of it.
            brake_torque[WHEEL_RL] += self.handbrake_torque_Nm
            brake_torque[WHEEL_RR] += self.handbrake_torque_Nm

        # --- wheel spin dynamics + averaged tire forces -------------------
        nw = max(int(self.wheel_substeps), 1)
        hw = h / nw
        f_long_acc = np.zeros(4, dtype=float)
        f_lat_acc = np.zeros(4, dtype=float)
        i_wheel = max(self.wheel_inertia_kg_m2, 1e-6)
        roll_eps = max(self.tire_model.config.slip_eps_m_s, 1e-6)

        for _ in range(nw):
            wheel_lin = self.wheel_speeds_rad_s * self.wheel_radius_m
            f_long, f_lat = self.tire_model.forces(
                v_long,
                v_lat,
                wheel_lin,
                self.wheel_fz_N,
                lateral_mu_scale=lateral_mu_scale,
                long_grip_bias=long_grip_bias,
            )
            f_long_acc += f_long
            f_lat_acc += f_lat

            # Rolling resistance is a torque that opposes wheel rotation.  The
            # wheel then slips slightly negative and the tire model turns that
            # slip into the chassis retarding force, which is how it works on a
            # real car (and avoids the reaction torque spinning the wheel up).
            roll_torque = (
                -self.rolling_resistance_coeff
                * self.wheel_fz_N
                * self.wheel_radius_m
                * np.tanh(wheel_lin / roll_eps)
            )

            # --- semi-implicit (backward-Euler) wheel spin update ----------
            # The slip model is *stiff* in its linear region, where
            #     dF_x/d(omega) = C_x * R / max(|u|, eps)
            # With I = 1.4, C_x = 5e4, R = 0.31 and eps = 0.5 that is ~6864/s,
            # i.e. a time constant of ~0.15 ms.  Explicit Euler stays stable
            # only while hw < 2*tau (~0.3 ms), which our 1/960 s wheel slice
            # exceeds -- and the denominator floor makes it worst at low speed.
            #
            # So linearise the tire force about the current wheel speed and
            # integrate implicitly:
            #
            #   F(omega') ~= F_k + s*(omega' - omega_k),   s = dF_x/d(omega)
            #   I*(omega' - omega_k)/hw = T_net - R*F(omega')
            #
            # The model is linear in omega within a region, so this solves
            # exactly in one step (no Newton iteration needed):
            #
            #   omega' = omega_k + T_net / (I/hw + R*s)
            #
            # with T_net = T_drive + T_roll - R*F_k.  Note the stiffness that
            # enters the torque balance is d(R*F_x)/d(omega) = R^2 * s, since
            # s = dF_x/d(omega*R); getting this wrong over-damps the wheel.
            # The update is unconditionally stable for s >= 0, and reduces to
            # explicit Euler when the tire is saturated (s = 0) -- which is
            # stable there because a clamped force no longer depends on the
            # wheel speed.
            stiffness = self.tire_model.longitudinal_slip_stiffness(
                v_long, wheel_lin, self.wheel_fz_N, long_grip_bias=long_grip_bias
            )
            net_torque = drive_torque + roll_torque - f_long * self.wheel_radius_m
            omega_free = self.wheel_speeds_rad_s + net_torque / (
                i_wheel / hw + stiffness * self.wheel_radius_m**2
            )

            # Braking torque (friction + regen) opposes rotation and may only
            # ever bring the wheel to a stop, never past it.  The applied
            # deceleration is clipped to exactly |omega_free|, i.e. the torque
            # T_brake = -omega * I / hw that stops the wheel within this slice.
            stop = np.minimum(brake_torque / i_wheel * hw, np.abs(omega_free))
            self.wheel_speeds_rad_s = np.sign(omega_free) * (
                np.abs(omega_free) - stop
            )

            # A wheel cannot rotate backwards while its contact patch is still
            # travelling forwards. Any overshoot through zero is discarded as
            # heat rather than fed back into the chassis. Note the floor is
            # zero, NOT the ground speed v_long/R: clamping to rolling speed
            # would kinematically lock the wheel to the road and destroy slip.
            reverse_guard = v_long > self.wheel_reverse_lock_speed_m_s
            self.wheel_speeds_rad_s = np.where(
                reverse_guard & (self.wheel_speeds_rad_s < 0.0),
                0.0,
                self.wheel_speeds_rad_s,
            )

        f_long = f_long_acc / nw
        f_lat = f_lat_acc / nw
        self.tire_long_force_N = f_long
        self.tire_lat_force_N = f_lat

        # --- chassis forces from the averaged tire forces -----------------
        fx_i = f_long * cos_d - f_lat * sin_d
        fy_i = f_long * sin_d + f_lat * cos_d

        f_x = float(np.sum(fx_i))
        f_y = float(np.sum(fy_i))
        m_z = float(np.sum(self.wheel_lx * fy_i - self.wheel_ly * fx_i))

        # --- aerodynamic drag --------------------------------------------
        speed = float(np.hypot(vx, vy))
        if speed > 1e-3:
            drag = 0.5 * self.air_density * self.drag_coefficient * self.frontal_area_m2 * speed**2
            f_x -= drag * vx / speed
            f_y -= drag * vy / speed

        # --- rigid body acceleration (rotating body frame) ---------------
        # The *inertial* accelerations along the body axes are F_x/m and F_y/m,
        # and those are what the body feels as "g" forces, so they are what
        # drive the weight transfer. Using the rotating-frame velocity
        # derivatives (u_dot, v_dot) instead would inject the centripetal terms
        # r*v and r*u, which at high yaw rate are enormous and produce
        # impossible per-wheel loads.
        a_x = f_x / self.mass_kg
        a_y = f_y / self.mass_kg
        self.accel_body_m_s2 = np.array([a_x, a_y], dtype=float)

        # Rotating-frame derivatives integrate the velocity state.
        u_dot = a_x + r * vy
        v_dot = a_y - r * vx
        r_dot = m_z / self.yaw_inertia_kg_m2

        self.velocity_body_m_s = self.velocity_body_m_s + np.array([u_dot, v_dot]) * h
        self.yaw_rate_rad_s += r_dot * h

        # --- integrate pose -------------------------------------------------
        self.heading_rad += self.yaw_rate_rad_s * h
        cos_h = math.cos(self.heading_rad)
        sin_h = math.sin(self.heading_rad)
        v_world = np.array(
            [
                vx * cos_h - vy * sin_h,
                vx * sin_h + vy * cos_h,
            ],
            dtype=float,
        )
        self.position_m = self.position_m + v_world * h
        self.time_s += h

        # --- static friction (sticking) ----------------------------------
        # With the brakes applied at walking pace the tire is in the stick
        # regime and simply holds the vehicle. Without this the discrete slip
        # model limit-cycles through zero velocity: the moment a braking
        # impulse overshoots, the slip changes sign and the tire begins
        # pushing the car forwards instead, producing the classic stick-slip
        # chatter (velocity dithering around zero, motor RPM flickering
        # positive/negative). This is an inequality constraint on the
        # friction force, not a smoothing term.
        speed_now = float(np.hypot(*self.velocity_body_m_s))
        if (controls.brake > 0.0 or controls.handbrake) and speed_now < self.stick_speed_m_s:
            self.velocity_body_m_s[:] = 0.0
            self.wheel_speeds_rad_s[:] = 0.0
            self.yaw_rate_rad_s = 0.0

        # --- tire diagnostics ---------------------------------------------
        wheel_lin = self.wheel_speeds_rad_s * self.wheel_radius_m
        self.slip_ratios = self.tire_model.slip_ratio(v_long, wheel_lin)
        self.slip_angles_rad = self.tire_model.slip_angle(v_lat, v_long)
        self.tire_utilisation = self.tire_model.saturation(
            v_long,
            v_lat,
            wheel_lin,
            self.wheel_fz_N,
            lateral_mu_scale=lateral_mu_scale,
            long_grip_bias=long_grip_bias,
        )
        # "Sliding" is judged from the combined slip magnitude, not from the
        # friction-ellipse utilisation.  Utilisation is pinned at 1.0 whenever
        # the (stiff) slip model saturates its force, which a trivially small
        # velocity difference can do at low speed -- so a perfectly coupled
        # wheel would be reported as sliding.  Combined slip only grows when
        # the wheel really is spinning or locked.
        self.wheel_sliding = self.tire_model.sliding_mask(
            self.slip_ratios, self.slip_angles_rad, v_long
        )
        self.is_sliding = bool(np.any(self.wheel_sliding))

        # --- battery -----------------------------------------------------
        elec_power = float(
            self.motor.power_required(self.motor_rpm, self.motor_torque_Nm)
        )
        current = self.pack.solve_current(elec_power, soc=self.pack.soc)
        self.pack.update(h, current)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def validate(self, *, emit_warnings: bool = True) -> List[str]:
        """Run consistency checks across the pack, motor and chassis.

        Args:
            emit_warnings: Also emit findings through :mod:`warnings`.

        Returns:
            A list of findings (empty means clean).
        """
        msgs: List[str] = []
        msgs.extend(self.motor.validate(emit_warnings=False))
        msgs.extend(
            self.pack.validate(
                controller_max_V=self.motor.controller_max_V,
                required_cont_power_W=self.motor.cont_power_W,
                emit_warnings=False,
            )
        )

        if abs(self.tire_model.config.rear_long_grip_bias) < 1e-6:
            msgs.append("rear_long_grip_bias is zero; the rear axle cannot drive.")

        if emit_warnings:
            import warnings

            for msg in msgs:
                warnings.warn(f"[Vehicle] {msg}", stacklevel=2)
        return msgs


# ======================================================================
# Factory used by the demo and the headless verification suite
# ======================================================================
def create_demo_vehicle(
    *,
    chemistry: str = "Li-Ion (NMC)",
    series: int = 92,
    parallel: int = 3,
    motor_id: str = "pmsm-150kw",
    initial_soc: float = 0.95,
    config_path: Optional[Path] = None,
) -> Vehicle:
    """Build a sensible EV (≈51 kWh, 150 kW) for the demo and tests.

    The default 92S3P NMC pack tops out at ``92 * 4.2 = 386 V`` so it sits
    comfortably under the 400 V controller of the ``pmsm-150kw`` motor.

    Args:
        chemistry: Chemistry name from ``batteries.json``.
        series: Series cell count ``S``.
        parallel: Parallel cell count ``P``.
        motor_id: Motor id from ``motors.json``.
        initial_soc: Starting state of charge.
        config_path: Optional override for the vehicle config.

    Returns:
        A configured :class:`Vehicle`.
    """
    pack = BatteryPack(series, parallel, chemistry, initial_soc=initial_soc)
    motor = Motor(motor_id)
    return Vehicle(pack, motor, config_path=config_path)


# ======================================================================
# Headless verification suite (proves the physics behaves as specified)
# ======================================================================
def _run_self_test() -> int:
    """Run the headless physics checks. Returns a process exit code."""
    failures: List[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not ok:
            failures.append(name)

    print("=== EV simulator physics self-test ===\n")

    # -- Test 1: low SOC visibly reduces acceleration ------------------
    def accel_run(soc: float, seconds: float, dt: float = 1.0 / 120.0) -> Tuple[float, float, float]:
        veh = create_demo_vehicle(initial_soc=soc)
        veh.pack.soc = soc
        peak_power = 0.0
        steps = int(seconds / dt)
        for _ in range(steps):
            state = veh.step(dt, Controls(throttle=1.0))
            peak_power = max(peak_power, state.battery["power_W"])
        return state.speed_kmh, state.position_m[0], peak_power

    high_soc_speed, high_soc_dist, high_power = accel_run(0.9, 6.0)
    low_soc_speed, low_soc_dist, low_power = accel_run(0.07, 6.0)
    check(
        "Low SOC reduces acceleration",
        low_soc_speed < high_soc_speed * 0.8 and low_soc_dist < high_soc_dist * 0.8,
        f"6 s full throttle: 90% SOC -> {high_soc_speed:.1f} km/h / "
        f"{high_soc_dist:.1f} m, 7% SOC -> {low_soc_speed:.1f} km/h / "
        f"{low_soc_dist:.1f} m",
    )
    check(
        "Pack power scales with SOC",
        low_power < high_power * 0.8,
        f"peak pack power: 90% SOC {high_power / 1000:.1f} kW vs "
        f"7% SOC {low_power / 1000:.1f} kW",
    )

    # -- Test 2: voltage sags under load -------------------------------
    veh = create_demo_vehicle(initial_soc=0.9)
    rest_v = veh.pack.terminal_voltage(0.0)
    loaded_v = None
    for _ in range(240):  # 2 s of hard acceleration
        state = veh.step(1.0 / 120.0, Controls(throttle=1.0))
        loaded_v = state.battery["voltage_V"]
    assert loaded_v is not None
    check(
        "Voltage sags under load",
        loaded_v < rest_v - 1.0,
        f"rest {rest_v:.1f} V -> loaded {loaded_v:.1f} V "
        f"(sag {rest_v - loaded_v:.1f} V)",
    )

    # -- Test 3: handbrake + steering initiates a controllable drift ----
    veh = create_demo_vehicle(initial_soc=0.9)
    dt = 1.0 / 120.0
    for _ in range(int(3.0 / dt)):  # build speed (~60 km/h)
        veh.step(dt, Controls(throttle=1.0))
    entry_speed = float(np.hypot(*veh.velocity_body_m_s))

    max_yaw = 0.0
    max_beta = 0.0
    max_utilisation = 0.0
    min_forward_speed = math.inf
    for _ in range(int(1.0 / dt)):  # handbrake flick + steer into the corner
        state = veh.step(dt, Controls(steer=1.0, handbrake=True))
        max_yaw = max(max_yaw, abs(state.yaw_rate_rad_s))
        beta = math.degrees(
            math.atan2(state.velocity_body_m_s[1], max(state.speed_m_s, 1e-6))
        )
        max_beta = max(max_beta, abs(beta))
        max_utilisation = max(max_utilisation, float(np.max(state.tire_utilisation)))
        min_forward_speed = min(min_forward_speed, state.velocity_body_m_s[0])

    check(
        "Handbrake initiates a drift",
        max_yaw > 1.0 and max_beta > 20.0 and max_utilisation > 1.0
        and min_forward_speed > 0.0,
        f"{entry_speed * 3.6:.0f} km/h entry -> peak yaw {max_yaw:.2f} rad/s, "
        f"peak sideslip {max_beta:.1f}deg, peak utilisation {max_utilisation:.2f}, "
        f"never driven backwards (min vx {min_forward_speed:.1f} m/s)",
    )

    # -- Test 4: friction ellipse saturation is reachable ---------------
    check(
        "Friction ellipse saturates (skid marks)",
        max_utilisation > 1.0,
        f"peak utilisation {max_utilisation:.2f} (>1 means sliding)",
    )

    # -- Test 5: the drift is controllable (counter-steer recovers) -----
    for _ in range(int(3.0 / dt)):
        state = veh.step(dt, Controls(steer=-0.6, throttle=0.5))
    recovered_yaw = abs(state.yaw_rate_rad_s)
    recovered_utilisation = float(np.max(state.tire_utilisation))
    check(
        "Drift is recoverable with counter-steer",
        recovered_yaw < max_yaw * 0.5
        and state.speed_m_s > 2.0
        and recovered_utilisation < 1.0,
        f"peak yaw {max_yaw:.2f} -> {recovered_yaw:.2f} rad/s after 3 s of "
        f"opposite lock, still doing {state.speed_kmh:.0f} km/h and back under "
        f"the friction limit (utilisation {recovered_utilisation:.2f})",
    )

    # -- Test 6: coasting decelerates (drag + rolling resistance) -------
    veh = create_demo_vehicle(initial_soc=0.9)
    for _ in range(int(4.0 / dt)):
        veh.step(dt, Controls(throttle=1.0))
    speed_before_coast = float(np.hypot(*veh.velocity_body_m_s))
    for _ in range(int(4.0 / dt)):
        veh.step(dt, Controls())
    speed_after_coast = float(np.hypot(*veh.velocity_body_m_s))
    check(
        "Coasting decelerates (drag + rolling resistance)",
        speed_after_coast < speed_before_coast - 0.5,
        f"coast 4 s: {speed_before_coast * 3.6:.1f} -> "
        f"{speed_after_coast * 3.6:.1f} km/h",
    )

    # -- Test 7: regen puts energy back into the pack -------------------
    veh = create_demo_vehicle(initial_soc=0.9)
    for _ in range(int(4.0 / dt)):
        veh.step(dt, Controls(throttle=1.0))
    before_wh = veh.pack.recovered_Wh
    for _ in range(int(4.0 / dt)):
        veh.step(dt, Controls(brake=1.0))
    regen_wh = veh.pack.recovered_Wh - before_wh
    check(
        "Regen recovers energy",
        regen_wh > 0.5 and float(np.hypot(*veh.velocity_body_m_s)) < 5.0,
        f"recovered {regen_wh:.1f} Wh braking to "
        f"{float(np.hypot(*veh.velocity_body_m_s)) * 3.6:.1f} km/h",
    )

    # -- Test 8: braking is smooth, sustained and locks no wheels -------
    veh = create_demo_vehicle(initial_soc=0.95)
    for _ in range(int(6.0 / dt)):
        veh.step(dt, Controls(throttle=1.0))
    entry_rpm = veh.motor_rpm

    negative_rpm_steps = 0
    zero_rpm_while_moving = 0
    max_substep_jump = 0.0
    regen_steps = 0
    rpms: List[float] = []
    previous_rpm: Optional[float] = None
    for _ in range(int(6.0 / dt)):
        state = veh.step(dt, Controls(brake=1.0))
        rpm = state.motor_rpm
        if previous_rpm is not None:
            max_substep_jump = max(max_substep_jump, abs(rpm - previous_rpm))
        previous_rpm = rpm
        rpms.append(rpm)
        if rpm < -1.0:
            negative_rpm_steps += 1
        if rpm == 0.0 and state.speed_kmh > 5.0:
            zero_rpm_while_moving += 1
        if state.battery["current_A"] < -0.5:
            regen_steps += 1

    check(
        "Braking: wheel speed stays a pure integrator",
        negative_rpm_steps == 0
        and zero_rpm_while_moving == 0
        and max_substep_jump < 1000.0,
        f"{entry_rpm:.0f} rpm -> 0 with no snap: negative-RPM substeps "
        f"{negative_rpm_steps}, zero-RPM-while-moving {zero_rpm_while_moving}, "
        f"max per-substep jump {max_substep_jump:.0f} rpm",
    )

    tail = rpms[-120:]
    check(
        "Braking: regen sustains and the car stops cleanly",
        regen_steps > 180 and max(tail) == 0.0 and min(tail) == 0.0,
        f"regen current on {regen_steps} substeps; at rest (last 1 s) rpm in "
        f"[{min(tail):.0f}, {max(tail):.0f}]",
    )

    # -- Test 9: no wheel-spin oscillation under full throttle ----------
    veh = create_demo_vehicle(initial_soc=0.95)
    spin_rpms: List[float] = []
    max_rpm_below_60 = 0.0
    for _ in range(int(10.0 / dt)):
        state = veh.step(dt, Controls(throttle=1.0))
        spin_rpms.append(state.motor_rpm)
        if state.speed_kmh < 60.0:
            max_rpm_below_60 = max(max_rpm_below_60, state.motor_rpm)
    jitter = max(
        abs(spin_rpms[i] - 0.5 * (spin_rpms[i - 1] + spin_rpms[i + 1]))
        for i in range(1, len(spin_rpms) - 1)
    )
    check(
        "No wheel-spin oscillation under full throttle",
        jitter < 200.0 and max_rpm_below_60 < 20000.0,
        f"peak RPM deviation from trend {jitter:.0f} rpm (limit 1000); max wheel "
        f"speed below 60 km/h {max_rpm_below_60:.0f} rpm (limit 20000)",
    )

    # -- Test 10: SLIDING is slip-based, not force-utilisation based ----
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as handle:
        base_cfg = json.load(handle)

    def low_grip_vehicle(mu: float, traction_control: bool) -> Vehicle:
        """Build the demo car on a slippery surface, with/without TC."""
        cfg = json.loads(json.dumps(base_cfg))
        cfg["tires"]["mu_base"] = mu
        cfg["tires"]["load_sensitivity_per_kN"] = 0.0
        cfg["drivetrain"]["traction_control"] = traction_control
        return Vehicle(
            BatteryPack(92, 3, "Li-Ion (NMC)", initial_soc=0.95),
            Motor("pmsm-150kw"),
            vehicle_config=cfg,
        )

    slippery = low_grip_vehicle(0.30, traction_control=False)
    for _ in range(int(6.0 / dt)):
        slippery.step(dt, Controls(throttle=0.12))
    slow_states = 0
    false_sliding = 0
    for _ in range(int(5.0 / dt)):
        state = slippery.step(dt, Controls(steer=0.5))
        if 2.5 < state.speed_kmh < 12.0 and abs(state.battery["current_A"]) < 1.0:
            slow_states += 1
            false_sliding += int(state.is_sliding)
    check(
        "No false SLIDING flag at low speed",
        slow_states > 0 and false_sliding == 0,
        f"{false_sliding} SLIDING substeps out of {slow_states} sampled while "
        f"coasting and steering between 2.5-12 km/h at ~0 A",
    )

    # -- Test 11: SLIDING still fires when grip is genuinely lost -------
    slippery = low_grip_vehicle(0.30, traction_control=False)
    for _ in range(int(6.0 / dt)):
        slippery.step(dt, Controls(throttle=1.0))
    above_30 = 0
    genuine_slide = 0
    for _ in range(int(3.0 / dt)):
        state = slippery.step(dt, Controls(throttle=1.0))
        if state.speed_kmh > 30.0:
            above_30 += 1
            genuine_slide += int(state.is_sliding)
    check(
        "SLIDING still fires on genuine grip loss above 30 km/h",
        above_30 > 0 and genuine_slide > 0,
        f"{genuine_slide}/{above_30} substeps above 30 km/h flagged while "
        f"spinning up a low-grip surface",
    )

    # -- Test 12: slip stays bounded, TC never boosts torque ------------
    veh = create_demo_vehicle(initial_soc=0.95)
    max_slip_raw = 0.0
    max_tc_scale = 0.0
    negative_rpm_moving = 0
    for _ in range(int(20.0 / dt)):
        state = veh.step(dt, Controls(throttle=1.0))
        max_slip_raw = max(max_slip_raw, float(np.max(np.abs(state.slip_ratios))))
        if veh.last_command is not None:
            max_tc_scale = max(max_tc_scale, veh.last_command.traction_control_scale)
        if state.motor_rpm < 0.0 and state.speed_kmh > 5.0:
            negative_rpm_moving += 1
    check(
        "Slip stays bounded and TC never boosts torque",
        max_slip_raw < 1.5 and max_tc_scale <= 1.0 + 1e-9 and negative_rpm_moving == 0,
        f"max slip_raw {max_slip_raw:.3f} (fail threshold 1.5), max TC scale "
        f"{max_tc_scale:.3f} (must be <= 1.0), negative-RPM-while-moving "
        f"{negative_rpm_moving}",
    )

    # -- Test 13: traction control holds steady (no limit cycle) --------
    slippery = low_grip_vehicle(0.35, traction_control=True)
    scales: List[float] = []
    for _ in range(int(20.0 / dt)):
        slippery.step(dt, Controls(throttle=1.0))
        if slippery.last_command is not None:
            scales.append(slippery.last_command.traction_control_scale)
    sampled = np.array(scales[:: int(0.5 / dt)])
    jumps = np.abs(np.diff(sampled)) if sampled.size > 1 else np.zeros(1)
    big_jumps = int((jumps > 0.3).sum())
    check(
        "Traction control holds steady under sustained wheelspin",
        big_jumps <= 1 and sampled.min() < 1.0,
        f"TC engaged (min scale {sampled.min():.2f}); {big_jumps} changes >0.3 "
        f"between print lines (a limit cycle would toggle repeatedly)",
    )

    # -- Test 14: battery / motor / vehicle validation ------------------
    veh = create_demo_vehicle()
    msgs = veh.validate(emit_warnings=False)
    check(
        "Default configuration validates cleanly",
        len(msgs) == 0,
        "no warnings" if not msgs else "; ".join(msgs),
    )

    print()
    if failures:
        print(f"RESULT: {len(failures)} check(s) failed -> {failures}")
        return 1
    print("RESULT: all physics checks passed.")
    return 0


# ======================================================================
# pygame debug view
# ======================================================================
# These constants only affect rendering of the debug overlay, never physics.
_DEMO_WIDTH = 1120
_DEMO_HEIGHT = 720
_DEMO_PIXELS_PER_METRE = 11.0
_DEMO_OVERHANG_M = 0.30
_DEMO_TIRE_WIDTH_M = 0.25
_DEMO_SKID_MAX_POINTS = 700
_DEMO_TRAIL_PER_WHEEL = 400


def _run_pygame_demo(initial_soc: float = 0.95, max_frames: int = 0) -> int:
    """Open the top-down pygame debug view. Returns a process exit code.

    Args:
        initial_soc: Starting pack state of charge.
        max_frames: If ``> 0``, stop after this many frames (used for smoke
            testing the renderer). ``0`` means run until the user quits.
    """
    import pygame

    vehicle = create_demo_vehicle(initial_soc=initial_soc)
    dt = 1.0 / 120.0

    pygame.init()
    screen = pygame.display.set_mode((_DEMO_WIDTH, _DEMO_HEIGHT))
    pygame.display.set_caption("EV Simulator - physics debug view")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    from collections import deque

    skid: "deque[Tuple[float, float]]" = deque(maxlen=_DEMO_SKID_MAX_POINTS)
    visual_spin = np.zeros(4, dtype=float)
    camera = np.array([0.0, 0.0], dtype=float)
    zoom = _DEMO_PIXELS_PER_METRE
    camera_mode = "chase"

    throttle = 0.0
    brake = 0.0
    steer = 0.0
    print_accum = 0.0
    frame = 0
    running = True

    def world_to_screen(xy: np.ndarray) -> Tuple[float, float]:
        sx = _DEMO_WIDTH / 2.0 + (xy[0] - camera[0]) * zoom
        sy = _DEMO_HEIGHT / 2.0 - (xy[1] - camera[1]) * zoom
        return sx, sy

    try:
        while running:
            frame += 1
            if max_frames > 0 and frame > max_frames:
                running = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        vehicle.reset(initial_soc=initial_soc)
                        skid.clear()
                        visual_spin[:] = 0.0
                    elif event.key == pygame.K_c:
                        camera_mode = "fixed" if camera_mode == "chase" else "chase"

            keys = pygame.key.get_pressed()
            throttle = float(keys[pygame.K_w] or keys[pygame.K_UP])
            brake = float(keys[pygame.K_s] or keys[pygame.K_DOWN])
            left = keys[pygame.K_a] or keys[pygame.K_LEFT]
            right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
            steer = float(left) - float(right)
            handbrake = bool(keys[pygame.K_SPACE])

            state = vehicle.step(
                dt,
                Controls(
                    throttle=throttle,
                    brake=brake,
                    steer=steer,
                    handbrake=handbrake,
                ),
            )

            # Visual wheel spin accumulation (cosmetic only).
            visual_spin = visual_spin + state.wheel_speeds_rad_s * dt

            # Skid marks only where a tire is genuinely sliding.
            if state.is_sliding:
                cos_h = math.cos(state.heading_rad)
                sin_h = math.sin(state.heading_rad)
                for idx in range(4):
                    if state.wheel_sliding[idx]:
                        lx = vehicle.wheel_lx[idx]
                        ly = vehicle.wheel_ly[idx]
                        wx = state.position_m[0] + lx * cos_h - ly * sin_h
                        wy = state.position_m[1] + lx * sin_h + ly * cos_h
                        skid.append((wx, wy))

            if camera_mode == "chase":
                camera = 0.9 * camera + 0.1 * state.position_m

            # ---- draw -------------------------------------------------
            screen.fill((18, 20, 24))

            # ground grid every 5 m
            grid = 5.0
            x0 = camera[0] - _DEMO_WIDTH / 2 / zoom
            x1 = camera[0] + _DEMO_WIDTH / 2 / zoom
            y0 = camera[1] - _DEMO_HEIGHT / 2 / zoom
            y1 = camera[1] + _DEMO_HEIGHT / 2 / zoom
            gx = math.floor(x0 / grid) * grid
            while gx <= x1:
                sx, _ = world_to_screen(np.array([gx, 0.0]))
                pygame.draw.line(screen, (30, 34, 40), (sx, 0), (sx, _DEMO_HEIGHT), 1)
                gx += grid
            gy = math.floor(y0 / grid) * grid
            while gy <= y1:
                _, sy = world_to_screen(np.array([0.0, gy]))
                pygame.draw.line(screen, (30, 34, 40), (0, sy), (_DEMO_WIDTH, sy), 1)
                gy += grid

            # skid marks
            for (wx, wy) in skid:
                sx, sy = world_to_screen(np.array([wx, wy]))
                pygame.draw.circle(screen, (60, 48, 48), (int(sx), int(sy)), 2)

            # car body
            cos_h = math.cos(state.heading_rad)
            sin_h = math.sin(state.heading_rad)
            half_len = (vehicle.lf + vehicle.lr) / 2.0 + _DEMO_OVERHANG_M
            half_wid = max(vehicle.track_front_m, vehicle.track_rear_m) / 2.0 + 0.15
            length_offset = (vehicle.lf - vehicle.lr) / 2.0
            corners_body = [
                (length_offset + half_len, half_wid),
                (length_offset + half_len, -half_wid),
                (length_offset - half_len, -half_wid),
                (length_offset - half_len, half_wid),
            ]
            corners_screen = []
            for (bx, by) in corners_body:
                wx = state.position_m[0] + bx * cos_h - by * sin_h
                wy = state.position_m[1] + bx * sin_h + by * cos_h
                corners_screen.append(world_to_screen(np.array([wx, wy])))
            pygame.draw.polygon(screen, (52, 120, 200), corners_screen)
            pygame.draw.polygon(screen, (150, 200, 255), corners_screen, 2)

            # wheels as lines that rotate
            for idx in range(4):
                lx = vehicle.wheel_lx[idx]
                ly = vehicle.wheel_ly[idx]
                wx = state.position_m[0] + lx * cos_h - ly * sin_h
                wy = state.position_m[1] + lx * sin_h + ly * cos_h
                delta = state.steer_angle_rad if idx in FRONT_WHEELS else 0.0
                wheel_heading = state.heading_rad + delta
                spoke = visual_spin[idx]
                # tire carcass line (along rolling direction)
                carcass_dir = np.array(
                    [math.cos(wheel_heading), math.sin(wheel_heading)]
                )
                tread_dir = np.array([-carcass_dir[1], carcass_dir[0]])
                centre = np.array([wx, wy])
                a = centre + carcass_dir * vehicle.wheel_radius_m
                b = centre - carcass_dir * vehicle.wheel_radius_m
                pygame.draw.line(
                    screen, (230, 230, 230), world_to_screen(a), world_to_screen(b), 3
                )
                # rotating spoke shows the wheel actually spinning
                spoke_vec = tread_dir * math.cos(spoke) + carcass_dir * math.sin(spoke)
                c = centre
                d = centre + spoke_vec * vehicle.wheel_radius_m
                colour = (255, 90, 90) if state.tire_utilisation[idx] > 1.0 else (255, 210, 90)
                pygame.draw.line(screen, colour, world_to_screen(c), world_to_screen(d), 2)

            # ---- HUD ---------------------------------------------------
            combined_slip = float(
                np.max(
                    np.sqrt(state.slip_ratios**2 + np.tan(state.slip_angles_rad) ** 2)
                )
            )
            tc_scale = (
                vehicle.last_command.traction_control_scale
                if vehicle.last_command is not None
                else 1.0
            )
            lines = [
                f"SOC        {state.battery['soc_pct']:5.1f} %",
                f"Pack V     {state.battery['voltage_V']:6.1f} V",
                f"Current    {state.battery['current_A']:6.1f} A",
                f"Power      {state.battery['power_W'] / 1000.0:6.1f} kW",
                f"Cell temp  {state.battery['cell_temp_C']:6.1f} C",
                f"Motor      {state.motor_rpm:6.0f} rpm",
                f"Motor T    {state.motor_torque_Nm:6.1f} Nm",
                f"Speed      {state.speed_kmh:6.1f} km/h",
                f"Yaw rate   {math.degrees(state.yaw_rate_rad_s):6.1f} deg/s",
                f"Steer      {math.degrees(state.steer_angle_rad):6.1f} deg",
                f"Comb slip  {combined_slip:5.3f} "
                f"{'SLIDING' if state.is_sliding else ''}",
                f"Tire use   {np.max(state.tire_utilisation):5.2f}    TC {tc_scale:4.2f}",
                f"Camera     {camera_mode}  (C)  reset (R)",
            ]
            for i, text in enumerate(lines):
                surface = font.render(text, True, (220, 225, 235))
                screen.blit(surface, (12, 10 + i * 18))

            pygame.display.flip()

            # console telemetry every 0.5 s
            print_accum += dt
            if print_accum >= 0.5:
                print_accum = 0.0
                print(
                    f"SOC {state.battery['soc_pct']:5.1f}% | "
                    f"{state.battery['voltage_V']:6.1f} V | "
                    f"{state.battery['current_A']:7.1f} A | "
                    f"{state.battery['power_W'] / 1000.0:7.1f} kW | "
                    f"{state.motor_rpm:6.0f} rpm | "
                    f"{state.speed_kmh:6.1f} km/h | "
                    f"{'SLIDING' if state.is_sliding else '       '} "
                    f"slip={combined_slip:5.3f} TC={tc_scale:4.2f}"
                )

            clock.tick(120)
    finally:
        pygame.quit()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point: run the headless self-test or the pygame debug view.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run the physics verification suite without opening a window",
    )
    parser.add_argument(
        "--soc",
        type=float,
        default=0.95,
        help="initial state of charge for the demo (0..1)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="stop the pygame demo after N frames (0 = run until quit)",
    )
    args = parser.parse_args(argv)

    if args.headless:
        return _run_self_test()

    try:
        return _run_pygame_demo(initial_soc=args.soc, max_frames=args.frames)
    except Exception as exc:  # pragma: no cover - depends on display
        print(f"Could not open the pygame window ({exc}).")
        print("Falling back to the headless physics self-test.\n")
        return _run_self_test()


if __name__ == "__main__":
    raise SystemExit(main())
