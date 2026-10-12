"""Drivetrain: blends throttle, regen and friction brakes into torque.

The drivetrain is the only place where the gear ratio and the final-drive
efficiency are applied.  It takes a *driver* intent (throttle, brake, handbrake
is handled by the tire model) and produces:

* a motor-shaft torque command (positive = drive, negative = regen), and
* a friction brake torque per wheel for the front and rear axles.

Brake blending rule
-------------------

Both pedals are normalised to ``0..1``.  The friction brakes have a fixed total
capacity; regen is used **first**, up to the smaller of the motor's regen
envelope at the current speed and whatever the battery will accept, and only
the remainder is handed to the friction brakes::

    T_brake_total = brake * 4 * T_friction_per_wheel
    T_regen_wheel = min(T_regen_motor * gear * eff, T_brake_total)
    T_friction    = T_brake_total - T_regen_wheel
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .motor import Motor

__all__ = ["DrivetrainConfig", "DrivetrainCommand", "Drivetrain"]


@dataclass(frozen=True)
class DrivetrainConfig:
    """Gearbox / final drive / brake hardware parameters.

    Attributes:
        gear_ratio: Total motor-to-wheel gear ratio (dimensionless).
        final_drive_efficiency: Mechanical efficiency from motor shaft to wheel.
        max_friction_brake_torque_Nm_per_wheel: Peak hydraulic brake torque at
            one wheel (Nm).
        front_brake_bias: Fraction of friction brake torque sent to the front
            axle (``0..1``).
        driven_wheels: Number of driven wheels (2 for RWD/FWD, 4 for AWD).
        wheel_radius_m: Effective rolling radius (m).
        driven_is_front: ``True`` when the driven axle is the front one.
        traction_control: When ``True`` the motor torque is cut back as the
            driven wheels start to spin.
        traction_control_slip_target: Driven-wheel slip ratio the controller
            tries to hold (e.g. 0.15).
        traction_control_min_scale: Lowest torque multiplier the controller may
            command, so it can never cut drive completely.
        traction_control_engage_slip: Slip ratio at which TC activates.
        traction_control_release_slip: Slip ratio below which TC starts
            considering release (hysteresis band).
        traction_control_release_hold: Substeps slip must stay below
            ``traction_control_release_slip`` before TC releases.
        traction_control_recovery_rate: Maximum increase of the TC scale per
            substep. The scale may always drop instantly (that is the safety
            direction) but must recover slowly, which is what stops the
            engage/release limit cycle.
    """

    gear_ratio: float
    final_drive_efficiency: float
    max_friction_brake_torque_Nm_per_wheel: float
    front_brake_bias: float
    driven_wheels: int
    wheel_radius_m: float
    driven_is_front: bool = False
    traction_control: bool = True
    traction_control_slip_target: float = 0.15
    traction_control_min_scale: float = 0.2
    traction_control_engage_slip: float = 0.18
    traction_control_release_slip: float = 0.08
    traction_control_release_hold: int = 5
    traction_control_recovery_rate: float = 0.02

    @classmethod
    def from_dict(
        cls, drivetrain: Dict[str, Any], wheels: Dict[str, Any]
    ) -> "DrivetrainConfig":
        """Build a config from the ``drivetrain`` and ``wheels`` JSON sections.

        Args:
            drivetrain: The ``drivetrain`` mapping from ``vehicle.json``.
            wheels: The ``wheels`` mapping from ``vehicle.json``.

        Returns:
            A validated :class:`DrivetrainConfig`.
        """
        axle = str(wheels.get("driven_axle", "rear")).lower()
        driven = 4 if axle in ("all", "awd", "4wd") else 2
        return cls(
            gear_ratio=float(drivetrain["gear_ratio"]),
            final_drive_efficiency=float(drivetrain["final_drive_efficiency"]),
            max_friction_brake_torque_Nm_per_wheel=float(
                drivetrain["max_friction_brake_torque_Nm_per_wheel"]
            ),
            front_brake_bias=float(wheels.get("front_brake_bias", 0.6)),
            driven_wheels=driven,
            wheel_radius_m=float(wheels["radius_m"]),
            driven_is_front=axle in ("front", "fwd"),
            traction_control=bool(drivetrain.get("traction_control", True)),
            traction_control_slip_target=float(
                drivetrain.get("traction_control_slip_target", 0.15)
            ),
            traction_control_min_scale=float(
                drivetrain.get("traction_control_min_scale", 0.2)
            ),
            traction_control_engage_slip=float(
                drivetrain.get("traction_control_engage_slip", 0.18)
            ),
            traction_control_release_slip=float(
                drivetrain.get("traction_control_release_slip", 0.08)
            ),
            traction_control_release_hold=int(
                drivetrain.get("traction_control_release_hold", 5)
            ),
            traction_control_recovery_rate=float(
                drivetrain.get("traction_control_recovery_rate", 0.02)
            ),
        )


@dataclass
class DrivetrainCommand:
    """Result of blending the pedals for one control step.

    Attributes:
        motor_torque_Nm: Torque command at the motor shaft (positive drives,
            negative regenerates).
        throttle_torque_Nm: The motoring part of the command before regen.
        regen_motor_torque_Nm: Magnitude of the regen torque applied (>= 0).
        friction_brake_front_Nm_per_wheel: Hydraulic brake torque per front wheel.
        friction_brake_rear_Nm_per_wheel: Hydraulic brake torque per rear wheel.
        total_brake_wheel_torque_Nm: Total braking torque requested at the wheels.
        regen_wheel_torque_Nm: Regen torque at the driven axle (positive magnitude).
        traction_control_scale: Torque multiplier applied by traction control
            (``1.0`` = not intervening).
    """

    motor_torque_Nm: float
    throttle_torque_Nm: float
    regen_motor_torque_Nm: float
    friction_brake_front_Nm_per_wheel: float
    friction_brake_rear_Nm_per_wheel: float
    total_brake_wheel_torque_Nm: float
    regen_wheel_torque_Nm: float = 0.0
    traction_control_scale: float = 1.0


class Drivetrain:
    """Converts driver inputs into motor and brake torques.

    Args:
        config: Gearbox, brake and wheel geometry parameters.
        motor: The :class:`~core.motor.Motor` driven by this drivetrain.

    Raises:
        ValueError: If the gear ratio or wheel radius is non-positive.
    """

    def __init__(self, config: DrivetrainConfig, motor: Motor) -> None:
        if config.gear_ratio <= 0.0:
            raise ValueError("gear_ratio must be > 0")
        if config.wheel_radius_m <= 0.0:
            raise ValueError("wheel_radius_m must be > 0")
        self.config = config
        self.motor = motor
        # --- traction control state (see DrivetrainConfig) ---------------
        self._tc_active: bool = False
        self._tc_scale: float = 1.0
        self._tc_below_release: int = 0

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Drivetrain(gear={self.config.gear_ratio:.2f}, "
            f"eff={self.config.final_drive_efficiency:.2f}, "
            f"driven={self.config.driven_wheels}, motor={self.motor.id})"
        )

    # ------------------------------------------------------------------
    # Torque / force conversions
    # ------------------------------------------------------------------
    @property
    def motor_to_wheel_torque_gain(self) -> float:
        """Multiplier from motor torque to total wheel torque."""
        return self.config.gear_ratio * self.config.final_drive_efficiency

    def wheel_torque(self, motor_torque_Nm: float) -> float:
        """Total wheel torque (Nm) produced by a motor-shaft torque.

        Losses are applied here, so regen is reduced by the same efficiency
        that reduces motoring.
        """
        return float(motor_torque_Nm) * self.motor_to_wheel_torque_gain

    def tractive_force(self, motor_torque_Nm: float) -> float:
        """Total longitudinal force (N) at the contact patches.

        Args:
            motor_torque_Nm: Motor-shaft torque (positive = drive).

        Returns:
            Force in newtons (positive = forward).
        """
        return self.wheel_torque(motor_torque_Nm) / self.config.wheel_radius_m

    def motor_torque_for_force(self, force_N: float) -> float:
        """Inverse of :meth:`tractive_force` (force in N -> motor torque in Nm)."""
        gain = self.motor_to_wheel_torque_gain
        if gain <= 0.0:
            return 0.0
        return float(force_N) * self.config.wheel_radius_m / gain

    def drive_torque_per_wheel(self, motor_torque_Nm: float) -> float:
        """Wheel torque delivered to each driven wheel (open differential).

        A symmetrical (open) differential splits torque equally, so each driven
        wheel receives ``total / driven_wheels``.
        """
        return self.wheel_torque(motor_torque_Nm) / max(self.config.driven_wheels, 1)

    # ------------------------------------------------------------------
    # Control blending
    # ------------------------------------------------------------------
    def command(
        self,
        throttle: float,
        brake: float,
        motor_rpm: float,
        *,
        drive_power_limit_W: Optional[float] = None,
        regen_torque_limit_Nm: Optional[float] = None,
        front_brake_bias: Optional[float] = None,
        max_total_brake_wheel_torque_Nm: Optional[float] = None,
        drive_slip: Optional[float] = None,
    ) -> DrivetrainCommand:
        """Blend the pedals into a motor torque and friction brake split.

        Args:
            throttle: Throttle pedal position, ``0..1``.
            brake: Brake pedal position, ``0..1``.
            motor_rpm: Current motor shaft speed (rpm).
            drive_power_limit_W: Optional cap on electrical drive power (W),
                typically the battery's available discharge power.
            regen_torque_limit_Nm: Optional cap on regen torque magnitude (Nm),
                typically derived from the battery's accepted charge current.
            front_brake_bias: Optional override for the front/rear friction
                brake split (``0..1``). The vehicle supplies this when it is
                distributing brake torque proportionally to axle load.
            max_total_brake_wheel_torque_Nm: Optional cap on the total braking
                torque at the wheels. The vehicle supplies the tire grip
                budget here so the friction brakes cannot demand more torque
                than the tires can react, which would lock every wheel.
            drive_slip: Current average slip ratio of the driven wheels, used
                by traction control. ``None`` disables the controller for this
                step.

        Returns:
            A :class:`DrivetrainCommand`.
        """
        cfg = self.config
        throttle = float(min(max(throttle, 0.0), 1.0))
        brake = float(min(max(brake, 0.0), 1.0))
        rpm = float(motor_rpm)
        bias = cfg.front_brake_bias if front_brake_bias is None else float(
            front_brake_bias
        )
        bias = min(max(bias, 0.0), 1.0)

        # --- motoring ----------------------------------------------------
        torque_available = float(self.motor.torque_at(rpm))
        throttle_torque = throttle * torque_available

        if throttle_torque > 0.0 and drive_power_limit_W is not None:
            omega = max(abs(rpm) * 2.0 * math.pi / 60.0, 1e-6)
            load = min(throttle, 1.0)
            eff = float(self.motor.efficiency_at(rpm, load))
            # Electrical limit -> mechanical torque limit.
            torque_power_limit = max(drive_power_limit_W, 0.0) * eff / omega
            throttle_torque = min(throttle_torque, torque_power_limit)

        # --- traction control --------------------------------------------
        # Slip regulator with hysteresis and a rate limit on recovery.
        # Without both, the controller chatters: it releases the moment slip
        # dips (letting the wheel spin straight back up), re-engages, and can
        # cascade into a blow-up. Engage above `engage_slip`, and only release
        # after slip has stayed below `release_slip` for several substeps.
        if cfg.traction_control and drive_slip is not None:
            slip = float(drive_slip)
            if not self._tc_active:
                if slip > cfg.traction_control_engage_slip:
                    self._tc_active = True
                    self._tc_below_release = 0
            elif slip < cfg.traction_control_release_slip:
                self._tc_below_release += 1
                if self._tc_below_release >= cfg.traction_control_release_hold:
                    self._tc_active = False
                    self._tc_below_release = 0
            else:
                self._tc_below_release = 0

            if self._tc_active:
                # Hysteresis means TC stays active well below the target slip,
                # where target/slip would exceed 1.0. Clamp it: traction
                # control may only ever REDUCE torque.
                target_scale = min(
                    1.0,
                    max(
                        cfg.traction_control_slip_target / max(slip, 1e-6),
                        cfg.traction_control_min_scale,
                    ),
                )
            else:
                target_scale = 1.0

            # The scale may drop instantly (safety direction) but recovers
            # slowly, which damps the loop.
            if target_scale < self._tc_scale:
                self._tc_scale = target_scale
            else:
                self._tc_scale = min(
                    target_scale,
                    self._tc_scale + cfg.traction_control_recovery_rate,
                    1.0,
                )
            self._tc_scale = min(self._tc_scale, 1.0)
            throttle_torque *= self._tc_scale
            tc_scale = self._tc_scale
        else:
            self._tc_scale = 1.0
            tc_scale = 1.0

        # --- regeneration ------------------------------------------------
        regen_cap = abs(float(self.motor.max_regen_torque(rpm)))
        if regen_torque_limit_Nm is not None:
            regen_cap = min(regen_cap, max(float(regen_torque_limit_Nm), 0.0))

        total_brake_torque = (
            brake * cfg.max_friction_brake_torque_Nm_per_wheel * 4.0
        )
        if max_total_brake_wheel_torque_Nm is not None:
            total_brake_torque = min(
                total_brake_torque,
                max(float(max_total_brake_wheel_torque_Nm), 0.0),
            )

        gain = self.motor_to_wheel_torque_gain
        regen_available_wheel = regen_cap * gain  # torque at the driven axle

        front_axle_demand = total_brake_torque * bias
        rear_axle_demand = total_brake_torque * (1.0 - bias)

        # Regen substitutes friction *on the axle it actually acts on*. If it
        # were merely subtracted from the total, the driven axle would receive
        # its full friction share *plus* the regen torque and lock instantly —
        # which kills regen (a locked wheel has zero speed) and makes the motor
        # RPM collapse.
        if cfg.driven_wheels >= 4:
            # All four wheels are driven, so regen is spread across them.
            regen_wheel_torque = min(regen_available_wheel, total_brake_torque)
            friction_total = total_brake_torque - regen_wheel_torque
            front_friction_per_wheel = friction_total * bias / 2.0
            rear_friction_per_wheel = friction_total * (1.0 - bias) / 2.0
        elif cfg.driven_is_front:
            regen_wheel_torque = min(regen_available_wheel, front_axle_demand)
            front_friction_per_wheel = (front_axle_demand - regen_wheel_torque) / 2.0
            rear_friction_per_wheel = rear_axle_demand / 2.0
        else:
            regen_wheel_torque = min(regen_available_wheel, rear_axle_demand)
            front_friction_per_wheel = front_axle_demand / 2.0
            rear_friction_per_wheel = (rear_axle_demand - regen_wheel_torque) / 2.0

        regen_motor_torque = regen_wheel_torque / gain if gain > 0.0 else 0.0

        return DrivetrainCommand(
            motor_torque_Nm=throttle_torque - regen_motor_torque,
            throttle_torque_Nm=throttle_torque,
            regen_motor_torque_Nm=regen_motor_torque,
            friction_brake_front_Nm_per_wheel=front_friction_per_wheel,
            friction_brake_rear_Nm_per_wheel=rear_friction_per_wheel,
            total_brake_wheel_torque_Nm=total_brake_torque,
            regen_wheel_torque_Nm=regen_wheel_torque,
            traction_control_scale=tc_scale,
        )

    # ------------------------------------------------------------------
    def max_regen_torque_from_battery(
        self, regen_power_W: float, motor_rpm: float
    ) -> float:
        """Convert an available regen power (W) into a regen torque cap (Nm).

        Args:
            regen_power_W: Electrical power the battery can absorb (>= 0).
            motor_rpm: Current motor shaft speed (rpm).

        Returns:
            Maximum regen torque magnitude in Nm.
        """
        omega = max(abs(motor_rpm) * 2.0 * math.pi / 60.0, 1e-6)
        load = 0.5
        eff = float(self.motor.efficiency_at(motor_rpm, load))
        return max(regen_power_W, 0.0) * eff / omega
