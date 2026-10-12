"""Slip-based tire model with a friction ellipse (two-track).

This module implements the forces that make vehicle dynamics emergent rather
than scripted.  Nothing here "decides" to drift: the rear axle simply loses
lateral capacity when it is asked for more combined force than the friction
ellipse allows, and the yaw rate follows.

Core relations
--------------

Slip ratio (longitudinal), with ``eps`` avoiding division by zero at rest:

.. math:: \\kappa = \\frac{\\omega_{wheel} R - u}{\\max(|u|, \\epsilon)}

Slip angle (lateral):

.. math:: \\alpha = \\operatorname{atan2}(v, \\max(|u|, \\epsilon))

Linear-then-saturating forces:

.. math:: F_x = C_x \\kappa, \\qquad F_y = -C_y \\alpha

clamped to the per-axis friction limits, then scaled onto a friction ellipse:

.. math::

    \\left(\\frac{F_x}{\\mu_x F_z}\\right)^2 +
    \\left(\\frac{F_y}{\\mu_y F_z}\\right)^2 \\le 1

Load sensitivity makes :math:`\\mu` fall slightly as the vertical load
:math:`F_z` rises, which is what makes weight transfer actually change grip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import numpy as np

__all__ = [
    "TireConfig",
    "TireModel",
    "slip_ratio",
    "slip_angle",
    "load_sensitive_mu",
    "friction_ellipse",
    "combined_slip",
    "effective_slip_ratio",
]

ArrayLike = Union[float, np.ndarray]


def slip_ratio(
    long_vel: ArrayLike, wheel_vel: ArrayLike, eps: float = 0.5
) -> np.ndarray:
    """Longitudinal slip ratio.

    Args:
        long_vel: Wheel-hub longitudinal velocity (m/s).
        wheel_vel: Equivalent wheel linear speed ``omega * R`` (m/s).
        eps: Velocity floor used to keep the ratio finite at rest.

    Returns:
        Slip ratio (dimensionless). ``-1`` means a fully locked wheel,
        positive values mean wheelspin.
    """
    u = np.asarray(long_vel, dtype=float)
    w = np.asarray(wheel_vel, dtype=float)
    denom = np.maximum(np.abs(u), eps)
    return (w - u) / denom


def slip_angle(
    lateral_vel: ArrayLike, long_vel: ArrayLike, eps: float = 0.5
) -> np.ndarray:
    """Tire slip angle in radians.

    Args:
        lateral_vel: Lateral velocity at the contact patch (m/s).
        long_vel: Longitudinal velocity at the contact patch (m/s).
        eps: Velocity floor used to keep the angle finite at rest.

    Returns:
        Slip angle in radians (positive = sliding toward +y / left).
    """
    v = np.asarray(lateral_vel, dtype=float)
    u = np.asarray(long_vel, dtype=float)
    return np.arctan2(v, np.maximum(np.abs(u), eps))


def load_sensitive_mu(
    fz: ArrayLike,
    mu_base: float,
    sensitivity_per_kN: float,
    fz_reference_N: float,
    mu_min: float,
) -> np.ndarray:
    """Friction coefficient after load sensitivity.

    Real tires lose a little grip per unit load (they gain less than
    proportionally), modelled here as a linear loss relative to a reference
    load.

    Args:
        fz: Vertical load on the tire (N).
        mu_base: Peak friction coefficient at the reference load.
        sensitivity_per_kN: ``mu`` lost per kN above the reference load.
        fz_reference_N: Reference vertical load (N).
        mu_min: Lower clamp for ``mu``.

    Returns:
        Effective friction coefficient per tire.
    """
    load = np.asarray(fz, dtype=float)
    drop = sensitivity_per_kN * (load - fz_reference_N) / 1000.0
    return np.clip(mu_base - drop, mu_min, mu_base)


def friction_ellipse(
    f_long_raw: ArrayLike,
    f_lat_raw: ArrayLike,
    limit_long: ArrayLike,
    limit_lat: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale a combined force demand onto the friction ellipse.

    When the demanded force exceeds what the tire can generate the whole force
    vector is scaled down, which is exactly the behaviour that makes the front
    push (understeer) or the rear come around (oversteer/drift) depending on
    which axle saturates first.

    Args:
        f_long_raw: Unclamped longitudinal force demand (N).
        f_lat_raw: Unclamped lateral force demand (N).
        limit_long: Longitudinal friction limit ``mu_x * Fz`` (N).
        limit_lat: Lateral friction limit ``mu_y * Fz`` (N).

    Returns:
        Tuple ``(f_long, f_lat)`` of scaled forces (N).
    """
    fx = np.asarray(f_long_raw, dtype=float)
    fy = np.asarray(f_lat_raw, dtype=float)
    lx = np.maximum(np.asarray(limit_long, dtype=float), 1e-9)
    ly = np.maximum(np.asarray(limit_lat, dtype=float), 1e-9)

    utilisation = (fx / lx) ** 2 + (fy / ly) ** 2
    scale = np.where(utilisation > 1.0, 1.0 / np.sqrt(np.maximum(utilisation, 1e-12)), 1.0)
    return fx * scale, fy * scale


def combined_slip(
    slip_ratio_value: ArrayLike, slip_angle_value: ArrayLike
) -> np.ndarray:
    """Magnitude of the combined slip demand.

    Both terms are velocity ratios (dimensionless), so they can be added in
    quadrature:

    .. math:: s = \\sqrt{\\kappa^2 + \\tan^2\\alpha}

    This is the right quantity to threshold for a "the tire is sliding" flag:
    unlike the friction-ellipse utilisation it stays small when the tire is
    merely loaded, and it grows without bound when the wheel really is spinning
    or locked.

    Args:
        slip_ratio_value: Longitudinal slip ratio(s).
        slip_angle_value: Slip angle(s) in radians.

    Returns:
        Combined slip magnitude (dimensionless, ``0`` = perfectly coupled).
    """
    kappa = np.asarray(slip_ratio_value, dtype=float)
    alpha = np.asarray(slip_angle_value, dtype=float)
    return np.sqrt(kappa * kappa + np.tan(alpha) ** 2)


def effective_slip_ratio(
    slip_ratio_value: ArrayLike, limit: float = 1.0
) -> np.ndarray:
    """Clip the slip ratio to its physically meaningful range.

    ``|kappa| = 1`` is the ceiling: ``+1`` is a wheel spinning at twice road
    speed, ``-1`` a fully locked wheel, and beyond that the tire force is
    already fully saturated. Letting ``kappa`` run to 8 or more therefore adds
    nothing to the force but does add stiffness and numerical noise, so all
    force math uses this clipped value while the raw value is retained for
    reporting.

    Args:
        slip_ratio_value: Raw slip ratio(s).
        limit: Symmetric clip bound (default 1.0).

    Returns:
        Clipped slip ratio(s).
    """
    return np.clip(np.asarray(slip_ratio_value, dtype=float), -limit, limit)


@dataclass(frozen=True)
class TireConfig:
    """Tire parameters (all sourced from ``config/vehicle.json``)."""

    cornering_stiffness_N_per_rad: float
    longitudinal_stiffness_N_per_slip: float
    mu_base: float
    load_sensitivity_per_kN: float
    fz_reference_N: float
    mu_min: float
    handbrake_lat_mu_factor: float
    slip_eps_m_s: float
    rear_long_grip_bias: float = 1.0
    sliding_slip_threshold: float = 0.2
    catastrophic_slip_threshold: float = 0.5
    low_speed_sliding_kmh: float = 10.0
    slip_ratio_limit: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TireConfig":
        """Build a config from the ``tires`` section of ``vehicle.json``."""
        return cls(
            cornering_stiffness_N_per_rad=float(
                data["cornering_stiffness_N_per_rad"]
            ),
            longitudinal_stiffness_N_per_slip=float(
                data["longitudinal_stiffness_N_per_slip"]
            ),
            mu_base=float(data["mu_base"]),
            load_sensitivity_per_kN=float(data["load_sensitivity_per_kN"]),
            fz_reference_N=float(data["fz_reference_N"]),
            mu_min=float(data["mu_min"]),
            handbrake_lat_mu_factor=float(data["handbrake_lat_mu_factor"]),
            slip_eps_m_s=float(data["slip_eps_m_s"]),
            rear_long_grip_bias=float(data.get("rear_long_grip_bias", 1.0)),
            sliding_slip_threshold=float(data.get("sliding_slip_threshold", 0.2)),
            catastrophic_slip_threshold=float(
                data.get("catastrophic_slip_threshold", 0.5)
            ),
            low_speed_sliding_kmh=float(data.get("low_speed_sliding_kmh", 10.0)),
            slip_ratio_limit=float(data.get("slip_ratio_limit", 1.0)),
        )


class TireModel:
    """Vectorised slip-based tire model operating on all wheels at once.

    Args:
        config: Tire parameters.
    """

    def __init__(self, config: TireConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    def slip_ratio(
        self, long_vel: ArrayLike, wheel_vel: ArrayLike
    ) -> np.ndarray:
        """Slip ratio; see :func:`slip_ratio`."""
        return slip_ratio(long_vel, wheel_vel, self.config.slip_eps_m_s)

    def slip_angle(
        self, lateral_vel: ArrayLike, long_vel: ArrayLike
    ) -> np.ndarray:
        """Slip angle in radians; see :func:`slip_angle`."""
        return slip_angle(lateral_vel, long_vel, self.config.slip_eps_m_s)

    def effective_mu(self, fz: ArrayLike) -> np.ndarray:
        """Load-sensitive friction coefficient for each tire."""
        cfg = self.config
        return load_sensitive_mu(
            fz,
            cfg.mu_base,
            cfg.load_sensitivity_per_kN,
            cfg.fz_reference_N,
            cfg.mu_min,
        )

    # ------------------------------------------------------------------
    def forces(
        self,
        long_vel: ArrayLike,
        lateral_vel: ArrayLike,
        wheel_speed: ArrayLike,
        fz: ArrayLike,
        *,
        lateral_mu_scale: ArrayLike = 1.0,
        long_grip_bias: ArrayLike = 1.0,
        slip_gain: ArrayLike = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute longitudinal and lateral tire forces.

        Args:
            long_vel: Contact-patch longitudinal velocity per wheel (m/s).
            lateral_vel: Contact-patch lateral velocity per wheel (m/s).
            wheel_speed: Wheel linear speed ``omega * R`` per wheel (m/s).
            fz: Vertical load per wheel (N).
            lateral_mu_scale: Multiplier on lateral capacity per wheel
                (0 drops lateral grip entirely, e.g. a locked handbrake).
            long_grip_bias: Per-wheel multiplier on longitudinal capacity,
                creating a friction ellipse rather than a circle.
            slip_gain: Per-wheel multiplier on the slip (used for surface
                changes; 1.0 on asphalt).

        Returns:
            Tuple ``(f_long, f_lat)`` in newtons. ``f_long`` is positive when
            driving the car forward, ``f_lat`` is positive to the left.
        """
        cfg = self.config
        gain = np.asarray(slip_gain, dtype=float)
        # Raw slip is unbounded by construction; clip it for the force math.
        kappa = effective_slip_ratio(
            self.slip_ratio(long_vel, wheel_speed) * gain, cfg.slip_ratio_limit
        )
        alpha = self.slip_angle(lateral_vel, long_vel) * np.asarray(
            slip_gain, dtype=float
        )

        mu = self.effective_mu(fz)
        limit_long = mu * np.asarray(fz, dtype=float) * np.asarray(
            long_grip_bias, dtype=float
        )
        limit_lat = mu * np.asarray(fz, dtype=float) * np.asarray(
            lateral_mu_scale, dtype=float
        )

        # Linear slip model, clamped to the per-axis friction limit.
        f_long = np.clip(
            cfg.longitudinal_stiffness_N_per_slip * kappa, -limit_long, limit_long
        )
        f_lat = np.clip(
            -cfg.cornering_stiffness_N_per_rad * alpha, -limit_lat, limit_lat
        )

        # Combined demand is then scaled onto the friction ellipse.
        return friction_ellipse(f_long, f_lat, limit_long, limit_lat)

    # ------------------------------------------------------------------
    def saturation(
        self,
        long_vel: ArrayLike,
        lateral_vel: ArrayLike,
        wheel_speed: ArrayLike,
        fz: ArrayLike,
        *,
        lateral_mu_scale: ArrayLike = 1.0,
        long_grip_bias: ArrayLike = 1.0,
    ) -> np.ndarray:
        """Utilisation of the friction ellipse per wheel (1.0 = fully sliding).

        Args:
            long_vel: Contact-patch longitudinal velocity per wheel (m/s).
            lateral_vel: Contact-patch lateral velocity per wheel (m/s).
            wheel_speed: Wheel linear speed ``omega * R`` per wheel (m/s).
            fz: Vertical load per wheel (N).
            lateral_mu_scale: See :meth:`forces`.
            long_grip_bias: See :meth:`forces`.

        Returns:
            Array of utilisation values; ``> 1`` means the tire is sliding and
            a skid mark should be laid down.
        """
        cfg = self.config
        kappa = self.slip_ratio(long_vel, wheel_speed)
        alpha = self.slip_angle(lateral_vel, long_vel)

        mu = self.effective_mu(fz)
        limit_long = np.maximum(
            mu * np.asarray(fz, dtype=float) * np.asarray(long_grip_bias, dtype=float),
            1e-9,
        )
        limit_lat = np.maximum(
            mu * np.asarray(fz, dtype=float) * np.asarray(lateral_mu_scale, dtype=float),
            1e-9,
        )

        f_long = np.clip(
            cfg.longitudinal_stiffness_N_per_slip * kappa, -limit_long, limit_long
        )
        f_lat = np.clip(
            -cfg.cornering_stiffness_N_per_rad * alpha, -limit_lat, limit_lat
        )
        return np.sqrt((f_long / limit_long) ** 2 + (f_lat / limit_lat) ** 2)

    # ------------------------------------------------------------------
    def cornering_stiffness(self, fz: ArrayLike) -> np.ndarray:
        """Axle cornering stiffness ``C_y`` in N/rad (informational)."""
        return np.full_like(np.asarray(fz, dtype=float), self.config.cornering_stiffness_N_per_rad)

    def combined_slip(
        self, slip_ratio_value: ArrayLike, slip_angle_value: ArrayLike
    ) -> np.ndarray:
        """Combined slip magnitude; see :func:`combined_slip`."""
        return combined_slip(slip_ratio_value, slip_angle_value)

    def sliding_mask(
        self,
        slip_ratio_value: ArrayLike,
        slip_angle_value: ArrayLike,
        long_vel: ArrayLike,
    ) -> np.ndarray:
        """Per-wheel boolean mask: is the tire genuinely sliding?

        Thresholds the *combined slip* rather than the friction-ellipse
        utilisation. Utilisation is the wrong quantity for this: at low speed
        the linear slip stiffness produces force levels beyond the friction
        limit for a trivially small velocity difference, so utilisation sits
        pinned at 1.0 and a perfectly coupled wheel gets flagged as sliding.
        Combined slip stays small in that situation and only grows when the
        wheel is really spinning or locked.

        Below :attr:`TireConfig.low_speed_sliding_kmh` the threshold is raised
        to :attr:`TireConfig.catastrophic_slip_threshold`, so slow-speed
        manoeuvring can only be flagged by a genuine lock-up or burnout.

        Args:
            slip_ratio_value: Longitudinal slip ratio per wheel.
            slip_angle_value: Slip angle (rad) per wheel.
            long_vel: Contact-patch longitudinal velocity per wheel (m/s).

        Returns:
            Boolean array, ``True`` where the tire is sliding.
        """
        cfg = self.config
        slip = combined_slip(slip_ratio_value, slip_angle_value)
        speed_kmh = np.abs(np.asarray(long_vel, dtype=float)) * 3.6
        threshold = np.where(
            speed_kmh < cfg.low_speed_sliding_kmh,
            cfg.catastrophic_slip_threshold,
            cfg.sliding_slip_threshold,
        )
        return slip > threshold

    def longitudinal_slip_stiffness(
        self,
        long_vel: ArrayLike,
        wheel_speed: ArrayLike,
        fz: ArrayLike,
        *,
        slip_gain: ArrayLike = 1.0,
        long_grip_bias: ArrayLike = 1.0,
    ) -> np.ndarray:
        """Local slope ``dF_long / d(wheel_speed)`` in N per (m/s).

        This is the derivative the semi-implicit wheel update needs. For the
        clamped linear slip model it is

        .. math::

            \\frac{\\partial F_x}{\\partial (\\omega R)} =
            \\begin{cases}
            \\dfrac{C_x}{\\max(|u|, \\epsilon)} & \\text{unsaturated} \\\\[4pt]
            0 & \\text{at the friction limit}
            \\end{cases}

        Once the force is clamped the derivative is zero and the update reduces
        to plain explicit Euler, which is stable there because the force no
        longer depends on the wheel speed.

        Args:
            long_vel: Contact-patch longitudinal velocity (m/s).
            wheel_speed: Wheel linear speed ``omega * R`` (m/s).
            fz: Vertical load per wheel (N).
            slip_gain: Per-wheel slip multiplier (see :meth:`forces`).
            long_grip_bias: Per-wheel longitudinal grip multiplier.

        Returns:
            Non-negative stiffness array (N per m/s of wheel speed).
        """
        cfg = self.config
        u = np.asarray(long_vel, dtype=float)
        gain = np.asarray(slip_gain, dtype=float)
        denom = np.maximum(np.abs(u), cfg.slip_eps_m_s)
        unsaturated_slope = cfg.longitudinal_stiffness_N_per_slip * gain / denom

        kappa = effective_slip_ratio(
            slip_ratio(u, wheel_speed, cfg.slip_eps_m_s) * gain, cfg.slip_ratio_limit
        )
        raw_force = cfg.longitudinal_stiffness_N_per_slip * kappa
        mu = self.effective_mu(fz)
        limit = np.abs(
            mu * np.asarray(fz, dtype=float) * np.asarray(long_grip_bias, dtype=float)
        )
        return np.where(np.abs(raw_force) < limit, unsaturated_slope, 0.0)
