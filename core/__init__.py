"""Physics core for the 2D electric vehicle simulator.

Modules:
    * :mod:`core.battery_pack` - data-driven electro-thermal battery model.
    * :mod:`core.motor` - traction motor torque envelope and efficiency map.
    * :mod:`core.tire_model` - slip-based tire model with a friction ellipse.
    * :mod:`core.drivetrain` - throttle / regen / friction brake blending.
    * :mod:`core.vehicle` - the fixed-timestep two-track chassis model.
"""

from __future__ import annotations

from .battery_pack import BatteryPack, Chemistry, available_chemistries
from .drivetrain import Drivetrain, DrivetrainCommand, DrivetrainConfig
from .motor import Motor, available_motors
from .tire_model import TireConfig, TireModel
from .vehicle import Controls, Vehicle, VehicleState, create_demo_vehicle

__all__ = [
    "BatteryPack",
    "Chemistry",
    "available_chemistries",
    "Motor",
    "available_motors",
    "TireModel",
    "TireConfig",
    "Drivetrain",
    "DrivetrainConfig",
    "DrivetrainCommand",
    "Vehicle",
    "VehicleState",
    "Controls",
    "create_demo_vehicle",
]
