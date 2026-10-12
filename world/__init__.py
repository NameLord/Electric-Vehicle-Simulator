"""World layer: procedural terrain, roads, props and chunk streaming.

Nothing here touches ``core/`` -- the physics package is consumed read-only
through the public ``Vehicle`` / ``Controls`` API.
"""

from __future__ import annotations

from .chunk_manager import Chunk, ChunkManager, RoadQuery
from .props import ImpactResult, Prop, PropGenerator, SpatialHash, resolve_car_collisions
from .roads import RoadEdge, RoadNetwork, RoadNode, build_segment_table, distance_to_segments
from .terrain import BIOMES, TerrainGenerator, fractal_noise, value_noise

__all__ = [
    "BIOMES",
    "TerrainGenerator",
    "value_noise",
    "fractal_noise",
    "RoadNetwork",
    "RoadEdge",
    "RoadNode",
    "build_segment_table",
    "distance_to_segments",
    "Prop",
    "PropGenerator",
    "SpatialHash",
    "resolve_car_collisions",
    "ImpactResult",
    "ChunkManager",
    "Chunk",
    "RoadQuery",
]
