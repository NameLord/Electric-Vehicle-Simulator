"""Chunk streaming: load/unload, caching, and the world queries the game needs.

The manager owns three things:

* the **chunk cache**, keyed by ``(seed, cx, cy)`` so regenerating is skipped and
  the world is reproducible;
* the **streaming radius**, loading chunks within ``load_radius_chunks`` of the
  player and unloading the rest (releasing their props from the spatial hash);
* the **queries** the rest of the game asks for -- what is the road under this
  point, and which colliders are inside this rectangle. Both are answered from
  per-chunk precomputed tables, never by scanning the world.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame

from .props import Prop, PropGenerator, SpatialHash
from .roads import (
    RoadEdge,
    RoadNode,
    RoadNetwork,
    build_segment_table,
    distance_to_segments,
)
from .terrain import TerrainGenerator

__all__ = ["Chunk", "RoadQuery", "ChunkManager"]

LOGGER = logging.getLogger(__name__)

Vector = np.ndarray


@dataclass
class Chunk:
    """A streamed chunk and everything generated for it.

    Attributes:
        cx: Chunk x index.
        cy: Chunk y index.
        bounds: ``(x0, y0, x1, y1)`` in metres.
        surface: Baked ground surface.
        edges: Road edges touching this chunk.
        intersections: Intersection nodes near this chunk.
        road_segments: Flattened ``(N, 5)`` road table for fast distance queries.
        props: Props spawned in this chunk.
    """

    cx: int
    cy: int
    bounds: Tuple[float, float, float, float]
    surface: pygame.Surface
    edges: List[RoadEdge]
    intersections: List[RoadNode]
    road_segments: np.ndarray
    props: List[Prop]
    dash_segments: np.ndarray

    @property
    def centre(self) -> Tuple[float, float]:
        """Chunk centre in metres."""
        x0, y0, x1, y1 = self.bounds
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


@dataclass
class RoadQuery:
    """Result of :meth:`ChunkManager.query_road_at`.

    Attributes:
        on_road: True when the point is within the road surface (lane area).
        distance_m: Distance from the point to the nearest road centreline.
        half_width_m: Half width of that road.
        lanes: Lane count of the nearest road.
        edge: The nearest :class:`~world.roads.RoadEdge`, if any.
        heading_rad: Direction of the nearest road, for spawn alignment.
    """

    on_road: bool = False
    distance_m: float = math.inf
    half_width_m: float = 0.0
    lanes: int = 0
    edge: Optional[RoadEdge] = None
    heading_rad: float = 0.0


class ChunkManager:
    """Streams chunks around the player and answers world queries.

    Args:
        config: Parsed ``config/world.json``.
        terrain: Terrain generator.
        roads: Road network.
        prop_generator: Prop generator.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        terrain: TerrainGenerator,
        roads: RoadNetwork,
        prop_generator: PropGenerator,
    ) -> None:
        self.config = config
        self.terrain = terrain
        self.roads = roads
        self.prop_generator = prop_generator
        self.seed = int(config["seed"])
        self.chunk_size_m = terrain.chunk_size_m
        self.radius = int(config["load_radius_chunks"])
        self.max_loads_per_update = int(config.get("max_loads_per_update", 2))
        self.spatial_hash = SpatialHash(
            float(config["collision"]["spatial_hash_cell_m"])
        )
        self._cache: Dict[Tuple[int, int, int], Chunk] = {}
        self._active: Dict[Tuple[int, int], Chunk] = {}
        self._last_player_chunk: Optional[Tuple[int, int]] = None
        self.loaded_chunks: int = 0
        self.generated_chunks: int = 0
        self.stream_events: int = 0
        LOGGER.info(
            "chunk streaming ready: seed=%d, radius=%d chunks (%d resident), "
            "max %d new chunk(s) per update",
            self.seed,
            self.radius,
            (2 * self.radius + 1) ** 2,
            self.max_loads_per_update,
        )

    # ------------------------------------------------------------------
    # Keying helpers
    # ------------------------------------------------------------------
    def chunk_coords(self, x: float, y: float) -> Tuple[int, int]:
        """Chunk indices containing a world point."""
        size = self.chunk_size_m
        return int(math.floor(x / size)), int(math.floor(y / size))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _generate(self, cx: int, cy: int) -> Chunk:
        """Generate (or fetch from cache) the chunk at ``(cx, cy)``."""
        key = (self.seed, cx, cy)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        size = self.chunk_size_m
        bounds = (cx * size, cy * size, (cx + 1) * size, (cy + 1) * size)

        surface = self.terrain.chunk_surface(cx, cy)
        edges = self.roads.edges_for_rect(*bounds)
        intersections = self.roads.intersections_for_rect(*bounds)
        segments = build_segment_table(edges)
        props = self.prop_generator.spawn_chunk(cx, cy, size, segments)

        # Lane markings are static, so bake them once here instead of walking
        # every spline on every frame.
        dash_blocks: List[np.ndarray] = []
        for edge in edges:
            dashes = self.roads.center_dashes(edge)
            if dashes:
                block = np.array(
                    [
                        [a[0], a[1], b[0], b[1]]
                        for a, b in dashes
                    ],
                    dtype=np.float64,
                )
                dash_blocks.append(block)
        dash_segments = (
            np.concatenate(dash_blocks, axis=0)
            if dash_blocks
            else np.zeros((0, 4), dtype=np.float64)
        )

        chunk = Chunk(
            cx=cx,
            cy=cy,
            bounds=bounds,
            surface=surface,
            edges=edges,
            intersections=intersections,
            road_segments=segments,
            props=props,
            dash_segments=dash_segments,
        )
        self._cache[key] = chunk
        self.generated_chunks += 1
        LOGGER.debug(
            "generated chunk (%d,%d): %d road edges, %d intersections, "
            "%d props, %d dash segments",
            cx,
            cy,
            len(edges),
            len(intersections),
            len(props),
            dash_segments.shape[0],
        )
        return chunk

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------
    def update(
        self, x: float, y: float, max_loads: Optional[int] = None
    ) -> Tuple[int, int]:
        """Load chunks near ``(x, y)`` and unload those out of range.

        Loading is budgeted (``max_loads_per_update``) so crossing a chunk
        border does not stall a frame generating the whole new column; the
        nearest missing chunks are always filled first.

        Args:
            x: Player x in metres.
            y: Player y in metres.
            max_loads: Override the per-call generation budget. ``None`` uses
                the configured value.

        Returns:
            ``(loaded, unloaded)`` counts for this call.
        """
        centre = self.chunk_coords(x, y)
        budget = self.max_loads_per_update if max_loads is None else int(max_loads)
        cached_centre = self._last_player_chunk

        loaded = 0
        unloaded = 0

        if cached_centre != centre:
            wanted = {
                (centre[0] + dx, centre[1] + dy)
                for dx in range(-self.radius, self.radius + 1)
                for dy in range(-self.radius, self.radius + 1)
            }
            for key in list(self._active.keys()):
                if key not in wanted:
                    chunk = self._active.pop(key)
                    self.spatial_hash.remove_many(chunk.props)
                    unloaded += 1
            self._last_player_chunk = centre
        else:
            wanted = {
                (centre[0] + dx, centre[1] + dy)
                for dx in range(-self.radius, self.radius + 1)
                for dy in range(-self.radius, self.radius + 1)
            }

        missing = [key for key in wanted if key not in self._active]
        if missing and budget > 0:
            if len(missing) > budget:
                missing.sort(
                    key=lambda key: (key[0] - centre[0]) ** 2 + (key[1] - centre[1]) ** 2
                )
                missing = missing[:budget]
            for key in missing:
                chunk = self._generate(key[0], key[1])
                self._active[key] = chunk
                self.spatial_hash.insert_many(chunk.props)
                loaded += 1

        if loaded or unloaded:
            LOGGER.info(
                "streamed around chunk (%d,%d): +%d new / -%d unloaded "
                "(%d resident, %d cached, %d generated total)",
                centre[0],
                centre[1],
                loaded,
                unloaded,
                len(self._active),
                len(self._cache),
                self.generated_chunks,
            )

        self.loaded_chunks = len(self._active)
        self.stream_events += 1
        return loaded, unloaded

    def warm_up(self, x: float, y: float) -> None:
        """Force a synchronous load around a point (used before the first frame)."""
        self._last_player_chunk = None
        for _ in range(64):
            loaded, _unloaded = self.update(x, y, max_loads=16)
            if loaded == 0:
                break

    def visible_chunks(self) -> List[Chunk]:
        """All currently loaded chunks, nearest centre first."""
        return list(self._active.values())

    def chunk_at(self, x: float, y: float) -> Optional[Chunk]:
        """Loaded chunk containing a point, or ``None`` when not resident."""
        return self._active.get(self.chunk_coords(x, y))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def query_road_at(self, x: float, y: float) -> RoadQuery:
        """What road (if any) is under a world point.

        Uses the containing chunk's precomputed segment table. Because
        :meth:`world.roads.RoadNetwork.edges_for_rect` keeps a generous margin,
        every road within roughly half a lattice cell of the chunk is present,
        so the containing chunk alone is enough.

        Args:
            x: Query x metres.
            y: Query y metres.

        Returns:
            A :class:`RoadQuery`.
        """
        chunk = self.chunk_at(x, y)
        if chunk is None or chunk.road_segments.shape[0] == 0:
            return RoadQuery()

        distance, half_width = distance_to_segments(
            np.array([x]), np.array([y]), chunk.road_segments
        )
        best_distance = float(distance[0])
        best_half = float(half_width[0])

        edge: Optional[RoadEdge] = None
        heading = 0.0
        best_gap = math.inf
        for candidate in chunk.edges:
            pts = candidate.points
            seg_a = pts[:-1]
            ab = pts[1:] - seg_a
            length_sq = np.maximum((ab * ab).sum(axis=1), 1e-12)
            t = ((x - seg_a[:, 0]) * ab[:, 0] + (y - seg_a[:, 1]) * ab[:, 1]) / length_sq
            np.clip(t, 0.0, 1.0, out=t)
            closest = seg_a + t[:, None] * ab
            gaps = np.hypot(closest[:, 0] - x, closest[:, 1] - y)
            index = int(np.argmin(gaps))
            if gaps[index] < best_gap:
                best_gap = float(gaps[index])
                edge = candidate
                heading = math.atan2(ab[index, 1], ab[index, 0])

        return RoadQuery(
            on_road=best_distance <= best_half,
            distance_m=best_distance,
            half_width_m=best_half,
            lanes=int(edge.lanes) if edge is not None else 0,
            edge=edge,
            heading_rad=heading,
        )

    def query_colliders_in_rect(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> List[Prop]:
        """Props whose spatial-hash cell overlaps a rectangle.

        Args:
            x0: Minimum x (metres).
            y0: Minimum y (metres).
            x1: Maximum x (metres).
            y1: Maximum y (metres).

        Returns:
            Broadphase candidate props (narrowphase still needed).
        """
        return self.spatial_hash.query_rect(x0, y0, x1, y1)

    def query_colliders_around(self, x: float, y: float, radius_m: float) -> List[Prop]:
        """Convenience wrapper: colliders within ``radius_m`` of a point."""
        return self.query_colliders_in_rect(
            x - radius_m, y - radius_m, x + radius_m, y + radius_m
        )

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, float]:
        """Diagnostics for the HUD."""
        prop_count = sum(len(chunk.props) for chunk in self._active.values())
        return {
            "loaded_chunks": float(self.loaded_chunks),
            "cached_chunks": float(len(self._cache)),
            "generated_chunks": float(self.generated_chunks),
            "active_props": float(prop_count),
            "hash_cells": float(self.spatial_hash.cell_count),
        }

    def clear(self) -> None:
        """Drop every loaded chunk and the cache."""
        for chunk in self._active.values():
            self.spatial_hash.remove_many(chunk.props)
        self._active.clear()
        self._cache.clear()
        self._last_player_chunk = None
        self.loaded_chunks = 0
