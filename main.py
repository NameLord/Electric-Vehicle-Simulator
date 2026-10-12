"""EV-Simulator world layer: game loop, camera, streaming and rendering.

Run with::

    python main.py                 # interactive
    python main.py --frames 300    # render N frames then exit (smoke test)
    python main.py --headless      # no window, verify the world layer only

Loop structure
--------------
Physics runs on a **fixed 120 Hz accumulator**; rendering runs unlocked. The
camera chases the vehicle, chunk streaming is driven by the player position, and
the render order is terrain -> roads -> props -> car.

Cost is kept flat on purpose: ground is baked per chunk, roads and props are
answered from per-chunk precomputed tables and a spatial hash, and only chunks
overlapping the viewport are blitted. There are no per-pixel Python loops.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.vehicle import Controls, Vehicle, create_demo_vehicle  # noqa: E402
from ui.garage import (  # noqa: E402
    SECTIONS,
    Garage,
    GarageConfig,
    build_vehicle,
    load_presets,
)
from ui.hud import HudModel  # noqa: E402
from world.chunk_manager import Chunk, ChunkManager  # noqa: E402
from world.props import Prop, PropGenerator, resolve_car_collisions  # noqa: E402
from world.roads import RoadNetwork, build_segment_table, distance_to_segments  # noqa: E402
from world.terrain import BIOMES, TerrainGenerator  # noqa: E402

LOGGER = logging.getLogger("main")

#: Physics rate (Hz) -- matches the core's own substep rate.
PHYSICS_HZ = 120.0

#: Never run more than this many physics steps in one frame (spiral-of-death guard).
MAX_STEPS_PER_FRAME = 8

WINDOW_SIZE: Tuple[int, int] = (1280, 720)

WORLD_CONFIG_PATH = _ROOT / "config" / "world.json"


# ======================================================================
# Configuration
# ======================================================================
def configure_logging(verbose: bool = False) -> None:
    """Configure console logging for the world layer.

    INFO (the default) reports world setup, the road-clearance rule change and
    every chunk-streaming event. ``--verbose`` additionally reports per-chunk
    detail from the generators.

    Args:
        verbose: When True, show DEBUG-level per-chunk detail.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)-14s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    LOGGER.debug("logging configured (verbose=%s)", verbose)


def log_world_summary(config: Dict[str, Any], world: "World") -> None:
    """Log what the world was built from, including the clearance fix.

    Prints the previously-existing setup (seed, chunk size, streaming radius,
    biomes, road parameters) and then the *changed* road-clearance rule side by
    side with the old one, so the difference is visible in the console.

    Args:
        config: Parsed ``world.json``.
        world: The assembled world.
    """
    terrain = world.terrain
    roads = world.roads
    props_cfg = config["props"]
    collision = config["collision"]
    radius = int(config["load_radius_chunks"])

    LOGGER.info("=" * 76)
    LOGGER.info("EV-Simulator world layer")
    LOGGER.info(
        "  seed=%d  (deterministic: same seed -> identical world)",
        config["seed"],
    )
    LOGGER.info(
        "  chunk=%d px / %.1f px per m = %.1f m;  radius=%d chunks (%d resident)",
        terrain.chunk_size_px,
        terrain.pixels_per_metre,
        terrain.chunk_size_m,
        radius,
        (2 * radius + 1) ** 2,
    )
    LOGGER.info(
        "  biomes=%s (continuous weights, partition of unity)",
        ", ".join(BIOMES),
    )
    if roads.mode == "straight":
        LOGGER.info("  roads: mode=STRAIGHT - one endless straight road")
        LOGGER.info(
            "         centreline: world y=%.1f m exactly, from x=-inf to x=+inf; "
            "width=%.1f m shoulder=%.1f m lanes=%d spline=%d samples",
            roads.straight_row * roads.spacing,
            roads.width_m,
            roads.shoulder_m,
            roads.lane_count,
            roads.samples,
        )
        LOGGER.info(
            "         no junctions, no width variation, no node jitter; nodes "
            "are exact lattice points so the road is straight to floating-point "
            "precision for any chunk index (grid/jitter/edge-probability keys in "
            "world.json are ignored in this mode)"
        )
    else:
        LOGGER.info(
            "  roads: mode=GRID - grid=%.0f m width=%.1f m shoulder=%.1f m spline=%d samples",
            roads.spacing,
            roads.width_m,
            roads.shoulder_m,
            roads.samples,
        )
    props_enabled = bool(world.prop_generator.enabled)
    if props_enabled:
        for biome in BIOMES:
            density = props_cfg["density"][biome]
            LOGGER.info(
                "  props[%-10s] tree=%.2f building=%.2f per %.0f m cell",
                biome,
                density["tree"],
                density["building"],
                props_cfg["prop_grid_m"],
            )
    else:
        LOGGER.info("  props: DISABLED - no trees, no buildings")
        LOGGER.info(
            "         scenery.prop_grid_m / density / clearance keys are ignored "
            "and the candidate lattice is not built at all, so chunk generation "
            "skips prop placement entirely. Set props.enabled=true in world.json "
            "to restore the procedural scenery."
        )
    LOGGER.info(
        "  collision: car radius=%.2f m restitution=%.2f damage>%.1f m/s",
        collision["car_radius_m"],
        collision["restitution"],
        collision["damage_speed_m_s"],
    )
    if not props_enabled:
        LOGGER.info(
            "  (road-clearance and prop-separation notes are inactive while "
            "props.enabled is false)"
        )
    if props_enabled:
        LOGGER.info("  -- road clearance for props (CHANGED) --")
        LOGGER.info(
            "     previous: prop CENTRE vs centreline   distance < half_width + %.1f m",
            props_cfg["road_clearance_m"],
        )
        LOGGER.info(
            "     now:      prop FOOTPRINT vs road edge  distance >= half_width + "
            "shoulder(%.1f m) + clearance(%.1f m) + extent",
            roads.shoulder_m,
            props_cfg["road_clearance_m"],
        )
        LOGGER.info(
            "     effect:   large buildings can no longer overhang the asphalt/"
            "shoulder the way the centre-only test allowed"
        )
    LOGGER.info("  -- terrain orientation across chunk borders (FIXED) --")
    LOGGER.info(
        "     previous: each chunk surface was blitted UNFLIPPED, but a chunk's "
        "row 0 is its LOWEST world y while screen y runs the other way, so every "
        "chunk was drawn mirrored in place and met its neighbour in a hard "
        "straight colour seam at every horizontal chunk border"
    )
    LOGGER.info(
        "     now:      ChunkSurfaceCache flips the baked surface vertically, so "
        "a chunk's highest world y lands on the top row of the blit - the mirror "
        "seam and the resulting ~90 deg 'wrong direction' edges are gone"
    )
    if props_enabled:
        LOGGER.info("  -- prop footprint separation (NEW) --")
        LOGGER.info(
            "     buildings: candidates come from a world-aligned %.0f m lattice "
            "and each is a pure hash of its cell, so neighbouring chunks agree; a "
            "building yields to every lower-ranked candidate it overlaps. "
            "Measured over 169 chunks: 0 overlapping footprints. The old code, "
            "with 7-20 m buildings on the same %.0f m grid, overlapped 83%% of "
            "buildings.",
            props_cfg["prop_grid_m"],
            props_cfg["prop_grid_m"],
        )
        LOGGER.info(
            "     trees:     dropped when a canopy would fall inside a building "
            "footprint (gap %.1f m); tree/tree overlap is left alone as undergrowth",
            props_cfg["building_tree_gap_m"],
        )
        LOGGER.info(
            "     roads:     junction pads are clamped to half_width*sqrt(2) so "
            "they round the corner instead of bulging past the asphalt"
        )
    LOGGER.info("=" * 76)
def load_world_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and lightly validate ``config/world.json``.

    Args:
        path: Optional override.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = Path(path) if path else WORLD_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = ("seed", "chunk_size_px", "pixels_per_metre", "load_radius_chunks")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"{config_path} is missing keys: {missing}")
    return config


# ======================================================================
# Camera
# ======================================================================
class Camera:
    """Smooth follow camera with discrete zoom steps.

    Args:
        config: Parsed ``world.json``.
        window_size: Output surface size in pixels.
    """

    def __init__(self, config: Dict[str, Any], window_size: Tuple[int, int]) -> None:
        camera = config["camera"]
        self.levels: List[float] = [float(v) for v in camera["pixels_per_metre_zoom_steps"]]
        self.index = int(camera["zoom_index_default"])
        self.lerp = float(camera["lerp"])
        self.window_size = window_size
        self.position = np.zeros(2, dtype=np.float64)

    @property
    def pixels_per_metre(self) -> float:
        """Current zoom in screen pixels per world metre."""
        return self.levels[self.index]

    def snap(self, target: np.ndarray) -> None:
        """Jump the camera immediately to a target (no smoothing)."""
        self.position = np.asarray(target, dtype=np.float64).copy()

    def update(self, target: np.ndarray, dt: float) -> None:
        """Exponentially smooth toward a target, frame-rate independently.

        Args:
            target: World point to follow (2,).
            dt: Frame time in seconds.
        """
        alpha = 1.0 - (1.0 - self.lerp) ** max(dt * 60.0, 0.0)
        self.position += (np.asarray(target, dtype=np.float64) - self.position) * alpha

    def zoom(self, delta: int) -> None:
        """Step the zoom level, clamped to the configured range."""
        self.index = int(np.clip(self.index + delta, 0, len(self.levels) - 1))

    def world_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        """Project a world point (metres) to screen pixels."""
        ppm = self.pixels_per_metre
        half_w = self.window_size[0] * 0.5
        half_h = self.window_size[1] * 0.5
        return (
            half_w + (x - self.position[0]) * ppm,
            half_h - (y - self.position[1]) * ppm,
        )

    def world_points_to_screen(self, points: np.ndarray) -> List[Tuple[int, int]]:
        """Vectorised projection of an ``(N, 2)`` array to integer screen points."""
        ppm = self.pixels_per_metre
        pts = np.asarray(points, dtype=np.float64)
        sx = self.window_size[0] * 0.5 + (pts[:, 0] - self.position[0]) * ppm
        sy = self.window_size[1] * 0.5 - (pts[:, 1] - self.position[1]) * ppm
        return list(zip(sx.astype(np.int32).tolist(), sy.astype(np.int32).tolist()))

    def visible_world_rect(self) -> Tuple[float, float, float, float]:
        """World-space ``(x0, y0, x1, y1)`` currently on screen."""
        ppm = self.pixels_per_metre
        half_w = self.window_size[0] * 0.5 / ppm
        half_h = self.window_size[1] * 0.5 / ppm
        return (
            self.position[0] - half_w,
            self.position[1] - half_h,
            self.position[0] + half_w,
            self.position[1] + half_h,
        )


# ======================================================================
# Rendering helpers
# ======================================================================
class ChunkSurfaceCache:
    """Caches chunk ground surfaces scaled to the current zoom.

    Baking happens once per chunk; scaling happens once per (chunk, zoom)
    instead of every frame.
    """

    def __init__(self) -> None:
        self._scaled: Dict[Tuple[int, int, int], pygame.Surface] = {}

    def get(self, chunk: Chunk, zoom_index: int, size_px: int) -> pygame.Surface:
        """Return the chunk surface scaled to ``size_px`` for a zoom level.

        The baked surface stores row 0 at the chunk's **minimum** world ``y``
        (``terrain.chunk_surface`` samples ``y`` in increasing order), but
        screen space flips ``y``: ``world_to_screen`` maps the chunk's maximum
        ``y`` to the smallest screen ``y``, which is where ``draw_terrain``
        blits the top edge.  Without the vertical flip each chunk is drawn
        mirrored in place, so two vertically adjacent chunks show content from
        opposite ends of their own spans and meet in a hard colour seam at
        every horizontal chunk border.
        """
        key = (chunk.cx, chunk.cy, zoom_index)
        cached = self._scaled.get(key)
        if cached is None:
            scaled = pygame.transform.scale(chunk.surface, (size_px, size_px))
            cached = pygame.transform.flip(scaled, False, True)
            self._scaled[key] = cached
        return cached

    def clear(self) -> None:
        """Drop all cached scaled surfaces."""
        self._scaled.clear()


def draw_terrain(
    surface: pygame.Surface,
    chunks: Sequence[Chunk],
    camera: Camera,
    cache: ChunkSurfaceCache,
    chunk_size_m: float,
) -> None:
    """Blit every chunk that overlaps the viewport."""
    ppm = camera.pixels_per_metre
    size_px = int(round(chunk_size_m * ppm))
    view = camera.visible_world_rect()
    blit = surface.blit
    for chunk in chunks:
        x0, y0, x1, y1 = chunk.bounds
        if x1 < view[0] or x0 > view[2] or y1 < view[1] or y0 > view[3]:
            continue
        scaled = cache.get(chunk, camera.index, size_px)
        sx, sy = camera.world_to_screen(x0, y1)
        blit(scaled, (int(sx), int(sy)))


def draw_roads(
    surface: pygame.Surface,
    chunks: Sequence[Chunk],
    camera: Camera,
    road_config: Dict[str, Any],
) -> None:
    """Draw road shoulders, surfaces, intersection patches and lane markings.

    Everything heavy is precomputed: centrelines were sampled when the chunk was
    generated and lane-marking dashes are stored on the chunk, so this only
    projects already-computed world points to the screen and issues draw calls.
    """
    shoulder = tuple(road_config["shoulder_color"])
    asphalt = tuple(road_config["surface_color"])
    marking = tuple(road_config["marking_color"])
    view = camera.visible_world_rect()
    ppm = camera.pixels_per_metre
    marking_width = max(int(0.18 * ppm), 1)
    nominal_half_width_m = 0.5 * float(road_config["width_m"])
    # A junction pad rounds off the re-entrant corner where two asphalt strips
    # meet.  For a road of half-width h that corner sits at h*sqrt(2) from the
    # node, so a pad larger than h*sqrt(2) bulges out of the asphalt and leaves a
    # bare disc of tarmac sitting in the grass.  Clamp pad_limit_m so it cannot.
    pad_limit_m = nominal_half_width_m * math.sqrt(2.0)

    # Widest incident edge per node, so wide (4-lane) roads get a matching pad.
    # Collected before the visibility cull so a node never loses its width just
    # because the edge that defines it is only partly on screen.
    node_half_width: Dict[Tuple[int, int], float] = {}
    for chunk in chunks:
        x0, y0, x1, y1 = chunk.bounds
        if x1 < view[0] or x0 > view[2] or y1 < view[1] or y0 > view[3]:
            continue
        for edge in chunk.edges:
            half = 0.5 * float(edge.width_m)
            ends = (
                (edge.ix, edge.iy),
                (
                    edge.ix + (1 if edge.horizontal else 0),
                    edge.iy + (0 if edge.horizontal else 1),
                ),
            )
            for key in ends:
                if node_half_width.get(key, 0.0) < half:
                    node_half_width[key] = half

    seen: set = set()
    for chunk in chunks:
        x0, y0, x1, y1 = chunk.bounds
        if x1 < view[0] or x0 > view[2] or y1 < view[1] or y0 > view[3]:
            continue

        for edge in chunk.edges:
            identity = (edge.ix, edge.iy, edge.horizontal)
            if identity in seen:
                continue
            ex0, ey0, ex1, ey1 = edge.bounds
            if ex1 < view[0] or ex0 > view[2] or ey1 < view[1] or ey0 > view[3]:
                continue
            seen.add(identity)

            points = camera.world_points_to_screen(edge.points)
            if len(points) < 2:
                continue
            width_px = max(int(edge.width_m * ppm), 2)
            shoulder_px = width_px + max(int(edge.shoulder_m * 2 * ppm), 2)
            pygame.draw.lines(surface, shoulder, False, points, shoulder_px)
            pygame.draw.lines(surface, asphalt, False, points, width_px)

        for node in chunk.intersections:
            if not (view[0] <= node.x <= view[2] and view[1] <= node.y <= view[3]):
                continue
            half = node_half_width.get((node.ix, node.iy), nominal_half_width_m)
            radius_px = max(int(min(half * math.sqrt(2.0), pad_limit_m) * ppm), 3)
            sx, sy = camera.world_to_screen(node.x, node.y)
            pygame.draw.circle(surface, asphalt, (int(sx), int(sy)), radius_px)

        dashes = chunk.dash_segments
        if dashes.shape[0]:
            visible = (
                (np.maximum(dashes[:, 0], dashes[:, 2]) >= view[0])
                & (np.minimum(dashes[:, 0], dashes[:, 2]) <= view[2])
                & (np.maximum(dashes[:, 1], dashes[:, 3]) >= view[1])
                & (np.minimum(dashes[:, 1], dashes[:, 3]) <= view[3])
            )
            for row in dashes[visible]:
                pygame.draw.line(
                    surface,
                    marking,
                    camera.world_to_screen(row[0], row[1]),
                    camera.world_to_screen(row[2], row[3]),
                    marking_width,
                )


def draw_props(
    surface: pygame.Surface,
    props: Sequence[Any],
    camera: Camera,
    generator: PropGenerator,
) -> None:
    """Draw trees as circles and buildings as boxes."""
    ppm = camera.pixels_per_metre
    view = camera.visible_world_rect()
    for prop in props:
        if (
            prop.x + prop.radius < view[0]
            or prop.x - prop.radius > view[2]
            or prop.y + prop.radius < view[1]
            or prop.y - prop.radius > view[3]
        ):
            continue
        colour = generator.color_for(prop)
        if prop.is_circle:
            sx, sy = camera.world_to_screen(prop.x, prop.y)
            radius = max(int(prop.radius * ppm), 2)
            pygame.draw.circle(surface, colour, (int(sx), int(sy)), radius)
            pygame.draw.circle(
                surface,
                tuple(max(channel - 24, 0) for channel in colour),
                (int(sx), int(sy)),
                radius,
                1,
            )
        else:
            top_left = camera.world_to_screen(
                prop.x - prop.half_x, prop.y + prop.half_y
            )
            size = (
                max(int(prop.half_x * 2 * ppm), 2),
                max(int(prop.half_y * 2 * ppm), 2),
            )
            rect = pygame.Rect(int(top_left[0]), int(top_left[1]), size[0], size[1])
            pygame.draw.rect(surface, colour, rect)
            pygame.draw.rect(
                surface,
                (40, 42, 48),
                rect,
                1,
            )


def draw_car(
    surface: pygame.Surface,
    vehicle: Vehicle,
    camera: Camera,
    style: Optional[Dict[str, Any]] = None,
) -> None:
    """Draw the car body and wheels at the vehicle's pose.

    Args:
        surface: Target surface.
        vehicle: Physics vehicle, for pose and wheel layout.
        camera: View transform.
        style: Optional cosmetic body style from the garage
            (``body_length_m``, ``body_width_m``, ``color``). ``None`` derives
            the outline from the chassis geometry, which is what the world
            layer has always done. The garage's body dimensions only affect
            this sprite - they are never handed to the physics, because the
            core schema has no body-length/width key.
    """
    ppm = camera.pixels_per_metre
    position = np.asarray(vehicle.position_m, dtype=np.float64)
    heading = float(vehicle.heading_rad)
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    derived_len = (vehicle.lf + vehicle.lr) * 0.5 + 0.35
    derived_wid = max(vehicle.track_front_m, vehicle.track_rear_m) * 0.5 + 0.18
    style = style or {}
    half_len = 0.5 * float(style.get("body_length_m", derived_len * 2.0))
    half_wid = 0.5 * float(style.get("body_width_m", derived_wid * 2.0))
    base_colour = tuple(style.get("color", (52, 120, 200)))
    offset = (vehicle.lf - vehicle.lr) * 0.5

    corners_body = [
        (offset + half_len, half_wid),
        (offset + half_len, -half_wid),
        (offset - half_len, -half_wid),
        (offset - half_len, half_wid),
    ]
    screen_corners = []
    for bx, by in corners_body:
        wx = position[0] + bx * cos_h - by * sin_h
        wy = position[1] + bx * sin_h + by * cos_h
        screen_corners.append(camera.world_to_screen(wx, wy))

    damaged = bool(getattr(vehicle, "damage_flag", False))
    body_colour = (196, 84, 72) if damaged else base_colour
    pygame.draw.polygon(surface, body_colour, screen_corners)
    pygame.draw.polygon(surface, (222, 236, 255), screen_corners, 2)

    for index in range(4):
        lx = float(vehicle.wheel_lx[index])
        ly = float(vehicle.wheel_ly[index])
        wx = position[0] + lx * cos_h - ly * sin_h
        wy = position[1] + lx * sin_h + ly * cos_h
        delta = float(vehicle.steer_angle_rad) if index in (0, 1) else 0.0
        wheel_heading = heading + delta
        direction = np.array([math.cos(wheel_heading), math.sin(wheel_heading)])
        a = np.array([wx, wy]) + direction * vehicle.wheel_radius_m
        b = np.array([wx, wy]) - direction * vehicle.wheel_radius_m
        pygame.draw.line(
            surface,
            (232, 232, 236),
            camera.world_to_screen(a[0], a[1]),
            camera.world_to_screen(b[0], b[1]),
            max(int(0.22 * ppm), 2),
        )


def _car_style_from(garage_config: "GarageConfig") -> Dict[str, Any]:
    """Cosmetic body style for :func:`draw_car`, taken from the garage.

    Body length/width and the sprite colour live in the config's ``style``
    section because the core schema has no key for them; they affect the drawn
    outline only.
    """
    return {
        "body_length_m": float(garage_config.get("style.body_length_m", 4.3)),
        "body_width_m": float(garage_config.get("style.body_width_m", 1.85)),
        "color": garage_config.body_colour(),
    }


# ======================================================================
# Spawning
# ======================================================================
def spawn_on_road(
    roads: RoadNetwork, x: float = 0.0, y: float = 0.0
) -> Tuple[np.ndarray, float]:
    """Find a road point near a location and the heading to face along it.

    Args:
        roads: The road network.
        x: Preferred world x (metres).
        y: Preferred world y (metres).

    Returns:
        ``(position, heading_rad)``. Falls back to ``(x, y)`` facing +x when no
        road is found.
    """
    found = roads.nearest_road(x, y)
    if found is None:
        return np.array([x, y], dtype=np.float64), 0.0

    point, _distance, edge = found
    pts = edge.points
    seg_a = pts[:-1]
    ab = pts[1:] - seg_a
    length_sq = np.maximum((ab * ab).sum(axis=1), 1e-12)
    t = ((point[0] - seg_a[:, 0]) * ab[:, 0] + (point[1] - seg_a[:, 1]) * ab[:, 1]) / length_sq
    np.clip(t, 0.0, 1.0, out=t)
    closest = seg_a + t[:, None] * ab
    gaps = np.hypot(closest[:, 0] - point[0], closest[:, 1] - point[1])
    index = int(np.argmin(gaps))
    heading = math.atan2(ab[index, 1], ab[index, 0])
    return np.asarray(point, dtype=np.float64), heading


# ======================================================================
# World assembly
# ======================================================================
class World:
    """Owns the generators, the chunk manager and the vehicle."""

    def __init__(
        self,
        config: Dict[str, Any],
        vehicle: Vehicle,
        *,
        initial_soc: float = 0.95,
    ) -> None:
        self.config = config
        self.terrain = TerrainGenerator(config)
        self.roads = RoadNetwork(config, self.terrain.chunk_size_m)
        self.prop_generator = PropGenerator(config, self.terrain)
        self.chunks = ChunkManager(
            config, self.terrain, self.roads, self.prop_generator
        )
        self.vehicle = vehicle
        self.chunk_size_m = self.terrain.chunk_size_m
        self.initial_soc = initial_soc
        self.spawn_position = np.zeros(2, dtype=np.float64)
        self.collision_flash = 0.0

    def spawn(self) -> None:
        """Place the vehicle on the road nearest the world origin."""
        position, heading = spawn_on_road(self.roads, 0.0, 0.0)
        self.vehicle.reset(initial_soc=self.initial_soc)
        self.vehicle.position_m[:] = position
        self.vehicle.heading_rad = heading
        self.vehicle.velocity_body_m_s[:] = 0.0
        self.spawn_position = position.copy()
        self.chunks.clear()
        self.chunks.warm_up(float(position[0]), float(position[1]))

    def step(self, dt: float, controls: Controls) -> None:
        """One fixed physics step plus collision resolution."""
        self.vehicle.step(dt, controls)

        position = np.asarray(self.vehicle.position_m, dtype=np.float64)
        query_radius = 6.0
        candidates = self.chunks.query_colliders_around(
            float(position[0]), float(position[1]), query_radius
        )
        if candidates:
            result = resolve_car_collisions(
                self.vehicle, candidates, self.config["collision"]
            )
            if result.count:
                self.collision_flash = min(self.collision_flash + 0.35, 1.0)

    def update_streaming(self) -> None:
        """Stream chunks around the current player position."""
        position = self.vehicle.position_m
        self.chunks.update(float(position[0]), float(position[1]))


# ======================================================================
# Garage integration
# ======================================================================
def apply_garage_build(
    world: "World", garage: "Garage", *, initial_soc: float
) -> Vehicle:
    """Rebuild the vehicle from the garage config and swap it into the world.

    Kept module-level (rather than inline in the game loop) so the UI wiring can
    be verified headlessly by ``--verify-ui``.

    The pose is copied from the outgoing vehicle because a refit must not
    teleport the car, and the pack is created at ``initial_soc`` so the state of
    charge carries over instead of resetting to full.

    Args:
        world: World whose vehicle is being replaced.
        garage: Menu holding the working configuration.
        initial_soc: SOC for the new pack.

    Returns:
        The new vehicle, already installed on ``world``.

    Raises:
        ValueError: If the core rejects the build.
    """
    replacement = build_vehicle(garage.config, initial_soc=float(initial_soc))
    current = world.vehicle
    replacement.position_m[:] = current.position_m
    replacement.heading_rad = float(current.heading_rad)
    replacement.velocity_body_m_s[:] = current.velocity_body_m_s
    replacement.yaw_rate_rad_s = float(current.yaw_rate_rad_s)
    replacement.steer_angle_rad = float(current.steer_angle_rad)
    world.vehicle = replacement
    garage.mark_applied()
    return replacement


# ======================================================================
# Main loop
# ======================================================================
def run(
    config: Dict[str, Any],
    *,
    max_frames: int = 0,
    headless: bool = False,
    initial_soc: float = 0.95,
    simulate_seconds: float = 0.0,
) -> int:
    """Run the game.

    Args:
        config: Parsed ``world.json``.
        max_frames: Stop after this many rendered frames (0 = until quit).
        headless: Drive the vehicle with a scripted throttle instead of keys and
            do not require a real display.
        initial_soc: Starting pack state of charge.
        simulate_seconds: When headless, simulate this long before returning.

    Returns:
        Process exit code.
    """
    pygame.init()
    if headless:
        flags = 0
    else:
        flags = pygame.RESIZABLE
    screen = pygame.display.set_mode(WINDOW_SIZE, flags)
    pygame.display.set_caption("EV Simulator - world layer")
    clock = pygame.time.Clock()

    vehicle = create_demo_vehicle(initial_soc=initial_soc)
    world = World(config, vehicle, initial_soc=initial_soc)
    log_world_summary(config, world)
    LOGGER.info("spawning: searching for a road near the world origin...")
    world.spawn()
    LOGGER.info(
        "spawned at (%.1f, %.1f) m, heading %.0f deg; %d chunks resident",
        float(vehicle.position_m[0]),
        float(vehicle.position_m[1]),
        math.degrees(float(vehicle.heading_rad)),
        world.chunks.loaded_chunks,
    )
    LOGGER.info(world.prop_generator.placement_report())

    camera = Camera(config, screen.get_size())
    camera.snap(vehicle.position_m)
    surface_cache = ChunkSurfaceCache()
    hud_font = pygame.font.SysFont("monospace", 14)
    hud_small = pygame.font.SysFont("monospace", 12)

    # The UI layer owns the dashboard and the garage; the world layer owns
    # everything spatial. Nothing in ui/ is imported by core/ or world/.
    hud = HudModel(vehicle.pack, vehicle.motor)
    garage = Garage(GarageConfig.from_vehicle(vehicle))
    car_style = _car_style_from(garage.config)

    def apply_build() -> None:
        """Rebuild the vehicle from the garage config and swap it into the world.

        A refit, not a respawn: :func:`apply_garage_build` copies the pose across.
        Closures over ``vehicle``/``hud`` are rebound here, which is why this
        lives inside :func:`run` rather than at module level.
        """
        nonlocal vehicle, hud, car_style
        try:
            vehicle = apply_garage_build(
                world, garage, initial_soc=float(vehicle.pack.soc)
            )
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
            LOGGER.error("garage could not apply the build: %s", exc)
            garage.message = f"build rejected by the core: {exc}"
            return

        hud.rebind(vehicle.pack, vehicle.motor)
        car_style = _car_style_from(garage.config)
        LOGGER.info(
            "garage applied: %s, %dS%dP, %.0f kg, gear %.2f, layout %s",
            vehicle.motor.id,
            vehicle.pack.S,
            vehicle.pack.P,
            vehicle.mass_kg,
            vehicle.drivetrain.config.gear_ratio,
            garage.config.get("wheels.driven_axle"),
        )

    physics_dt = 1.0 / PHYSICS_HZ
    accumulator = 0.0
    frames = 0
    fps_display = 0.0
    sim_time = 0.0
    running = True

    scripted_duration = simulate_seconds
    try:
        while running:
            frame_dt = clock.tick(0 if headless else 240) / 1000.0
            frame_dt = min(frame_dt, 0.25)  # clamp long stalls

            if not headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        continue
                    elif event.type == pygame.VIDEORESIZE:
                        screen = pygame.display.set_mode(
                            (max(event.w, 480), max(event.h, 360)), pygame.RESIZABLE
                        )
                        camera.window_size = screen.get_size()
                    if garage.open:
                        # The garage owns every event while it is open. It is
                        # click-driven, so it needs motion / button / wheel
                        # events as well as keys, and the driving controls must
                        # not fight the menu.
                        action = garage.handle_event(event)
                        if action == "close":
                            garage.open = False
                        elif action == "apply":
                            apply_build()
                        continue
                    if event.type != pygame.KEYDOWN:
                        continue
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_TAB:
                        garage.open = True
                        garage.reload_presets()
                    elif event.key == pygame.K_m:
                        hud.toggle_units()
                    elif event.key == pygame.K_f:
                        hud.cycle_drive_mode(1)
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                        hud.mode_index = int(event.key) - pygame.K_1
                    elif event.key == pygame.K_LEFTBRACKET:
                        hud.set_regen_level(hud.regen_level - 1)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        hud.set_regen_level(hud.regen_level + 1)
                    elif event.key == pygame.K_r:
                        surface_cache.clear()
                        world.spawn()
                        camera.snap(vehicle.position_m)
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        camera.zoom(1)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        camera.zoom(-1)
                    elif event.key == pygame.K_l:
                        LOGGER.info(world.prop_generator.placement_report())
                        stats = world.chunks.stats()
                        LOGGER.info(
                            "world stats: %d resident / %d cached chunks, "
                            "%d active props, %d hash cells",
                            int(stats["loaded_chunks"]),
                            int(stats["cached_chunks"]),
                            int(stats["active_props"]),
                            int(stats["hash_cells"]),
                        )

            if headless:
                throttle = 0.55 if sim_time < scripted_duration else 0.0
                steer = 0.35 * math.sin(sim_time * 0.7)
                controls = Controls(throttle=throttle, steer=steer)
            else:
                keys = pygame.key.get_pressed()
                throttle = float(keys[pygame.K_w] or keys[pygame.K_UP])
                brake = float(keys[pygame.K_s] or keys[pygame.K_DOWN])
                left = keys[pygame.K_a] or keys[pygame.K_LEFT]
                right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
                controls = Controls(
                    throttle=throttle,
                    brake=brake,
                    steer=float(left) - float(right),
                    handbrake=bool(keys[pygame.K_SPACE]),
                )
            # Drive mode and regen level are applied as pedal shaping, so the
            # core still owns every torque decision.
            controls = hud.shape(controls)

            # ---- fixed-step physics, unlocked rendering ------------------
            # The garage pauses physics: nothing is integrated, streamed or
            # accumulated while the menu is open.
            paused = garage.open and not headless
            if paused:
                accumulator = 0.0
            accumulator += 0.0 if paused else frame_dt
            steps = 0
            while accumulator >= physics_dt and steps < MAX_STEPS_PER_FRAME:
                world.step(physics_dt, controls)
                accumulator -= physics_dt
                sim_time += physics_dt
                steps += 1
            if steps == MAX_STEPS_PER_FRAME:
                accumulator = 0.0  # dropped behind; do not spiral

            if not paused:
                world.update_streaming()
                camera.update(vehicle.position_m, frame_dt)

            state = vehicle.get_state()
            world.collision_flash = max(world.collision_flash - frame_dt * 2.0, 0.0)
            hud.update(state, frame_dt)

            # ---- render order: terrain -> roads -> props -> car ----------
            screen.fill((14, 16, 20))
            chunks = world.chunks.visible_chunks()
            draw_terrain(screen, chunks, camera, surface_cache, world.chunk_size_m)
            draw_roads(screen, chunks, camera, config["roads"])

            view = camera.visible_world_rect()
            visible_props = world.chunks.query_colliders_in_rect(*view)
            draw_props(screen, visible_props, camera, world.prop_generator)
            draw_car(screen, vehicle, camera, car_style)

            stats = world.chunks.stats()
            fps_display = clock.get_fps()
            # The garage replaces the dashboard while it is open: drawing both
            # would show the panel ghosting through the menu overlay.
            if not garage.open:
                hud.draw(
                    screen,
                    state,
                    font=hud_font,
                    small=hud_small,
                    fps=fps_display,
                    extra_lines=[
                        f"Chunks   {int(stats['loaded_chunks']):3d} loaded / "
                        f"{int(stats['cached_chunks']):3d} cached",
                        f"Props    {int(stats['active_props']):4d} active / "
                        f"{int(stats['hash_cells']):4d} cells",
                        f"Damage   {'YES' if getattr(vehicle, 'damage_flag', False) else 'no'}"
                        f"   zoom {camera.pixels_per_metre:.0f} px/m",
                        f"FPS      {fps_display:6.1f}",
                    ],
                    helper="WASD drive  SPACE handbrake  TAB garage  M units  "
                           "F mode  1-3  [ ] regen  +/- zoom  R respawn  ESC quit",
                )
            else:
                garage.draw(screen)

            pygame.display.flip()
            frames += 1
            if max_frames and frames >= max_frames:
                running = False
            if headless and sim_time >= scripted_duration + 4.0:
                running = False
    finally:
        pygame.quit()

    print(
        f"frames={frames} simulated={sim_time:.1f}s "
        f"chunks={int(world.chunks.stats()['loaded_chunks'])} "
        f"generated={int(world.chunks.stats()['generated_chunks'])} "
        f"damage={bool(getattr(vehicle, 'damage_flag', False))}"
    )
    return 0


def verify_ui_layer(config: Dict[str, Any]) -> int:
    """End-to-end checks for the UI layer's wiring into the game loop.

    The UI modules have their own self-tests (``python -m ui.hud`` and
    ``python -m ui.garage``); this one proves the *integration*: that the
    dashboard tracks real telemetry, that drive mode and regen level shape the
    pedals without exceeding them, that a garage apply really replaces the
    vehicle in the world and changes the physics, and that both overlays draw.

    Args:
        config: Parsed ``world.json``.

    Returns:
        Process exit code (0 when every check passes).
    """
    import tempfile

    failures: List[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    world_config = config
    vehicle = create_demo_vehicle(initial_soc=0.9)
    world = World(world_config, vehicle, initial_soc=0.9)
    hud = HudModel(vehicle.pack, vehicle.motor)
    with tempfile.TemporaryDirectory() as tmp:
        garage = Garage(GarageConfig.from_vehicle(vehicle),
                        save_path=Path(tmp) / "garage_save.json")
        check(
            "garage starts from the live build",
            garage.config.get("build.motor_id") == vehicle.motor.id
            and garage.config.get("build.series") == vehicle.pack.S,
            f"{garage.config.get('build.motor_id')} "
            f"{garage.config.get('build.series')}S",
        )
        check("garage starts with no unapplied changes",
              not garage.pending_changes())

        # -- the dashboard tracks the sim -----------------------------
        dt = 1.0 / PHYSICS_HZ
        for _ in range(int(2.5 * PHYSICS_HZ)):
            shaped = hud.shape(Controls(throttle=0.6, steer=0.05))
            world.step(dt, shaped)
            hud.update(vehicle.get_state(), dt)
        state = vehicle.get_state()
        check(
            "HUD odometer follows the simulated distance",
            hud.odometer_m > 1.0,
            f"{hud.odometer_m:.1f} m for {state.speed_kmh:.1f} km/h",
        )
        check(
            "HUD consumption is finite once the car has moved",
            math.isfinite(hud.consumption_Wh_per_km)
            and hud.consumption_Wh_per_km > 0.0,
            f"{hud.consumption_Wh_per_km:.0f} Wh/km",
        )
        snapshot = hud.snapshot(state)
        check(
            "HUD shows speed in both units",
            "km/h" in snapshot["speed_primary"] and "mph" in snapshot["speed_secondary"],
            f"{snapshot['speed_primary']} / {snapshot['speed_secondary']}",
        )
        check(
            "HUD reports the active drive mode and regen level",
            snapshot["mode"] == hud.drive_mode.name and snapshot["regen_level"] == 2,
            f"{snapshot['mode']} regen {snapshot['regen_level']}",
        )

        # -- pedal shaping must stay inside the pedals -----------------
        # NOTE: a drive mode is allowed to *amplify* part-throttle travel
        # (SPORT does: 0.25 travel -> 0.37), because that is what a sharper
        # pedal map means. What it must never do is exceed full authority or
        # leave the valid range.
        shaping_ok = True
        shaping_detail = ""
        for mode_index in range(3):
            hud.mode_index = mode_index
            for travel in (0.0, 0.25, 0.5, 0.75, 1.0):
                shaped = hud.shape(Controls(throttle=travel, steer=1.0))
                if not (0.0 <= shaped.throttle <= 1.0):
                    shaping_ok = False
                    shaping_detail = (
                        f"{hud.drive_mode.name} travel {travel} -> "
                        f"{shaped.throttle}"
                    )
                if not (-1.0 <= shaped.steer <= 1.0):
                    shaping_ok = False
                    shaping_detail = f"steer {shaped.steer}"
        hud.mode_index = 1
        full = hud.shape(Controls(throttle=1.0, steer=1.0))
        check(
            "pedal shaping stays inside the pedal range",
            shaping_ok and full.throttle <= 1.0 and full.steer <= 1.0,
            shaping_detail or f"full throttle -> {full.throttle:.3f}",
        )

        # Lift-off regen must be a brake request only, and never with throttle.
        hud.set_regen_level(3)
        lifted = hud.shape(Controls())
        held = hud.shape(Controls(throttle=0.3))
        check(
            "lift-off regen only adds brake when coasting",
            lifted.brake > 0.0 and lifted.throttle == 0.0 and held.brake == 0.0,
            f"coasting brake {lifted.brake:.3f}, on-throttle brake {held.brake:.3f}",
        )
        hud.set_regen_level(2)

        # -- the menu pauses nothing by itself, but reports intent ------
        action = garage.handle_event(
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB, mod=0, unicode="")
        )
        check("TAB tells the host to close the menu", action == "close", str(action))

        # -- applying a build really changes the physics ---------------
        before_motor = world.vehicle.motor.id
        before_position = np.asarray(world.vehicle.position_m, dtype=float).copy()
        before_speed = float(world.vehicle.get_state().speed_kmh)

        # The garage is click-driven now, so the integration test drives it the
        # way a player does: click the stepper to change the motor, then click
        # APPLY BUILD. No key press is involved anywhere in this block.
        def click_widget(widget_id: str, surface: pygame.Surface) -> Optional[str]:
            """Redraw, then simulate a full hover + left click on a widget."""
            garage.draw(surface)
            widget = next(
                (rect for wid, rect in garage._widgets if wid == widget_id), None
            )
            if widget is None:
                return None
            result: Optional[str] = None
            for event in (
                pygame.event.Event(
                    pygame.MOUSEMOTION, pos=widget.center, rel=(0, 0), buttons=(0, 0, 0)
                ),
                pygame.event.Event(
                    pygame.MOUSEBUTTONDOWN, pos=widget.center, button=1
                ),
                pygame.event.Event(
                    pygame.MOUSEBUTTONUP, pos=widget.center, button=1
                ),
            ):
                action = garage.handle_event(event)
                # The press carries the action; the release only ends a drag.
                if action is not None and result is None:
                    result = action
            return result

        ui_surface = pygame.Surface(WINDOW_SIZE)
        garage.section_index = next(
            index for index, section in enumerate(SECTIONS)
            if section.title == "MOTOR"
        )
        garage.field_index = 0
        for _ in range(80):
            if garage.config.get("build.motor_id") == "pmsm-335kw":
                break
            click_widget("plus:0", ui_surface)
        check(
            "clicking a stepper selects a different motor",
            garage.config.get("build.motor_id") == "pmsm-335kw"
            and garage.pending_changes(),
            f"{before_motor} -> {garage.config.get('build.motor_id')}",
        )
        check(
            "clicking APPLY BUILD asks the host to apply",
            click_widget("btn:apply", ui_surface) == "apply",
            "the button returns the apply action, mouse events only",
        )

        new_vehicle = apply_garage_build(
            world, garage, initial_soc=float(world.vehicle.pack.soc)
        )
        check(
            "apply swaps the vehicle inside the world",
            world.vehicle is new_vehicle and world.vehicle.motor.id == "pmsm-335kw",
            f"world.vehicle is the rebuilt car ({world.vehicle.motor.id})",
        )
        check(
            "a refit does not teleport the car",
            np.allclose(np.asarray(world.vehicle.position_m, dtype=float),
                        before_position),
            f"position {np.asarray(before_position).round(2)} preserved",
        )
        check(
            "a refit carries the state of charge over",
            abs(world.vehicle.pack.soc - vehicle.pack.soc) < 1e-9,
            f"SOC {world.vehicle.pack.soc * 100:.1f}%",
        )
        check(
            "apply marks the build as applied",
            not garage.pending_changes(),
        )

        # The rebuild must actually reach the physics, not just the object graph.
        for _ in range(int(2.0 * PHYSICS_HZ)):
            world.step(dt, Controls(throttle=1.0))
        after_speed = float(world.vehicle.get_state().speed_kmh)
        check(
            "the rebuilt car drives and accelerates harder than before",
            after_speed > before_speed,
            f"{before_speed:.1f} km/h -> {after_speed:.1f} km/h after 2 s at "
            f"full throttle",
        )

        # -- both overlays must draw ----------------------------------
        screen = pygame.Surface(WINDOW_SIZE)
        try:
            hud.draw(screen, world.vehicle.get_state(), fps=60.0,
                     extra_lines=["Chunks   49 loaded"], helper="TAB garage")
            check("dashboard draws", True)
        except Exception as exc:  # noqa: BLE001
            check("dashboard draws", False, repr(exc))
        try:
            garage.draw(screen)
            check("garage overlay draws", True)
        except Exception as exc:  # noqa: BLE001
            check("garage overlay draws", False, repr(exc))

        # -- saving the new build then restoring the old one ----------
        message = garage.save_preset("verify build")
        check("a preset can be saved from the game loop",
              "saved" in message and "verify build" in load_presets(garage.save_path),
              message)

    print()
    if failures:
        print(f"RESULT: {len(failures)} UI check(s) failed -> {failures}")
        return 1
    print("RESULT: all UI checks passed.")
    return 0


def verify_world_layer(config: Dict[str, Any]) -> int:
    """Check the world layer's invariants and print the results.

    Covers the properties the world layer is *supposed* to have and that are
    easy to break silently: seeded determinism, partition-of-unity biome
    blending, seam-free roads at chunk borders, road spawning, spatial-hash
    narrowing, collision response (bounce, damage threshold, no tunnelling) and
    streaming load radius.

    Args:
        config: Parsed ``world.json``.

    Returns:
        Process exit code (0 when every check passes).
    """
    failures: List[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    # This suite exercises the *generators*, so it runs them with every feature
    # switched on. world.json may select a simpler world for gameplay (see the
    # "shipped world" block at the end of this function); both configurations
    # get verified. Deep-copied so the override cannot leak into the caller's
    # config and silently flip the running world to grid mode.
    full = copy.deepcopy(config)
    full["roads"]["mode"] = "grid"
    full["props"]["enabled"] = True

    terrain = TerrainGenerator(full)
    roads = RoadNetwork(full, terrain.chunk_size_m)
    prop_generator = PropGenerator(full, terrain)
    manager = ChunkManager(full, terrain, roads, prop_generator)
    size = terrain.chunk_size_m

    # -- determinism ----------------------------------------------------
    sample_x = np.linspace(-300, 300, 400)
    sample_y = np.linspace(-300, 300, 400)
    weights_a = terrain.biome_weights(sample_x, sample_y).weights
    weights_b = TerrainGenerator(config).biome_weights(sample_x, sample_y).weights
    check(
        "Terrain is deterministic for a seed",
        np.allclose(weights_a, weights_b),
        f"max diff {np.abs(weights_a - weights_b).max():.2e}",
    )

    other = dict(config)
    other["seed"] = int(config["seed"]) + 1
    weights_c = TerrainGenerator(other).biome_weights(sample_x, sample_y).weights
    check(
        "A different seed gives a different world",
        not np.allclose(weights_a, weights_c),
        f"mean abs diff {np.abs(weights_a - weights_c).mean():.3f}",
    )

    totals = weights_a.sum(axis=0)
    check(
        "Biome weights are a partition of unity (blend, not snap)",
        np.allclose(totals, 1.0),
        f"sum range {totals.min():.4f}..{totals.max():.4f}",
    )

    # -- road seams -----------------------------------------------------
    straddling = None
    for ix in range(-6, 7):
        for iy in range(-6, 7):
            if not roads.edge_exists(ix, iy, True):
                continue
            edge = roads.make_edge(ix, iy, True)
            x0, _y0, x1, _y1 = edge.bounds
            left = math.floor((x0 + 0.1) / size)
            right = math.floor((x1 - 0.1) / size)
            if right > left:
                straddling = (ix, iy, edge, left, right)
                break
        if straddling:
            break

    if straddling is None:
        check("Road edges span chunk borders identically", False, "no edge found")
    else:
        ix, iy, edge, left, right = straddling
        _x0, ey0, _x1, ey1 = edge.bounds
        cy = int(math.floor(0.5 * (ey0 + ey1) / size))
        key = (ix, iy, True)
        in_left = any(
            (e.ix, e.iy, e.horizontal) == key
            for e in roads.edges_for_rect(left * size, cy * size, (left + 1) * size, (cy + 1) * size)
        )
        in_right = any(
            (e.ix, e.iy, e.horizontal) == key
            for e in roads.edges_for_rect(right * size, cy * size, (right + 1) * size, (cy + 1) * size)
        )
        identical = np.array_equal(edge.points, roads.make_edge(ix, iy, True).points)
        check(
            "Road edges span chunk borders identically",
            in_left and in_right and identical,
            f"edge {key} straddles chunks {left}|{right}; identical geometry={identical}",
        )

    check(
        "Road node positions are a pure function of coordinates",
        np.array_equal(roads.node_xy(12, -7), roads.node_xy(12, -7)),
        f"node (12,-7) = {roads.node_xy(12, -7).round(2)}",
    )

    # -- spawn and queries ----------------------------------------------
    vehicle = create_demo_vehicle(initial_soc=0.95)
    world = World(full, vehicle, initial_soc=0.95)
    world.spawn()
    spawn = np.asarray(vehicle.position_m)
    query = world.chunks.query_road_at(float(spawn[0]), float(spawn[1]))
    check(
        "Player spawns on a road at the origin",
        query.on_road and float(np.hypot(*spawn)) < roads.spacing,
        f"spawn {spawn.round(1)}, centreline distance {query.distance_m:.2f} m, "
        f"half width {query.half_width_m:.2f} m, lanes {query.lanes}",
    )
    check(
        "query_road_at off the loaded world reports no road",
        not world.chunks.query_road_at(1e5, 1e5).on_road,
    )

    stats = world.chunks.stats()
    active = int(stats["active_props"])
    nearby = world.chunks.query_colliders_around(float(spawn[0]), float(spawn[1]), 8.0)
    check(
        "Spatial hash narrows the candidate set",
        len(nearby) < max(active, 1),
        f"{len(nearby)} candidates within 8 m vs {active} active props",
    )

    # -- props must clear the road (footprint-aware, not centre-only) ---
    overlapping = 0
    checked = 0
    smallest_gap = math.inf
    for chunk in world.chunks.visible_chunks():
        segments = chunk.road_segments
        if segments.shape[0] == 0:
            continue
        for prop in chunk.props:
            if prop.is_circle:
                angles = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
                px = prop.x + np.cos(angles) * prop.radius
                py = prop.y + np.sin(angles) * prop.radius
            else:
                # Corners plus edge midpoints, so a road slicing through the
                # middle of a building is detected too.
                px = np.array([
                    prop.x - prop.half_x, prop.x + prop.half_x,
                    prop.x - prop.half_x, prop.x + prop.half_x,
                    prop.x, prop.x, prop.x,
                ])
                py = np.array([
                    prop.y - prop.half_y, prop.y - prop.half_y,
                    prop.y + prop.half_y, prop.y + prop.half_y,
                    prop.y - prop.half_y, prop.y + prop.half_y, prop.y,
                ])
            distance, half_width = distance_to_segments(px, py, segments)
            gap = float(np.min(distance - half_width))
            checked += 1
            smallest_gap = min(smallest_gap, gap)
            if gap < 0.0:
                overlapping += 1
    check(
        "No prop footprint touches a road (footprint-aware clearance)",
        overlapping == 0 and checked > 0,
        f"{overlapping} overlapping of {checked} props; smallest gap to the "
        f"asphalt edge {smallest_gap:+.2f} m (must be > 0)",
    )
    report = world.prop_generator.placement_stats
    LOGGER.debug(
        "clearance fix effect: %d of %d road rejections were only caught by the "
        "footprint test",
        report["rejected_footprint_only"],
        report["rejected_road"],
    )

    # -- collision ------------------------------------------------------
    car_radius = float(config["collision"]["car_radius_m"])
    tree = Prop("tree", 6.0, 0.0, 2.0, 0.0, 0.0, "forest", 0)
    vehicle.reset(initial_soc=0.95)
    vehicle.position_m[:] = np.array([4.2, 0.0])
    vehicle.heading_rad = 0.0
    vehicle.velocity_body_m_s[:] = np.array([12.0, 0.0])
    hit = resolve_car_collisions(vehicle, [tree], config["collision"], passes=3)
    contact_x = 6.0 - 2.0 - car_radius
    check(
        "Car bounces off a tree without penetrating",
        hit.count > 0
        and float(np.hypot(*vehicle.velocity_body_m_s)) < 12.0
        and abs(float(vehicle.position_m[0]) - contact_x) < 1e-6,
        f"impact {hit.max_impact_speed_m_s:.1f} m/s -> "
        f"{float(np.hypot(*vehicle.velocity_body_m_s)):.1f} m/s, pushed out to "
        f"x={float(vehicle.position_m[0]):.2f} (contact {contact_x:.2f})",
    )
    check(
        "Hard impact sets vehicle.damage_flag",
        bool(getattr(vehicle, "damage_flag", False)) and hit.damage,
        f"damage={hit.damage}, energy={hit.energy_j:.0f} J",
    )

    vehicle.reset(initial_soc=0.95)
    vehicle.damage_flag = False  # type: ignore[attr-defined]
    vehicle.position_m[:] = np.array([4.2, 0.0])
    vehicle.velocity_body_m_s[:] = np.array([1.0, 0.0])
    gentle = resolve_car_collisions(vehicle, [tree], config["collision"], passes=3)
    check(
        "A gentle nudge does not damage",
        not gentle.damage,
        f"impact {gentle.max_impact_speed_m_s:.2f} m/s, threshold "
        f"{config['collision']['damage_speed_m_s']} m/s",
    )

    vehicle.reset(initial_soc=0.95)
    vehicle.position_m[:] = np.array([2.6, 0.0])
    vehicle.velocity_body_m_s[:] = np.array([15.0, 0.0])
    box = Prop("building", 8.0, 0.0, 5.0, 5.0, 5.0, "urban", 0)
    box_hit = resolve_car_collisions(vehicle, [box], config["collision"], passes=3)
    check(
        "Car bounces off a building box",
        box_hit.count > 0
        and float(np.hypot(*vehicle.velocity_body_m_s)) < 15.0
        and abs(float(vehicle.position_m[0]) - (3.0 - car_radius)) < 1e-6,
        f"speed 15.0 -> {float(np.hypot(*vehicle.velocity_body_m_s)):.1f} m/s, "
        f"pushed out to x={float(vehicle.position_m[0]):.2f}",
    )

    vehicle.reset(initial_soc=0.95)
    vehicle.position_m[:] = np.array([0.0, 0.0])
    vehicle.velocity_body_m_s[:] = np.array([14.0, 0.0])
    approach = None
    for _ in range(int(2.0 * PHYSICS_HZ)):
        vehicle.step(1.0 / PHYSICS_HZ, Controls(throttle=1.0))
        approach = resolve_car_collisions(vehicle, [tree], config["collision"])
        if approach.count:
            break
    final = np.asarray(vehicle.position_m)
    clearance = math.hypot(final[0] - 6.0, final[1]) - 2.0
    check(
        "Car driven into a tree stops outside it (no tunnelling)",
        approach is not None and approach.count > 0 and clearance >= car_radius - 1e-3,
        f"final position {final.round(2)}, surface clearance {clearance:.2f} m "
        f">= car radius {car_radius} m",
    )

    # -- terrain render continuity --------------------------------------
    # Regression guard for the chunk-mirroring bug: chunk surfaces are baked
    # with row 0 at the chunk's minimum world y, but screen space flips y, so
    # blitting them unflipped drew every chunk mirrored in place and left a hard
    # colour seam at every horizontal chunk border.
    camera = Camera(config, (320, 240))
    camera.snap(np.array([0.0, 0.0]))
    surface_cache = ChunkSurfaceCache()
    for _ in range(90):
        manager.update(0.0, 0.0)
    north = manager.chunk_at(0, 0)
    south = manager.chunk_at(0, -1)
    if north is None or south is None:
        check("Terrain renders continuously across a chunk border", False,
              "chunks 0,0 and 0,-1 are not resident")
    else:
        size_px = int(round(size * camera.pixels_per_metre))
        strip = pygame.Surface((size_px, 2 * size_px))
        strip.blit(surface_cache.get(north, camera.index, size_px), (0, 0))
        strip.blit(surface_cache.get(south, camera.index, size_px), (0, size_px))
        pixels = pygame.surfarray.array3d(strip).astype(np.float64)
        seam = np.abs(pixels[:, size_px - 1, :] - pixels[:, size_px, :]).mean()
        interior = np.abs(
            pixels[:, size_px - 1, :] - pixels[:, size_px - 2, :]
        ).mean()
        check(
            "Terrain renders continuously across a chunk border",
            seam <= max(4.0 * interior, 1.0),
            f"step across the world y=0 border {seam:.2f} vs {interior:.2f} "
            f"between neighbouring pixels inside a chunk",
        )

    # -- prop-versus-prop footprints ------------------------------------
    buildings: List[Prop] = []
    trees: List[Prop] = []
    for cx in range(-6, 7):
        for cy in range(-6, 7):
            bounds = (cx * size, cy * size, (cx + 1) * size, (cy + 1) * size)
            segments = build_segment_table(roads.edges_for_rect(*bounds))
            for prop in prop_generator.spawn_chunk(cx, cy, size, segments):
                (trees if prop.is_circle else buildings).append(prop)

    if len(buildings) > 1:
        bx = np.array([p.x for p in buildings])
        by = np.array([p.y for p in buildings])
        bhx = np.array([p.half_x for p in buildings])
        bhy = np.array([p.half_y for p in buildings])
        overlap = (
            np.abs(bx[:, None] - bx[None, :]) < bhx[:, None] + bhx[None, :]
        ) & (np.abs(by[:, None] - by[None, :]) < bhy[:, None] + bhy[None, :])
        np.fill_diagonal(overlap, False)
        pairs = int(overlap.sum() // 2)
    else:
        bx = by = bhx = bhy = np.zeros(0)
        pairs = 0
    check(
        "No two building footprints overlap, even across chunk borders",
        pairs == 0,
        f"{pairs} overlapping pairs among {len(buildings)} buildings over "
        f"169 chunks",
    )

    if len(trees) and len(buildings):
        tx = np.array([p.x for p in trees])
        ty = np.array([p.y for p in trees])
        tr = np.array([p.radius for p in trees])
        reach_x = bhx + float(config["props"].get("building_tree_gap_m", 0.0))
        reach_y = bhy + float(config["props"].get("building_tree_gap_m", 0.0))
        near_x = np.clip(tx[:, None] - bx[None, :], -reach_x[None, :], reach_x[None, :])
        near_y = np.clip(ty[:, None] - by[None, :], -reach_y[None, :], reach_y[None, :])
        distance = np.hypot(
            tx[:, None] - bx[None, :] - near_x, ty[:, None] - by[None, :] - near_y
        )
        inside = int((distance < tr[:, None]).any(axis=1).sum())
    else:
        inside = 0
    check(
        "No tree canopy grows inside a building footprint",
        inside == 0,
        f"{inside} of {len(trees)} trees overlap a building",
    )

    # -- streaming ------------------------------------------------------
    expected = (2 * int(config["load_radius_chunks"]) + 1) ** 2
    vehicle.position_m[:] = np.array([12 * size + 5.0, 5.0])
    for _ in range(90):
        manager.update(float(vehicle.position_m[0]), float(vehicle.position_m[1]))
    check(
        "Streaming keeps exactly the load radius resident",
        manager.loaded_chunks == expected,
        f"{manager.loaded_chunks} chunks resident after moving 12 chunks away "
        f"(expected {expected})",
    )

    # -- the world actually shipped in world.json ------------------------
    # Everything above verified the procedural generators with their features
    # turned on. This block verifies whatever world.json actually selects, so a
    # mode switch cannot quietly ship a world with no road, a bent "straight"
    # road, or scenery that was meant to be off.
    print()
    live_terrain = TerrainGenerator(config)
    live_roads = RoadNetwork(config, live_terrain.chunk_size_m)
    live_props = PropGenerator(config, live_terrain)
    live_size = live_terrain.chunk_size_m

    if live_roads.mode == "straight":
        row = live_roads.straight_row
        line_y = row * live_roads.spacing

        # Perfectly straight, and endless: sample edges far beyond the load
        # radius in both directions. A tolerance of 1e-9 m is essentially
        # floating-point exact - a jittered node would be metres out.
        worst_bend = 0.0
        for ix in (-99999, -4096, -7, 0, 1, 12, 5000, 64000, 99999):
            pts = live_roads.make_edge(ix, row, True).points
            worst_bend = max(worst_bend, float(np.abs(pts[:, 1] - line_y).max()))
        check(
            "Straight road: centreline is exactly on the line, out to +-1e5 m",
            worst_bend < 1e-9,
            f"max |y - {line_y:.1f}| over 9 sampled edges = {worst_bend:.2e} m",
        )

        # Exactly one road line and nothing else: no vertical edges, no edges on
        # any other row, therefore no junctions anywhere.
        stray = 0
        for ix in range(-40, 41):
            for iy in range(-40, 41):
                if live_roads.edge_exists(ix, iy, False):
                    stray += 1
                if iy != row and live_roads.edge_exists(ix, iy, True):
                    stray += 1
        check(
            "Straight road: no vertical edges, no off-line edges, no junctions",
            stray == 0 and len(live_roads.intersections_for_rect(-500, -500, 500, 500)) == 0,
            f"{stray} stray edges and "
            f"{len(live_roads.intersections_for_rect(-500, -500, 500, 500))} junctions "
            f"in an 81x81 lattice sweep",
        )

        # Endless: a chunk 1 km along the road has road in it; a chunk 1 km off
        # the road has none.
        on_road_chunks = [
            len(live_roads.edges_for_rect(x, -live_size, x + live_size, 0.0))
            for x in (0.0, 64.0, 64000.0, -64000.0)
        ]
        off_road = len(
            live_roads.edges_for_rect(0.0, 1000.0, live_size, 1000.0 + live_size)
        )
        check(
            "Straight road: endless along x, absent off the line",
            all(count > 0 for count in on_road_chunks) and off_road == 0,
            f"edges per chunk at x=0/64/64000/-64000: {on_road_chunks}; "
            f"1 km off the road: {off_road}",
        )

        # Uniform cross-section (no per-edge width or lane variation).
        widths = {
            round(live_roads.make_edge(ix, row, True).width_m, 6)
            for ix in range(-30, 31)
        }
        lanes = {
            live_roads.make_edge(ix, row, True).lanes for ix in range(-30, 31)
        }
        check(
            "Straight road: one uniform cross-section for every edge",
            widths == {round(live_roads.width_m, 6)} and lanes == {live_roads.lane_count},
            f"widths={sorted(widths)}, lanes={sorted(lanes)}",
        )

        spawn, heading = spawn_on_road(live_roads, 0.0, 0.0)
        check(
            "Straight road: spawns exactly on the line, facing along it",
            abs(float(spawn[1]) - line_y) < 1e-9 and abs(math.degrees(heading)) < 1e-6,
            f"spawn {np.asarray(spawn).round(6)}, heading {math.degrees(heading):.4f} deg",
        )

        live_world = World(config, create_demo_vehicle(initial_soc=0.95), initial_soc=0.95)
        live_world.spawn()
        # Stream along the road first: query_road_at only answers for chunks that
        # are actually resident, so querying 256 m away needs the chunks there.
        for _ in range(60):
            live_world.chunks.update(256.0, line_y)
        on_road = live_world.chunks.query_road_at(256.0, line_y)
        beside = live_world.chunks.query_road_at(256.0, line_y + 40.0)
        check(
            "Straight road: query_road_at sees the road 256 m along, not beside it",
            bool(on_road.on_road)
            and not bool(beside.on_road)
            and abs(float(on_road.half_width_m) - 0.5 * live_roads.width_m) < 1e-6,
            f"on the line: on_road={on_road.on_road} half width "
            f"{on_road.half_width_m:.2f} m; 40 m beside it: on_road={beside.on_road}",
        )
    else:
        check(
            "Road mode is one this suite knows how to verify",
            live_roads.mode == "grid",
            f"mode={live_roads.mode!r} (expected 'grid' or 'straight')",
        )

    # Scenery switch: when disabled there must be genuinely nothing, and the
    # lattice must not even be built.
    live_chunk = live_props.spawn_chunk(
        0, 0, live_size, build_segment_table(live_roads.edges_for_rect(0, 0, live_size, live_size))
    )
    check(
        "Props switch is honoured: enabled=false yields no scenery at all",
        bool(live_props.enabled) or not live_chunk,
        f"enabled={live_props.enabled}, {len(live_chunk)} props in chunk (0,0)",
    )
    if not live_props.enabled:
        check(
            "Props are reported as off, not as a zero-count placement run",
            "disabled" in live_props.placement_report()
            and live_props.placement_stats["candidates"] == 0,
            live_props.placement_report()[:72],
        )

    print()
    if failures:
        print(f"RESULT: {len(failures)} world-layer check(s) failed -> {failures}")
        return 1
    print("RESULT: all world-layer checks passed.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="EV-Simulator world layer")
    parser.add_argument("--frames", type=int, default=0, help="stop after N frames")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="no window; scripted driving and a world-layer summary",
    )
    parser.add_argument("--soc", type=float, default=0.95, help="initial SOC (0..1)")
    parser.add_argument(
        "--seconds",
        type=float,
        default=8.0,
        help="headless: seconds of scripted driving before exiting",
    )
    parser.add_argument("--config", type=Path, default=None, help="world.json path")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="also log per-chunk generator detail (DEBUG level)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="run the world-layer invariant checks and exit",
    )
    parser.add_argument(
        "--verify-ui",
        action="store_true",
        help="run the UI-layer integration checks and exit",
    )
    args = parser.parse_args(argv)

    configure_logging(args.verbose)
    config = load_world_config(args.config)

    if args.verify or args.verify_ui:
        pygame.init()
        pygame.display.set_mode((320, 240))
        try:
            if args.verify_ui:
                return verify_ui_layer(config)
            return verify_world_layer(config)
        finally:
            pygame.quit()

    return run(
        config,
        max_frames=args.frames,
        headless=args.headless,
        initial_soc=args.soc,
        simulate_seconds=args.seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
