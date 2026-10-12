"""Props (trees, buildings), their deterministic placement, and collision.

Placement
---------
Props are dropped on a jittered grid inside each chunk, then filtered by biome
density and finally rejected if they would sit on a road. All of that is
vectorised over the candidate array; only the surviving handful of props is
touched by Python-level loops.

Queries
-------
A :class:`SpatialHash` buckets props into fixed world-space cells so
``query_rect`` only visits the cells overlapping the query. Collision never
iterates the full prop set, which is what keeps the cost flat as the world
grows.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

from .roads import distance_to_segments
from .terrain import BIOMES, TerrainGenerator, _hash01

__all__ = [
    "Prop",
    "PropGenerator",
    "SpatialHash",
    "resolve_car_collisions",
    "ImpactResult",
]

LOGGER = logging.getLogger(__name__)

Vector = np.ndarray


@dataclass(frozen=True)
class Prop:
    """A static world object with a simple collision shape.

    Attributes:
        kind: ``"tree"`` (circle) or ``"building"`` (axis-aligned box).
        x: Centre x in metres.
        y: Centre y in metres.
        radius: Bounding radius, used for broadphase and drawing.
        half_x: AABB half width (``0`` for trees).
        half_y: AABB half height (``0`` for trees).
        biome: Dominant biome that spawned it (drives colour).
        variant: Small deterministic index for visual variety.
    """

    kind: str
    x: float
    y: float
    radius: float
    half_x: float
    half_y: float
    biome: str
    variant: int

    @property
    def is_circle(self) -> bool:
        """True when the collider is a circle rather than a box."""
        return self.kind == "tree"


@dataclass
class ImpactResult:
    """Outcome of a collision-resolution pass.

    Attributes:
        count: Number of contacts resolved.
        energy_j: Total dissipated impact energy (J).
        damage: True when any impact exceeded the damage threshold.
        max_impact_speed_m_s: Largest normal closing speed seen (m/s).
    """

    count: int = 0
    energy_j: float = 0.0
    damage: bool = False
    max_impact_speed_m_s: float = 0.0


class PropGenerator:
    """Deterministic per-chunk prop placement.

    Args:
        config: Parsed ``config/world.json``.
        terrain: Terrain generator (provides biome lookups).
    """

    def __init__(self, config: Dict[str, Any], terrain: TerrainGenerator) -> None:
        self.config = config
        self.terrain = terrain
        self.seed = int(config["seed"])
        props = config["props"]
        #: ``False`` disables scenery entirely: no trees, no buildings. The
        #: placement rules below are then never exercised, which is exactly what
        #: a plain open road wants - and it makes chunk generation cheaper,
        #: since the candidate lattice is not even built.
        self.enabled = bool(props.get("enabled", True))
        self.grid_m = float(props["prop_grid_m"])
        self.max_per_chunk = int(props["max_props_per_chunk"])
        # Minimum gap between a prop's FOOTPRINT and the edge of the drawn road
        # (asphalt + shoulder), not between its centre and the centreline.
        self.road_margin_m = float(props["road_clearance_m"])
        self.shoulder_m = float(config["roads"]["shoulder_m"])
        # Buildings must not interpenetrate. The prop grid is far coarser than a
        # building footprint (9 m grid vs 7-20 m buildings), so without this rule
        # ~80% of buildings overlapped a neighbour and merged into grey blobs.
        self.building_gap_m = float(props.get("building_min_gap_m", 0.0))
        self.building_tree_gap_m = float(props.get("building_tree_gap_m", 0.0))
        self._placement_stats: Dict[str, int] = {
            "candidates": 0,
            "spawned": 0,
            "rejected_road": 0,
            "rejected_footprint_only": 0,
            "rejected_building_overlap": 0,
            "rejected_tree_in_building": 0,
        }
        self.tree_min = float(props["tree_radius_m"][0])
        self.tree_max = float(props["tree_radius_m"][1])
        self.building_min = float(props["building_radius_m"][0])
        self.building_max = float(props["building_radius_m"][1])
        # How far past the chunk edge the candidate lattice is grown so a chunk
        # can see - and avoid - the buildings its neighbours place near the
        # shared border. Two buildings can only collide if their centres are
        # closer than the sum of their half-extents, so the margin must reach the
        # widest possible pair.
        self.building_aspect_range = (0.6, 1.5)
        worst_half_x = 0.5 * self.building_max * self.building_aspect_range[1]
        worst_half_y = 0.5 * self.building_max / self.building_aspect_range[0]
        self.placement_margin_m = (
            2.0 * max(worst_half_x, worst_half_y) + self.building_gap_m
        )
        self.tree_colors = [
            tuple(props["tree_color"]),
            tuple(props["tree_color_alt"]),
        ]
        self.building_colors = [
            tuple(props["building_color"]),
            tuple(props["building_color_alt"]),
        ]
        # (4,) density tables indexed by biome index.
        self.tree_density = np.array(
            [props["density"][name]["tree"] for name in BIOMES], dtype=np.float64
        )
        self.building_density = np.array(
            [props["density"][name]["building"] for name in BIOMES],
            dtype=np.float64,
        )
        if self.enabled:
            LOGGER.info(
                "prop road clearance: footprint-aware, distance >= half_width + "
                "shoulder(%.1f m) + clearance(%.1f m) + extent",
                self.shoulder_m,
                self.road_margin_m,
            )
        else:
            LOGGER.info(
                "prop generator DISABLED (props.enabled=false): no trees, no "
                "buildings, no candidate lattice"
            )

    @property
    def placement_stats(self) -> Dict[str, int]:
        """Counters describing prop placement since construction."""
        return dict(self._placement_stats)

    def placement_report(self) -> str:
        """Human-readable summary of prop placement and the clearance fix."""
        if not self.enabled:
            return (
                "prop placement: disabled (props.enabled=false in world.json) - "
                "no trees and no buildings are generated"
            )
        stats = self._placement_stats
        return (
            f"prop placement: {stats['spawned']} spawned of "
            f"{stats['candidates']} candidates; {stats['rejected_road']} rejected by "
            f"road clearance, of which {stats['rejected_footprint_only']} were only "
            f"caught by the footprint test (the old centre-only test would have "
            f"kept them and let them overhang the road); "
            f"{stats['rejected_building_overlap']} buildings dropped because their "
            f"footprint overlapped an already-placed building; "
            f"{stats['rejected_tree_in_building']} trees dropped because their "
            f"canopy fell inside a building footprint"
        )

    # ------------------------------------------------------------------
    def spawn_chunk(
        self,
        cx: int,
        cy: int,
        chunk_size_m: float,
        road_segments: np.ndarray,
    ) -> List[Prop]:
        """Generate the props for one chunk.

        Args:
            cx: Chunk x index.
            cy: Chunk y index.
            chunk_size_m: Chunk edge length in metres.
            road_segments: Road segment table for the chunk (used for road
                clearance), from :func:`world.roads.build_segment_table`.

        Returns:
            List of :class:`Prop`, deterministically derived from the seed and
            the chunk coordinates. Empty when ``props.enabled`` is ``False``.
        """
        if not self.enabled:
            return []
        x0 = cx * chunk_size_m
        y0 = cy * chunk_size_m
        x1 = x0 + chunk_size_m
        y1 = y0 + chunk_size_m
        step = self.grid_m
        # Candidates live on a WORLD-aligned lattice of integer cell indices, and
        # every one of their properties (jitter, biome roll, kind, size, aspect,
        # variant) is a pure hash of (seed, cell index).  Nothing is drawn from a
        # per-chunk random stream, so a given cell has the SAME position and the
        # same kind in every chunk that can see it.  That is what makes the
        # "do not interpenetrate" rule below agree across chunk borders.
        #
        # The lattice is grown by `placement_margin_m` past the chunk edges so a
        # chunk can see - and dodge - the footprints its neighbours place near
        # the shared border; a chunk then only emits props whose centre is
        # inside itself, so ownership stays unambiguous.
        margin = self.placement_margin_m
        ix_range = np.arange(
            int(np.floor((x0 - margin) / step)),
            int(np.ceil((x1 + margin) / step)) + 1,
            dtype=np.int64,
        )
        iy_range = np.arange(
            int(np.floor((y0 - margin) / step)),
            int(np.ceil((y1 + margin) / step)) + 1,
            dtype=np.int64,
        )
        if ix_range.size == 0 or iy_range.size == 0:
            return []
        grid_ix, grid_iy = np.meshgrid(ix_range, iy_range)
        ix = grid_ix.ravel()
        iy = grid_iy.ravel()
        count = ix.size

        px = ix.astype(np.float64) * step + (_hash01(self.seed, ix, iy, 401) - 0.5) * step
        py = iy.astype(np.float64) * step + (_hash01(self.seed, ix, iy, 409) - 0.5) * step

        # Ownership: a chunk only emits props whose centre lies inside it, so the
        # lattice margin cannot duplicate props along the borders.
        owned = (px >= x0) & (px < x1) & (py >= y0) & (py < y1)

        biome = self.terrain.dominant_biome(px, py).astype(np.int64)
        tree_prob = self.tree_density[biome]
        building_prob = self.building_density[biome]
        tree_mask = _hash01(self.seed, ix, iy, 419) < tree_prob
        building_mask = (~tree_mask) & (_hash01(self.seed, ix, iy, 421) < building_prob)

        # Draw geometry for EVERY candidate up front, so the road-clearance test
        # below knows each prop's real footprint size.
        tree_radii = self.tree_min + _hash01(self.seed, ix, iy, 431) * (
            self.tree_max - self.tree_min
        )
        building_sizes = self.building_min + _hash01(self.seed, ix, iy, 433) * (
            self.building_max - self.building_min
        )
        aspect_lo, aspect_hi = self.building_aspect_range
        building_aspects = aspect_lo + _hash01(self.seed, ix, iy, 439) * (
            aspect_hi - aspect_lo
        )
        half_x = building_sizes * building_aspects * 0.5
        half_y = building_sizes / np.maximum(building_aspects, 1e-6) * 0.5
        building_radii = np.hypot(half_x, half_y)
        extent = np.where(tree_mask, tree_radii, building_radii)

        # Push everything off the roads.
        #
        # OLD rule (buggy): `distance < half_width + clearance`, where
        # `distance` is measured to the prop's *centre*. A building can be tens
        # of metres across, so its centre can clear the road while its box still
        # overhangs the asphalt by several metres.
        #
        # NEW rule: measure the prop's *footprint* against the edge of the drawn
        # road (asphalt half-width plus the shoulder we actually render):
        #     distance >= half_width + shoulder + clearance + extent
        would_be = tree_mask | building_mask
        rejected = 0
        rejected_footprint_only = 0
        if road_segments.shape[0] > 0 and would_be.any():
            distance, half_width = distance_to_segments(px, py, road_segments)
            required = half_width + self.shoulder_m + self.road_margin_m + extent
            keep = distance >= required
            centre_only_keep = distance >= (half_width + self.road_margin_m)
            # Only count rejections for candidates this chunk actually owns.
            rejected = int((would_be & ~keep & owned).sum())
            rejected_footprint_only = int(
                (would_be & centre_only_keep & ~keep & owned).sum()
            )
            tree_mask &= keep
            building_mask &= keep

        self._placement_stats["candidates"] += int(owned.sum())
        self._placement_stats["rejected_road"] += rejected
        self._placement_stats["rejected_footprint_only"] += rejected_footprint_only

        # Props must not interpenetrate each other either.
        rejected_building_overlap, rejected_tree_in_building = (
            self._resolve_footprint_conflicts(
                ix, iy, px, py, half_x, half_y, tree_radii, tree_mask, building_mask
            )
        )
        self._placement_stats["rejected_building_overlap"] += rejected_building_overlap
        self._placement_stats["rejected_tree_in_building"] += rejected_tree_in_building

        # Ownership: a chunk only emits props whose centre lies inside it, so the
        # lattice margin above cannot duplicate props along the borders.
        tree_mask &= owned
        building_mask &= owned

        LOGGER.debug(
            "chunk (%d,%d): %d road segments, %d trees + %d buildings kept, "
            "%d rejected on road clearance (%d only by the footprint test), "
            "%d buildings dropped to stop them interpenetrating, "
            "%d trees dropped from building footprints",
            cx,
            cy,
            road_segments.shape[0],
            int(tree_mask.sum()),
            int(building_mask.sum()),
            rejected,
            rejected_footprint_only,
            rejected_building_overlap,
            rejected_tree_in_building,
        )

        tree_idx = np.flatnonzero(tree_mask)
        building_idx = np.flatnonzero(building_mask)

        # Deterministic cap: keep the lowest candidate indices so the same
        # chunk always yields the same props regardless of arrival order.
        budget = self.max_per_chunk
        if tree_idx.size + building_idx.size > budget:
            all_idx = np.concatenate([tree_idx, building_idx])
            kinds = np.concatenate(
                [
                    np.zeros(tree_idx.size, dtype=bool),
                    np.ones(building_idx.size, dtype=bool),
                ]
            )
            order = np.argsort(all_idx, kind="stable")
            all_idx = all_idx[order][:budget]
            kinds = kinds[order][:budget]
            tree_idx = all_idx[~kinds]
            building_idx = all_idx[kinds]

        # Visual variety index, also a pure function of the lattice cell so it
        # does not shift when the set of surviving props changes.
        tree_variants = (_hash01(self.seed, ix[tree_idx], iy[tree_idx], 443) * 2).astype(
            np.int64
        )
        building_variants = (
            _hash01(self.seed, ix[building_idx], iy[building_idx], 449) * 2
        ).astype(np.int64)

        props: List[Prop] = []
        if tree_idx.size:
            for slot, index in enumerate(tree_idx):
                props.append(
                    Prop(
                        kind="tree",
                        x=float(px[index]),
                        y=float(py[index]),
                        radius=float(tree_radii[index]),
                        half_x=0.0,
                        half_y=0.0,
                        biome=BIOMES[int(biome[index])],
                        variant=int(tree_variants[slot]),
                    )
                )
        if building_idx.size:
            for slot, index in enumerate(building_idx):
                props.append(
                    Prop(
                        kind="building",
                        x=float(px[index]),
                        y=float(py[index]),
                        radius=float(building_radii[index]),
                        half_x=float(half_x[index]),
                        half_y=float(half_y[index]),
                        biome=BIOMES[int(biome[index])],
                        variant=int(building_variants[slot]),
                    )
                )
        self._placement_stats["spawned"] += len(props)
        return props

    def color_for(self, prop: Prop) -> Tuple[int, int, int]:
        """Return the render colour for a prop."""
        palette = self.tree_colors if prop.is_circle else self.building_colors
        return palette[prop.variant % len(palette)]

    def _resolve_footprint_conflicts(
        self,
        ix: np.ndarray,
        iy: np.ndarray,
        px: np.ndarray,
        py: np.ndarray,
        half_x: np.ndarray,
        half_y: np.ndarray,
        tree_radii: np.ndarray,
        tree_mask: np.ndarray,
        building_mask: np.ndarray,
    ) -> Tuple[int, int]:
        """Reject props whose footprints collide with another prop's footprint.

        Candidates are drawn independently on a grid that is coarser than a
        building (9 m grid vs 7-20 m buildings), so without this pass most
        buildings overlapped a neighbour and merged into a single grey blob.

        Conflict is resolved by a fixed priority: a candidate is dropped when it
        clashes with a *lower-ranking* candidate that survived.  Ranking uses the
        lattice cell ``(iy, ix)``, not array order, and every candidate's
        geometry is a hash of its cell - so a chunk that owns a building makes
        the same accept/reject decision as the neighbour that can merely see it.
        Without that, the two chunks disagreed and buildings still overlapped
        across borders.

        Buildings are resolved first; trees that would grow inside a surviving
        building footprint are then dropped.

        Args:
            ix: Candidate lattice x indices, ``(count,)``.
            iy: Candidate lattice y indices, ``(count,)``.
            px: Candidate x positions in metres, ``(count,)``.
            py: Candidate y positions in metres, ``(count,)``.
            half_x: Building half extent along x, ``(count,)``.
            half_y: Building half extent along y, ``(count,)``.
            tree_radii: Tree canopy radii, ``(count,)``.
            tree_mask: Boolean mask of tree candidates; updated in place.
            building_mask: Boolean mask of building candidates; updated in place.

        Returns:
            ``(buildings_rejected, trees_rejected)``.
        """
        rejected_buildings = 0
        building_idx = np.flatnonzero(building_mask)
        if building_idx.size > 1:
            gap = self.building_gap_m
            # Ascending rank: sort by iy, then ix.
            ranked = building_idx[np.lexsort((ix[building_idx], iy[building_idx]))]
            rank_pos = np.arange(ranked.size)
            order_half_x = half_x[ranked]
            order_half_y = half_y[ranked]
            order_px = px[ranked]
            order_py = py[ranked]
            clash = (
                np.abs(order_px[:, None] - order_px[None, :])
                < order_half_x[:, None] + order_half_x[None, :] + gap
            ) & (
                np.abs(order_py[:, None] - order_py[None, :])
                < order_half_y[:, None] + order_half_y[None, :] + gap
            )
            # A building yields to every LOWER-ranked candidate it overlaps,
            # whether or not that candidate itself survives. Depending only on
            # lower-ranked neighbours keeps the decision inside the lattice
            # margin, so two neighbouring chunks always agree; it also guarantees
            # no two survivors can overlap, since the higher-ranked half of any
            # clashing pair is always dropped.
            blocked = (clash & (rank_pos[None, :] < rank_pos[:, None])).any(axis=1)
            survivors = ranked[~blocked]
            rejected_buildings = int(blocked.sum())
            building_mask[:] = False
            building_mask[survivors] = True

        rejected_trees = 0
        tree_idx = np.flatnonzero(tree_mask)
        building_idx = np.flatnonzero(building_mask)
        if tree_idx.size and building_idx.size:
            tx = px[tree_idx]
            ty = py[tree_idx]
            bx = px[building_idx]
            by = py[building_idx]
            reach_x = half_x[building_idx] + self.building_tree_gap_m
            reach_y = half_y[building_idx] + self.building_tree_gap_m
            nearest_x = np.clip(
                tx[:, None] - bx[None, :], -reach_x[None, :], reach_x[None, :]
            )
            nearest_y = np.clip(
                ty[:, None] - by[None, :], -reach_y[None, :], reach_y[None, :]
            )
            distance = np.hypot(
                tx[:, None] - bx[None, :] - nearest_x,
                ty[:, None] - by[None, :] - nearest_y,
            )
            inside = (distance < tree_radii[tree_idx][:, None]).any(axis=1)
            rejected_trees = int(inside.sum())
            tree_mask[tree_idx[inside]] = False

        return rejected_buildings, rejected_trees


class SpatialHash:
    """Fixed-cell spatial hash over props, for broadphase collision queries.

    Args:
        cell_size_m: Cell edge length in metres.
    """

    def __init__(self, cell_size_m: float) -> None:
        self.cell_size_m = max(float(cell_size_m), 1e-3)
        self._cells: Dict[Tuple[int, int], Set[Prop]] = {}

    def _range(self, prop: Prop) -> Tuple[int, int, int, int]:
        """Inclusive cell index range covering a prop's bounding box."""
        size = self.cell_size_m
        return (
            int(math.floor((prop.x - prop.radius) / size)),
            int(math.floor((prop.y - prop.radius) / size)),
            int(math.floor((prop.x + prop.radius) / size)),
            int(math.floor((prop.y + prop.radius) / size)),
        )

    def insert_many(self, props: Iterable[Prop]) -> None:
        """Insert props into every cell their bounding box touches."""
        for prop in props:
            ix0, iy0, ix1, iy1 = self._range(prop)
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    self._cells.setdefault((ix, iy), set()).add(prop)

    def remove_many(self, props: Iterable[Prop]) -> None:
        """Remove props previously inserted."""
        for prop in props:
            ix0, iy0, ix1, iy1 = self._range(prop)
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    bucket = self._cells.get((ix, iy))
                    if bucket is not None:
                        bucket.discard(prop)
                        if not bucket:
                            del self._cells[(ix, iy)]

    def query_rect(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> List[Prop]:
        """All props whose cell overlaps the rectangle.

        Args:
            x0: Minimum x (metres).
            y0: Minimum y (metres).
            x1: Maximum x (metres).
            y1: Maximum y (metres).

        Returns:
            De-duplicated list of candidate props.
        """
        size = self.cell_size_m
        ix0 = int(math.floor(x0 / size))
        iy0 = int(math.floor(y0 / size))
        ix1 = int(math.floor(x1 / size))
        iy1 = int(math.floor(y1 / size))
        found: Set[Prop] = set()
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                bucket = self._cells.get((ix, iy))
                if bucket:
                    found.update(bucket)
        return list(found)

    @property
    def cell_count(self) -> int:
        """Number of occupied cells (diagnostics)."""
        return len(self._cells)


def _body_to_world(velocity: Vector, heading: float) -> Vector:
    """Rotate a body-frame velocity into world axes."""
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    return np.array(
        [
            cos_h * velocity[0] - sin_h * velocity[1],
            sin_h * velocity[0] + cos_h * velocity[1],
        ],
        dtype=np.float64,
    )


def _world_to_body(velocity: Vector, heading: float) -> Vector:
    """Rotate a world-frame velocity back into the body frame."""
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    return np.array(
        [
            cos_h * velocity[0] + sin_h * velocity[1],
            -sin_h * velocity[0] + cos_h * velocity[1],
        ],
        dtype=np.float64,
    )


def _box_contact(
    position: Vector, prop: Prop, car_radius: float
) -> Tuple[Vector, float]:
    """Circle-vs-AABB contact normal and penetration depth.

    Args:
        position: Car centre (world metres).
        prop: Building prop.
        car_radius: Car collision radius (m).

    Returns:
        ``(normal, penetration)``; penetration ``<= 0`` means no contact. The
        normal points from the box toward the car.
    """
    x0 = prop.x - prop.half_x
    x1 = prop.x + prop.half_x
    y0 = prop.y - prop.half_y
    y1 = prop.y + prop.half_y
    closest = np.array(
        [min(max(position[0], x0), x1), min(max(position[1], y0), y1)],
        dtype=np.float64,
    )
    delta = position - closest
    distance = float(np.hypot(delta[0], delta[1]))

    if distance > 1e-6:
        if distance >= car_radius:
            return np.zeros(2), 0.0
        return delta / distance, car_radius - distance

    # Centre is inside the box: escape through the nearest face.
    to_left = position[0] - x0
    to_right = x1 - position[0]
    to_bottom = position[1] - y0
    to_top = y1 - position[1]
    best = min(to_left, to_right, to_bottom, to_top)
    if best == to_left:
        return np.array([-1.0, 0.0]), car_radius + to_left
    if best == to_right:
        return np.array([1.0, 0.0]), car_radius + to_right
    if best == to_bottom:
        return np.array([0.0, -1.0]), car_radius + to_bottom
    return np.array([0.0, 1.0]), car_radius + to_top


def resolve_car_collisions(
    vehicle: Any,
    props: Sequence[Prop],
    collision_config: Dict[str, Any],
    *,
    passes: int = 2,
) -> ImpactResult:
    """Resolve car-vs-prop collisions with an impulse response.

    The car is treated as a circle. Each contact is pushed out along its normal
    and the normal velocity component is reflected with restitution; the
    tangential component is damped. The car's velocity lives in the body frame,
    so it is rotated to world axes for the impulse and back afterwards. The
    vehicle's own physics is untouched -- this only writes its position and
    velocity, both of which are part of its public state.

    Args:
        vehicle: A :class:`core.vehicle.Vehicle` (must expose ``position_m``,
            ``velocity_body_m_s``, ``heading_rad`` and ``mass_kg``).
        props: Candidate props (already broadphase-filtered).
        collision_config: The ``collision`` section of ``world.json``.
        passes: Number of resolution iterations.

    Returns:
        An :class:`ImpactResult`; also sets ``vehicle.damage_flag`` when an
        impact exceeds ``damage_speed_m_s``.
    """
    result = ImpactResult()
    if not props:
        return result

    car_radius = float(collision_config["car_radius_m"])
    restitution = float(collision_config["restitution"])
    friction = float(collision_config["tangential_friction"])
    damage_speed = float(collision_config["damage_speed_m_s"])
    mass = float(getattr(vehicle, "mass_kg", 1200.0))

    position = np.asarray(vehicle.position_m, dtype=np.float64)
    heading = float(vehicle.heading_rad)
    velocity_world = _body_to_world(
        np.asarray(vehicle.velocity_body_m_s, dtype=np.float64), heading
    )

    moved = False
    for _ in range(max(int(passes), 1)):
        contacts = 0
        for prop in props:
            if prop.is_circle:
                delta = position - np.array([prop.x, prop.y])
                distance = float(np.hypot(delta[0], delta[1]))
                minimum = car_radius + prop.radius
                if distance >= minimum:
                    continue
                if distance < 1e-6:
                    normal = np.array([1.0, 0.0])
                else:
                    normal = delta / distance
                penetration = minimum - distance
            else:
                normal, penetration = _box_contact(position, prop, car_radius)
                if penetration <= 0.0:
                    continue

            position = position + normal * penetration
            moved = True
            contacts += 1

            closing = float(np.dot(velocity_world, normal))
            if closing < 0.0:
                impact = -closing
                result.max_impact_speed_m_s = max(result.max_impact_speed_m_s, impact)
                velocity_world = velocity_world - (1.0 + restitution) * closing * normal
                tangential = velocity_world - np.dot(velocity_world, normal) * normal
                velocity_world = velocity_world - friction * tangential
                if impact > damage_speed:
                    result.damage = True
                    result.energy_j += 0.5 * mass * impact * impact
        result.count += contacts
        if contacts == 0:
            break

    if moved:
        vehicle.position_m[:] = position
        vehicle.velocity_body_m_s[:] = _world_to_body(velocity_world, heading)

    if result.damage:
        setattr(vehicle, "damage_flag", True)
        setattr(
            vehicle,
            "damage_energy_j",
            float(getattr(vehicle, "damage_energy_j", 0.0)) + result.energy_j,
        )
    return result
