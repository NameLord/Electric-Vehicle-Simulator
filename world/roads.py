"""Procedural road network: deterministic lattice of spline roads.

Design
------
Nodes live on a world-wide square lattice of spacing ``grid_spacing_m``, each
jittered by a hash of its own integer coordinates. Because a node's position is a
pure function of ``(seed, ix, iy)``, two neighbouring chunks compute *identical*
positions for a shared node, so roads join seamlessly across chunk borders with
no stitching pass.

Edges (the road segments between adjacent nodes) exist with a fixed probability,
also hashed. Each edge is sampled as a **cubic Hermite spline** whose end
tangents come from central differences along the same lattice row/column. That
makes consecutive collinear edges C1-continuous, so a chain of edges reads as one
smooth road rather than a series of kinks.

Nodes where both a horizontal and a vertical edge exist become intersections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .terrain import _hash01

__all__ = [
    "RoadEdge",
    "RoadNode",
    "RoadNetwork",
    "distance_to_segments",
    "build_segment_table",
]

LOGGER = logging.getLogger(__name__)

Vector = np.ndarray


def _hermite(
    p0: Vector, p1: Vector, m0: Vector, m1: Vector, samples: int
) -> np.ndarray:
    """Sample a cubic Hermite spline between ``p0`` and ``p1``.

    Args:
        p0: Start point (2,).
        p1: End point (2,).
        m0: Unit tangent at ``p0``.
        m1: Unit tangent at ``p1``.
        samples: Number of output points (>= 2).

    Returns:
        ``(samples, 2)`` array of points from ``p0`` to ``p1``.
    """
    n = max(int(samples), 2)
    t = np.linspace(0.0, 1.0, n)[:, None]
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2

    length = float(np.linalg.norm(p1 - p0))
    return (
        h00 * p0[None, :]
        + h10 * (m0 * length)[None, :]
        + h01 * p1[None, :]
        + h11 * (m1 * length)[None, :]
    )


def _polyline_point_at(
    polyline: np.ndarray, cumulative: np.ndarray, distance: float
) -> np.ndarray:
    """Point on a polyline at a given arc length (clamped at both ends).

    Args:
        polyline: ``(N, 2)`` points.
        cumulative: ``(N,)`` arc length at each point.
        distance: Arc length to sample.

    Returns:
        ``(2,)`` interpolated point.
    """
    if distance <= 0.0:
        return polyline[0].copy()
    if distance >= cumulative[-1]:
        return polyline[-1].copy()
    index = int(np.searchsorted(cumulative, distance, side="right") - 1)
    index = max(0, min(index, polyline.shape[0] - 2))
    span = max(float(cumulative[index + 1] - cumulative[index]), 1e-9)
    t = (distance - float(cumulative[index])) / span
    return polyline[index] + (polyline[index + 1] - polyline[index]) * t


@dataclass
class RoadEdge:
    """One road segment between two adjacent lattice nodes."""

    ix: int
    iy: int
    horizontal: bool
    width_m: float
    lanes: int
    shoulder_m: float
    points: np.ndarray  # (N, 2) sampled centreline

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Axis-aligned bounding box of the sampled centreline."""
        return (
            float(self.points[:, 0].min()),
            float(self.points[:, 1].min()),
            float(self.points[:, 0].max()),
            float(self.points[:, 1].max()),
        )


@dataclass(frozen=True)
class RoadNode:
    """A lattice node, promoted to an intersection when its degree is >= 3."""

    ix: int
    iy: int
    x: float
    y: float
    degree: int

    @property
    def is_intersection(self) -> bool:
        """True when three or more road arms meet here."""
        return self.degree >= 3


def build_segment_table(
    edges: Sequence[RoadEdge],
) -> np.ndarray:
    """Flatten road centrelines into a ``(N, 5)`` segment table.

    Columns are ``x0, y0, x1, y1, half_width``. A flat table is what makes the
    per-frame and per-prop distance queries vectorised rather than looping over
    road objects.

    Args:
        edges: Road edges to flatten.

    Returns:
        ``(N, 5)`` float array, possibly empty.
    """
    chunks: List[np.ndarray] = []
    for edge in edges:
        pts = edge.points
        if pts.shape[0] < 2:
            continue
        block = np.empty((pts.shape[0] - 1, 5), dtype=np.float64)
        block[:, 0:2] = pts[:-1]
        block[:, 2:4] = pts[1:]
        block[:, 4] = edge.width_m * 0.5
        chunks.append(block)
    if not chunks:
        return np.zeros((0, 5), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def distance_to_segments(
    px: np.ndarray, py: np.ndarray, segments: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Closest distance from points to a segment table.

    Fully vectorised over both the query points and the segments.

    Args:
        px: Query x coordinates, shape ``(Q,)``.
        py: Query y coordinates, shape ``(Q,)``.
        segments: ``(N, 5)`` table from :func:`build_segment_table`.

    Returns:
        ``(distance, half_width)`` arrays of shape ``(Q,)`` holding the distance
        to the nearest centreline and that segment's half width. Distances are
        ``inf`` where there are no segments.
    """
    q = np.asarray(px, dtype=np.float64).reshape(-1)
    if segments.shape[0] == 0 or q.size == 0:
        empty = np.full(q.shape, np.inf)
        return empty, np.zeros(q.shape)

    x0 = segments[:, 0][None, :]
    y0 = segments[:, 1][None, :]
    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    length_sq = dx * dx + dy * dy
    length_sq = np.where(length_sq < 1e-12, 1e-12, length_sq)

    qx = q[:, None]
    qy = np.asarray(py, dtype=np.float64).reshape(-1)[:, None]

    t = ((qx - x0) * dx[None, :] + (qy - y0) * dy[None, :]) / length_sq[None, :]
    np.clip(t, 0.0, 1.0, out=t)

    cx = x0 + t * dx[None, :]
    cy = y0 + t * dy[None, :]
    dist = np.hypot(qx - cx, qy - cy)

    best = np.argmin(dist, axis=1)
    rows = np.arange(q.shape[0])
    return dist[rows, best], segments[best, 4]


class RoadNetwork:
    """Deterministic spline road network over the whole world.

    Args:
        config: Parsed ``config/world.json``.
        chunk_size_m: Chunk edge length in metres.
    """

    def __init__(self, config: Dict[str, Any], chunk_size_m: float) -> None:
        roads = config["roads"]
        self.config = config
        self.seed = int(config["seed"])
        self.chunk_size_m = float(chunk_size_m)
        #: ``"grid"`` builds the procedural lattice network; ``"straight"`` builds
        #: a single perfectly straight road of endless length along one lattice
        #: row. Straight mode reuses every other part of this class unchanged:
        #: the same spline sampler, the same width/shoulder/dash helpers and the
        #: same ``distance_to_segments`` table, so no consumer needs to know.
        self.mode = str(roads.get("mode", "grid")).strip().lower()
        if self.mode not in ("grid", "straight"):
            raise ValueError(
                f"roads.mode must be 'grid' or 'straight' (got {self.mode!r})"
            )
        #: Lattice row the straight road runs along (world y = row * spacing).
        self.straight_row = int(roads.get("straight_row", 0))
        self.spacing = float(roads["grid_spacing_m"])
        self.jitter = float(roads["node_jitter_m"])
        self.prob_h = float(roads["horiz_edge_probability"])
        self.prob_v = float(roads["vert_edge_probability"])
        self.force_origin = bool(roads.get("force_origin_road", True))
        self.width_m = float(roads["width_m"])
        self.width_variance = float(roads["width_variance"])
        self.lane_count = int(roads["lane_count"])
        self.wide_lane_count = int(roads["wide_lane_count"])
        self.wide_probability = float(roads["wide_probability"])
        self.shoulder_m = float(roads["shoulder_m"])
        self.samples = int(roads["spline_samples"])
        self.intersection_radius_m = float(roads["intersection_radius_m"])
        self.dash_m = float(roads["lane_marking_dash_m"])
        self.gap_m = float(roads["lane_marking_gap_m"])
        self._edge_cache: Dict[Tuple[int, int, bool], RoadEdge] = {}
        if self.mode == "straight":
            LOGGER.info(
                "road network ready: mode=STRAIGHT, single endless road along "
                "world y=%.1f m, spacing=%.1f m, width=%.1f m, shoulder=%.1f m, "
                "no intersections, no width variation",
                self.straight_row * self.spacing,
                self.spacing,
                self.width_m,
                self.shoulder_m,
            )
            LOGGER.info(
                "  splines: cubic Hermite, %d samples/edge (nodes are exact "
                "lattice points with zero jitter, so the centreline is straight "
                "to floating-point precision for any chunk index)",
                self.samples,
            )
            return
        LOGGER.info(
            "road network ready: mode=GRID, grid=%.0f m, jitter=+-%.0f m, edge "
            "p=(%.2f h, %.2f v), width=%.1f m +-%.0f%%, lanes=%d/%d, "
            "shoulder=%.1f m",
            self.spacing,
            self.jitter,
            self.prob_h,
            self.prob_v,
            self.width_m,
            self.width_variance * 50.0,
            self.lane_count,
            self.wide_lane_count,
            self.shoulder_m,
        )
        LOGGER.info(
            "  splines: cubic Hermite, %d samples/edge, %d-row lattice (node "
            "positions are pure functions of (seed, ix, iy), so roads join "
            "across chunk borders with no stitching)",
            self.samples,
            4,
        )

    # ------------------------------------------------------------------
    # Lattice geometry
    # ------------------------------------------------------------------
    def node_xy(self, ix: int, iy: int) -> np.ndarray:
        """World position of a lattice node (deterministic in world space)."""
        if self.mode == "straight":
            # Zero jitter: every node sits exactly on the road line, so the
            # Hermite spline through them is a perfectly straight centreline and
            # the dashes stay evenly spaced for any chunk index.
            return np.array(
                [ix * self.spacing, iy * self.spacing], dtype=np.float64
            )
        jx = (_hash01(self.seed, ix, iy, salt=31) - 0.5) * 2.0 * self.jitter
        jy = (_hash01(self.seed, ix, iy, salt=37) - 0.5) * 2.0 * self.jitter
        return np.array(
            [ix * self.spacing + float(jx), iy * self.spacing + float(jy)],
            dtype=np.float64,
        )

    def _tangent(self, ix: int, iy: int, horizontal: bool) -> np.ndarray:
        """Unit tangent along a road through ``(ix, iy)`` from central differences."""
        if horizontal:
            a = self.node_xy(ix - 1, iy)
            b = self.node_xy(ix + 1, iy)
        else:
            a = self.node_xy(ix, iy - 1)
            b = self.node_xy(ix, iy + 1)
        delta = b - a
        norm = float(np.linalg.norm(delta))
        return delta / norm if norm > 1e-9 else np.array([1.0, 0.0])

    def _origin_node(self) -> Tuple[int, int]:
        """Lattice node index nearest the world origin."""
        return int(round(0.0 / self.spacing)), int(round(0.0 / self.spacing))

    def _forced(self, ix: int, iy: int, horizontal: bool) -> bool:
        """True when an edge is force-enabled so the spawn point has a road."""
        if not self.force_origin:
            return False
        ox, oy = self._origin_node()
        if horizontal:
            return iy == oy and ox - 1 <= ix <= ox + 1
        return ix == ox and oy - 1 <= iy <= oy + 1

    def edge_exists(self, ix: int, iy: int, horizontal: bool) -> bool:
        """Whether the lattice edge leaving ``(ix, iy)`` exists.

        In ``"straight"`` mode there is exactly one road line: the horizontal
        edges along row :attr:`straight_row`, for every ``ix`` (which is what
        makes the road endless). Every other edge, including all vertical ones,
        is absent - so no junction can ever form and ``intersections_for_rect``
        always returns an empty list.
        """
        if self.mode == "straight":
            return bool(horizontal) and iy == self.straight_row
        if self._forced(ix, iy, horizontal):
            return True
        probability = self.prob_h if horizontal else self.prob_v
        salt = 91 if horizontal else 97
        return bool(_hash01(self.seed, ix, iy, salt=salt) < probability)

    def _edge_geometry(self, ix: int, iy: int) -> Tuple[float, int]:
        """Width and lane count for an edge.

        Straight mode returns the nominal width and lane count with no
        variation, so the road reads as a single uniform ribbon rather than a
        sequence of differently-sized segments.
        """
        if self.mode == "straight":
            return self.width_m, int(self.lane_count)
        roll = float(_hash01(self.seed, ix, iy, salt=113))
        width = self.width_m * (1.0 + self.width_variance * (roll - 0.5) * 2.0)
        lanes = (
            self.wide_lane_count if roll < self.wide_probability else self.lane_count
        )
        return width, int(lanes)

    def make_edge(self, ix: int, iy: int, horizontal: bool) -> RoadEdge:
        """Build (and cache) the sampled spline for one edge."""
        key = (ix, iy, horizontal)
        cached = self._edge_cache.get(key)
        if cached is not None:
            return cached

        if horizontal:
            p0 = self.node_xy(ix, iy)
            p1 = self.node_xy(ix + 1, iy)
            m0 = self._tangent(ix, iy, True)
            m1 = self._tangent(ix + 1, iy, True)
        else:
            p0 = self.node_xy(ix, iy)
            p1 = self.node_xy(ix, iy + 1)
            m0 = self._tangent(ix, iy, False)
            m1 = self._tangent(ix, iy + 1, False)

        width, lanes = self._edge_geometry(ix, iy)
        edge = RoadEdge(
            ix=ix,
            iy=iy,
            horizontal=horizontal,
            width_m=width,
            lanes=lanes,
            shoulder_m=self.shoulder_m,
            points=_hermite(p0, p1, m0, m1, self.samples),
        )
        self._edge_cache[key] = edge
        return edge

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def edges_for_rect(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> List[RoadEdge]:
        """Every road edge whose spline could pass through a world rectangle.

        Args:
            x0: Minimum x (metres).
            y0: Minimum y (metres).
            x1: Maximum x (metres).
            y1: Maximum y (metres).

        Returns:
            List of :class:`RoadEdge` (may be empty).
        """
        margin = self.jitter + self.width_m + self.shoulder_m + self.spacing * 0.5
        i_start = int(np.floor((x0 - margin) / self.spacing))
        i_end = int(np.ceil((x1 + margin) / self.spacing))
        j_start = int(np.floor((y0 - margin) / self.spacing))
        j_end = int(np.ceil((y1 + margin) / self.spacing))

        found: List[RoadEdge] = []
        for ix in range(i_start, i_end + 1):
            for iy in range(j_start, j_end + 1):
                for horizontal in (True, False):
                    if not self.edge_exists(ix, iy, horizontal):
                        continue
                    edge = self.make_edge(ix, iy, horizontal)
                    ex0, ey0, ex1, ey1 = edge.bounds
                    half = edge.width_m * 0.5 + edge.shoulder_m
                    if (
                        ex1 + half >= x0
                        and ex0 - half <= x1
                        and ey1 + half >= y0
                        and ey0 - half <= y1
                    ):
                        found.append(edge)
        return found

    def intersections_for_rect(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> List[RoadNode]:
        """Intersection nodes (degree >= 3) inside or near a rectangle."""
        margin = self.jitter + self.intersection_radius_m + self.spacing * 0.5
        i_start = int(np.floor((x0 - margin) / self.spacing))
        i_end = int(np.ceil((x1 + margin) / self.spacing))
        j_start = int(np.floor((y0 - margin) / self.spacing))
        j_end = int(np.ceil((y1 + margin) / self.spacing))

        nodes: List[RoadNode] = []
        for ix in range(i_start, i_end + 1):
            for iy in range(j_start, j_end + 1):
                west = self.edge_exists(ix - 1, iy, True)
                east = self.edge_exists(ix, iy, True)
                south = self.edge_exists(ix, iy - 1, False)
                north = self.edge_exists(ix, iy, False)
                degree = int(west) + int(east) + int(south) + int(north)
                if degree >= 3:
                    position = self.node_xy(ix, iy)
                    nodes.append(
                        RoadNode(
                            ix=ix,
                            iy=iy,
                            x=float(position[0]),
                            y=float(position[1]),
                            degree=degree,
                        )
                    )
        return nodes

    def nearest_road(
        self, x: float, y: float, max_radius_m: float = 400.0
    ) -> Optional[Tuple[np.ndarray, float, RoadEdge]]:
        """Closest point on any road to ``(x, y)``.

        Args:
            x: Query x metres.
            y: Query y metres.
            max_radius_m: Give up beyond this search radius.

        Returns:
            ``(point, distance, edge)`` or ``None`` when no road is found.
        """
        radius = self.spacing
        while radius <= max_radius_m:
            edges = self.edges_for_rect(x - radius, y - radius, x + radius, y + radius)
            best_distance = np.inf
            best_point: Optional[np.ndarray] = None
            best_edge: Optional[RoadEdge] = None
            for edge in edges:
                pts = edge.points
                seg_a = pts[:-1]
                seg_b = pts[1:]
                ab = seg_b - seg_a
                length_sq = np.maximum((ab * ab).sum(axis=1), 1e-12)
                t = ((x - seg_a[:, 0]) * ab[:, 0] + (y - seg_a[:, 1]) * ab[:, 1]) / length_sq
                np.clip(t, 0.0, 1.0, out=t)
                closest = seg_a + t[:, None] * ab
                dist = np.hypot(closest[:, 0] - x, closest[:, 1] - y)
                index = int(np.argmin(dist))
                if dist[index] < best_distance:
                    best_distance = float(dist[index])
                    best_point = closest[index]
                    best_edge = edge
            if best_point is not None and best_edge is not None:
                return best_point, best_distance, best_edge
            radius *= 2.0
        return None

    # ------------------------------------------------------------------
    def center_dashes(
        self, edge: RoadEdge
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Dashed centreline segments for lane markings.

        Walks the sampled spline accumulating arc length and emits a dash every
        ``dash + gap`` metres. Wide roads get an extra dashed lane divider at a
        quarter and three-quarter width.

        Args:
            edge: The road edge to mark.

        Returns:
            List of ``((x0, y0), (x1, y1))`` world-space segments.
        """
        pts = edge.points
        if pts.shape[0] < 2:
            return []

        directions = np.gradient(pts, axis=0)
        magnitudes = np.maximum(np.hypot(directions[:, 0], directions[:, 1]), 1e-9)
        normals = np.stack(
            [-directions[:, 1] / magnitudes, directions[:, 0] / magnitudes], axis=1
        )
        cumulative = np.concatenate(
            [[0.0], np.cumsum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))]
        )
        total = float(cumulative[-1])
        if total <= 0.0:
            return []

        offsets = [0.0]
        if edge.lanes >= 4:
            offsets = [-edge.width_m * 0.25, edge.width_m * 0.25]

        period = self.dash_m + self.gap_m
        dashes: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for offset in offsets:
            line = pts + normals * offset
            start = 0.0
            while start < total:
                end = min(start + self.dash_m, total)
                a = _polyline_point_at(line, cumulative, start)
                b = _polyline_point_at(line, cumulative, end)
                dashes.append(
                    ((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
                )
                start += period
        return dashes

    @staticmethod
    def shoulder_outline(
        edge: RoadEdge, half_width_m: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Left and right boundary polylines offset from the centreline."""
        pts = edge.points
        directions = np.gradient(pts, axis=0)
        norms = np.maximum(np.hypot(directions[:, 0], directions[:, 1]), 1e-9)
        normals = np.stack(
            [-directions[:, 1] / norms, directions[:, 0] / norms], axis=1
        )
        left = pts + normals * half_width_m
        right = pts - normals * half_width_m
        return left, right
