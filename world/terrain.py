"""Chunked procedural terrain: seeded value noise, biome blending, baked ground.

Everything here is a pure function of ``(seed, world position)``, so the same
seed always produces the same world and any chunk can be generated in isolation
or in any order.

* Noise is hash-based value noise with smoothstep interpolation, evaluated with
  NumPy over whole grids -- there are no per-pixel Python loops.
* Biome weights are *continuous* (soft thresholds over three noise fields), so
  biomes blend across their borders instead of butting up against each other.
* Ground is baked once per chunk into a cached :class:`pygame.Surface` by
  combining a seamless per-biome procedural tile with the biome weight map.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pygame

__all__ = [
    "BIOMES",
    "value_noise",
    "fractal_noise",
    "mix_seed",
    "TerrainGenerator",
]

LOGGER = logging.getLogger(__name__)

#: Biome order is fixed; index into it everywhere (weights, colours, tiles).
BIOMES: Tuple[str, ...] = ("forest", "plains", "urban", "industrial")

_UINT32: int = 0xFFFFFFFF


def mix_seed(seed: int, *values: int) -> int:
    """Fold integers into a valid non-negative NumPy seed.

    ``numpy.random.default_rng`` rejects negative seeds, and chunk coordinates
    are routinely negative (the world extends in all four directions), so they
    are folded through a 32-bit FNV-style mix instead of being passed directly.

    Args:
        seed: World seed.
        *values: Extra integers (chunk coordinates, salts, ...).

    Returns:
        A non-negative integer usable as a seed.
    """
    mixed = int(seed) & _UINT32
    for value in values:
        mixed = (mixed ^ (int(value) & _UINT32)) & _UINT32
        mixed = (mixed * 16777619) & _UINT32
        mixed = (mixed ^ (mixed >> 15)) & _UINT32
    return int(mixed)


def _hash01(
    seed: int, ix: np.ndarray, iy: np.ndarray, salt: int = 0
) -> np.ndarray:
    """Vectorised integer hash mapped to ``[0, 1)``.

    Args:
        seed: World seed.
        ix: Integer lattice x coordinates (any shape).
        iy: Integer lattice y coordinates (same shape as ``ix``).
        salt: Extra constant so several independent noise fields can share a
            seed.

    Returns:
        Float array of the same shape as ``ix``, values in ``[0, 1)``.
    """
    x = np.asarray(ix, dtype=np.int64)
    y = np.asarray(iy, dtype=np.int64)
    h = (x * 374761393 + y * 668265263 + seed * 1013904223 + salt * 362437) & _UINT32
    h = (h ^ (h >> 13)) & _UINT32
    h = (h * 1274126177) & _UINT32
    h = (h ^ (h >> 16)) & _UINT32
    return h.astype(np.float64) / 4294967296.0


def value_noise(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    scale: float,
    salt: int = 0,
) -> np.ndarray:
    """Smoothstep-interpolated value noise in ``[0, 1]``.

    Args:
        seed: World seed.
        x: World x coordinates in metres (any shape).
        y: World y coordinates in metres.
        scale: Lattice cells per metre. Larger is noisier.
        salt: Field selector.

    Returns:
        Noise values in ``[0, 1]``, same shape as ``x``.
    """
    fx = np.asarray(x, dtype=np.float64) * scale
    fy = np.asarray(y, dtype=np.float64) * scale
    ix = np.floor(fx)
    iy = np.floor(fy)
    tx = fx - ix
    ty = fy - iy
    # Smoothstep removes the lattice creases of plain bilinear interpolation.
    sx = tx * tx * (3.0 - 2.0 * tx)
    sy = ty * ty * (3.0 - 2.0 * ty)

    i0 = ix.astype(np.int64)
    j0 = iy.astype(np.int64)

    v00 = _hash01(seed, i0, j0, salt)
    v10 = _hash01(seed, i0 + 1, j0, salt)
    v01 = _hash01(seed, i0, j0 + 1, salt)
    v11 = _hash01(seed, i0 + 1, j0 + 1, salt)

    return (v00 * (1.0 - sx) + v10 * sx) * (1.0 - sy) + (
        v01 * (1.0 - sx) + v11 * sx
    ) * sy


def fractal_noise(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    scale: float,
    octaves: int,
    salt: int = 0,
) -> np.ndarray:
    """Sum of ``octaves`` value-noise octaves at doubling frequency.

    Args:
        seed: World seed.
        x: World x metres.
        y: World y metres.
        scale: Base lattice cells per metre.
        octaves: Number of octaves (>= 1).
        salt: Field selector.

    Returns:
        Normalised fractal noise in ``[0, 1]``.
    """
    total = np.zeros_like(np.asarray(x, dtype=np.float64))
    amplitude = 1.0
    norm = 0.0
    freq = scale
    for octave in range(max(int(octaves), 1)):
        total = total + amplitude * value_noise(
            seed, x, y, freq, salt + octave * 17
        )
        norm += amplitude
        amplitude *= 0.5
        freq *= 2.0
    return total / max(norm, 1e-9)


def _periodic_value_noise(
    rng: np.random.Generator, lattice: int, out: int
) -> np.ndarray:
    """Seamless ``out x out`` noise tile from a periodic ``lattice`` grid.

    The lattice wraps, so tiling the result leaves no visible seam.

    Args:
        rng: Seeded generator.
        lattice: Lattice resolution (wraps).
        out: Output tile size in pixels.

    Returns:
        ``(out, out)`` float array in ``[0, 1]`` with mean ~0.5.
    """
    base = rng.random((lattice, lattice))
    idx = np.arange(out, dtype=np.float64) * (lattice / out)
    i0 = np.floor(idx).astype(np.int64)
    t = idx - i0
    s = t * t * (3.0 - 2.0 * t)
    i0m = i0 % lattice
    i1m = (i0 + 1) % lattice

    a = base[np.ix_(i0m, i0m)]
    b = base[np.ix_(i0m, i1m)]
    c = base[np.ix_(i1m, i0m)]
    d = base[np.ix_(i1m, i1m)]
    sx = s[:, None]
    sy = s[None, :]
    return (a * (1.0 - sx) + c * sx) * (1.0 - sy) + (
        b * (1.0 - sx) + d * sx
    ) * sy


def _resample_corner_grid(source: np.ndarray, out_res: int) -> np.ndarray:
    """Resample a chunk-corner-sampled field onto the bake pixel grid.

    ``source`` is an ``(n+1, n+1)`` field sampled at the **exact chunk corners**,
    so source index ``i`` sits at world offset ``i * chunk / n``. Output pixel
    ``p`` covers ``[p, p+1) * chunk / out_res``, so its centre is at world offset
    ``(p + 0.5) * chunk / out_res`` and maps to source coordinate
    ``(p + 0.5) * n / out_res``.

    Keeping that mapping world-aligned is what makes chunk borders invisible. The
    earlier version sampled at cell *centres* (``0.5, 1.5, ... n-0.5`` scaled) and
    then stretched that grid over the whole chunk, which

    * scaled the field by ``chunk / (chunk - chunk/n)`` inside every chunk, so the
      terrain appeared to change size between chunks, and
    * reset the field's phase at every border, leaving a hard seam wherever a
      biome boundary crossed a chunk edge (reading as a sharp, straight,
      roughly 90-degree edge instead of a blend).

    Args:
        source: ``(n+1, n+1)`` field sampled at chunk corners.
        out_res: Output edge length in pixels.

    Returns:
        ``(out_res, out_res)`` resampled field.
    """
    n = source.shape[0] - 1
    if n < 1:
        return np.full((out_res, out_res), float(source.flat[0]))
    idx = (np.arange(out_res, dtype=np.float64) + 0.5) * (n / out_res)
    i0 = np.floor(idx).astype(np.int64)
    np.clip(i0, 0, n - 1, out=i0)
    i1 = i0 + 1
    t = idx - i0
    rows = source[i0] * (1.0 - t)[:, None] + source[i1] * t[:, None]
    return rows[:, i0] * (1.0 - t)[None, :] + rows[:, i1] * t[None, :]


@dataclass(frozen=True)
class BiomeWeights:
    """Per-point biome blend weights plus the dominant biome index."""

    weights: np.ndarray  # (4, N)
    dominant: np.ndarray  # (N,) argmax index


class TerrainGenerator:
    """Deterministic, seeded terrain generator.

    Args:
        config: Parsed ``config/world.json``.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.seed = int(config["seed"])
        self.pixels_per_metre = float(config["pixels_per_metre"])
        self.chunk_size_px = int(config["chunk_size_px"])
        self.chunk_size_m = self.chunk_size_px / self.pixels_per_metre

        terrain = config["terrain"]
        self.noise_scale = float(terrain["noise_scale"])
        self.noise_octaves = int(terrain["noise_octaves"])
        self.texture_tile_px = int(terrain["texture_tile_px"])
        self.texture_contrast = float(terrain["texture_contrast"])
        self.biome_colors = np.array(
            [terrain["biome_colors"][name] for name in BIOMES], dtype=np.float64
        )
        thresholds = terrain["biome_thresholds"]
        self._urban_low = float(thresholds["urban_low"])
        self._urban_high = float(thresholds["urban_high"])
        self._ind_low = float(thresholds["industrial_low"])
        self._ind_high = float(thresholds["industrial_high"])
        self._forest_low = float(thresholds["forest_low"])
        self._forest_high = float(thresholds["forest_high"])

        self._tiles: Dict[int, np.ndarray] = {}
        self._luma_cache: Dict[int, np.ndarray] = {}
        self.biome_sample_resolution = max(
            int(config.get("biome_sample_resolution", 16)), 2
        )
        LOGGER.info(
            "terrain ready: seed=%d, chunk=%d px (%.1f m), bake=%d px, "
            "biome samples=%d px, tile=%d px",
            self.seed,
            self.chunk_size_px,
            self.chunk_size_m,
            int(config.get("bake_resolution_px", 128)),
            self.biome_sample_resolution,
            self.texture_tile_px,
        )
        LOGGER.info(
            "  biomes: %s - continuous noise weights (partition of unity), so "
            "borders cross-fade",
            ", ".join(BIOMES),
        )
        LOGGER.info(
            "  noise: value-noise fBm, scale=%.4f cells/m x %d octaves, "
            "hash-seeded so any chunk generates standalone",
            self.noise_scale,
            self.noise_octaves,
        )

    # ------------------------------------------------------------------
    def _rng(self, salt: int = 0) -> np.random.Generator:
        """Return a fresh generator derived from the world seed."""
        return np.random.default_rng((self.seed, salt))

    def biome_weights(self, x: np.ndarray, y: np.ndarray) -> BiomeWeights:
        """Continuous biome blend weights at world positions.

        Three independent noise fields (urbanity, industry, forestiness) are
        passed through smoothstep ramps and combined, so each point gets a
        normalised weight per biome. Because the ramps are smooth, biome
        borders cross-fade rather than snapping.

        Args:
            x: World x metres (any shape).
            y: World y metres.

        Returns:
            :class:`BiomeWeights` holding a ``(4, N)`` weight array and the
            dominant biome index per point.
        """
        shape = np.shape(x)
        urbanity = fractal_noise(
            self.seed, x, y, self.noise_scale, self.noise_octaves, salt=0
        )
        industry = fractal_noise(
            self.seed, x, y, self.noise_scale, self.noise_octaves, salt=211
        )
        forestry = fractal_noise(
            self.seed, x, y, self.noise_scale, self.noise_octaves, salt=577
        )

        def ramp(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
            return np.clip((v - lo) / max(hi - lo, 1e-9), 0.0, 1.0)

        w_urban = ramp(urbanity, self._urban_low, self._urban_high)
        w_ind = ramp(industry, self._ind_low, self._ind_high) * (1.0 - 0.5 * w_urban)
        w_forest = (
            ramp(forestry, self._forest_low, self._forest_high)
            * (1.0 - w_urban)
            * (1.0 - w_ind)
        )
        w_plains = np.clip(1.0 - w_urban - w_ind - w_forest, 0.0, None)

        raw = np.stack([w_forest, w_plains, w_urban, w_ind], axis=0)
        total = raw.sum(axis=0, keepdims=True)
        weights = raw / np.maximum(total, 1e-9)
        flat = weights.reshape(len(BIOMES), -1)
        dominant = np.argmax(flat, axis=0).reshape(shape)
        return BiomeWeights(weights=flat, dominant=dominant)

    def dominant_biome(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Index of the strongest biome at each point (batched, for props).

        Args:
            x: World x metres.
            y: World y metres.

        Returns:
            Integer array of :data:`BIOMES` indices, same shape as ``x``.
        """
        return self.biome_weights(x, y).dominant

    # ------------------------------------------------------------------
    def _tile(self, biome_index: int) -> np.ndarray:
        """Return the cached seamless luma tile for a biome, in ``[0.4, 1.5]``."""
        cached = self._tiles.get(biome_index)
        if cached is not None:
            return cached

        rng = self._rng(salt=1000 + biome_index)
        size = self.texture_tile_px
        tile = np.zeros((size, size), dtype=np.float64)
        weight = 1.0
        norm = 0.0
        for lattice in (2, 4, 8, 16):
            tile += weight * _periodic_value_noise(rng, lattice, size)
            norm += weight
            weight *= 0.55
        tile /= max(norm, 1e-9)
        tile -= tile.mean()
        tile = 1.0 + tile * self.texture_contrast * 2.0
        np.clip(tile, 0.35, 1.6, out=tile)
        self._tiles[biome_index] = tile
        return tile

    def _tiled_luma(self, resolution: int) -> np.ndarray:
        """Stack of per-biome seamless tiles tiled up to the chunk resolution.

        The result only depends on the bake resolution, so it is cached rather
        than rebuilt for every chunk.

        Args:
            resolution: Baked chunk resolution in pixels.

        Returns:
            ``(4, resolution, resolution)`` luma array.
        """
        cached = self._luma_cache.get(resolution)
        if cached is not None:
            return cached

        size = self.texture_tile_px
        reps = int(np.ceil(resolution / size))
        stack = np.empty((len(BIOMES), resolution, resolution), dtype=np.float64)
        for index in range(len(BIOMES)):
            tiled = np.tile(self._tile(index), (reps, reps))[:resolution, :resolution]
            stack[index] = tiled
        self._luma_cache[resolution] = stack
        return stack

    # ------------------------------------------------------------------
    def chunk_surface(self, cx: int, cy: int) -> pygame.Surface:
        """Bake (or rebuild) the ground surface for a chunk.

        The bake resolution is decoupled from the chunk's pixel size, so the
        expensive part is small and the result is scaled once. Because the
        chunk size is an exact multiple of the texture tile, tiled ground stays
        continuous across chunk borders.

        Args:
            cx: Chunk x index.
            cy: Chunk y index.

        Returns:
            A ``chunk_size_px x chunk_size_px`` surface.
        """
        resolution = int(self.config.get("bake_resolution_px", 128))
        chunk_m = self.chunk_size_m
        x0 = cx * chunk_m
        y0 = cy * chunk_m

        # Biome weights are smooth, so sample them coarsely and resample up.
        # Sampling happens at the chunk CORNERS (sample_res + 1 points spanning
        # exactly [x0, x0 + chunk_m]) so the field is world-aligned and adjacent
        # chunks agree exactly on their shared edge.
        sample_res = self.biome_sample_resolution
        sample_step = chunk_m / sample_res
        sample_x = x0 + np.arange(sample_res + 1, dtype=np.float64) * sample_step
        sample_y = y0 + np.arange(sample_res + 1, dtype=np.float64) * sample_step
        grid_x, grid_y = np.meshgrid(sample_x, sample_y)
        low = self.biome_weights(grid_x, grid_y).weights.reshape(
            len(BIOMES), sample_res + 1, sample_res + 1
        )
        weights = np.stack(
            [_resample_corner_grid(low[i], resolution) for i in range(len(BIOMES))]
        )
        weights /= np.maximum(weights.sum(axis=0, keepdims=True), 1e-9)

        luma = self._tiled_luma(resolution)

        out = np.zeros((resolution, resolution, 3), dtype=np.float64)
        for index in range(len(BIOMES)):
            colour = self.biome_colors[index]
            out += (
                weights[index][..., None] * colour[None, None, :]
            ) * luma[index][..., None]

        rgb = np.clip(out, 0, 255).astype(np.uint8)
        # pygame.surfarray wants [x][y][channel].
        surface = pygame.surfarray.make_surface(
            np.ascontiguousarray(np.transpose(rgb, (1, 0, 2)))
        )
        if self.config.get("prescale_chunk_surface", True):
            surface = pygame.transform.scale(
                surface, (self.chunk_size_px, self.chunk_size_px)
            )
        return surface

    # ------------------------------------------------------------------
    def biome_at_point(self, x: float, y: float) -> str:
        """Name of the dominant biome at a single world point."""
        dominant = self.dominant_biome(
            np.array([x], dtype=np.float64), np.array([y], dtype=np.float64)
        )
        return BIOMES[int(dominant[0])]

    def biome_density(self, x: float, y: float) -> Dict[str, float]:
        """Spawn densities for the dominant biome at a point.

        Args:
            x: World x metres.
            y: World y metres.

        Returns:
            Mapping of prop kind to probability (from ``world.json``).
        """
        return dict(self.config["props"]["density"][self.biome_at_point(x, y)])
