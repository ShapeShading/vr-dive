#!/usr/bin/env python3
"""Build the low-resolution Gyirong debris-flow scene assets.

Inputs are public Terrarium elevation tiles plus OpenStreetMap building and
waterway Overpass responses. The generated assets are intentionally compact
enough to ship with the demo while retaining one uniform metre-based coordinate
system.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image


ZOOM = 12
TILE_SIZE = 256
TILE_X0, TILE_X1 = 3018, 3022
TILE_Y0, TILE_Y1 = 1711, 1713
# Keep the event corridor at roughly the original 90–100 m DEM spacing while
# adding several kilometres of real terrain around every previous edge. A
# still coarser procedural apron is added by the renderer beyond these bounds.
WEST, EAST = 85.29, 85.63
SOUTH, NORTH = 28.20, 28.36
GRID_WIDTH, GRID_HEIGHT = 337, 193
PORT_LAT, PORT_LON = 28.2809, 85.3779
SOURCE_LAT, SOURCE_LON = 28.281051, 85.545404
# OSM way 937405875 is the mapped downstream centreline of the Donglin
# Tsangpo/Lende Khola from the upper-valley confluence to Gyirong Port.  The
# event source is in a steep, incompletely mapped tributary, so the connector
# to this way is derived hydrologically from the DEM instead of drawing a
# source-to-port line.
DOWNSTREAM_LENDE_WAY_ID = 937405875


def lon_to_global_x(lon: np.ndarray | float) -> np.ndarray | float:
    return (lon + 180.0) / 360.0 * (2**ZOOM) * TILE_SIZE


def lat_to_global_y(lat: np.ndarray | float) -> np.ndarray | float:
    radians = np.radians(lat)
    return (
        1.0 - np.arcsinh(np.tan(radians)) / math.pi
    ) * 0.5 * (2**ZOOM) * TILE_SIZE


def read_mosaic(tile_dir: Path) -> np.ndarray:
    rows = []
    for tile_y in range(TILE_Y0, TILE_Y1 + 1):
        row = []
        for tile_x in range(TILE_X0, TILE_X1 + 1):
            path = tile_dir / f"{tile_x}_{tile_y}.png"
            rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
            height = rgb[:, :, 0] * 256.0 + rgb[:, :, 1] + rgb[:, :, 2] / 256.0 - 32768.0
            row.append(height)
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, image.shape[1] - 1.001)
    y = np.clip(y, 0.0, image.shape[0] - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = x - x0
    fy = y - y0
    return (
        image[y0, x0] * (1.0 - fx) * (1.0 - fy)
        + image[y0, x1] * fx * (1.0 - fy)
        + image[y1, x0] * (1.0 - fx) * fy
        + image[y1, x1] * fx * fy
    )


def sample_grid(mosaic: np.ndarray) -> np.ndarray:
    longitudes = np.linspace(WEST, EAST, GRID_WIDTH, dtype=np.float64)
    latitudes = np.linspace(SOUTH, NORTH, GRID_HEIGHT, dtype=np.float64)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    local_x = lon_to_global_x(lon_grid) - TILE_X0 * TILE_SIZE
    local_y = lat_to_global_y(lat_grid) - TILE_Y0 * TILE_SIZE
    return bilinear_sample(mosaic, local_x, local_y).astype("<f4")


def sample_geo_height(grid: np.ndarray, latitude: float, longitude: float) -> float:
    x = (longitude - WEST) / (EAST - WEST) * (GRID_WIDTH - 1)
    y = (latitude - SOUTH) / (NORTH - SOUTH) * (GRID_HEIGHT - 1)
    return float(bilinear_sample(grid, np.asarray(x), np.asarray(y)))


def parse_number(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    return float(match.group()) if match else None


def building_height(tags: dict[str, str]) -> float:
    explicit = parse_number(tags.get("height"))
    if explicit is not None:
        return min(max(explicit, 2.5), 45.0)
    levels = parse_number(tags.get("building:levels"))
    if levels is not None:
        return min(max(levels * 3.2, 3.2), 45.0)
    kind = tags.get("building", "yes")
    if kind in {"industrial", "warehouse", "hangar"}:
        return 9.0
    return 7.5


def process_buildings(overpass_path: Path) -> list[dict[str, object]]:
    payload = json.loads(overpass_path.read_text())
    buildings = []
    for element in payload.get("elements", []):
        geometry = element.get("geometry") or []
        footprint = [[point["lon"], point["lat"]] for point in geometry]
        if len(footprint) >= 2 and footprint[0] == footprint[-1]:
            footprint.pop()
        if len(footprint) < 3:
            continue
        tags = element.get("tags", {})
        buildings.append(
            {
                "sourceElementID": int(element["id"]),
                "heightMeters": building_height(tags),
                "footprintLonLat": footprint,
            }
        )
    return buildings


def process_waterway_flow_path(
    grid: np.ndarray,
    overpass_path: Path,
) -> list[list[float]]:
    """Join the source tributary to the mapped Lende centreline.

    OSM currently maps the winding main channel continuously from the
    upper-valley confluence to the port, but the glacier-side tributary is
    fragmented.  Dijkstra routing is therefore used only for the missing
    source-to-confluence section.  Its very high uphill penalty makes it follow
    connected descending DEM cells.  From the confluence onwards the exact OSM
    way geometry is the hard constraint.
    """

    def cell(latitude: float, longitude: float) -> tuple[int, int]:
        x = round((longitude - WEST) / (EAST - WEST) * (GRID_WIDTH - 1))
        y = round((latitude - SOUTH) / (NORTH - SOUTH) * (GRID_HEIGHT - 1))
        return int(x), int(y)

    payload = json.loads(overpass_path.read_text())
    element_by_id = {
        int(element["id"]): element
        for element in payload.get("elements", [])
        if element.get("type") == "way"
    }
    try:
        lende_geometry = element_by_id[DOWNSTREAM_LENDE_WAY_ID]["geometry"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"OSM way {DOWNSTREAM_LENDE_WAY_ID} is missing from waterways input"
        ) from error
    if len(lende_geometry) < 2:
        raise RuntimeError("mapped Lende centreline has insufficient geometry")

    # Orient the mapped way downhill.  In the current OSM data the first point
    # is the upstream confluence and the final point is beside the border port,
    # but elevation-based orientation keeps regeneration robust to an OSM way
    # reversal.
    first_height = sample_geo_height(
        grid, lende_geometry[0]["lat"], lende_geometry[0]["lon"]
    )
    last_height = sample_geo_height(
        grid, lende_geometry[-1]["lat"], lende_geometry[-1]["lon"]
    )
    if first_height < last_height:
        lende_geometry.reverse()

    start = cell(SOURCE_LAT, SOURCE_LON)
    goal = cell(lende_geometry[0]["lat"], lende_geometry[0]["lon"])
    longitude_cell_meters = (
        (EAST - WEST)
        * 111_320.0
        * math.cos(math.radians(PORT_LAT))
        / (GRID_WIDTH - 1)
    )
    latitude_cell_meters = (NORTH - SOUTH) * 110_540.0 / (GRID_HEIGHT - 1)
    queue: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    best_cost: dict[tuple[int, int], float] = {start: 0.0}

    while queue:
        _, cost, current = heapq.heappop(queue)
        if current == goal:
            break
        if cost != best_cost.get(current):
            continue
        x, y = current
        current_elevation = float(grid[y, x])
        for dx, dy in (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ):
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= GRID_WIDTH or ny < 0 or ny >= GRID_HEIGHT:
                continue
            step_meters = math.hypot(
                dx * longitude_cell_meters,
                dy * latitude_cell_meters,
            )
            next_elevation = float(grid[ny, nx])
            uphill = max(next_elevation - current_elevation, 0.0)
            next_cost = (
                cost
                + step_meters
                # A 1 m rise costs more than three full DEM cells.  This is
                # intentionally much stronger than the old source-to-port A*,
                # which could shortcut a ridge to reduce horizontal distance.
                + uphill * 300.0
            )
            neighbor = (nx, ny)
            if next_cost >= best_cost.get(neighbor, math.inf):
                continue
            best_cost[neighbor] = next_cost
            came_from[neighbor] = current
            heuristic = math.hypot(
                (nx - goal[0]) * longitude_cell_meters,
                (ny - goal[1]) * latitude_cell_meters,
            )
            heapq.heappush(queue, (next_cost + heuristic, next_cost, neighbor))

    if goal not in came_from:
        raise RuntimeError("unable to find the descending source tributary")
    connector = [goal]
    while connector[-1] != start:
        connector.append(came_from[connector[-1]])
    connector.reverse()

    longitude_meters = (
        (EAST - WEST) * 111_320.0 * math.cos(math.radians(PORT_LAT))
    )
    latitude_meters = (NORTH - SOUTH) * 110_540.0
    route = [
        np.asarray(
            [x / (GRID_WIDTH - 1), y / (GRID_HEIGHT - 1)],
            dtype=np.float64,
        )
        for x, y in connector
    ]
    for point in lende_geometry[1:]:
        route.append(
            np.asarray(
                [
                    (point["lon"] - WEST) / (EAST - WEST),
                    (point["lat"] - SOUTH) / (NORTH - SOUTH),
                ],
                dtype=np.float64,
            )
        )

    # Equal-distance samples make GPU progress proportional to actual travel
    # distance and retain the OSM bends without paying for hundreds of uneven
    # source vertices in every particle lookup.
    scale = np.asarray([longitude_meters, latitude_meters])
    segment_lengths = [
        float(np.linalg.norm((b - a) * scale))
        for a, b in zip(route, route[1:])
    ]
    cumulative = np.concatenate(
        [np.asarray([0.0]), np.cumsum(segment_lengths, dtype=np.float64)]
    )
    total_length = float(cumulative[-1])
    sample_count = max(2, math.ceil(total_length / 65.0) + 1)
    sample_distances = np.linspace(0.0, total_length, sample_count)
    result: list[list[float]] = []
    segment_index = 0
    for distance in sample_distances:
        while (
            segment_index + 1 < len(cumulative) - 1
            and cumulative[segment_index + 1] < distance
        ):
            segment_index += 1
        segment_length = max(segment_lengths[segment_index], 1e-9)
        t = min(
            max((distance - cumulative[segment_index]) / segment_length, 0.0),
            1.0,
        )
        point = route[segment_index] * (1.0 - t) + route[segment_index + 1] * t
        result.append([float(point[0]), float(point[1])])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--osm", type=Path, required=True)
    parser.add_argument("--waterways", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    grid = sample_grid(read_mosaic(args.tiles))
    port_elevation = sample_geo_height(grid, PORT_LAT, PORT_LON)
    source_elevation = sample_geo_height(grid, SOURCE_LAT, SOURCE_LON)
    flow_path = process_waterway_flow_path(grid, args.waterways)
    grid.tofile(args.output / "gyirong_terrain.f32")

    latitude_meters = (NORTH - SOUTH) * 110_540.0
    longitude_meters = (EAST - WEST) * 111_320.0 * math.cos(math.radians(PORT_LAT))
    metadata = {
        "version": 1,
        "terrain": {
            "width": GRID_WIDTH,
            "height": GRID_HEIGHT,
            "west": WEST,
            "east": EAST,
            "south": SOUTH,
            "north": NORTH,
            "physicalWidthMeters": longitude_meters,
            "physicalHeightMeters": latitude_meters,
            "heightFile": "gyirong_terrain.f32",
            "portDatumElevationMeters": port_elevation,
        },
        "event": {
            "portLatitude": PORT_LAT,
            "portLongitude": PORT_LON,
            "sourceLatitude": SOURCE_LAT,
            "sourceLongitude": SOURCE_LON,
            "sourceElevationMeters": source_elevation,
            "reportedMonitoringWaterLevelMeters": 7.0,
            "scenarioPeakDischargeCubicMetersPerSecond": 15000.0,
            "scenarioReleasedVolumeCubicMeters": 17300000.0,
            "scenarioHydraulicBoreDepthMeters": 12.0,
            "scenarioSprayHeightMeters": 25.0,
            "sourceStatus": "provisional",
            "eventDate": "2026-08-26",
        },
        "flowPathUV": flow_path,
        "buildings": process_buildings(args.osm),
    }
    (args.output / "gyirong_scene.json").write_text(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n"
    )

    print(
        f"terrain={GRID_WIDTH}x{GRID_HEIGHT} "
        f"elevation={grid.min():.1f}..{grid.max():.1f}m "
        f"port={port_elevation:.1f}m source={source_elevation:.1f}m "
        f"buildings={len(metadata['buildings'])} flowPath={len(flow_path)}"
    )


if __name__ == "__main__":
    main()
