# Gyirong debris-flow scene data

This folder contains a deliberately low-resolution reconstruction for the
August 26, 2026 Gyirong–Rasuwagadhi flood/debris-flow visualization.

- `gyirong_terrain.f32` is a 337 × 193 little-endian Float32 elevation grid.
  It was sampled from the public AWS Terrain Tiles Terrarium dataset (SRTM and
  other open elevation sources), zoom 12, over 28.20–28.36° N and
  85.29–85.63° E. This adds several kilometres of real terrain around every
  previous edge. Elevations remain in metres and no vertical exaggeration is
  applied. The runtime joins a very coarse procedural mountain apron beyond
  these downloaded bounds; that apron is contextual, not measured terrain.
- Building footprints in `gyirong_scene.json` come from OpenStreetMap via the
  Overpass API, queried on August 26, 2026. OpenStreetMap data is © OpenStreetMap
  contributors and available under the Open Database License (ODbL).
- The Donglin Tsangpo/Lende Khola centreline comes from the same OpenStreetMap
  dataset via Overpass, queried on August 27, 2026. The routed section records
  OSM way 937405875 explicitly so later map revisions can be audited.
- `flowPathUV` no longer solves directly from the source to the port. Its
  incompletely mapped upper tributary follows the lowest connected DEM route
  to the main-valley confluence at approximately 28.3306° N, 85.4311° E; the
  downstream section follows OpenStreetMap Donglin Tsangpo/Lende Khola way
  937405875 through every mapped turn to the port. The combined 23.1 km route is
  resampled at equal 65 m intervals. Runtime hydrologically conditions only a
  220 m band around it to remove false DEM dams caused by the approximately
  90–100 m cells. This is still a reconstruction guide, not a surveyed channel
  cross-section or an official inundation map.
- The procedural Chinese-side gate model is separate from the much larger Nepal
  Rasuwagadhi dry port. Its position, facade bearing and approximately 83 × 51 m
  footprint principal dimensions come from OpenStreetMap way 904894059,
  queried August 27, 2026. The resulting centre is 28.279511° N, 85.377742° E,
  about 149 m south of the earlier CCTV/plaza reference. Its approximately
  27 m height remains estimated from border-gate photographs and freight
  vehicles. OSM geometry and the regional border layout place the 160 × 126 m
  reconstructed flat inspection apron north of the gate and the river along
  its east side; this platform geometry is image-derived, not a cadastral
  survey. Runtime aligns reset to the Chinese approach and lowers only excess
  DEM terrain along a 60 m road floor with 180 m blended shoulders, because the
  approximately 90–100 m source cells cannot resolve the road-width valley.
- The provisional avalanche/source coordinate is 28.281051° N,
  85.545404° E. The port reference is 28.2809° N, 85.3779° E. These are early
  public assessments, not a final official reconstruction.
- The 7 m figure is an early reported monitoring-station water level. It is a
  scenario calibration input, not a verified peak inundation depth at every
  point in the port.
- The 15,000 m³/s peak, 17.3 million m³ volume, 12 m hydraulic bore, and 25 m
  spray/debris wall are a video-constrained central scenario. They are stored
  explicitly in `gyirong_scene.json` so later official hydrology can replace
  them. The wider visual-inference range is 5,000–40,000 m³/s; none of these
  values is presented as an official measurement.

Regenerate the two data files with:

```sh
python3 tools/generate_gyirong_data.py \
  --tiles /tmp/gyirong-terrain \
  --osm /tmp/gyirong-buildings.json \
  --waterways /tmp/gyirong-waterways.json \
  --output vr-dive/Demos/GyirongDebrisFlow/Data
```

Sources:

- AWS Terrain Tiles: https://registry.opendata.aws/terrain-tiles/
- OpenStreetMap copyright and licence: https://www.openstreetmap.org/copyright
- Gyirong gate footprint: https://www.openstreetmap.org/way/904894059
- Chinese-side 4,000 m² inspection/parking report:
  https://www.gzhsfy.gov.cn/web/content?gid=7428&lmdm=1029
- Chinese-side border-gate photographs:
  https://www.chinadaily.com.cn/a/202207/04/WS62c24798a310fd2b29e6a272.html
  and https://rikaze.xzdw.gov.cn/xwzx_455/ttxw/202602/t20260206_647111.html
- Nepal Rasuwagadhi dry-port scale (kept distinct from the gate model):
  https://www.myrepublica.nagariknetwork.com/news/agreement-to-build-rasuwagadhi-dry-port-signed
- Event CCTV and flood-front descriptions:
  https://www.ndtv.com/world-news/video-people-buses-buildings-engulfed-by-devastating-floods-in-nepal-11961906
  and https://www.rainews.it/video/2026/08/londa-di-fango-travolge-tutto-le-drammatiche-immagini-al-confine-tra-tibet-e-nepal-dbb66c2d-9ba3-4fe6-ba16-80cc7c3cc97c.html
- Narrow-valley context (reported minimum width above 100 m):
  https://www.jiemian.com/article/14999118.html
- Published debris-flow case using the Costa volume/peak-discharge relation:
  https://nhess.copernicus.org/articles/24/4179/2024/nhess-24-4179-2024.html
- USACE guidance on dam-breach parameter uncertainty:
  https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/6.6/performing-a-dam-break-study-with-hec-ras/estimating-dam-breach-parameters/estimating-breach-parameters
- Event reporting and uncertainty are documented in the demo-level README.
