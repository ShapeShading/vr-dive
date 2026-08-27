# Gyirong–Rasuwagadhi debris-flow reconstruction

This is a low-resolution, interactive first reconstruction of the August 26,
2026 event. It is designed for spatial exploration, not operational hazard
forecasting or forensic attribution.

## Current evidence used

- China's Ministry of Emergency Management reported that the disaster began on
  the Nepal side and caused major casualties and missing persons at Gyirong Port
  at about 10:30 China Standard Time.
- AP, citing ICIMOD, reported the early interpretation that a Nepal-side
  ice-rock avalanche blocked the Lhende Khola and released a sudden surge.
- A preliminary scientist report placed the possible source near
  28.281051° N, 85.545404° E and mentioned a 7 m monitoring-station level. Both
  values remain provisional.
- Nepal's Department of Hydrology and Meteorology and independent specialists
  continued to list an avalanche-dam failure, glacial-lake outburst, or a
  compound event as possibilities on the day of the event.

## Simulation model

- All terrain, building, water-height, and velocity values use one 1:1 metre
  scale. There is no vertical exaggeration.
- Terrain is a 337 × 193 sample of public elevation data covering roughly
  33.3 × 17.7 km, split into 66 spatial tiles. Each tile independently selects
  1×, 2×, 4×, or 8× sampling according
  to horizontal camera distance. Nearby tiles therefore retain every available
  DEM sample and add clamped Catmull-Rom half-cell interpolation, replacing
  each large source triangle with eight curved sub-triangles. Remote mountains
  use substantially fewer triangles. Terrain
  is rendered two-sided. The 18 tiles intersecting the routed river stay on the
  highest half-cell LOD regardless of camera distance and omit vertical edge
  skirts. Tiles outside this river band retain only a shallow 4 m overlap to
  close T-junction cracks where adjacent tiles select different sampling
  rates; the former 160 m skirts produced visible canyon-like tile walls and
  were removed. Terrain triangles are no longer deleted beneath the port, so
  the black, grid-aligned holes visible on device cannot expose the clear color.
  Beyond the downloaded DEM, a kilometre-scale procedural apron is joined to
  the real boundary elevations with a small overlap and continues broad
  mountain ridges towards the horizon, avoiding an abrupt map-edge cliff.
- Terrain colour follows the contrast visible in current aerial reporting:
  dark green forest and alpine vegetation at lower elevations, grey-brown bare
  rock on steep faces, bright retained snow above roughly 4,700 m, and a
  blue-white glacier component in flatter basins above roughly 5,250 m.
  Four-octave world-space noise adds forest, scrub, meadow and exposed-rock
  mottling continuously across tile boundaries instead of assigning one flat
  colour to each coarse terrain triangle. A finer procedural normal field adds
  metre-scale surface relief without increasing the distant triangle budget.
- A closed 48 × 24-segment, 200 km physical sky dome is drawn as ordinary
  world-space geometry with a grey overcast horizon-to-zenith gradient. It uses
  exactly the same scene presentation, navigation and stereo projection path
  as the terrain. The dome is rendered first and writes its finite 200 km
  reverse-Z depth across every opaque sky pixel; closer terrain and structures
  then replace it through the normal greater-depth test. This is required on
  device because Compositor Services presents an opaque color pixel only when
  it also has a valid perspective depth for reprojection. Earlier versions
  wrote grey color but deliberately left the sky depth at the zero clear value,
  which the simulator tolerated but the device exposed as black foveation
  tiles. The dome does not depend on clear color, untouched reverse-Z depth, an
  eye-local transform, or a clip-space fullscreen primitive. The device reports
  an infinite far plane and a 0.1 m near plane; the previous diagnostic
  labelled those two values backwards. The higher tessellation also avoids
  asking six enormous cube-face triangles to cover the nonuniform foveated
  viewport.
- 144 OpenStreetMap building footprints near the port are extruded using tagged
  heights/levels when available and conservative defaults otherwise.
- The Chinese border gate is anchored to OpenStreetMap way 904894059 at
  28.279511° N, 85.377742° E. Its measured footprint principal axes are about
  83 × 51 m, with the long facade at a 70.47° bearing. This corrects the
  former CCTV-reference placement, which was about
  149 m too far north, and rotates the long facade onto the mapped building
  bearing instead of into the mountainside. The procedural gate is about 84 m
  wide, 30 m deep and 27 m high, with an open central portal, four rows of
  recessed windows, facade bands and stone/glass material detail. Height is
  still inferred from imagery and freight-vehicle scale, not surveyed.
- OSM geometry and the regional north/south border layout place the broad
  Chinese inspection apron north of the gate, with the river immediately along its eastern
  edge. A flat 160 × 126 m structural terrace around the documented gate and
  near inspection apron now sits over continuously conditioned terrain rather
  than a deleted mesh region. It includes lane markings, booths, a
  west-side service building and a segmented river-edge barrier. The former
  invented elevated parking deck was removed.
- The presentation frame is aligned to the gate itself rather than the old
  CCTV coordinate: reset places the gate 650 m ahead, centres the portal, looks
  from the Chinese approach into the Nepal-side valley, and puts the mapped
  river on the viewer's right. The former implementation used the correct
  facade line but reversed the short-axis sign, placing the Chinese apron on
  the Nepal side; the corrected positive-forward axis now points south-southeast
  through the portal toward Nepal. A continuous 16 m road ribbon passes through the
  portal. Because the source DEM cells are wider than the road, a 60 m valley
  floor with 180 m blended shoulders lowers only obstructing terrain for 520 m
  on the Chinese side and 420 m into the Nepal-side valley.
- 52 scattered 1.7 m columns around the gate road and apron provide a human
  scale reference corresponding to the dozens of people visible before impact
  in the CCTV. They are not intended to identify or reproduce individuals.
- A two-dimensional shallow-water approximation evolves water depth, east/north
  velocity, friction, and sediment concentration over the DEM. The provisional
  source injects a synthetic hydrograph calibrated around the early 7 m gauge
  report. Reset starts an event timeline: about 6 real seconds of snow/rock
  avalanche, about 7 seconds of blockage and impoundment, breach at roughly
  13 seconds, then about 66 seconds of valley routing before the bore reaches
  the port. Because the roughly 90–100 m DEM cells cannot preserve the narrow
  river continuously, the route is built in two evidence tiers. From the
  provisional source it follows a minimum-uphill DEM drainage route to the
  mapped upper-valley confluence at approximately 28.3306° N, 85.4311° E.
  From there it follows OpenStreetMap Donglin Tsangpo/Lende Khola way
  937405875 point-for-point through its turns to the port, rather than solving
  a direct source-to-port line. The resulting 23.1 km route has 359 equal 65 m
  samples. A 220 m-wide hydrological-conditioning band imposes a monotonically
  descending centre floor where the 90–100 m DEM would otherwise sample an
  adjacent canyon wall; terrain outside that narrow band is unchanged. The
  routed corridor supplements the shallow-water grid while retaining local
  gravity, depth, velocity, friction, continuity, and sediment updates.
  In the final two percent of the route, the guide widens only around the
  mapped port. The gate lies about 97 m from the mapped centreline, below the
  source DEM's cell size. The terminal overtopping field is evaluated in the
  gate's measured local frame: a dominant branch passes the viewer-right/east
  side of the building, a 56%-strength branch passes the left side, and a
  smaller central spill enters the portal. All three continue downstream
  toward China and retain shallow-water depth, velocity, friction and sediment
  updates rather than drawing a source-to-building shortcut.
- The central unvalidated flood scenario is 15,000 m³/s peak discharge and
  17.3 million m³ released volume. The hydraulic core of the advancing bore is
  represented as about 12 m deep, with entrained spray, snow dust, sediment and
  debris particles reaching about 25 m. Most particles are concentrated into a
  broad moving front wall, with smaller clasts and spray retained in the wake,
  so the flow path remains visible at full mountain scale. The apparent seven-to-eight-storey
  video wall is therefore not treated as 25 m of static water. A broad visual
  cross-section estimate gives an uncertainty range of roughly
  5,000–40,000 m³/s; no official discharge or volume was available on the event
  date. The central volume is consistent in order of magnitude with the Costa
  barrier-lake peak-discharge relation, but it is not a measurement.
  Peak flow and volume also define a roughly 2,307 simulated-second triangular
  source hydrograph. An effective 100 m active width and 12 m bore depth are
  used to derive an 18.4 m/s section velocity. Adding shallow-water celerity
  gives about 29.2 m/s over the 23.1 km routed path, which sets the 790-second
  simulated travel time instead of choosing front speed separately.
- 196,608 physical debris carriers (24 times the original count) each render
  four decorrelated opaque tetrahedral micro-clasts, producing 786,432 visible
  fragments—four times the previous visible density. Tetrahedra retain hard
  faces, real depth occlusion and deterministic tumbling while using 12 rather
  than 24 vertices per fragment. Their size is reduced again and four replicas
  receive small independent side/height/forward offsets. Up to nine short
  physics substeps per frame apply water drag,
  gravity along the conditioned DEM slope, density-dependent vertical gravity,
  turbulence, surface collision, restitution and friction. The mapped river
  tangent and corridor-gradient force are also applied at every substep, so
  particles turn inside the gorge instead of taking a ballistic shortcut
  through a mountain spur. New particles spawn within 42 m of only the part of
  the route already reached by the flood front; in the final port reach the
  spawn distribution assigns roughly 62% of new fragments to the right-side
  branch, 26% to the left and 12% to the portal. Invalid half-float water cells
  are sanitized before simulation and rendering so unstable NaN cells cannot
  appear as large black water-grid polygons.
- Pattern-navigation translation is multiplied by 250 for practical travel over
  the regional corridor. Either L1 or R1 applies a 16× boost (4,000× effective);
  holding both stacks it twice for 256× (64,000× effective). This changes
  navigation speed only; scene geometry stays at 1:1 scale. Reset restores the
  aerial origin and restarts the event from the avalanche.

## Important limitations

The source coordinate, release volume, hydrograph, sediment rheology, peak
depth, and building condition are not yet field-validated. The 30–100 m terrain
sampling cannot resolve individual channels, levees, bridges, or building-scale
flow interaction. Update the metadata and solver calibration when official
satellite analysis, surveyed cross-sections, and gauge hydrographs become
available.

## Sources

- https://www.mem.gov.cn/xw/yjglbgzdt/202608/t20260826_708718.shtml
- https://apnews.com/article/climate-change-nepal-flash-floods-7262dac22e31258955efa5c28c8fe917
- https://kathmandupost.com/national/2026/08/26/scientists-suspect-ice-avalanche-triggered-bhotekoshi-flood
- https://thetourismtimes.com/news/t3-special/avalanche-blocks-lhende-river-in-tibet-causing-bhote-koshi-flood-say-initial-reports
- https://english.onlinekhabar.com/cause-rasuwa-flood-unclear.html
- https://registry.opendata.aws/terrain-tiles/
- https://www.openstreetmap.org/copyright
- https://www.openstreetmap.org/way/904894059
- https://www.gzhsfy.gov.cn/web/content?gid=7428&lmdm=1029
- https://www.chinadaily.com.cn/a/202207/04/WS62c24798a310fd2b29e6a272.html
- https://rikaze.xzdw.gov.cn/xwzx_455/ttxw/202602/t20260206_647111.html
- https://www.ndtv.com/world-news/video-people-buses-buildings-engulfed-by-devastating-floods-in-nepal-11961906
- https://commons.wikimedia.org/wiki/File:China_Gyirong_Port_of_Entry_building.png
- https://www.globaltimes.cn/galleries/2587.html
- https://www.rainews.it/video/2026/08/londa-di-fango-travolge-tutto-le-drammatiche-immagini-al-confine-tra-tibet-e-nepal-dbb66c2d-9ba3-4fe6-ba16-80cc7c3cc97c.html
- https://www.jiemian.com/article/14999118.html
- https://nhess.copernicus.org/articles/24/4179/2024/nhess-24-4179-2024.html
- https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/6.6/performing-a-dam-break-study-with-hec-ras/estimating-dam-breach-parameters/estimating-breach-parameters
