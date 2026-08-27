import Metal
import simd

struct GyirongTerrainMetadata: Decodable {
  let width: Int
  let height: Int
  let west: Double
  let east: Double
  let south: Double
  let north: Double
  let physicalWidthMeters: Double
  let physicalHeightMeters: Double
  let heightFile: String
  let portDatumElevationMeters: Double
}

struct GyirongEventMetadata: Decodable {
  let portLatitude: Double
  let portLongitude: Double
  let sourceLatitude: Double
  let sourceLongitude: Double
  let sourceElevationMeters: Double
  let reportedMonitoringWaterLevelMeters: Double
  let scenarioPeakDischargeCubicMetersPerSecond: Double
  let scenarioReleasedVolumeCubicMeters: Double
  let scenarioHydraulicBoreDepthMeters: Double
  let scenarioSprayHeightMeters: Double
  let sourceStatus: String
  let eventDate: String
}

struct GyirongBuildingMetadata: Decodable {
  let sourceElementID: Int64
  let heightMeters: Float
  let footprintLonLat: [[Double]]
}

struct GyirongSceneMetadata: Decodable {
  let version: Int
  let terrain: GyirongTerrainMetadata
  let event: GyirongEventMetadata
  let flowPathUV: [[Float]]
  let buildings: [GyirongBuildingMetadata]
}

/// Shared CPU/GPU contract. Keep this layout synchronized with the Metal struct.
struct GyirongDebrisFlowUniforms {
  var viewCount: UInt32
  var gridWidth: UInt32
  var gridHeight: UInt32
  /// Number of DEM-derived flood-path points in the GPU path buffer.
  var flags: UInt32

  var simulationTime: Float
  var simulationDelta: Float
  var navigationSpeedScale: Float
  var particleCount: UInt32

  /// north span, east span, port datum elevation, reported water level (metres)
  var terrainSizeAndDatum: SIMD4<Float>
  /// port u/v and provisional source u/v in the terrain grid
  var portSourceUV: SIMD4<Float>
  /// local scene offset; y places port ground below the initial headset position
  var sceneOrigin: SIMD4<Float>
  /// breach start, routed travel time, hydraulic bore depth, spray/debris height
  var floodScenario: SIMD4<Float>
  /// peak discharge, released volume, effective channel width, reserved
  var floodHydrology: SIMD4<Float>
  /// surveyed gate u/v, reconstructed terrace elevation, overflow influence radius
  var facilityUVAndElevation: SIMD4<Float>
  /// gate local across east/north, followed by forward east/north
  var facilityFrame: SIMD4<Float>
  /// Inverse of the accelerated pattern-navigation transform for mesh rendering.
  var navigationInverse: simd_float4x4
}

struct GyirongMeshVertex {
  var position: SIMD3<Float>
  var normal: SIMD3<Float>
  var color: SIMD4<Float>
}

struct GyirongWaterVertex {
  var uv: SIMD2<Float>
}

struct GyirongFlowParticle {
  /// u, v, vertical spray offset in metres, remaining life in seconds
  var uvHeightLife: SIMD4<Float>
  /// east velocity, north velocity, vertical velocity, deterministic seed
  var velocitySeed: SIMD4<Float>
}

struct GyirongMeshLOD {
  let vertexBuffer: MTLBuffer
  let indexBuffer: MTLBuffer
  let indexCount: Int
  let sampleStep: Int
}

struct GyirongTerrainTile {
  let centerSceneXZ: SIMD2<Float>
  /// Tiles intersecting the routed river stay at the highest terrain LOD and
  /// omit vertical skirts, which would otherwise appear as walls in the gorge.
  let isFlowCorridor: Bool
  let lods: [GyirongMeshLOD]
}
