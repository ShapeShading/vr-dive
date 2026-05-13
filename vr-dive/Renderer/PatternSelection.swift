import Foundation
import Observation

enum VisualPatternKind: String, CaseIterable, Identifiable {
  case pongWar
  case cubeField
  case fiveCellProjection
  case eightCellProjection
  case sixteenCellProjection
  case twentyFourCellProjection
  case oneHundredTwentyCellProjection
  case sixHundredCellProjection
  case lorenzAttractor
  case fourWingAttractor
  case aizawaAttractor
  case julia3D
  case pagoda
  case tetris3D
  case snake3D
  case rhombicDodecahedron
  case quatPolynomial
  case huashan
  case rayMarchingDemo
  case cubeRayMarchDemo
  case metaball
  case synthwaveSunset
  case tunnel
  case interferenceCascadeCube
  case cubicSpaceDivision
  case voxelEdges
  case pathTilesCube
  case gyroidEchoCube
  case waveLatticeCube
  case waveySpheres
  case glowingMountainLines
  case magnetar
  case spiraledLayers
  case angleFire
  case orbitalSphereCube
  case apollonianIIv4
  case platonicMirror
  case glassBox
  case cartoonFractalCube
  case fractalFlythrough
  case hyperbolicGroupLimitSet
  case apolloSpiral
  case voxelTunnel
  case nearLoxodrome
  case shield
  case digitalLines
  case cloudyCrystal
  case shaderdoughFairy
  case crystalCubeLatticinioCore1
  case fireTornado
  case reflectiveWythoffPolyhedra
  case apollonian
  case magneticLinesThatDrawInGold
  case lanterns
  case laceTunnel
  case torusFan
  case apollonianElevator
  case torusKnotInR4
  case threeDFire
  case bubbleRings
  case ether
  case fiberSpiral
  case saturdayTorus
  case tesseractCornerFractal
  case hexwaves
  case milosRose
  case recursiveLotus
  case blueFlower
  case flowerTest
  case floreus
  case sineBud
  case saturdayWeirdness
  case soulstone
  case boxOfStars
  case mirrorLooping
  case greatDodecaheadroll
  case playingMarble
  case novaMarble
  case dirtBall
  case fractal49Gaz
  case starryPlanes
  case fractal77Gaz
  case poincareBallHoneycomb
  case anotherMarble
  case marbleMovingRemix
  case slicesInMarbles
  case logSphericalKIFSZoomer
  case petalsFractal
  case goldenApollian
  case tunnelingThroughApollianFrac
  case fractalCity
  case starTrails
  case particleRain
  case apollonianTwist

  var id: String { rawValue }

  var supportsOriginCellInspection: Bool {
    switch self {
    case .fiveCellProjection, .eightCellProjection, .sixteenCellProjection,
      .twentyFourCellProjection,
      .oneHundredTwentyCellProjection, .sixHundredCellProjection:
      return true
    default:
      return false
    }
  }

  var displayName: String {
    switch self {
    case .pongWar:
      return "PongWar"
    case .cubeField:
      return "杂物空间"
    case .fiveCellProjection:
      return "正五胞体投影"
    case .eightCellProjection:
      return "正八胞体投影"
    case .sixteenCellProjection:
      return "正十六胞体投影"
    case .twentyFourCellProjection:
      return "正二十四胞体投影"
    case .oneHundredTwentyCellProjection:
      return "正一百二十胞体投影"
    case .sixHundredCellProjection:
      return "正六百胞体投影"
    case .lorenzAttractor:
      return "Lorenz 吸引子"
    case .fourWingAttractor:
      return "Four-Wing Attractor"
    case .aizawaAttractor:
      return "Aizawa 吸引子"
    case .julia3D:
      return "Julia 3D"
    case .pagoda:
      return "大雁塔"
    case .tetris3D:
      return "3D 俄罗斯方块"
    case .snake3D:
      return "3D 贪食蛇"
    case .rhombicDodecahedron:
      return "菱形十二面体镜室"
    case .metaball:
      return "圆球水滴"
    case .quatPolynomial:
      return "四元数多项式根"
    case .glassBox:
      return "玻璃魔方"
    case .platonicMirror:
      return "反射多面体"
    case .synthwaveSunset:
      return "合成波日落"
    case .tunnel:
      return "Tunnel"
    case .cubicSpaceDivision:
      return "Cubic Space Division"
    case .voxelEdges:
      return "Voxel Edges"
    case .pathTilesCube:
      return "PathTilesCube"
    case .cartoonFractalCube:
      return "CartoonFractalCube"
    case .gyroidEchoCube:
      return "GyroidEchoCube"
    case .waveLatticeCube:
      return "WaveLatticeCube"
    case .waveySpheres:
      return "Wavey spheres"
    case .fractalFlythrough:
      return "Fractal Flythrough"
    case .apollonianIIv4:
      return "Apollonian-II-v4"
    case .magnetar:
      return "Magnetar"
    case .spiraledLayers:
      return "Spiraled Layers"
    case .angleFire:
      return "Angle Fire"
    case .glowingMountainLines:
      return "Glowing Mountain Lines"
    case .interferenceCascadeCube:
      return "InterferenceCascadeCube"
    case .orbitalSphereCube:
      return "OrbitalSphereCube"
    case .huashan:
      return "华山3D扫描"
    case .rayMarchingDemo:
      return "Ray Marching 演示"
    case .cubeRayMarchDemo:
      return "方块内 Ray Marching"
    case .hyperbolicGroupLimitSet:
      return "Hyperbolic Group Limit Set"
    case .apolloSpiral:
      return "Apollo Spiral"
    case .voxelTunnel:
      return "Voxel tunnel"
    case .nearLoxodrome:
      return "Near Loxodrome"
    case .shield:
      return "Shield"
    case .digitalLines:
      return "Digital Lines"
    case .cloudyCrystal:
      return "Cloudy Crystal"
    case .shaderdoughFairy:
      return "Shaderdough Fairy"
    case .crystalCubeLatticinioCore1:
      return "Crystal Cube Latticinio core 1"
    case .fireTornado:
      return "Fire Tornado"
    case .reflectiveWythoffPolyhedra:
      return "Reflective Wythoff polyhedra"
    case .apollonian:
      return "apollonian"
    case .apollonianTwist:
      return "Apollonian Twist"
    case .magneticLinesThatDrawInGold:
      return "Magnetic lines that draw in gold"
    case .lanterns:
      return "Lanterns"
    case .laceTunnel:
      return "Lace Tunnel"
    case .torusFan:
      return "Torus fan"
    case .apollonianElevator:
      return "Apollonian Elevator"
    case .torusKnotInR4:
      return "Torus Knot in ℝ⁴"
    case .threeDFire:
      return "3D Fire"
    case .bubbleRings:
      return "Bubble rings"
    case .ether:
      return "Ether"
    case .fiberSpiral:
      return "Fiber Spiral"
    case .saturdayTorus:
      return "Saturday Torus"
    case .tesseractCornerFractal:
      return "Tesseract Corner Fractal"
    case .hexwaves:
      return "hexwaves"
    case .milosRose:
      return "Milo's Rose"
    case .recursiveLotus:
      return "Recursive Lotus"
    case .blueFlower:
      return "Blue Flower"
    case .flowerTest:
      return "Flower Test"
    case .floreus:
      return "Floreus"
    case .sineBud:
      return "Sine bud"
    case .saturdayWeirdness:
      return "Saturday weirdness"
    case .soulstone:
      return "Soulstone"
    case .boxOfStars:
      return "Box of Stars"
    case .mirrorLooping:
      return "Mirror Looping"
    case .greatDodecaheadroll:
      return "Great Dodecaheadroll"
    case .playingMarble:
      return "Playing marble"
    case .novaMarble:
      return "Nova Marble"
    case .dirtBall:
      return "Dirt Ball"
    case .fractal49Gaz:
      return "Fractal 49_gaz"
    case .starryPlanes:
      return "Starry planes"
    case .fractal77Gaz:
      return "Fractal 77_gaz"
    case .poincareBallHoneycomb:
      return "Poincare Ball Honeycomb"
    case .anotherMarble:
      return "Another Marble"
    case .marbleMovingRemix:
      return "marble moving remix"
    case .slicesInMarbles:
      return "slices in marbles"
    case .logSphericalKIFSZoomer:
      return "Log Spherical KIFS Zoomer"
    case .petalsFractal:
      return "Petals Fractal"
    case .goldenApollian:
      return "Golden apollian"
    case .tunnelingThroughApollianFrac:
      return "Tunneling through apollian frac"
    case .fractalCity:
      return "Fractal city"
    case .starTrails:
      return "星轨延时"
    case .particleRain:
      return "流光雨"
    }
  }
}

enum RayMarchingProbeDimTarget: Int, CaseIterable {
  case none
  case sphere
  case torus

  var buttonTitle: String {
    switch self {
    case .none:
      return "灰化: 无"
    case .sphere:
      return "灰化: 球"
    case .torus:
      return "灰化: 圆环"
    }
  }

  func next() -> RayMarchingProbeDimTarget {
    switch self {
    case .none:
      return .sphere
    case .sphere:
      return .torus
    case .torus:
      return .none
    }
  }
}

final class PatternCoordinator {
  static let minHuashanSampleRatio: Float = 0.05
  static let maxHuashanSampleRatio: Float = 1.0
  static let defaultHuashanSampleRatio: Float = 0.45

  static func clampedHuashanSampleRatio(_ ratio: Float) -> Float {
    min(max(ratio, minHuashanSampleRatio), maxHuashanSampleRatio)
  }

  private let queue = DispatchQueue(label: "vr-dive.pattern.coordinator", attributes: .concurrent)
  private var _current: VisualPatternKind = .apollonianTwist
  private var _isPaused: Bool = false
  private var _shouldReset: Bool = false
  private var _speedMultiplier: Float = 1.0
  private var _originCellInspectionEnabled: Bool = false
  private var _rayMarchingProbeDimTarget: RayMarchingProbeDimTarget = .none
  private var _huashanSampleRatio: Float = PatternCoordinator.defaultHuashanSampleRatio

  func currentPattern() -> VisualPatternKind {
    queue.sync { _current }
  }

  func setPattern(_ pattern: VisualPatternKind) {
    queue.async(flags: .barrier) { self._current = pattern }
  }

  func isPaused() -> Bool {
    queue.sync { _isPaused }
  }

  func setPaused(_ paused: Bool) {
    queue.async(flags: .barrier) { self._isPaused = paused }
  }

  func shouldReset() -> Bool {
    queue.sync { _shouldReset }
  }

  func triggerReset() {
    queue.async(flags: .barrier) { self._shouldReset = true }
  }

  func clearResetFlag() {
    queue.async(flags: .barrier) { self._shouldReset = false }
  }

  func speedMultiplier() -> Float {
    queue.sync { _speedMultiplier }
  }

  func setSpeedMultiplier(_ multiplier: Float) {
    queue.async(flags: .barrier) { self._speedMultiplier = multiplier }
  }

  func originCellInspectionEnabled() -> Bool {
    queue.sync { _originCellInspectionEnabled }
  }

  func setOriginCellInspectionEnabled(_ enabled: Bool) {
    queue.async(flags: .barrier) { self._originCellInspectionEnabled = enabled }
  }

  func rayMarchingProbeDimTarget() -> RayMarchingProbeDimTarget {
    queue.sync { _rayMarchingProbeDimTarget }
  }

  func setRayMarchingProbeDimTarget(_ target: RayMarchingProbeDimTarget) {
    queue.async(flags: .barrier) { self._rayMarchingProbeDimTarget = target }
  }

  func huashanSampleRatio() -> Float {
    queue.sync { _huashanSampleRatio }
  }

  func setHuashanSampleRatio(_ ratio: Float) {
    let clamped = Self.clampedHuashanSampleRatio(ratio)
    queue.async(flags: .barrier) { self._huashanSampleRatio = clamped }
  }
}

@MainActor
@Observable
final class PatternMenuModel {
  var selectedPattern: VisualPatternKind {
    didSet {
      if !selectedPattern.supportsOriginCellInspection && originCellInspectionEnabled {
        originCellInspectionEnabled = false
      }
      coordinator.setPattern(selectedPattern)
    }
  }

  var isPaused: Bool = false {
    didSet {
      coordinator.setPaused(isPaused)
    }
  }

  var speedMultiplier: Float = 1.0 {
    didSet {
      coordinator.setSpeedMultiplier(speedMultiplier)
    }
  }

  var originCellInspectionEnabled: Bool = false {
    didSet {
      coordinator.setOriginCellInspectionEnabled(originCellInspectionEnabled)
      if originCellInspectionEnabled {
        isPaused = true
      }
    }
  }

  var rayMarchingProbeDimTarget: RayMarchingProbeDimTarget = .none {
    didSet {
      coordinator.setRayMarchingProbeDimTarget(rayMarchingProbeDimTarget)
    }
  }

  var huashanSampleRatio: Float = PatternCoordinator.defaultHuashanSampleRatio {
    didSet {
      coordinator.setHuashanSampleRatio(huashanSampleRatio)
    }
  }

  private let coordinator: PatternCoordinator

  init(coordinator: PatternCoordinator) {
    self.coordinator = coordinator
    self.selectedPattern = coordinator.currentPattern()
    self.isPaused = coordinator.isPaused()
    self.originCellInspectionEnabled = coordinator.originCellInspectionEnabled()
    self.rayMarchingProbeDimTarget = coordinator.rayMarchingProbeDimTarget()
    self.huashanSampleRatio = coordinator.huashanSampleRatio()
  }

  func refreshFromCoordinator() {
    selectedPattern = coordinator.currentPattern()
    isPaused = coordinator.isPaused()
    originCellInspectionEnabled = coordinator.originCellInspectionEnabled()
    rayMarchingProbeDimTarget = coordinator.rayMarchingProbeDimTarget()
    huashanSampleRatio = coordinator.huashanSampleRatio()
  }

  func reset() {
    coordinator.triggerReset()
  }

  func toggleSpeed() {
    speedMultiplier = speedMultiplier > 1.0 ? 1.0 : 5.0
  }

  func cycleRayMarchingProbeDimTarget() {
    rayMarchingProbeDimTarget = rayMarchingProbeDimTarget.next()
  }

  func adjustHuashanSampleRatio(by delta: Float) {
    huashanSampleRatio = PatternCoordinator.clampedHuashanSampleRatio(huashanSampleRatio + delta)
  }

  var huashanSampleRatioPercentText: String {
    "\(Int((huashanSampleRatio * 100).rounded()))%"
  }
}
