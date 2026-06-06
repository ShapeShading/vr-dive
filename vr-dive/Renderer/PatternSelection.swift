import Foundation
import Observation
import simd

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
  case simoneOrbit3D
  case apollonianTwist
  case steampunkOrb
  case apollonianWires
  case kuKo
  case sonicAndTails
  case followYourLight
  case weirdSurface
  case neonShells

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
    case .steampunkOrb:
      return "Steampunk Orb"
    case .apollonianWires:
      return "Apollonian Wires"
    case .kuKo:
      return "KuKo"
    case .sonicAndTails:
      return "Sonic & Tails"
    case .followYourLight:
      return "Follow Your Light"
    case .weirdSurface:
      return "Weird Surface"
    case .neonShells:
      return "Neon Shells"
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
    case .simoneOrbit3D:
      return "Simone Orbit 3D"
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

enum SimoneOrbit3DPreset: String, CaseIterable, Identifiable {
  case preset01
  case preset02
  case preset03
  case preset04
  case preset05
  case preset06
  case preset07
  case preset08
  case preset09
  case preset10
  case preset11
  case preset12
  case preset13
  case preset14
  case preset15
  case preset16
  case preset17
  case preset18
  case preset19
  case preset20
  case preset21
  case preset22
  case preset23
  case preset24

  var id: String { rawValue }

  var parameters: SIMD3<Float> {
    switch self {
    case .preset01: return SIMD3<Float>(2.203284, 3.682889, 2.468944)
    case .preset02: return SIMD3<Float>(1.628974, 0.930842, 0.349066)
    case .preset03: return SIMD3<Float>(2.327106, 5.817764, 4.537856)
    case .preset04: return SIMD3<Float>(1.369087, 4.212377, 2.810484)
    case .preset05: return SIMD3<Float>(4.654211, 3.723369, 0.465421)
    case .preset06: return SIMD3<Float>(1.548957, 1.995704, 2.219864)
    case .preset07: return SIMD3<Float>(3.331636, 3.911041, 4.126978)
    case .preset08: return SIMD3<Float>(2.095198, 5.750775, 4.322093)
    case .preset09: return SIMD3<Float>(1.237973, 0.912589, 1.669799)
    case .preset10: return SIMD3<Float>(2.660594, 3.259471, 3.987449)
    case .preset11: return SIMD3<Float>(5.119632, 2.559816, 1.279908)
    case .preset12: return SIMD3<Float>(3.142631, 4.374982, 3.507344)
    case .preset13: return SIMD3<Float>(2.625194, 0.433779, 2.202473)
    case .preset14: return SIMD3<Float>(5.872473, 5.984790, 5.495428)
    case .preset15: return SIMD3<Float>(1.073784, 2.655868, 4.320235)
    case .preset16: return SIMD3<Float>(2.700000, 2.320000, -2.510000)
    case .preset17: return SIMD3<Float>(2.700000, 2.320000, 2.510000)
    case .preset18: return SIMD3<Float>(5.817764, 3.956080, 0.930842)
    case .preset19: return SIMD3<Float>(2.792527, 0.465421, 1.163553)
    case .preset20: return SIMD3<Float>(2.222355, 5.498354, 1.969796)
    case .preset21: return SIMD3<Float>(1.419006, 2.241834, 4.075525)
    case .preset22: return SIMD3<Float>(1.636273, 1.173366, 5.913431)
    case .preset23: return SIMD3<Float>(3.257948, 6.050475, 4.886922)
    case .preset24: return SIMD3<Float>(3.025237, 5.585054, 5.003277)
    }
  }

  var pickerTitle: String {
    switch self {
    case .preset01: return "Filament 01"
    case .preset02: return "Sparse 02"
    case .preset03: return "Filament 03"
    case .preset04: return "Filament 04"
    case .preset05: return "Filament 05"
    case .preset06: return "Filament 06"
    case .preset07: return "Orbital 07"
    case .preset08: return "Sparse 08"
    case .preset09: return "Sparse 09"
    case .preset10: return "Sparse 10"
    case .preset11: return "Sparse 11"
    case .preset12: return "Filament 12"
    case .preset13: return "Sparse 13"
    case .preset14: return "Sparse 14"
    case .preset15: return "Sparse 15"
    case .preset16: return "Filament 16"
    case .preset17: return "Filament 17"
    case .preset18: return "Sparse 18"
    case .preset19: return "Sparse 19"
    case .preset20: return "Sparse 20"
    case .preset21: return "Filament 21"
    case .preset22: return "Sparse 22"
    case .preset23: return "Sparse 23"
    case .preset24: return "Sparse 24"
    }
  }

  var metricsSummary: String {
    switch self {
    case .preset01: return "vox 883 depth 0.068"
    case .preset02: return "vox 707 depth 0.086"
    case .preset03: return "vox 846 depth 0.058"
    case .preset04: return "vox 848 depth 0.087"
    case .preset05: return "vox 887 depth 0.038"
    case .preset06: return "vox 858 depth 0.036"
    case .preset07: return "vox 751 depth 0.089"
    case .preset08: return "vox 764 depth 0.056"
    case .preset09: return "vox 768 depth 0.044"
    case .preset10: return "vox 569 depth 0.075"
    case .preset11: return "vox 594 depth 0.074"
    case .preset12: return "vox 875 depth 0.023"
    case .preset13: return "vox 884 depth 0.004"
    case .preset14: return "vox 881 depth 0.018"
    case .preset15: return "vox 886 depth 0.002"
    case .preset16: return "vox 871 depth 0.002"
    case .preset17: return "vox 866 depth 0.002"
    case .preset18: return "vox 872 depth 0.003"
    case .preset19: return "vox 867 depth 0.111"
    case .preset20: return "vox 578 depth 0.079"
    case .preset21: return "vox 812 depth 0.012"
    case .preset22: return "vox 826 depth 0.102"
    case .preset23: return "vox 807 depth 0.005"
    case .preset24: return "vox 801 depth 0.122"
    }
  }

  func next() -> SimoneOrbit3DPreset {
    let all = Self.allCases
    guard let index = all.firstIndex(of: self) else { return self }
    return all[(index + 1) % all.count]
  }

  func previous() -> SimoneOrbit3DPreset {
    let all = Self.allCases
    guard let index = all.firstIndex(of: self) else { return self }
    return all[(index + all.count - 1) % all.count]
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
  private var _current: VisualPatternKind = .simoneOrbit3D
  private var _isPaused: Bool = false
  private var _shouldReset: Bool = false
  private var _speedMultiplier: Float = 1.0
  private var _originCellInspectionEnabled: Bool = false
  private var _rayMarchingProbeDimTarget: RayMarchingProbeDimTarget = .none
  private var _huashanSampleRatio: Float = PatternCoordinator.defaultHuashanSampleRatio
  private var _simoneOrbit3DPreset: SimoneOrbit3DPreset = .preset01

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

  func simoneOrbit3DPreset() -> SimoneOrbit3DPreset {
    queue.sync { _simoneOrbit3DPreset }
  }

  func setSimoneOrbit3DPreset(_ preset: SimoneOrbit3DPreset) {
    queue.async(flags: .barrier) { self._simoneOrbit3DPreset = preset }
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

  var simoneOrbit3DPreset: SimoneOrbit3DPreset = .preset01 {
    didSet {
      coordinator.setSimoneOrbit3DPreset(simoneOrbit3DPreset)
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
    self.simoneOrbit3DPreset = coordinator.simoneOrbit3DPreset()
  }

  func refreshFromCoordinator() {
    selectedPattern = coordinator.currentPattern()
    isPaused = coordinator.isPaused()
    originCellInspectionEnabled = coordinator.originCellInspectionEnabled()
    rayMarchingProbeDimTarget = coordinator.rayMarchingProbeDimTarget()
    huashanSampleRatio = coordinator.huashanSampleRatio()
    simoneOrbit3DPreset = coordinator.simoneOrbit3DPreset()
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

  var simoneOrbit3DPrincipleText: String {
    "离线脚本现在优先筛选 filament 型 3D 轨道: x'=sin(x²-y²-z²+a), y'=cos(2xy+b), z'=sin(2xz+c)。面板里的预设更偏向可见曲线骨架，而不是高密度云团。"
  }
}
