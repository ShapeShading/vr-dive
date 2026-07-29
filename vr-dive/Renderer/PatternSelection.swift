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
  case lunarSurface
  case dynamicBox

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
    case .lunarSurface:
      return "月面日出"
    case .dynamicBox:
      return "动态着色器"
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
  case preset25
  case preset26
  case preset27
  case preset28
  case preset29
  case preset30
  case preset31
  case preset32
  case preset33
  case preset34
  case preset35
  case preset36
  case preset37
  case preset38
  case preset39
  case preset40
  case preset41
  case preset42
  case preset43
  case preset44
  case preset45
  case preset46
  case preset47
  case preset48

  var id: String { rawValue }

  var parameters: SIMD3<Float> {
    switch self {
    case .preset01: return SIMD3<Float>(1.462086, 1.582345, 4.446544)
    case .preset02: return SIMD3<Float>(1.683307, 4.972426, 3.582771)
    case .preset03: return SIMD3<Float>(5.871389, 5.575985, 4.540637)
    case .preset04: return SIMD3<Float>(5.729566, 3.817284, 4.553311)
    case .preset05: return SIMD3<Float>(1.649747, 5.027885, 2.701418)
    case .preset06: return SIMD3<Float>(3.043993, 5.618190, 5.022583)
    case .preset07: return SIMD3<Float>(2.268079, 2.239146, 0.972088)
    case .preset08: return SIMD3<Float>(2.219517, 3.718163, 2.364348)
    case .preset09: return SIMD3<Float>(1.694355, 1.152851, 0.253725)
    case .preset10: return SIMD3<Float>(2.884511, 3.356605, 3.901848)
    case .preset11: return SIMD3<Float>(6.031906, 2.902234, 0.751461)
    case .preset12: return SIMD3<Float>(2.886869, 0.679263, 1.103803)
    case .preset13: return SIMD3<Float>(1.539831, 4.250501, 2.498236)
    case .preset14: return SIMD3<Float>(2.446127, 2.632377, 5.380263)
    case .preset15: return SIMD3<Float>(5.405390, 2.501911, 0.979672)
    case .preset16: return SIMD3<Float>(6.052144, 0.824742, 2.061010)
    case .preset17: return SIMD3<Float>(2.867907, 5.795400, 4.079892)
    case .preset18: return SIMD3<Float>(5.974421, 0.540345, 1.046620)
    case .preset19: return SIMD3<Float>(6.027619, 0.351632, 5.311029)
    case .preset20: return SIMD3<Float>(2.887478, 5.906425, 2.155681)
    case .preset21: return SIMD3<Float>(5.891320, 0.170534, 5.447242)
    case .preset22: return SIMD3<Float>(1.686444, 2.189827, 6.093671)
    case .preset23: return SIMD3<Float>(3.121399, 3.344856, 5.179220)
    case .preset24: return SIMD3<Float>(5.094475, 2.717053, 1.188711)
    case .preset25: return SIMD3<Float>(2.518217, 0.373488, 1.103066)
    case .preset26: return SIMD3<Float>(2.291239, 0.706262, 0.910705)
    case .preset27: return SIMD3<Float>(0.169816, 5.603922, 3.566132)
    case .preset28: return SIMD3<Float>(6.274131, 2.554442, 1.865317)
    case .preset29: return SIMD3<Float>(1.553157, 1.992974, 0.508493)
    case .preset30: return SIMD3<Float>(4.769333, 1.977667, 0.440707)
    case .preset31: return SIMD3<Float>(6.188916, 2.833792, 2.160711)
    case .preset32: return SIMD3<Float>(5.469113, 2.699814, 1.400029)
    case .preset33: return SIMD3<Float>(2.558385, 3.678280, 4.381503)
    case .preset34: return SIMD3<Float>(1.533633, 4.114744, 2.849692)
    case .preset35: return SIMD3<Float>(1.528342, 2.377421, 5.858646)
    case .preset36: return SIMD3<Float>(2.296071, 0.215828, 1.012999)
    case .preset37: return SIMD3<Float>(6.080375, 0.335794, 0.774542)
    case .preset38: return SIMD3<Float>(5.082344, 5.744661, 4.525501)
    case .preset39: return SIMD3<Float>(1.511192, 5.328898, 2.576730)
    case .preset40: return SIMD3<Float>(1.574113, 1.611801, 1.126520)
    case .preset41: return SIMD3<Float>(4.554356, 4.590237, 4.561001)
    case .preset42: return SIMD3<Float>(5.462157, 2.635644, 0.589428)
    case .preset43: return SIMD3<Float>(5.085448, 3.114017, 5.354014)
    case .preset44: return SIMD3<Float>(5.658926, 0.728864, 1.045564)
    case .preset45: return SIMD3<Float>(1.554969, 5.916431, 3.247268)
    case .preset46: return SIMD3<Float>(2.173314, 5.838365, 2.480228)
    case .preset47: return SIMD3<Float>(2.157378, 3.550493, 3.812191)
    case .preset48: return SIMD3<Float>(5.313185, 0.365443, 5.689588)
    }
  }

  var pickerTitle: String {
    switch self {
    case .preset01: return "Orbital 01"
    case .preset02: return "Orbital 02"
    case .preset03: return "Orbital 03"
    case .preset04: return "Orbital 04"
    case .preset05: return "Orbital 05"
    case .preset06: return "Orbital 06"
    case .preset07: return "Orbital 07"
    case .preset08: return "Orbital 08"
    case .preset09: return "Orbital 09"
    case .preset10: return "Orbital 10"
    case .preset11: return "Orbital 11"
    case .preset12: return "Orbital 12"
    case .preset13: return "Orbital 13"
    case .preset14: return "Orbital 14"
    case .preset15: return "Orbital 15"
    case .preset16: return "Orbital 16"
    case .preset17: return "Orbital 17"
    case .preset18: return "Orbital 18"
    case .preset19: return "Orbital 19"
    case .preset20: return "Orbital 20"
    case .preset21: return "Orbital 21"
    case .preset22: return "Orbital 22"
    case .preset23: return "Orbital 23"
    case .preset24: return "Orbital 24"
    case .preset25: return "Orbital 25"
    case .preset26: return "Orbital 26"
    case .preset27: return "Orbital 27"
    case .preset28: return "Orbital 28"
    case .preset29: return "Sparse 29"
    case .preset30: return "Sparse 30"
    case .preset31: return "Sparse 31"
    case .preset32: return "Sparse 32"
    case .preset33: return "Sparse 33"
    case .preset34: return "Sparse 34"
    case .preset35: return "Sparse 35"
    case .preset36: return "Sparse 36"
    case .preset37: return "Sparse 37"
    case .preset38: return "Sparse 38"
    case .preset39: return "Sparse 39"
    case .preset40: return "Filament 40"
    case .preset41: return "Filament 41"
    case .preset42: return "Filament 42"
    case .preset43: return "Filament 43"
    case .preset44: return "Filament 44"
    case .preset45: return "Filament 45"
    case .preset46: return "Filament 46"
    case .preset47: return "Dense 47"
    case .preset48: return "Dense 48"
    }
  }

  var metricsSummary: String {
    switch self {
    case .preset01: return "vox 947 depth 0.086 orb 0.830"
    case .preset02: return "vox 869 depth 0.107 orb 0.905"
    case .preset03: return "vox 892 depth 0.101 orb 0.909"
    case .preset04: return "vox 883 depth 0.102 orb 0.900"
    case .preset05: return "vox 889 depth 0.112 orb 0.928"
    case .preset06: return "vox 862 depth 0.102 orb 0.892"
    case .preset07: return "vox 920 depth 0.083 orb 0.805"
    case .preset08: return "vox 919 depth 0.089 orb 0.847"
    case .preset09: return "vox 844 depth 0.107 orb 0.900"
    case .preset10: return "vox 890 depth 0.088 orb 0.860"
    case .preset11: return "vox 920 depth 0.091 orb 0.881"
    case .preset12: return "vox 928 depth 0.086 orb 0.854"
    case .preset13: return "vox 904 depth 0.103 orb 0.921"
    case .preset14: return "vox 900 depth 0.110 orb 0.928"
    case .preset15: return "vox 902 depth 0.097 orb 0.876"
    case .preset16: return "vox 896 depth 0.089 orb 0.866"
    case .preset17: return "vox 963 depth 0.089 orb 0.875"
    case .preset18: return "vox 892 depth 0.110 orb 0.953"
    case .preset19: return "vox 854 depth 0.111 orb 0.938"
    case .preset20: return "vox 915 depth 0.121 orb 0.930"
    case .preset21: return "vox 1026 depth 0.092 orb 0.873"
    case .preset22: return "vox 810 depth 0.113 orb 0.891"
    case .preset23: return "vox 1043 depth 0.084 orb 0.826"
    case .preset24: return "vox 858 depth 0.086 orb 0.828"
    case .preset25: return "vox 966 depth 0.110 orb 0.967"
    case .preset26: return "vox 901 depth 0.111 orb 0.924"
    case .preset27: return "vox 906 depth 0.100 orb 0.879"
    case .preset28: return "vox 985 depth 0.104 orb 0.955"
    case .preset29: return "vox 973 depth 0.079 orb 0.826"
    case .preset30: return "vox 850 depth 0.077 orb 0.774"
    case .preset31: return "vox 775 depth 0.080 orb 0.767"
    case .preset32: return "vox 954 depth 0.071 orb 0.776"
    case .preset33: return "vox 775 depth 0.088 orb 0.788"
    case .preset34: return "vox 965 depth 0.071 orb 0.802"
    case .preset35: return "vox 745 depth 0.087 orb 0.786"
    case .preset36: return "vox 796 depth 0.075 orb 0.746"
    case .preset37: return "vox 726 depth 0.081 orb 0.775"
    case .preset38: return "vox 892 depth 0.058 orb 0.717"
    case .preset39: return "vox 958 depth 0.064 orb 0.750"
    case .preset40: return "vox 913 depth 0.075 orb 0.742"
    case .preset41: return "vox 890 depth 0.072 orb 0.723"
    case .preset42: return "vox 1087 depth 0.084 orb 0.797"
    case .preset43: return "vox 762 depth 0.076 orb 0.724"
    case .preset44: return "vox 842 depth 0.080 orb 0.736"
    case .preset45: return "vox 858 depth 0.064 orb 0.687"
    case .preset46: return "vox 990 depth 0.063 orb 0.720"
    case .preset47: return "vox 1032 depth 0.066 orb 0.732"
    case .preset48: return "vox 1108 depth 0.071 orb 0.741"
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
  private var _current: VisualPatternKind = .lunarSurface
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

  // ─── DynamicBox shader loading ────────────────────────────────────────────
  /// Called by Renderer after DynamicBoxRenderer is created.
  func setDynamicBoxLoadAction(_ action: @escaping @MainActor (String) async -> Void) {
    queue.async(flags: .barrier) { self._dynamicBoxLoadAction = action }
  }

  func loadDynamicBoxShader(named name: String) {
    let action = queue.sync { self._dynamicBoxLoadAction }
    guard let action else { return }
    Task { @MainActor in
      await action(name)
    }
  }

  private var _dynamicBoxLoadAction: (@MainActor (String) async -> Void)?
  private var _dynamicBoxStatus: String = "default"

  /// Called by the Renderer's load action to report status back to the UI.
  func setDynamicBoxStatus(_ status: String) {
    queue.async(flags: .barrier) { self._dynamicBoxStatus = status }
  }

  func dynamicBoxStatus() -> String {
    queue.sync { _dynamicBoxStatus }
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
      if selectedPattern == .dynamicBox {
        refreshShaderList()
      }
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

  // ─── DynamicBox ──────────────────────────────────────────────────────────
  /// Shader server URL. Change this to your Mac's local IP if the device can't connect.
  static let shaderServerBaseURL = "http://192.168.31.49:8888"
  /// Available shader names fetched from the server.
  var dynamicBoxAvailableShaders: [String] = ["default"]
  /// Currently selected shader in the Picker.
  var dynamicBoxSelectedShader: String = "default" {
    didSet {
      if dynamicBoxSelectedShader != oldValue {
        loadDynamicBoxShader()
      }
    }
  }
  /// Displayed status: current shader name, "Loading…", or error text.
  var dynamicBoxStatus: String = "default"

  func loadDynamicBoxShader() {
    let name = dynamicBoxSelectedShader
    dynamicBoxStatus = "Loading…"
    coordinator.setDynamicBoxStatus("Loading…")
    coordinator.loadDynamicBoxShader(named: name)
  }

  func refreshShaderList() {
    Task {
      let url = URL(string: "\(Self.shaderServerBaseURL)/shaders")!
      guard let (data, response) = try? await URLSession.shared.data(from: url),
        let httpResp = response as? HTTPURLResponse,
        httpResp.statusCode == 200,
        let names = try? JSONDecoder().decode([String].self, from: data),
        !names.isEmpty
      else {
        print(
          "[DynamicBox] refreshShaderList: server at \(Self.shaderServerBaseURL) unreachable."
        )
        if dynamicBoxAvailableShaders.count <= 1 {
          dynamicBoxStatus = "服务器不可达 (\(Self.shaderServerBaseURL))"
        }
        return
      }
      // "default" is always available (embedded in app)
      var allNames = names
      if !allNames.contains("default") {
        allNames.insert("default", at: 0)
      }
      print("[DynamicBox] Shaders available: \(allNames)")
      dynamicBoxAvailableShaders = allNames
      if dynamicBoxAvailableShaders.count > 1 && dynamicBoxStatus.hasPrefix("服务器") {
        dynamicBoxStatus = "default"
      }
      // Ensure selected shader still exists; if not, fall back to first available
      if !allNames.contains(dynamicBoxSelectedShader) {
        dynamicBoxSelectedShader = allNames[0]
      }
    }
  }

  /// Call periodically from the UI to refresh DynamicBox status from the coordinator.
  func refreshDynamicBoxStatus() {
    let s = coordinator.dynamicBoxStatus()
    if s != dynamicBoxStatus {
      dynamicBoxStatus = s
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
