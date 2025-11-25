import Metal
import simd

private typealias MeshBuffers = (vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, indexCount: Int)

private struct EdgeVertex {
  var position: SIMD3<Float>
  var normal: SIMD3<Float>
  var axisMask: SIMD3<Float>
  var center: SIMD3<Float>
}

private struct PongWarBall {
  var position: SIMD3<Float>
  var velocity: SIMD3<Float>
  var color: SIMD3<Float>
  var zoneIndex: Int
}

final class PongWarRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .pongWar
  let preferredClearColor = MTLClearColor(red: 0.01, green: 0.01, blue: 0.02, alpha: 1)

  private let pipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  private let cubeVertexBuffer: MTLBuffer
  private let cubeIndexBuffer: MTLBuffer
  private let sphereVertexBuffer: MTLBuffer
  private let sphereIndexBuffer: MTLBuffer
  private let cubeInstanceBuffer: MTLBuffer
  private let sphereInstanceBuffer: MTLBuffer
  private let cubeIndexCount: Int
  private let sphereIndexCount: Int

  private var cubeStates: [PongWarInstanceState]
  private var sphereStates: [PongWarInstanceState]
  private var balls: [PongWarBall]
  private var cubeZones: [UInt8]
  private var cubeStateDirty = true

  private let device: MTLDevice
  private let maxViewCount: Int

  private var lastSimulationTimestamp: Float = 0

  private let gridDimension = 16  // 16x16x16 = 4096个格子，性能可接受
  private let worldCubeSize: Float = 4.0  // 恢复到4
  private let minSpeed: Float = 2.5
  private let maxSpeed: Float = 5.0
  private let velocityNoise: Float = 0.2
  private let boundaryEdgeScale: Float = 0.65
  private let interiorEdgeScale: Float = 0.012

  // 8种高对比度颜色（在黑色背景上清晰可辨，彼此差异最大化）
  // 使用色轮均匀分布 + 亮度变化，确保任意两色都有足够对比
  private let zoneColors: [SIMD3<Float>] = [
    PongWarRenderer.colorFromHex("#FF3333"),  // 0: 亮红
    PongWarRenderer.colorFromHex("#33AAFF"),  // 1: 天蓝
    PongWarRenderer.colorFromHex("#33FF55"),  // 2: 亮绿
    PongWarRenderer.colorFromHex("#FFAA22"),  // 3: 橙色
    PongWarRenderer.colorFromHex("#AA44FF"),  // 4: 亮紫
    PongWarRenderer.colorFromHex("#22DDDD"),  // 5: 青色
    PongWarRenderer.colorFromHex("#FF55AA"),  // 6: 玫红
    PongWarRenderer.colorFromHex("#FFFF44"),  // 7: 亮黄
  ]

  private lazy var effectUniforms = PongWarUniforms(
    edgeHighlight: 0.45,
    baseGlow: 0.25,
    ballGlow: 0.55
  )

  private let cellSize: Float
  private let halfWorldSize: Float
  private let ballRadius: Float

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.device = device
    self.maxViewCount = max(1, maxViewCount)
    cellSize = worldCubeSize / Float(gridDimension)
    halfWorldSize = worldCubeSize * 0.5
    ballRadius = cellSize * 0.45  // 小球半径

    pipelineState = try PongWarRenderer.makePipelineState(
      device: device,
      library: library,
      maxViewCount: self.maxViewCount
    )
    depthStencilState = PongWarRenderer.makeDepthStencilState(device: device)

    let edgeGeometry = PongWarRenderer.makeEdgeGeometry(device: device)
    cubeVertexBuffer = edgeGeometry.vertexBuffer
    cubeIndexBuffer = edgeGeometry.indexBuffer
    cubeIndexCount = edgeGeometry.indexCount

    let sphereGeometry = PongWarRenderer.makeSphereGeometry(device: device, segments: 24, stacks: 16)
    sphereVertexBuffer = sphereGeometry.vertexBuffer
    sphereIndexBuffer = sphereGeometry.indexBuffer
    sphereIndexCount = sphereGeometry.indexCount

    cubeZones = PongWarRenderer.makeInitialZones(gridDimension: gridDimension)
    balls = PongWarRenderer.makeInitialBalls(
      gridDimension: gridDimension,
      center: .zero,
      halfWorld: halfWorldSize,
      colors: zoneColors
    )

    cubeStates = PongWarRenderer.makeCubeStates(
      gridDimension: gridDimension,
      center: .zero,
      cellSize: cellSize,
      zones: cubeZones,
      zoneColors: zoneColors,
      boundaryThickness: boundaryEdgeScale,
      interiorThickness: interiorEdgeScale
    )

    sphereStates = PongWarRenderer.makeSphereInstances(from: balls, radius: ballRadius)

    cubeInstanceBuffer = device.makeBuffer(
      bytes: cubeStates,
      length: MemoryLayout<PongWarInstanceState>.stride * cubeStates.count,
      options: [.storageModeShared]
    )!

    sphereInstanceBuffer = device.makeBuffer(
      bytes: sphereStates,
      length: MemoryLayout<PongWarInstanceState>.stride * sphereStates.count,
      options: [.storageModeShared]
    )!
  }

  func updateSimulation(_ context: PatternSimulationContext) {
    let elapsed = context.time - lastSimulationTimestamp
    if lastSimulationTimestamp == 0 {
      lastSimulationTimestamp = context.time
      return
    }

    let clampedDelta = max(1.0 / 240.0, min(elapsed, 1.0 / 45.0))
    integrateBalls(deltaTime: clampedDelta)
    updateSphereInstances()

    if cubeStateDirty {
      cubeStates = PongWarRenderer.makeCubeStates(
        gridDimension: gridDimension,
        center: .zero,
        cellSize: cellSize,
        zones: cubeZones,
        zoneColors: zoneColors,
        boundaryThickness: boundaryEdgeScale,
        interiorThickness: interiorEdgeScale
      )
      cubeStateDirty = false
    }
    
    // 每帧更新外边界显示状态（基于小球位置）
    updateOuterBoundaryVisibility()
    
    cubeStates.withUnsafeBytes { buffer in
      guard let base = buffer.baseAddress else { return }
      memcpy(cubeInstanceBuffer.contents(), base, buffer.count)
    }

    sphereStates.withUnsafeBytes { buffer in
      guard let base = buffer.baseAddress else { return }
      memcpy(sphereInstanceBuffer.contents(), base, buffer.count)
    }

    lastSimulationTimestamp = context.time
  }
  
  /// 更新外边界棱的可见性：只有当小球接近特定小立方体时才显示该立方体的外边界棱
  func updateOuterBoundaryVisibility() {
    let threshold = cellSize * 1.0  // 小球距离小立方体一个格子长度以内才显示
    
    // 重置所有外边界显示状态
    for i in cubeStates.indices {
      cubeStates[i].edgeData.z = 0
    }
    
    // 对每个边界上的小立方体，检查是否有小球足够近
    for idx in 0..<cubeStates.count {
      let (x, y, z) = PongWarRenderer.indices(for: idx, grid: gridDimension)
      
      // 计算该立方体的外边界掩码（哪些面在大立方体边界上）
      var outerMask: UInt8 = 0
      if x == 0 { outerMask |= 1 << 0 }                    // -X面
      if x == gridDimension - 1 { outerMask |= 1 << 1 }   // +X面
      if y == 0 { outerMask |= 1 << 2 }                    // -Y面
      if y == gridDimension - 1 { outerMask |= 1 << 3 }   // +Y面
      if z == 0 { outerMask |= 1 << 4 }                    // -Z面
      if z == gridDimension - 1 { outerMask |= 1 << 5 }   // +Z面
      
      // 如果不是边界立方体，跳过
      if outerMask == 0 { continue }
      
      // 获取该立方体的中心位置
      let ps = cubeStates[idx].positionAndScale
      let cubePos = SIMD3<Float>(ps.x, ps.y, ps.z)
      
      // 检查是否有小球足够近
      var activeMask: UInt8 = 0
      for ball in balls {
        let dist = simd_distance(ball.position, cubePos)
        if dist < threshold {
          // 小球足够近，显示该立方体的外边界棱
          activeMask = outerMask
          break
        }
      }
      
      cubeStates[idx].edgeData.z = Float(activeMask)
    }
  }

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.back)
    encoder.setFrontFacing(.counterClockwise)
    encoder.setTriangleFillMode(.fill)

    context.applyViewConfiguration(on: encoder)

    var sceneUniforms = SceneUniforms(
      time: context.time,
      layerCount: UInt32(context.viewData.viewCount)
    )
    var viewMatrices = context.viewData.viewProjectionMatrices
    if viewMatrices.isEmpty {
      viewMatrices = [matrix_identity_float4x4]
    }

    var uniforms = effectUniforms

    encoder.setVertexBytes(&sceneUniforms, length: MemoryLayout<SceneUniforms>.stride, index: 2)
    viewMatrices.withUnsafeBytes {
      if let base = $0.baseAddress { encoder.setVertexBytes(base, length: $0.count, index: 3) }
    }
    encoder.setVertexBytes(&uniforms, length: MemoryLayout<PongWarUniforms>.stride, index: 4)
    encoder.setFragmentBytes(&sceneUniforms, length: MemoryLayout<SceneUniforms>.stride, index: 0)
    encoder.setFragmentBytes(&uniforms, length: MemoryLayout<PongWarUniforms>.stride, index: 1)

    encoder.setRenderPipelineState(pipelineState)

    encoder.setVertexBuffer(cubeVertexBuffer, offset: 0, index: 0)
    encoder.setVertexBuffer(cubeInstanceBuffer, offset: 0, index: 1)
    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: cubeIndexCount,
      indexType: .uint16,
      indexBuffer: cubeIndexBuffer,
      indexBufferOffset: 0,
      instanceCount: cubeStates.count
    )

    encoder.setVertexBuffer(sphereVertexBuffer, offset: 0, index: 0)
    encoder.setVertexBuffer(sphereInstanceBuffer, offset: 0, index: 1)
    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: sphereIndexCount,
      indexType: .uint16,
      indexBuffer: sphereIndexBuffer,
      indexBufferOffset: 0,
      instanceCount: sphereStates.count
    )
  }

  func resetToInitialState() {
    cubeZones = PongWarRenderer.makeInitialZones(gridDimension: gridDimension)
    balls = PongWarRenderer.makeInitialBalls(
      gridDimension: gridDimension,
      center: .zero,
      halfWorld: halfWorldSize,
      colors: zoneColors
    )
    cubeStateDirty = true
    lastSimulationTimestamp = 0
  }
}

private extension PongWarRenderer {
  static func makePipelineState(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: "pongWarVertexShader")
    descriptor.fragmentFunction = library.makeFunction(name: "pongWarFragmentShader")
    descriptor.colorAttachments[0].pixelFormat = .rgba16Float
    descriptor.depthAttachmentPixelFormat = .depth32Float
    descriptor.inputPrimitiveTopology = .triangle

    let vertexDescriptor = MTLVertexDescriptor()
    vertexDescriptor.attributes[0].format = .float3
    vertexDescriptor.attributes[0].offset = 0
    vertexDescriptor.attributes[0].bufferIndex = 0

    vertexDescriptor.attributes[1].format = .float3
    vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
    vertexDescriptor.attributes[1].bufferIndex = 0

    vertexDescriptor.attributes[2].format = .float3
    vertexDescriptor.attributes[2].offset = MemoryLayout<SIMD3<Float>>.stride * 2
    vertexDescriptor.attributes[2].bufferIndex = 0

    vertexDescriptor.attributes[3].format = .float3
    vertexDescriptor.attributes[3].offset = MemoryLayout<SIMD3<Float>>.stride * 3
    vertexDescriptor.attributes[3].bufferIndex = 0

    vertexDescriptor.layouts[0].stride = MemoryLayout<EdgeVertex>.stride
    descriptor.vertexDescriptor = vertexDescriptor
    descriptor.maxVertexAmplificationCount = max(maxViewCount, 1)

    return try device.makeRenderPipelineState(descriptor: descriptor)
  }

  static func makeDepthStencilState(device: MTLDevice) -> MTLDepthStencilState {
    let descriptor = MTLDepthStencilDescriptor()
    descriptor.depthCompareFunction = .greater
    descriptor.isDepthWriteEnabled = true
    return device.makeDepthStencilState(descriptor: descriptor)!
  }

  static func makeInitialZones(gridDimension: Int) -> [UInt8] {
    let total = gridDimension * gridDimension * gridDimension
    return (0..<total).map { index in
      let (x, y, z) = PongWarRenderer.indices(for: index, grid: gridDimension)
      let zone = PongWarRenderer.zoneIndex(x: x, y: y, z: z, grid: gridDimension)
      return UInt8(zone)
    }
  }

  static func makeInitialBalls(
    gridDimension: Int,
    center: SIMD3<Float>,
    halfWorld: Float,
    colors: [SIMD3<Float>]
  ) -> [PongWarBall] {
    precondition(colors.count >= 8)
    var balls: [PongWarBall] = []
    balls.reserveCapacity(8)

    // offsets顺序必须与zoneIndex匹配：index = (xBit<<2) | (yBit<<1) | zBit
    // 0: x<0,y<0,z<0  1: x<0,y<0,z>0  2: x<0,y>0,z<0  3: x<0,y>0,z>0
    // 4: x>0,y<0,z<0  5: x>0,y<0,z>0  6: x>0,y>0,z<0  7: x>0,y>0,z>0
    let offsets: [SIMD3<Float>] = [
      SIMD3(-0.5, -0.5, -0.5),  // 0: ---
      SIMD3(-0.5, -0.5,  0.5),  // 1: --+
      SIMD3(-0.5,  0.5, -0.5),  // 2: -+-
      SIMD3(-0.5,  0.5,  0.5),  // 3: -++
      SIMD3( 0.5, -0.5, -0.5),  // 4: +--
      SIMD3( 0.5, -0.5,  0.5),  // 5: +-+
      SIMD3( 0.5,  0.5, -0.5),  // 6: ++-
      SIMD3( 0.5,  0.5,  0.5),  // 7: +++
    ]

    for (index, offset) in offsets.enumerated() {
      // 小球在各自象限的中心位置
      let position = center + offset * (halfWorld * 0.5)
      // 随机方向，速度一致
      let randomDir = simd_normalize(SIMD3<Float>(
        Float.random(in: -1...1),
        Float.random(in: -1...1),
        Float.random(in: -1...1)
      ))
      let speed: Float = 3.0
      let velocity = randomDir * speed
      balls.append(
        PongWarBall(
          position: position,
          velocity: velocity,
          color: colors[index],
          zoneIndex: index
        )
      )
    }

    return balls
  }

  func integrateBalls(deltaTime: Float) {
    for idx in balls.indices {
      updateBall(at: idx, deltaTime: deltaTime)
    }
  }

  func updateBall(at index: Int, deltaTime: Float) {
    var ball = balls[index]
    let ballZone = UInt8(ball.zoneIndex)
    
    // 预测下一步位置
    var nextPos = ball.position + ball.velocity * deltaTime
    var velocity = ball.velocity
    
    // 1. 检查外边界碰撞
    let bounds = halfWorldSize - ballRadius
    if nextPos.x > bounds {
      nextPos.x = bounds
      velocity.x = -velocity.x
    } else if nextPos.x < -bounds {
      nextPos.x = -bounds
      velocity.x = -velocity.x
    }
    if nextPos.y > bounds {
      nextPos.y = bounds
      velocity.y = -velocity.y
    } else if nextPos.y < -bounds {
      nextPos.y = -bounds
      velocity.y = -velocity.y
    }
    if nextPos.z > bounds {
      nextPos.z = bounds
      velocity.z = -velocity.z
    } else if nextPos.z < -bounds {
      nextPos.z = -bounds
      velocity.z = -velocity.z
    }
    
    // 2. 检查是否进入不同颜色区域
    if let (cx, cy, cz, linear) = cellIndex(for: nextPos) {
      let targetZone = cubeZones[linear]
      if targetZone != ballZone {
        // 碰到不同颜色的立方体，变色并反弹
        cubeZones[linear] = ballZone
        cubeStateDirty = true
        
        // 根据进入方向反弹
        let cellCenter = worldPosition(x: cx, y: cy, z: cz)
        let fromCenter = ball.position - cellCenter
        let absFrom = simd_abs(fromCenter)
        
        if absFrom.x >= absFrom.y && absFrom.x >= absFrom.z {
          velocity.x = -velocity.x
          nextPos.x = ball.position.x  // 不进入该格子
        } else if absFrom.y >= absFrom.x && absFrom.y >= absFrom.z {
          velocity.y = -velocity.y
          nextPos.y = ball.position.y
        } else {
          velocity.z = -velocity.z
          nextPos.z = ball.position.z
        }
      }
    }
    
    // 更新位置和速度
    ball.position = nextPos
    ball.velocity = velocity
    
    // 添加随机扰动
    ball.velocity += SIMD3<Float>(
      Float.random(in: -velocityNoise...velocityNoise),
      Float.random(in: -velocityNoise...velocityNoise),
      Float.random(in: -velocityNoise...velocityNoise)
    ) * deltaTime
    
    // 限制速度范围
    let speed = simd_length(ball.velocity)
    if speed < minSpeed {
      ball.velocity = simd_normalize(ball.velocity) * minSpeed
    } else if speed > maxSpeed {
      ball.velocity = simd_normalize(ball.velocity) * maxSpeed
    }
    
    balls[index] = ball
  }

  func updateSphereInstances() {
    sphereStates = balls.map { ball in
      PongWarInstanceState(
        positionAndScale: SIMD4<Float>(ball.position, ballRadius),
        color: SIMD4<Float>(ball.color, 1),
        edgeData: SIMD4<Float>(Float(ball.zoneIndex), 1, 0, 1)
      )
    }
  }

  func cellIndex(for position: SIMD3<Float>) -> (Int, Int, Int, Int)? {
    let local = position + SIMD3<Float>(repeating: halfWorldSize)
    guard local.x >= 0, local.y >= 0, local.z >= 0,
      local.x <= worldCubeSize, local.y <= worldCubeSize, local.z <= worldCubeSize
    else { return nil }

    let fx = min(Float(gridDimension - 1), max(0, local.x / worldCubeSize * Float(gridDimension)))
    let fy = min(Float(gridDimension - 1), max(0, local.y / worldCubeSize * Float(gridDimension)))
    let fz = min(Float(gridDimension - 1), max(0, local.z / worldCubeSize * Float(gridDimension)))

    let ix = Int(fx)
    let iy = Int(fy)
    let iz = Int(fz)
    let linear = PongWarRenderer.linearIndex(x: ix, y: iy, z: iz, grid: gridDimension)
    return (ix, iy, iz, linear)
  }

  func worldPosition(x: Int, y: Int, z: Int) -> SIMD3<Float> {
    let start = -halfWorldSize + cellSize * 0.5
    return SIMD3<Float>(
      start + Float(x) * cellSize,
      start + Float(y) * cellSize,
      start + Float(z) * cellSize
    )
  }

  static func makeCubeStates(
    gridDimension: Int,
    center: SIMD3<Float>,
    cellSize: Float,
    zones: [UInt8],
    zoneColors: [SIMD3<Float>],
    boundaryThickness: Float,
    interiorThickness: Float
  ) -> [PongWarInstanceState] {
    var states: [PongWarInstanceState] = []
    states.reserveCapacity(zones.count)
    let halfWorld = cellSize * Float(gridDimension) * 0.5

    // 获取6个方向的边界掩码：哪些面与不同颜色相邻
    // bit0: -X, bit1: +X, bit2: -Y, bit3: +Y, bit4: -Z, bit5: +Z
    func getBoundaryMask(x: Int, y: Int, z: Int, zone: UInt8) -> UInt8 {
      var mask: UInt8 = 0
      let neighbors: [(Int, Int, Int, UInt8)] = [
        (x - 1, y, z, 1 << 0),  // -X
        (x + 1, y, z, 1 << 1),  // +X
        (x, y - 1, z, 1 << 2),  // -Y
        (x, y + 1, z, 1 << 3),  // +Y
        (x, y, z - 1, 1 << 4),  // -Z
        (x, y, z + 1, 1 << 5),  // +Z
      ]
      for (nx, ny, nz, bit) in neighbors {
        // 超出边界的邻居不计入（外边界由动态更新处理）
        guard nx >= 0, ny >= 0, nz >= 0,
              nx < gridDimension, ny < gridDimension, nz < gridDimension
        else { continue }
        
        let idx = PongWarRenderer.linearIndex(x: nx, y: ny, z: nz, grid: gridDimension)
        if zones[idx] != zone {
          mask |= bit
        }
      }
      return mask
    }

    for index in 0..<zones.count {
      let (x, y, z) = PongWarRenderer.indices(for: index, grid: gridDimension)
      let position = SIMD3<Float>(
        -halfWorld + cellSize * (Float(x) + 0.5),
        -halfWorld + cellSize * (Float(y) + 0.5),
        -halfWorld + cellSize * (Float(z) + 0.5)
      ) + center

      let zone = zones[index]
      let color = zoneColors[Int(zone)]
      let boundaryMask = getBoundaryMask(x: x, y: y, z: z, zone: zone)

      states.append(
        PongWarInstanceState(
          positionAndScale: SIMD4<Float>(position, cellSize),
          color: SIMD4<Float>(color, 1),
          // edgeData: (zone, boundaryMask, outerMask(动态更新), isSphere=0)
          edgeData: SIMD4<Float>(Float(zone), Float(boundaryMask), 0, 0)
        )
      )
    }

    return states
  }

  static func makeSphereInstances(from balls: [PongWarBall], radius: Float) -> [PongWarInstanceState] {
    balls.map { ball in
      PongWarInstanceState(
        positionAndScale: SIMD4<Float>(ball.position, radius),
        color: SIMD4<Float>(ball.color, 1),
        edgeData: SIMD4<Float>(Float(ball.zoneIndex), 1, 0, 1)
      )
    }
  }

  static func linearIndex(x: Int, y: Int, z: Int, grid: Int) -> Int {
    return (z * grid * grid) + (y * grid) + x
  }

  static func indices(for index: Int, grid: Int) -> (Int, Int, Int) {
    let x = index % grid
    let y = (index / grid) % grid
    let z = index / (grid * grid)
    return (x, y, z)
  }

  static func zoneIndex(x: Int, y: Int, z: Int, grid: Int) -> Int {
    let half = grid / 2
    let xBit = x >= half ? 1 : 0
    let yBit = y >= half ? 1 : 0
    let zBit = z >= half ? 1 : 0
    return (xBit << 2) | (yBit << 1) | zBit
  }

  static func colorFromHex(_ hex: String) -> SIMD3<Float> {
    var value: UInt64 = 0
    Scanner(string: hex.replacingOccurrences(of: "#", with: "")).scanHexInt64(&value)
    let r = Float((value >> 16) & 0xFF) / 255
    let g = Float((value >> 8) & 0xFF) / 255
    let b = Float(value & 0xFF) / 255
    return SIMD3<Float>(r, g, b)
  }
}

private extension PongWarRenderer {
  static func makeEdgeGeometry(device: MTLDevice) -> MeshBuffers {
    var vertices: [EdgeVertex] = []
    var indices: [UInt16] = []
    
    let halfLength: Float = 0.5  // 满尺寸，margin在shader中根据粗细应用
    let edgeHalfThickness: Float = 0.06  // 棱的粗细

    // 每条棱属于两个面，用 faceMask 编码
    // bit0: -X, bit1: +X, bit2: -Y, bit3: +Y, bit4: -Z, bit5: +Z
    func appendBox(center: SIMD3<Float>, halfSize: SIMD3<Float>, axisMask: SIMD3<Float>, faceMask: SIMD3<Float>) {
      let corners: [SIMD3<Float>] = [
        SIMD3(-halfSize.x, -halfSize.y, halfSize.z),
        SIMD3(halfSize.x, -halfSize.y, halfSize.z),
        SIMD3(halfSize.x, halfSize.y, halfSize.z),
        SIMD3(-halfSize.x, halfSize.y, halfSize.z),
        SIMD3(-halfSize.x, -halfSize.y, -halfSize.z),
        SIMD3(halfSize.x, -halfSize.y, -halfSize.z),
        SIMD3(halfSize.x, halfSize.y, -halfSize.z),
        SIMD3(-halfSize.x, halfSize.y, -halfSize.z),
      ]

      let faces: [(SIMD3<Float>, [Int])] = [
        (SIMD3(0, 0, 1), [0, 1, 2, 3]),
        (SIMD3(0, 0, -1), [5, 4, 7, 6]),
        (SIMD3(-1, 0, 0), [4, 0, 3, 7]),
        (SIMD3(1, 0, 0), [1, 5, 6, 2]),
        (SIMD3(0, 1, 0), [3, 2, 6, 7]),
        (SIMD3(0, -1, 0), [4, 5, 1, 0]),
      ]

      for (normal, faceIndices) in faces {
        let base = UInt16(vertices.count)
        for idx in faceIndices {
          let corner = corners[idx] + center
          // axisMask 存储棱的方向，center 存储 faceMask（两个面的掩码）
          vertices.append(
            EdgeVertex(position: corner, normal: normal, axisMask: axisMask, center: faceMask)
          )
        }
        indices.append(contentsOf: [
          base + 0, base + 1, base + 2,
          base + 0, base + 2, base + 3,
        ])
      }
    }

    // 12条棱，每条棱属于两个面
    // X轴方向的4条棱（位于 Y-Z 平面的四角）
    // 棱在 y=-0.5, z=-0.5: 属于 -Y(bit2) 和 -Z(bit4) 面 -> faceMask = 4+16=20
    appendBox(center: SIMD3(0, -halfLength, -halfLength), 
              halfSize: SIMD3(halfLength, edgeHalfThickness, edgeHalfThickness), 
              axisMask: SIMD3(1, 0, 0), 
              faceMask: SIMD3(20, 0, 0))  // -Y, -Z
    // 棱在 y=+0.5, z=-0.5: 属于 +Y(bit3) 和 -Z(bit4) 面 -> faceMask = 8+16=24
    appendBox(center: SIMD3(0, halfLength, -halfLength), 
              halfSize: SIMD3(halfLength, edgeHalfThickness, edgeHalfThickness), 
              axisMask: SIMD3(1, 0, 0), 
              faceMask: SIMD3(24, 0, 0))  // +Y, -Z
    // 棱在 y=-0.5, z=+0.5: 属于 -Y(bit2) 和 +Z(bit5) 面 -> faceMask = 4+32=36
    appendBox(center: SIMD3(0, -halfLength, halfLength), 
              halfSize: SIMD3(halfLength, edgeHalfThickness, edgeHalfThickness), 
              axisMask: SIMD3(1, 0, 0), 
              faceMask: SIMD3(36, 0, 0))  // -Y, +Z
    // 棱在 y=+0.5, z=+0.5: 属于 +Y(bit3) 和 +Z(bit5) 面 -> faceMask = 8+32=40
    appendBox(center: SIMD3(0, halfLength, halfLength), 
              halfSize: SIMD3(halfLength, edgeHalfThickness, edgeHalfThickness), 
              axisMask: SIMD3(1, 0, 0), 
              faceMask: SIMD3(40, 0, 0))  // +Y, +Z

    // Y轴方向的4条棱（位于 X-Z 平面的四角）
    // 棱在 x=-0.5, z=-0.5: 属于 -X(bit0) 和 -Z(bit4) 面 -> faceMask = 1+16=17
    appendBox(center: SIMD3(-halfLength, 0, -halfLength), 
              halfSize: SIMD3(edgeHalfThickness, halfLength, edgeHalfThickness), 
              axisMask: SIMD3(0, 1, 0), 
              faceMask: SIMD3(17, 0, 0))  // -X, -Z
    // 棱在 x=+0.5, z=-0.5: 属于 +X(bit1) 和 -Z(bit4) 面 -> faceMask = 2+16=18
    appendBox(center: SIMD3(halfLength, 0, -halfLength), 
              halfSize: SIMD3(edgeHalfThickness, halfLength, edgeHalfThickness), 
              axisMask: SIMD3(0, 1, 0), 
              faceMask: SIMD3(18, 0, 0))  // +X, -Z
    // 棱在 x=-0.5, z=+0.5: 属于 -X(bit0) 和 +Z(bit5) 面 -> faceMask = 1+32=33
    appendBox(center: SIMD3(-halfLength, 0, halfLength), 
              halfSize: SIMD3(edgeHalfThickness, halfLength, edgeHalfThickness), 
              axisMask: SIMD3(0, 1, 0), 
              faceMask: SIMD3(33, 0, 0))  // -X, +Z
    // 棱在 x=+0.5, z=+0.5: 属于 +X(bit1) 和 +Z(bit5) 面 -> faceMask = 2+32=34
    appendBox(center: SIMD3(halfLength, 0, halfLength), 
              halfSize: SIMD3(edgeHalfThickness, halfLength, edgeHalfThickness), 
              axisMask: SIMD3(0, 1, 0), 
              faceMask: SIMD3(34, 0, 0))  // +X, +Z

    // Z轴方向的4条棱（位于 X-Y 平面的四角）
    // 棱在 x=-0.5, y=-0.5: 属于 -X(bit0) 和 -Y(bit2) 面 -> faceMask = 1+4=5
    appendBox(center: SIMD3(-halfLength, -halfLength, 0), 
              halfSize: SIMD3(edgeHalfThickness, edgeHalfThickness, halfLength), 
              axisMask: SIMD3(0, 0, 1), 
              faceMask: SIMD3(5, 0, 0))  // -X, -Y
    // 棱在 x=+0.5, y=-0.5: 属于 +X(bit1) 和 -Y(bit2) 面 -> faceMask = 2+4=6
    appendBox(center: SIMD3(halfLength, -halfLength, 0), 
              halfSize: SIMD3(edgeHalfThickness, edgeHalfThickness, halfLength), 
              axisMask: SIMD3(0, 0, 1), 
              faceMask: SIMD3(6, 0, 0))  // +X, -Y
    // 棱在 x=-0.5, y=+0.5: 属于 -X(bit0) 和 +Y(bit3) 面 -> faceMask = 1+8=9
    appendBox(center: SIMD3(-halfLength, halfLength, 0), 
              halfSize: SIMD3(edgeHalfThickness, edgeHalfThickness, halfLength), 
              axisMask: SIMD3(0, 0, 1), 
              faceMask: SIMD3(9, 0, 0))  // -X, +Y
    // 棱在 x=+0.5, y=+0.5: 属于 +X(bit1) 和 +Y(bit3) 面 -> faceMask = 2+8=10
    appendBox(center: SIMD3(halfLength, halfLength, 0), 
              halfSize: SIMD3(edgeHalfThickness, edgeHalfThickness, halfLength), 
              axisMask: SIMD3(0, 0, 1), 
              faceMask: SIMD3(10, 0, 0))  // +X, +Y

    let vertexBuffer = device.makeBuffer(
      bytes: vertices,
      length: MemoryLayout<EdgeVertex>.stride * vertices.count,
      options: [.storageModeShared]
    )!
    let indexBuffer = device.makeBuffer(
      bytes: indices,
      length: MemoryLayout<UInt16>.stride * indices.count,
      options: [.storageModeShared]
    )!

    return (vertexBuffer, indexBuffer, indices.count)
  }

  static func makeSphereGeometry(device: MTLDevice, segments: Int, stacks: Int) -> MeshBuffers {
    var vertices: [EdgeVertex] = []
    var indices: [UInt16] = []

    for stack in 0...stacks {
      let v = Float(stack) / Float(stacks)
      let phi = v * Float.pi
      let y = cos(phi)
      let sinPhi = sin(phi)
      for segment in 0...segments {
        let u = Float(segment) / Float(segments)
        let theta = u * 2 * Float.pi
        let x = sinPhi * cos(theta)
        let z = sinPhi * sin(theta)
        let normal = simd_normalize(SIMD3<Float>(x, y, z))
        vertices.append(
          EdgeVertex(
            position: normal * 0.5,
            normal: normal,
            axisMask: SIMD3<Float>(repeating: 1),
            center: .zero
          )
        )
      }
    }

    let rowLength = segments + 1
    for stack in 0..<stacks {
      for segment in 0..<segments {
        let topLeft = stack * rowLength + segment
        let bottomLeft = (stack + 1) * rowLength + segment
        let topRight = topLeft + 1
        let bottomRight = bottomLeft + 1
        indices.append(contentsOf: [
          UInt16(topLeft), UInt16(bottomLeft), UInt16(topRight),
          UInt16(topRight), UInt16(bottomLeft), UInt16(bottomRight),
        ])
      }
    }

    let vertexBuffer = device.makeBuffer(
      bytes: vertices,
      length: MemoryLayout<EdgeVertex>.stride * vertices.count,
      options: [.storageModeShared]
    )!
    let indexBuffer = device.makeBuffer(
      bytes: indices,
      length: MemoryLayout<UInt16>.stride * indices.count,
      options: [.storageModeShared]
    )!

    return (vertexBuffer, indexBuffer, indices.count)
  }
}
