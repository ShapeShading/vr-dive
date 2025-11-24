import Metal
import simd

final class PongWars3DRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .pongWars3D
  let preferredClearColor = MTLClearColor(red: 0.02, green: 0.02, blue: 0.02, alpha: 1)
  
  private let device: MTLDevice
  private let maxViewCount: Int
  private let computePipelineState: MTLComputePipelineState
  private let edgePipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  
  private let ballStateBuffer: MTLBuffer
  private let voxelDataBuffer: MTLBuffer
  private var edgeVertexBuffer: MTLBuffer?
  private var edgeVertexCount: Int = 0
  
  private var lastSimulationTimestamp: Float = 0
  private var needsEdgeRebuild: Bool = true
  private var edgeRebuildCounter: Int = 0
  
  private static let gridSize: UInt32 = 16
  private static let worldSize: Float = 4.0
  private static let ballCount: UInt32 = 8
  private static let voxelSize: Float = worldSize / Float(gridSize)
  
  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.device = device
    self.maxViewCount = max(1, maxViewCount)
    
    computePipelineState = try PongWars3DRenderer.makeComputePipelineState(
      device: device, library: library)
    edgePipelineState = try PongWars3DRenderer.makeEdgePipelineState(
      device: device, library: library, maxViewCount: self.maxViewCount)
    depthStencilState = PongWars3DRenderer.makeDepthStencilState(device: device)
    
    // Initialize ball states
    ballStateBuffer = PongWars3DRenderer.makeInitialBallStates(device: device)
    
    // Initialize voxel data (16x16x16 = 4096 voxels)
    let voxelCount = Int(PongWars3DRenderer.gridSize * PongWars3DRenderer.gridSize * PongWars3DRenderer.gridSize)
    voxelDataBuffer = PongWars3DRenderer.makeInitialVoxelData(device: device, voxelCount: voxelCount)
  }
  
  func updateSimulation(_ context: PatternSimulationContext) {
    let deltaTime = max(0, context.time - lastSimulationTimestamp)
    guard deltaTime > 0 else { return }
    
    var uniforms = PongWarsSimulationUniforms(
      deltaTime: min(deltaTime, 1.0 / 30.0),
      globalTime: context.time,
      gridSize: PongWars3DRenderer.gridSize,
      ballCount: PongWars3DRenderer.ballCount,
      worldSize: PongWars3DRenderer.worldSize,
      voxelSize: PongWars3DRenderer.voxelSize,
      padding1: 0,
      padding2: 0
    )
    
    guard let commandBuffer = context.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder()
    else { return }
    
    encoder.setComputePipelineState(computePipelineState)
    encoder.setBuffer(ballStateBuffer, offset: 0, index: 0)
    encoder.setBuffer(voxelDataBuffer, offset: 0, index: 1)
    encoder.setBytes(&uniforms, length: MemoryLayout<PongWarsSimulationUniforms>.stride, index: 2)
    
    let threadWidth = min(computePipelineState.maxTotalThreadsPerThreadgroup, 32)
    let threadsPerThreadgroup = MTLSize(width: threadWidth, height: 1, depth: 1)
    let threadgroups = MTLSize(
      width: (Int(PongWars3DRenderer.ballCount) + threadWidth - 1) / threadWidth,
      height: 1,
      depth: 1
    )
    
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    lastSimulationTimestamp = context.time
    
    // Rebuild edges every 10 frames to reflect voxel ownership changes
    edgeRebuildCounter += 1
    if edgeRebuildCounter >= 10 {
      needsEdgeRebuild = true
      edgeRebuildCounter = 0
    }
  }
  
  func resetToInitialState() {
    // Reset ball states
    let ballStates = PongWars3DRenderer.generateInitialBallStates()
    ballStates.withUnsafeBytes { buffer in
      guard let baseAddress = buffer.baseAddress, buffer.count > 0 else { return }
      memcpy(ballStateBuffer.contents(), baseAddress, buffer.count)
    }
    
    // Reset voxel data
    let voxelCount = Int(PongWars3DRenderer.gridSize * PongWars3DRenderer.gridSize * PongWars3DRenderer.gridSize)
    let voxelData = PongWars3DRenderer.generateInitialVoxelData(voxelCount: voxelCount)
    voxelData.withUnsafeBytes { buffer in
      guard let baseAddress = buffer.baseAddress, buffer.count > 0 else { return }
      memcpy(voxelDataBuffer.contents(), baseAddress, buffer.count)
    }
    
    lastSimulationTimestamp = 0
    needsEdgeRebuild = true
    edgeRebuildCounter = 0
  }
  
  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    // Rebuild edge geometry if needed
    if needsEdgeRebuild {
      rebuildEdgeGeometry()
      needsEdgeRebuild = false
    }
    
    guard let edgeBuffer = edgeVertexBuffer, edgeVertexCount > 0 else { return }
    
    encoder.setRenderPipelineState(edgePipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)
    encoder.setFrontFacing(.counterClockwise)
    
    context.applyViewConfiguration(on: encoder)
    
    encoder.setVertexBuffer(edgeBuffer, offset: 0, index: 0)
    
    var sceneUniforms = PongWarsSceneUniforms(
      time: context.time,
      layerCount: UInt32(context.viewData.viewCount),
      padding: .zero
    )
    
    var viewMatrices = context.viewData.viewProjectionMatrices
    if viewMatrices.isEmpty {
      viewMatrices = [matrix_identity_float4x4]
    }
    
    encoder.setVertexBytes(&sceneUniforms, length: MemoryLayout<PongWarsSceneUniforms>.stride, index: 1)
    viewMatrices.withUnsafeBytes {
      if let baseAddress = $0.baseAddress, $0.count > 0 {
        encoder.setVertexBytes(baseAddress, length: $0.count, index: 2)
      }
    }
    encoder.setFragmentBytes(&sceneUniforms, length: MemoryLayout<PongWarsSceneUniforms>.stride, index: 0)
    
    // Draw edges as lines
    encoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: edgeVertexCount)
  }
  
  private func rebuildEdgeGeometry() {
    // Read voxel data from buffer (work directly with buffer to avoid copying)
    let voxelCount = Int(PongWars3DRenderer.gridSize * PongWars3DRenderer.gridSize * PongWars3DRenderer.gridSize)
    let voxelPtr = voxelDataBuffer.contents().bindMemory(to: VoxelData.self, capacity: voxelCount)
    
    // Generate edge vertices
    var vertices: [EdgeVertex] = []
    let halfWorld = PongWars3DRenderer.worldSize * 0.5
    let voxelSize = PongWars3DRenderer.voxelSize
    
    for z in 0..<Int(PongWars3DRenderer.gridSize) {
      for y in 0..<Int(PongWars3DRenderer.gridSize) {
        for x in 0..<Int(PongWars3DRenderer.gridSize) {
          let index = x + y * Int(PongWars3DRenderer.gridSize) + z * Int(PongWars3DRenderer.gridSize) * Int(PongWars3DRenderer.gridSize)
          let owner = voxelPtr[index].ownerAndFlags & 0xFF
          
          // Check if this voxel is on a boundary
          let isEdge = PongWars3DRenderer.isEdgeVoxel(x: x, y: y, z: z, voxelPtr: voxelPtr, gridSize: Int(PongWars3DRenderer.gridSize))
          
          if isEdge {
            let color = PongWars3DRenderer.colorForRegion(Int(owner))
            let basePos = SIMD3<Float>(
              Float(x) * voxelSize - halfWorld,
              Float(y) * voxelSize - halfWorld,
              Float(z) * voxelSize - halfWorld
            )
            
            // Draw edges of the voxel (12 edges of a cube)
            let corners: [SIMD3<Float>] = [
              basePos + SIMD3<Float>(0, 0, 0),
              basePos + SIMD3<Float>(voxelSize, 0, 0),
              basePos + SIMD3<Float>(voxelSize, voxelSize, 0),
              basePos + SIMD3<Float>(0, voxelSize, 0),
              basePos + SIMD3<Float>(0, 0, voxelSize),
              basePos + SIMD3<Float>(voxelSize, 0, voxelSize),
              basePos + SIMD3<Float>(voxelSize, voxelSize, voxelSize),
              basePos + SIMD3<Float>(0, voxelSize, voxelSize),
            ]
            
            // 12 edges defined by pairs of corners
            let edges: [(Int, Int)] = [
              (0, 1), (1, 2), (2, 3), (3, 0), // Bottom face
              (4, 5), (5, 6), (6, 7), (7, 4), // Top face
              (0, 4), (1, 5), (2, 6), (3, 7)  // Vertical edges
            ]
            
            for edge in edges {
              vertices.append(EdgeVertex(position: corners[edge.0], color: color))
              vertices.append(EdgeVertex(position: corners[edge.1], color: color))
            }
          }
        }
      }
    }
    
    edgeVertexCount = vertices.count
    
    if !vertices.isEmpty {
      edgeVertexBuffer = device.makeBuffer(
        bytes: vertices,
        length: MemoryLayout<EdgeVertex>.stride * vertices.count,
        options: [.storageModeShared]
      )
    }
  }
}

// Helper structs for edge rendering
struct EdgeVertex {
  var position: SIMD3<Float>
  var color: SIMD3<Float>
}

// Static helper methods
extension PongWars3DRenderer {
  fileprivate static func makeComputePipelineState(device: MTLDevice, library: MTLLibrary) throws -> MTLComputePipelineState {
    let function = library.makeFunction(name: "simulatePongWarsBalls")!
    return try device.makeComputePipelineState(function: function)
  }
  
  fileprivate static func makeEdgePipelineState(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: "edgeVertexShader")
    descriptor.fragmentFunction = library.makeFunction(name: "edgeFragmentShader")
    descriptor.colorAttachments[0].pixelFormat = .rgba16Float
    descriptor.depthAttachmentPixelFormat = .depth32Float
    descriptor.inputPrimitiveTopology = .line
    
    let vertexDescriptor = MTLVertexDescriptor()
    vertexDescriptor.attributes[0].format = .float3
    vertexDescriptor.attributes[0].offset = 0
    vertexDescriptor.attributes[0].bufferIndex = 0
    vertexDescriptor.attributes[1].format = .float3
    vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
    vertexDescriptor.attributes[1].bufferIndex = 0
    vertexDescriptor.layouts[0].stride = MemoryLayout<EdgeVertex>.stride
    descriptor.vertexDescriptor = vertexDescriptor
    descriptor.maxVertexAmplificationCount = max(maxViewCount, 1)
    
    return try device.makeRenderPipelineState(descriptor: descriptor)
  }
  
  fileprivate static func makeDepthStencilState(device: MTLDevice) -> MTLDepthStencilState {
    let descriptor = MTLDepthStencilDescriptor()
    descriptor.depthCompareFunction = .greater
    descriptor.isDepthWriteEnabled = true
    return device.makeDepthStencilState(descriptor: descriptor)!
  }
  
  fileprivate static func makeInitialBallStates(device: MTLDevice) -> MTLBuffer {
    let states = generateInitialBallStates()
    return device.makeBuffer(
      bytes: states,
      length: MemoryLayout<BallState>.stride * states.count,
      options: [.storageModeShared]
    )!
  }
  
  fileprivate static func generateInitialBallStates() -> [BallState] {
    var states: [BallState] = []
    let halfWorld = worldSize * 0.5
    let quarterWorld = worldSize * 0.25
    
    // 8 regions in a 2x2x2 configuration
    let regionOffsets: [SIMD3<Float>] = [
      SIMD3<Float>(-quarterWorld, -quarterWorld, -quarterWorld), // Region 0
      SIMD3<Float>(quarterWorld, -quarterWorld, -quarterWorld),  // Region 1
      SIMD3<Float>(-quarterWorld, quarterWorld, -quarterWorld),  // Region 2
      SIMD3<Float>(quarterWorld, quarterWorld, -quarterWorld),   // Region 3
      SIMD3<Float>(-quarterWorld, -quarterWorld, quarterWorld),  // Region 4
      SIMD3<Float>(quarterWorld, -quarterWorld, quarterWorld),   // Region 5
      SIMD3<Float>(-quarterWorld, quarterWorld, quarterWorld),   // Region 6
      SIMD3<Float>(quarterWorld, quarterWorld, quarterWorld),    // Region 7
    ]
    
    for i in 0..<8 {
      let regionCenter = regionOffsets[i]
      
      // Randomize position within region
      let randomOffset = SIMD3<Float>(
        Float.random(in: -0.3...0.3),
        Float.random(in: -0.3...0.3),
        Float.random(in: -0.3...0.3)
      )
      
      let position = regionCenter + randomOffset
      
      // Random velocity
      let speed = Float.random(in: 1.2...2.0)
      let velocity = SIMD3<Float>(
        Float.random(in: -1...1),
        Float.random(in: -1...1),
        Float.random(in: -1...1)
      )
      let normalizedVelocity = simd_normalize(velocity) * speed
      
      let color = colorForRegion(i)
      let radius: Float = 0.08
      
      states.append(BallState(
        position: SIMD4<Float>(position, Float(i)),
        velocity: SIMD4<Float>(normalizedVelocity, radius),
        colorAndPadding: SIMD4<Float>(color, 0)
      ))
    }
    
    return states
  }
  
  fileprivate static func makeInitialVoxelData(device: MTLDevice, voxelCount: Int) -> MTLBuffer {
    let voxelData = generateInitialVoxelData(voxelCount: voxelCount)
    return device.makeBuffer(
      bytes: voxelData,
      length: MemoryLayout<VoxelData>.stride * voxelCount,
      options: [.storageModeShared]
    )!
  }
  
  fileprivate static func generateInitialVoxelData(voxelCount: Int) -> [VoxelData] {
    var voxels: [VoxelData] = []
    voxels.reserveCapacity(voxelCount)
    
    let halfGrid = Int(gridSize) / 2
    
    for z in 0..<Int(gridSize) {
      for y in 0..<Int(gridSize) {
        for x in 0..<Int(gridSize) {
          // Determine which region this voxel belongs to (2x2x2 regions)
          let regionX = x < halfGrid ? 0 : 1
          let regionY = y < halfGrid ? 0 : 1
          let regionZ = z < halfGrid ? 0 : 1
          let regionIndex = regionX + regionY * 2 + regionZ * 4
          
          voxels.append(VoxelData(ownerAndFlags: UInt32(regionIndex)))
        }
      }
    }
    
    return voxels
  }
  
  fileprivate static func colorForRegion(_ region: Int) -> SIMD3<Float> {
    // 8 distinct colors for 8 regions
    let colors: [SIMD3<Float>] = [
      SIMD3<Float>(1.0, 0.3, 0.3),   // Red
      SIMD3<Float>(0.3, 1.0, 0.3),   // Green
      SIMD3<Float>(0.3, 0.3, 1.0),   // Blue
      SIMD3<Float>(1.0, 1.0, 0.3),   // Yellow
      SIMD3<Float>(1.0, 0.3, 1.0),   // Magenta
      SIMD3<Float>(0.3, 1.0, 1.0),   // Cyan
      SIMD3<Float>(1.0, 0.6, 0.2),   // Orange
      SIMD3<Float>(0.6, 0.3, 1.0),   // Purple
    ]
    return colors[region % colors.count]
  }
  
  fileprivate static func isEdgeVoxel(x: Int, y: Int, z: Int, voxelPtr: UnsafePointer<VoxelData>, gridSize: Int) -> Bool {
    let centerIndex = x + y * gridSize + z * gridSize * gridSize
    let centerOwner = voxelPtr[centerIndex].ownerAndFlags & 0xFF
    
    // Check 6 neighbors
    let offsets: [(Int, Int, Int)] = [
      (1, 0, 0), (-1, 0, 0),
      (0, 1, 0), (0, -1, 0),
      (0, 0, 1), (0, 0, -1)
    ]
    
    for offset in offsets {
      let nx = x + offset.0
      let ny = y + offset.1
      let nz = z + offset.2
      
      if nx < 0 || nx >= gridSize || ny < 0 || ny >= gridSize || nz < 0 || nz >= gridSize {
        return true // Boundary of world
      }
      
      let neighborIndex = nx + ny * gridSize + nz * gridSize * gridSize
      let neighborOwner = voxelPtr[neighborIndex].ownerAndFlags & 0xFF
      
      if neighborOwner != centerOwner {
        return true // Different color neighbor
      }
    }
    
    return false
  }
}
