import Foundation
import Metal
import MetalKit
import simd

struct PagodaInstance {
  var modelMatrix: matrix_float4x4
  var color: SIMD4<Float>
}

final class PagodaRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .pagoda
  let preferredClearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.15, alpha: 1)

  private let device: MTLDevice
  private let pipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  private let instanceBuffer: MTLBuffer
  private let instanceCount: Int
  private let meshVertexBuffer: MTLBuffer
  private let meshIndexBuffer: MTLBuffer
  private let meshIndexCount: Int

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.device = device

    // 1. Create Pipeline State
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: "pagodaVertexShader")
    descriptor.fragmentFunction = library.makeFunction(name: "pagodaFragmentShader")
    descriptor.colorAttachments[0].pixelFormat = .rgba16Float
    descriptor.depthAttachmentPixelFormat = .depth32Float
    descriptor.inputPrimitiveTopology = .triangle

    let vertexDescriptor = MTLVertexDescriptor()
    vertexDescriptor.attributes[0].format = .float3  // Position
    vertexDescriptor.attributes[0].offset = 0
    vertexDescriptor.attributes[0].bufferIndex = 0
    vertexDescriptor.attributes[1].format = .float3  // Normal
    vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
    vertexDescriptor.attributes[1].bufferIndex = 0
    vertexDescriptor.layouts[0].stride = MemoryLayout<MeshVertex>.stride
    descriptor.vertexDescriptor = vertexDescriptor
    // Ensure amplification is at least 1, but typically 2 for stereo
    descriptor.maxVertexAmplificationCount = max(maxViewCount, 1)

    self.pipelineState = try device.makeRenderPipelineState(descriptor: descriptor)

    // 2. Create Depth Stencil State
    let depthDescriptor = MTLDepthStencilDescriptor()
    depthDescriptor.depthCompareFunction = .greater
    depthDescriptor.isDepthWriteEnabled = true
    self.depthStencilState = device.makeDepthStencilState(descriptor: depthDescriptor)!

    // 3. Create Mesh (Unit Cube)
    let (vBuffer, iBuffer, iCount) = PagodaRenderer.makeCubeMesh(device: device)
    self.meshVertexBuffer = vBuffer
    self.meshIndexBuffer = iBuffer
    self.meshIndexCount = iCount

    // 4. Generate Instances
    let instances = PagodaRenderer.generatePagodaGeometry()
    self.instanceCount = instances.count
    self.instanceBuffer = device.makeBuffer(
      bytes: instances, length: MemoryLayout<PagodaInstance>.stride * instances.count,
      options: .storageModeShared)!

    print("[PagodaRenderer] Generated \(instanceCount) instances.")
  }

  func updateSimulation(_ context: PatternSimulationContext) {
    // Static geometry, no simulation needed
  }

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    encoder.setRenderPipelineState(pipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none) // Disable culling for wireframe/stick look and to avoid stereo issues
    encoder.setFrontFacing(.counterClockwise)

    context.applyViewConfiguration(on: encoder)

    var sceneUniforms = SceneUniforms(
      time: context.time,
      layerCount: UInt32(context.viewData.viewCount)
    )

    encoder.setVertexBytes(&sceneUniforms, length: MemoryLayout<SceneUniforms>.stride, index: 2)

    var viewMatrices = context.viewData.viewProjectionMatrices
    if viewMatrices.isEmpty { viewMatrices = [matrix_identity_float4x4] }

    viewMatrices.withUnsafeBytes {
      if let baseAddress = $0.baseAddress, $0.count > 0 {
        encoder.setVertexBytes(baseAddress, length: $0.count, index: 3)
      }
    }

    encoder.setFragmentBytes(&sceneUniforms, length: MemoryLayout<SceneUniforms>.stride, index: 0)

    encoder.setVertexBuffer(meshVertexBuffer, offset: 0, index: 0)
    encoder.setVertexBuffer(instanceBuffer, offset: 0, index: 1)

    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: meshIndexCount,
      indexType: .uint16,
      indexBuffer: meshIndexBuffer,
      indexBufferOffset: 0,
      instanceCount: instanceCount
    )
  }

  func resetToInitialState() {}

  // MARK: - Geometry Generation

  private static func makeCubeMesh(device: MTLDevice) -> (MTLBuffer, MTLBuffer, Int) {
    // Standard unit cube [-0.5, 0.5]
    let vertices: [MeshVertex] = [
      // Front
      MeshVertex(position: [-0.5, -0.5, 0.5], normal: [0, 0, 1]),
      MeshVertex(position: [0.5, -0.5, 0.5], normal: [0, 0, 1]),
      MeshVertex(position: [0.5, 0.5, 0.5], normal: [0, 0, 1]),
      MeshVertex(position: [-0.5, 0.5, 0.5], normal: [0, 0, 1]),
      // Back
      MeshVertex(position: [0.5, -0.5, -0.5], normal: [0, 0, -1]),
      MeshVertex(position: [-0.5, -0.5, -0.5], normal: [0, 0, -1]),
      MeshVertex(position: [-0.5, 0.5, -0.5], normal: [0, 0, -1]),
      MeshVertex(position: [0.5, 0.5, -0.5], normal: [0, 0, -1]),
      // Left
      MeshVertex(position: [-0.5, -0.5, -0.5], normal: [-1, 0, 0]),
      MeshVertex(position: [-0.5, -0.5, 0.5], normal: [-1, 0, 0]),
      MeshVertex(position: [-0.5, 0.5, 0.5], normal: [-1, 0, 0]),
      MeshVertex(position: [-0.5, 0.5, -0.5], normal: [-1, 0, 0]),
      // Right
      MeshVertex(position: [0.5, -0.5, 0.5], normal: [1, 0, 0]),
      MeshVertex(position: [0.5, -0.5, -0.5], normal: [1, 0, 0]),
      MeshVertex(position: [0.5, 0.5, -0.5], normal: [1, 0, 0]),
      MeshVertex(position: [0.5, 0.5, 0.5], normal: [1, 0, 0]),
      // Top
      MeshVertex(position: [-0.5, 0.5, 0.5], normal: [0, 1, 0]),
      MeshVertex(position: [0.5, 0.5, 0.5], normal: [0, 1, 0]),
      MeshVertex(position: [0.5, 0.5, -0.5], normal: [0, 1, 0]),
      MeshVertex(position: [-0.5, 0.5, -0.5], normal: [0, 1, 0]),
      // Bottom
      MeshVertex(position: [-0.5, -0.5, -0.5], normal: [0, -1, 0]),
      MeshVertex(position: [0.5, -0.5, -0.5], normal: [0, -1, 0]),
      MeshVertex(position: [0.5, -0.5, 0.5], normal: [0, -1, 0]),
      MeshVertex(position: [-0.5, -0.5, 0.5], normal: [0, -1, 0]),
    ]

    let indices: [UInt16] = [
      0, 1, 2, 0, 2, 3,  // Front
      4, 5, 6, 4, 6, 7,  // Back
      8, 9, 10, 8, 10, 11,  // Left
      12, 13, 14, 12, 14, 15,  // Right
      16, 17, 18, 16, 18, 19,  // Top
      20, 21, 22, 20, 22, 23,  // Bottom
    ]

    let vBuffer = device.makeBuffer(
      bytes: vertices, length: MemoryLayout<MeshVertex>.stride * vertices.count,
      options: .storageModeShared)!
    let iBuffer = device.makeBuffer(
      bytes: indices, length: MemoryLayout<UInt16>.stride * indices.count,
      options: .storageModeShared)!

    return (vBuffer, iBuffer, indices.count)
  }

  private static func generatePagodaGeometry() -> [PagodaInstance] {
    var instances: [PagodaInstance] = []

    // Blueprint Style Colors
    let blueprintColor = SIMD4<Float>(0.7, 0.85, 1.0, 1.0) // Blue-white
    let stickThickness: Float = 0.02 // 2cm diameter

    // Helper to add a stick (cylinder/box) from point A to point B
    func addStick(from: SIMD3<Float>, to: SIMD3<Float>, thickness: Float, color: SIMD4<Float>) {
        let vector = to - from
        let length = simd_length(vector)
        if length < 0.001 { return }
        
        let direction = vector / length
        let center = (from + to) * 0.5
        
        // Create rotation basis to align Z axis with direction
        // If direction is parallel to Up (0,1,0), use Right (1,0,0) as temp Up
        let upRef = abs(direction.y) > 0.99 ? SIMD3<Float>(1, 0, 0) : SIMD3<Float>(0, 1, 0)
        let right = simd_normalize(simd_cross(upRef, direction))
        let up = simd_cross(direction, right)
        
        var matrix = matrix_identity_float4x4
        // Scale columns: Right * thickness, Up * thickness, Forward * length
        matrix.columns.0 = SIMD4<Float>(right * thickness, 0)
        matrix.columns.1 = SIMD4<Float>(up * thickness, 0)
        matrix.columns.2 = SIMD4<Float>(direction * length, 0)
        matrix.columns.3 = SIMD4<Float>(center, 1)
        
        instances.append(PagodaInstance(modelMatrix: matrix, color: color))
    }

    // Helper to draw a wireframe rectangle (horizontal)
    func addRect(center: SIMD3<Float>, size: SIMD2<Float>, color: SIMD4<Float>) {
        let halfW = size.x * 0.5
        let halfD = size.y * 0.5
        let y = center.y
        
        let p1 = SIMD3<Float>(center.x - halfW, y, center.z - halfD)
        let p2 = SIMD3<Float>(center.x + halfW, y, center.z - halfD)
        let p3 = SIMD3<Float>(center.x + halfW, y, center.z + halfD)
        let p4 = SIMD3<Float>(center.x - halfW, y, center.z + halfD)
        
        addStick(from: p1, to: p2, thickness: stickThickness, color: color)
        addStick(from: p2, to: p3, thickness: stickThickness, color: color)
        addStick(from: p3, to: p4, thickness: stickThickness, color: color)
        addStick(from: p4, to: p1, thickness: stickThickness, color: color)
    }

    // --- A. Base ---
    let baseWidth: Float = 30.0
    let baseHeight: Float = 4.0
    let baseLayers = 8
    
    for i in 0...baseLayers {
      let progress = Float(i) / Float(baseLayers)
      let w = baseWidth * (1.0 - progress * 0.1)
      let h = baseHeight / Float(baseLayers)
      let y = Float(i) * h - 2.0
      
      // Draw outline of each step
      addRect(center: SIMD3<Float>(0, y, 0), size: SIMD2<Float>(w, w), color: blueprintColor)
      
      // Vertical connectors at corners for base
      if i < baseLayers {
          let nextProgress = Float(i + 1) / Float(baseLayers)
          let nextW = baseWidth * (1.0 - nextProgress * 0.1)
          let nextY = Float(i + 1) * h - 2.0
          
          let corners = [
              SIMD2<Float>(-1, -1), SIMD2<Float>(1, -1), SIMD2<Float>(1, 1), SIMD2<Float>(-1, 1)
          ]
          
          for corner in corners {
              let p1 = SIMD3<Float>(corner.x * w * 0.5, y, corner.y * w * 0.5)
              let p2 = SIMD3<Float>(corner.x * nextW * 0.5, nextY, corner.y * nextW * 0.5)
              addStick(from: p1, to: p2, thickness: stickThickness, color: blueprintColor)
          }
      }
    }

    // --- B. Body (7 Layers) ---
    var currentY: Float = 2.0
    var currentWidth: Float = 24.0
    let topWidth: Float = 10.0
    let widthStep = (currentWidth - topWidth) / 6.0

    var currentLayerHeight: Float = 5.0
    let minLayerHeight: Float = 3.0
    let heightStep = (currentLayerHeight - minLayerHeight) / 6.0

    for layer in 0..<7 {
      let layerHeight = currentLayerHeight
      let halfW = currentWidth * 0.5
      
      // Floor
      addRect(center: SIMD3<Float>(0, currentY, 0), size: SIMD2<Float>(currentWidth, currentWidth), color: blueprintColor)
      
      // Ceiling
      addRect(center: SIMD3<Float>(0, currentY + layerHeight, 0), size: SIMD2<Float>(currentWidth, currentWidth), color: blueprintColor)
      
      // Vertical Posts (Corners + Pilasters)
      let bays = (layer < 2) ? 9 : (layer < 4 ? 7 : 5)
      let pilasterCount = bays + 1
      
      for side in 0..<4 {
          // 0: Front, 1: Right, 2: Back, 3: Left
          // We iterate along the face
          for p in 0..<pilasterCount {
              let t = Float(p) / Float(bays)
              let offset = -halfW + t * currentWidth
              
              var pos = SIMD3<Float>(0, 0, 0)
              if side == 0 { pos = SIMD3<Float>(offset, 0, halfW) } // Front
              else if side == 1 { pos = SIMD3<Float>(halfW, 0, offset) } // Right
              else if side == 2 { pos = SIMD3<Float>(offset, 0, -halfW) } // Back
              else { pos = SIMD3<Float>(-halfW, 0, offset) } // Left
              
              let start = SIMD3<Float>(pos.x, currentY, pos.z)
              let end = SIMD3<Float>(pos.x, currentY + layerHeight, pos.z)
              addStick(from: start, to: end, thickness: stickThickness, color: blueprintColor)
          }
      }

      // 2. Eaves (Corbelling)
      // Stack of rectangles expanding out
      let eaveStartY = currentY + layerHeight
      let eaveLayers = 5
      let maxEaveOut: Float = 1.5
      let eaveHeightPerLayer: Float = 0.15

      for e in 0..<eaveLayers {
        let progress = Float(e) / Float(eaveLayers - 1)
        let expansion = progress * maxEaveOut
        let w = currentWidth + expansion * 2.0
        let y = eaveStartY + Float(e) * eaveHeightPerLayer

        addRect(center: SIMD3<Float>(0, y, 0), size: SIMD2<Float>(w, w), color: blueprintColor)
        
        // Add rafters (lines radiating out)
        // Connect previous layer to this layer
        if e > 0 {
            let prevProgress = Float(e - 1) / Float(eaveLayers - 1)
            let prevExpansion = prevProgress * maxEaveOut
            let prevW = currentWidth + prevExpansion * 2.0
            let prevY = eaveStartY + Float(e - 1) * eaveHeightPerLayer
            
            // Draw rafters at corners and some intervals
            let rafterCount = 8
            for r in 0..<rafterCount {
                // Simple distribution along one side for now, or just corners
                // Let's just do corners for simplicity + midpoints
                // Actually, let's do a loop around the perimeter
                // Just corners:
                let corners = [SIMD2<Float>(-1,-1), SIMD2<Float>(1,-1), SIMD2<Float>(1,1), SIMD2<Float>(-1,1)]
                for corner in corners {
                    let p1 = SIMD3<Float>(corner.x * prevW * 0.5, prevY, corner.y * prevW * 0.5)
                    let p2 = SIMD3<Float>(corner.x * w * 0.5, y, corner.y * w * 0.5)
                    addStick(from: p1, to: p2, thickness: stickThickness, color: blueprintColor)
                }
            }
        }
      }

      // Roof slope (Retracting)
      let roofLayers = 4
      let roofStartY = eaveStartY + Float(eaveLayers) * eaveHeightPerLayer
      for r in 0..<roofLayers {
        let progress = Float(r) / Float(roofLayers)
        let nextW = currentWidth - widthStep
        let startW = currentWidth + maxEaveOut * 2.0
        let w = startW - (startW - nextW) * progress
        let y = roofStartY + Float(r) * eaveHeightPerLayer

        addRect(center: SIMD3<Float>(0, y, 0), size: SIMD2<Float>(w, w), color: blueprintColor)
      }

      currentY = roofStartY + Float(roofLayers) * eaveHeightPerLayer
      currentWidth -= widthStep
      currentLayerHeight -= heightStep
    }

    // --- C. Finial ---
    let finialBaseY = currentY
    let finialSize = currentWidth
    let pyramidLayers = 10
    let pyramidHeight: Float = 3.0

    for p in 0...pyramidLayers {
      let progress = Float(p) / Float(pyramidLayers)
      let w = finialSize * (1.0 - progress)
      let h = pyramidHeight / Float(pyramidLayers)
      let y = finialBaseY + Float(p) * h
      
      addRect(center: SIMD3<Float>(0, y, 0), size: SIMD2<Float>(w, w), color: blueprintColor)
      
      // Corner spines
      if p < pyramidLayers {
          let nextProgress = Float(p + 1) / Float(pyramidLayers)
          let nextW = finialSize * (1.0 - nextProgress)
          let nextY = finialBaseY + Float(p + 1) * h
          
          let corners = [SIMD2<Float>(-1,-1), SIMD2<Float>(1,-1), SIMD2<Float>(1,1), SIMD2<Float>(-1,1)]
          for corner in corners {
              let p1 = SIMD3<Float>(corner.x * w * 0.5, y, corner.y * w * 0.5)
              let p2 = SIMD3<Float>(corner.x * nextW * 0.5, nextY, corner.y * nextW * 0.5)
              addStick(from: p1, to: p2, thickness: stickThickness, color: blueprintColor)
          }
      }
    }

    // Spire
    let spireY = finialBaseY + pyramidHeight
    addStick(from: SIMD3<Float>(0, spireY, 0), to: SIMD3<Float>(0, spireY + 4.0, 0), thickness: stickThickness * 2, color: blueprintColor)

    return instances
  }
}
