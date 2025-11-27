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
  private let boxInstanceBuffer: MTLBuffer
  private let boxInstanceCount: Int
  private let meshVertexBuffer: MTLBuffer
  private let meshIndexBuffer: MTLBuffer
  private let meshIndexCount: Int
  private let tubeVertexBuffer: MTLBuffer
  private let tubeIndexBuffer: MTLBuffer
  private let tubeIndexCount: Int
  private let tubeInstanceBuffer: MTLBuffer
  private let tubeInstanceCount: Int

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

    // 3. Create Meshes (Unit Cube + Unit Tube)
    let (vBuffer, iBuffer, iCount) = PagodaRenderer.makeCubeMesh(device: device)
    self.meshVertexBuffer = vBuffer
    self.meshIndexBuffer = iBuffer
    self.meshIndexCount = iCount

    let (tvBuffer, tiBuffer, tiCount) = PagodaRenderer.makeTubeMesh(device: device, sides: 16)
    self.tubeVertexBuffer = tvBuffer
    self.tubeIndexBuffer = tiBuffer
    self.tubeIndexCount = tiCount

    // 4. Generate Instances (tube + box)
    let geometry = PagodaRenderer.generatePagodaGeometry()
    self.tubeInstanceCount = geometry.tube.count
    self.boxInstanceCount = geometry.box.count
    self.tubeInstanceBuffer = device.makeBuffer(
      bytes: geometry.tube,
      length: MemoryLayout<PagodaInstance>.stride * max(geometry.tube.count, 1),
      options: .storageModeShared)!
    self.boxInstanceBuffer = device.makeBuffer(
      bytes: geometry.box,
      length: MemoryLayout<PagodaInstance>.stride * max(geometry.box.count, 1),
      options: .storageModeShared)!

    print("[PagodaRenderer] Generated tube=\(tubeInstanceCount) box=\(boxInstanceCount) instances.")
  }

  func updateSimulation(_ context: PatternSimulationContext) {
    // Static geometry, no simulation needed
  }

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    encoder.setRenderPipelineState(pipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)  // Disable culling for wireframe/stick look and to avoid stereo issues
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

    // Draw tubes first (round sticks)
    if tubeInstanceCount > 0 {
      encoder.setVertexBuffer(tubeVertexBuffer, offset: 0, index: 0)
      encoder.setVertexBuffer(tubeInstanceBuffer, offset: 0, index: 1)
      encoder.drawIndexedPrimitives(
        type: .triangle,
        indexCount: tubeIndexCount,
        indexType: .uint16,
        indexBuffer: tubeIndexBuffer,
        indexBufferOffset: 0,
        instanceCount: tubeInstanceCount
      )
    }

    // Then draw boxes (flat sticks)
    if boxInstanceCount > 0 {
      encoder.setVertexBuffer(meshVertexBuffer, offset: 0, index: 0)
      encoder.setVertexBuffer(boxInstanceBuffer, offset: 0, index: 1)
      encoder.drawIndexedPrimitives(
        type: .triangle,
        indexCount: meshIndexCount,
        indexType: .uint16,
        indexBuffer: meshIndexBuffer,
        indexBufferOffset: 0,
        instanceCount: boxInstanceCount
      )
    }
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

  private static func makeTubeMesh(device: MTLDevice, sides: Int = 16) -> (
    MTLBuffer, MTLBuffer, Int
  ) {
    let N = max(8, sides)
    var vertices: [MeshVertex] = []
    var indices: [UInt16] = []

    // Unit cylinder: radius=0.5, z in [-0.5, 0.5]
    let radius: Float = 0.5
    let zTop: Float = 0.5
    let zBottom: Float = -0.5

    // Side wall vertices
    for i in 0..<N {
      let theta = Float(i) * 2.0 * .pi / Float(N)
      let nx = cos(theta)
      let ny = sin(theta)
      let px = radius * nx
      let py = radius * ny
      // bottom
      vertices.append(MeshVertex(position: [px, py, zBottom], normal: [nx, ny, 0]))
      // top
      vertices.append(MeshVertex(position: [px, py, zTop], normal: [nx, ny, 0]))
    }

    // Side wall indices (quad per segment -> 2 triangles)
    for i in 0..<N {
      let i0 = UInt16(i * 2)
      let i1 = UInt16(i * 2 + 1)
      let i2 = UInt16(((i + 1) % N) * 2)
      let i3 = UInt16(((i + 1) % N) * 2 + 1)
      // triangle 1: i0, i2, i1
      indices.append(contentsOf: [i0, i2, i1])
      // triangle 2: i1, i2, i3
      indices.append(contentsOf: [i1, i2, i3])
    }

    // Top cap center and ring
    let topCenterIndex = vertices.count
    vertices.append(MeshVertex(position: [0, 0, zTop], normal: [0, 0, 1]))
    for i in 0..<N {
      let theta = Float(i) * 2.0 * .pi / Float(N)
      let nx = cos(theta)
      let ny = sin(theta)
      let px = radius * nx
      let py = radius * ny
      vertices.append(MeshVertex(position: [px, py, zTop], normal: [0, 0, 1]))
    }
    for i in 0..<N {
      let a = UInt16(topCenterIndex)
      let b = UInt16(topCenterIndex + 1 + i)
      let c = UInt16(topCenterIndex + 1 + ((i + 1) % N))
      indices.append(contentsOf: [a, b, c])
    }

    // Bottom cap center and ring
    let bottomCenterIndex = vertices.count
    vertices.append(MeshVertex(position: [0, 0, zBottom], normal: [0, 0, -1]))
    for i in 0..<N {
      let theta = Float(i) * 2.0 * .pi / Float(N)
      let nx = cos(theta)
      let ny = sin(theta)
      let px = radius * nx
      let py = radius * ny
      vertices.append(MeshVertex(position: [px, py, zBottom], normal: [0, 0, -1]))
    }
    for i in 0..<N {
      let a = UInt16(bottomCenterIndex)
      let c = UInt16(bottomCenterIndex + 1 + i)
      let b = UInt16(bottomCenterIndex + 1 + ((i + 1) % N))
      indices.append(contentsOf: [a, b, c])
    }

    let vBuffer = device.makeBuffer(
      bytes: vertices,
      length: MemoryLayout<MeshVertex>.stride * vertices.count,
      options: .storageModeShared)!
    let iBuffer = device.makeBuffer(
      bytes: indices,
      length: MemoryLayout<UInt16>.stride * indices.count,
      options: .storageModeShared)!
    return (vBuffer, iBuffer, indices.count)
  }

  private static func generatePagodaGeometry() -> (tube: [PagodaInstance], box: [PagodaInstance]) {
    var tubeInstances: [PagodaInstance] = []
    var boxInstances: [PagodaInstance] = []
    struct SegmentKey: Hashable {
      let ax: Int32, ay: Int32, az: Int32
      let bx: Int32, by: Int32, bz: Int32
      init(_ a: SIMD3<Float>, _ b: SIMD3<Float>, eps: Float) {
        func q(_ v: Float) -> Int32 { Int32(lroundf(v / eps)) }
        let A = (q(a.x), q(a.y), q(a.z))
        let B = (q(b.x), q(b.y), q(b.z))
        if A < B {
          (ax, ay, az, bx, by, bz) = (A.0, A.1, A.2, B.0, B.1, B.2)
        } else {
          (ax, ay, az, bx, by, bz) = (B.0, B.1, B.2, A.0, A.1, A.2)
        }
      }
    }
    var seenSegments: Set<SegmentKey> = []

    // Blueprint Style Colors
    let blueprintColor = SIMD4<Float>(0.7, 0.85, 1.0, 1.0)
    let stickThickness: Float = 0.02
    let detailLevel = 2

    // Helper to add a stick (cylinder/box) from point A to point B
    func addStick(
      from: SIMD3<Float>, to: SIMD3<Float>, thickness: Float, color: SIMD4<Float>,
      asTube: Bool = true
    ) {
      let vector = to - from
      let length = simd_length(vector)
      if length < 0.001 { return }
      let key = SegmentKey(from, to, eps: 0.001)
      if seenSegments.contains(key) { return }
      seenSegments.insert(key)

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

      if asTube {
        tubeInstances.append(PagodaInstance(modelMatrix: matrix, color: color))
      } else {
        boxInstances.append(PagodaInstance(modelMatrix: matrix, color: color))
      }
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

      addStick(from: p1, to: p2, thickness: stickThickness, color: color, asTube: false)
      addStick(from: p2, to: p3, thickness: stickThickness, color: color, asTube: false)
      addStick(from: p3, to: p4, thickness: stickThickness, color: color, asTube: false)
      addStick(from: p4, to: p1, thickness: stickThickness, color: color, asTube: false)
    }

    func addPolyline(
      points: [SIMD3<Float>], thickness: Float, color: SIMD4<Float>, asTube: Bool = true
    ) {
      guard points.count > 1 else { return }
      for i in 0..<(points.count - 1) {
        addStick(
          from: points[i], to: points[i + 1], thickness: thickness, color: color, asTube: asTube)
      }
    }

    func addRailing(perimeterCenterY: Float, width: Float, bays: Int) {
      let halfW = width * 0.5
      let railTopY = perimeterCenterY + 1.0
      let railBottomY = perimeterCenterY + 0.5
      let segments = max(4, bays)
      // Horizontal rails on four sides
      addRect(
        center: SIMD3<Float>(0, railTopY, 0), size: SIMD2<Float>(width, width),
        color: blueprintColor)
      addRect(
        center: SIMD3<Float>(0, railBottomY, 0), size: SIMD2<Float>(width, width),
        color: blueprintColor)
      // Vertical posts at intervals
      for side in 0..<4 {
        for p in 0...segments {
          let t = Float(p) / Float(segments)
          let offset = -halfW + t * width
          var pos = SIMD3<Float>(0, 0, 0)
          if side == 0 {
            pos = SIMD3<Float>(offset, 0, halfW)
          } else if side == 1 {
            pos = SIMD3<Float>(halfW, 0, offset)
          } else if side == 2 {
            pos = SIMD3<Float>(offset, 0, -halfW)
          } else {
            pos = SIMD3<Float>(-halfW, 0, offset)
          }
          let start = SIMD3<Float>(pos.x, railBottomY, pos.z)
          let end = SIMD3<Float>(pos.x, railTopY, pos.z)
          addStick(
            from: start, to: end, thickness: stickThickness * 0.8, color: blueprintColor,
            asTube: true)
        }
      }
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
          SIMD2<Float>(-1, -1), SIMD2<Float>(1, -1), SIMD2<Float>(1, 1), SIMD2<Float>(-1, 1),
        ]

        for corner in corners {
          let p1 = SIMD3<Float>(corner.x * w * 0.5, y, corner.y * w * 0.5)
          let p2 = SIMD3<Float>(corner.x * nextW * 0.5, nextY, corner.y * nextW * 0.5)
          addStick(from: p1, to: p2, thickness: stickThickness, color: blueprintColor, asTube: true)
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
      addRect(
        center: SIMD3<Float>(0, currentY, 0), size: SIMD2<Float>(currentWidth, currentWidth),
        color: blueprintColor)

      // Ceiling
      addRect(
        center: SIMD3<Float>(0, currentY + layerHeight, 0),
        size: SIMD2<Float>(currentWidth, currentWidth), color: blueprintColor)

      // Vertical Posts (Corners + Pilasters)
      let bays = (layer < 2) ? 9 : (layer < 4 ? 7 : 5)
      let baysLOD = detailLevel == 0 ? max(3, bays - 2) : (detailLevel == 1 ? bays : bays + 2)
      let pilasterCount = baysLOD + 1

      for side in 0..<4 {
        // 0: Front, 1: Right, 2: Back, 3: Left
        // We iterate along the face
        for p in 0..<pilasterCount {
          let t = Float(p) / Float(bays)
          let offset = -halfW + t * currentWidth

          var pos = SIMD3<Float>(0, 0, 0)
          if side == 0 {
            pos = SIMD3<Float>(offset, 0, halfW)
          }  // Front
          else if side == 1 {
            pos = SIMD3<Float>(halfW, 0, offset)
          }  // Right
          else if side == 2 {
            pos = SIMD3<Float>(offset, 0, -halfW)
          }  // Back
          else {
            pos = SIMD3<Float>(-halfW, 0, offset)
          }  // Left

          let start = SIMD3<Float>(pos.x, currentY, pos.z)
          let end = SIMD3<Float>(pos.x, currentY + layerHeight, pos.z)
          addStick(
            from: start, to: end, thickness: stickThickness, color: blueprintColor, asTube: true)
        }
      }

      // Railings around the perimeter (skip on low detail for lower layers)
      if detailLevel >= 1 {
        addRailing(
          perimeterCenterY: currentY + layerHeight * 0.5, width: currentWidth, bays: baysLOD)
      }

      // Window frames on front/back faces (medium/high)
      if detailLevel >= 1 {
        let windowRows = detailLevel == 2 ? 3 : 2
        let windowCols = max(2, baysLOD / (detailLevel == 2 ? 2 : 3))
        let frameInset: Float = currentWidth * 0.08
        let frameThickness = stickThickness * 0.7
        // For front and back faces
        for face in [0, 2] {
          for c in 0..<windowCols {
            for r in 0..<windowRows {
              let colT = Float(c + 1) / Float(windowCols + 1)
              let rowT = Float(r + 1) / Float(windowRows + 1)
              let px = -halfW + colT * currentWidth
              let py = currentY + rowT * layerHeight
              let pz: Float = (face == 0) ? halfW - frameInset : -halfW + frameInset
              let w: Float = currentWidth / Float(windowCols) * 0.5
              let h: Float = layerHeight / Float(windowRows) * 0.4
              let p1 = SIMD3<Float>(px - w, py - h, pz)
              let p2 = SIMD3<Float>(px + w, py - h, pz)
              let p3 = SIMD3<Float>(px + w, py + h, pz)
              let p4 = SIMD3<Float>(px - w, py + h, pz)
              addStick(
                from: p1, to: p2, thickness: frameThickness, color: blueprintColor, asTube: true)
              addStick(
                from: p2, to: p3, thickness: frameThickness, color: blueprintColor, asTube: true)
              addStick(
                from: p3, to: p4, thickness: frameThickness, color: blueprintColor, asTube: true)
              addStick(
                from: p4, to: p1, thickness: frameThickness, color: blueprintColor, asTube: true)
              // inner cross
              addStick(
                from: SIMD3<Float>(px, py - h, pz), to: SIMD3<Float>(px, py + h, pz),
                thickness: frameThickness * 0.7, color: blueprintColor, asTube: true)
              addStick(
                from: SIMD3<Float>(px - w, py, pz), to: SIMD3<Float>(px + w, py, pz),
                thickness: frameThickness * 0.7, color: blueprintColor, asTube: true)
            }
          }
        }
      }

      // 2. Eaves (Corbelling)
      // Stack of rectangles expanding out
      let eaveStartY = currentY + layerHeight
      let eaveLayers = detailLevel == 2 ? 6 : 5
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
          let rafterCount = detailLevel == 0 ? 4 : (detailLevel == 1 ? 8 : 12)
          for _ in 0..<rafterCount {
            // Simple distribution along one side for now, or just corners
            // Let's just do corners for simplicity + midpoints
            // Actually, let's do a loop around the perimeter
            // Just corners:
            let corners = [
              SIMD2<Float>(-1, -1), SIMD2<Float>(1, -1), SIMD2<Float>(1, 1), SIMD2<Float>(-1, 1),
            ]
            for corner in corners {
              let p1 = SIMD3<Float>(corner.x * prevW * 0.5, prevY, corner.y * prevW * 0.5)
              let p2 = SIMD3<Float>(corner.x * w * 0.5, y, corner.y * w * 0.5)
              addStick(
                from: p1, to: p2, thickness: stickThickness, color: blueprintColor, asTube: true)
            }
            let beamsPerSide = detailLevel == 0 ? 12 : (detailLevel == 1 ? 28 : 60)
            let drop: Float = detailLevel == 2 ? 0.25 : 0.2
            let outExtra: Float = detailLevel == 2 ? 0.5 : 0.3
            let halfPrev = prevW * 0.5
            let halfCurr = w * 0.5
            for i in 0...beamsPerSide {
              let t = Float(i) / Float(beamsPerSide)
              let xo = -halfCurr + t * w
              let zo = -halfCurr + t * w
              let sf = SIMD3<Float>(xo, prevY, halfPrev)
              let ef = SIMD3<Float>(xo, y - drop, halfCurr + outExtra)
              addStick(
                from: sf, to: ef, thickness: stickThickness * 2.4, color: blueprintColor,
                asTube: true)
              let sr = SIMD3<Float>(halfPrev, prevY, zo)
              let er = SIMD3<Float>(halfCurr + outExtra, y - drop, zo)
              addStick(
                from: sr, to: er, thickness: stickThickness * 2.4, color: blueprintColor,
                asTube: true)
              let sb = SIMD3<Float>(xo, prevY, -halfPrev)
              let eb = SIMD3<Float>(xo, y - drop, -halfCurr - outExtra)
              addStick(
                from: sb, to: eb, thickness: stickThickness * 2.4, color: blueprintColor,
                asTube: true)
              let sl = SIMD3<Float>(-halfPrev, prevY, zo)
              let el = SIMD3<Float>(-halfCurr - outExtra, y - drop, zo)
              addStick(
                from: sl, to: el, thickness: stickThickness * 2.4, color: blueprintColor,
                asTube: true)
            }
          }
        }
      }

      // Corner bells (short vertical + diagonal)
      if detailLevel >= 1 {
        let cornerOffset = currentWidth * 0.5 + maxEaveOut
        let bellLen: Float = 0.4
        let bellY = eaveStartY + eaveHeightPerLayer * Float(eaveLayers)
        let corners = [
          SIMD2<Float>(-1, -1), SIMD2<Float>(1, -1), SIMD2<Float>(1, 1), SIMD2<Float>(-1, 1),
        ]
        for c in corners {
          let base = SIMD3<Float>(c.x * cornerOffset * 0.5, bellY, c.y * cornerOffset * 0.5)
          addStick(
            from: base, to: base + SIMD3<Float>(0, -bellLen, 0), thickness: stickThickness * 0.6,
            color: blueprintColor, asTube: true)
          addStick(
            from: base + SIMD3<Float>(0, -bellLen, 0),
            to: base + SIMD3<Float>(c.x * 0.2, -bellLen * 1.5, c.y * 0.2),
            thickness: stickThickness * 0.5, color: blueprintColor, asTube: true)
        }
      }

      // Roof slope (Retracting)
      let roofLayers = detailLevel == 2 ? 5 : 4
      let roofStartY = eaveStartY + Float(eaveLayers) * eaveHeightPerLayer
      for r in 0..<roofLayers {
        let progress = Float(r) / Float(roofLayers)
        let nextW = currentWidth - widthStep
        let startW = currentWidth + maxEaveOut * 2.0
        let w = startW - (startW - nextW) * progress
        let y = roofStartY + Float(r) * eaveHeightPerLayer

        addRect(center: SIMD3<Float>(0, y, 0), size: SIMD2<Float>(w, w), color: blueprintColor)

        // Slight curved edge using polyline (tube)
        if detailLevel >= 2 {
          let edgePointsCount = 6
          let halfW = w * 0.5
          // Front edge
          var frontEdge: [SIMD3<Float>] = []
          for i in 0..<edgePointsCount {
            let t = Float(i) / Float(edgePointsCount - 1)
            let x = -halfW + t * w
            let z = halfW
            let lift = Float(0.1) * sinf(t * Float.pi)
            frontEdge.append(SIMD3<Float>(x, y + lift, z))
          }
          addPolyline(
            points: frontEdge, thickness: stickThickness * 0.8, color: blueprintColor, asTube: true)
          // Back edge
          var backEdge: [SIMD3<Float>] = []
          for i in 0..<edgePointsCount {
            let t = Float(i) / Float(edgePointsCount - 1)
            let x = -halfW + t * w
            let z = -halfW
            let lift = Float(0.1) * sinf(t * Float.pi)
            backEdge.append(SIMD3<Float>(x, y + lift, z))
          }
          addPolyline(
            points: backEdge, thickness: stickThickness * 0.8, color: blueprintColor, asTube: true)
          // Left edge
          var leftEdge: [SIMD3<Float>] = []
          for i in 0..<edgePointsCount {
            let t = Float(i) / Float(edgePointsCount - 1)
            let z = -halfW + t * w
            let x = -halfW
            let lift = Float(0.1) * sinf(t * Float.pi)
            leftEdge.append(SIMD3<Float>(x, y + lift, z))
          }
          addPolyline(
            points: leftEdge, thickness: stickThickness * 0.8, color: blueprintColor, asTube: true)
          // Right edge
          var rightEdge: [SIMD3<Float>] = []
          for i in 0..<edgePointsCount {
            let t = Float(i) / Float(edgePointsCount - 1)
            let z = -halfW + t * w
            let x = halfW
            let lift = Float(0.1) * sinf(t * Float.pi)
            rightEdge.append(SIMD3<Float>(x, y + lift, z))
          }
          addPolyline(
            points: rightEdge, thickness: stickThickness * 0.8, color: blueprintColor, asTube: true)
        }
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

        let corners = [
          SIMD2<Float>(-1, -1), SIMD2<Float>(1, -1), SIMD2<Float>(1, 1), SIMD2<Float>(-1, 1),
        ]
        for corner in corners {
          let p1 = SIMD3<Float>(corner.x * w * 0.5, y, corner.y * w * 0.5)
          let p2 = SIMD3<Float>(corner.x * nextW * 0.5, nextY, corner.y * nextW * 0.5)
          addStick(from: p1, to: p2, thickness: stickThickness, color: blueprintColor, asTube: true)
        }
      }
    }

    // Spire
    let spireY = finialBaseY + pyramidHeight
    addStick(
      from: SIMD3<Float>(0, spireY, 0), to: SIMD3<Float>(0, spireY + 4.0, 0),
      thickness: stickThickness * 2, color: blueprintColor, asTube: true)

    // --- D. Ground Grid & Scale Reference ---
    let groundY: Float = -2.5
    let gridHalf: Float = detailLevel == 0 ? 15.0 : (detailLevel == 1 ? 20.0 : 25.0)
    let coarseStep: Float = 1.0
    let fineStep: Float = detailLevel == 2 ? 0.25 : 0.5
    // Coarse grid lines (tube)
    var x: Float = -gridHalf
    while x <= gridHalf {
      addStick(
        from: SIMD3<Float>(x, groundY, -gridHalf), to: SIMD3<Float>(x, groundY, gridHalf),
        thickness: stickThickness * 0.6, color: blueprintColor, asTube: true)
      x += coarseStep
    }
    var z: Float = -gridHalf
    while z <= gridHalf {
      addStick(
        from: SIMD3<Float>(-gridHalf, groundY, z), to: SIMD3<Float>(gridHalf, groundY, z),
        thickness: stickThickness * 0.6, color: blueprintColor, asTube: true)
      z += coarseStep
    }
    // Fine grid (sparingly)
    if detailLevel >= 2 {
      var xf: Float = -gridHalf
      while xf <= gridHalf {
        addStick(
          from: SIMD3<Float>(xf, groundY + 0.001, -gridHalf),
          to: SIMD3<Float>(xf, groundY + 0.001, gridHalf), thickness: stickThickness * 0.4,
          color: blueprintColor, asTube: true)
        xf += fineStep
      }
      var zf: Float = -gridHalf
      while zf <= gridHalf {
        addStick(
          from: SIMD3<Float>(-gridHalf, groundY + 0.001, zf),
          to: SIMD3<Float>(gridHalf, groundY + 0.001, zf), thickness: stickThickness * 0.4,
          color: blueprintColor, asTube: true)
        zf += fineStep
      }
    }

    // Height ruler (1.8m)
    let rulerHeight: Float = 1.8
    let rulerBase = SIMD3<Float>(gridHalf * 0.6, groundY, 0)
    addStick(
      from: rulerBase, to: rulerBase + SIMD3<Float>(0, rulerHeight, 0),
      thickness: stickThickness * 0.8, color: blueprintColor, asTube: true)
    // Marks every 0.2m
    for i in 1...9 {
      let y = Float(i) * 0.2
      addStick(
        from: rulerBase + SIMD3<Float>(-0.1, y, 0), to: rulerBase + SIMD3<Float>(0.1, y, 0),
        thickness: stickThickness * 0.5, color: blueprintColor, asTube: true)
    }

    return (tubeInstances, boxInstances)
  }
}
