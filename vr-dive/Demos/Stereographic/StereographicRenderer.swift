import Metal
import simd

enum RegularPolychoronKind {
  case sixteenCell
  case twentyFourCell
  case oneHundredTwentyCell
  case sixHundredCell
}

final class StereographicRenderer: VisualPatternController {
  private struct PolychoronEdge {
    let start: Int
    let end: Int
  }

  private struct PolychoronDefinition {
    let kind: RegularPolychoronKind
    let vertices4D: [SIMD4<Float>]
    let edges: [PolychoronEdge]
    let edgeColors: [SIMD4<Float>]
  }

  private struct PolychoronStyle {
    let edgeSegments: Int
    let radialSegments: Int
    let sphereLatitudeSegments: Int
    let sphereLongitudeSegments: Int
    let worldScale: Float
    let worldOffset: SIMD3<Float>
    let baseRadius: Float
    let minimumRadius: Float
    let maximumRadius: Float
    let junctionRadiusScale: Float
    let junctionColor: SIMD4<Float>
  }

  private struct MeshLayout {
    let edgeStride: Int
    let sphereStride: Int
    let totalVertexCount: Int
  }

  let identifier: VisualPatternKind
  let preferredClearColor = MTLClearColor(red: 0.01, green: 0.01, blue: 0.015, alpha: 1)

  private static let maxDeltaTime: Float = 1.0 / 45.0
  private static let primaryRotationSpeed: Float = 0.18
  private static let secondaryRotationSpeed: Float = 0.09
  private static let baseOrientationAngles = SIMD3<Float>(-0.8, -0.66, 0.96)
  private static let junctionHighlight = SIMD4<Float>(0.93, 0.94, 0.97, 1.0)
  private static let e1 = SIMD4<Float>(1.0 / sqrt(2.0), -1.0 / sqrt(2.0), 0.0, 0.0)
  private static let e2 = SIMD4<Float>(0.5, 0.5, -0.5, -0.5)
  private static let e3 = SIMD4<Float>(0.0, 0.0, 1.0 / sqrt(2.0), -1.0 / sqrt(2.0))
  private static let nPole = SIMD4<Float>(repeating: 0.5)
  private static let edgePalette: [SIMD4<Float>] = [
    SIMD4<Float>(0.90, 0.28, 0.34, 1.0),
    SIMD4<Float>(0.96, 0.66, 0.22, 1.0),
    SIMD4<Float>(0.95, 0.90, 0.27, 1.0),
    SIMD4<Float>(0.23, 0.78, 0.50, 1.0),
    SIMD4<Float>(0.27, 0.62, 0.96, 1.0),
    SIMD4<Float>(0.72, 0.42, 0.96, 1.0),
  ]
  private static let definitionsByKind = buildPolychoronDefinitions()

  private let pipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  private let vertexBuffer: MTLBuffer
  private let indexBuffer: MTLBuffer
  private let indexCount: Int
  private let definition: PolychoronDefinition
  private let style: PolychoronStyle
  private let meshLayout: MeshLayout
  private var animationTime: Float = 0
  private var lastSimulationTimestamp: Float?

  init(
    device: MTLDevice,
    library: MTLLibrary,
    patternKind: VisualPatternKind,
    polychoronKind: RegularPolychoronKind,
    maxViewCount: Int
  ) throws {
    guard let definition = Self.definitionsByKind[polychoronKind] else {
      preconditionFailure("Missing polychoron definition for \(polychoronKind)")
    }

    self.identifier = patternKind
    self.definition = definition
    self.style = Self.style(for: polychoronKind)
    self.meshLayout = Self.meshLayout(for: definition, style: style)

    pipelineState = try Self.makePipelineState(
      device: device,
      library: library,
      maxViewCount: max(1, maxViewCount)
    )

    let depthDescriptor = MTLDepthStencilDescriptor()
    depthDescriptor.depthCompareFunction = .greater
    depthDescriptor.isDepthWriteEnabled = true
    depthStencilState = device.makeDepthStencilState(descriptor: depthDescriptor)!

    let initialVertices = Self.generateMeshVertices(
      definition: definition,
      style: style,
      meshLayout: meshLayout,
      animationTime: 0
    )
    vertexBuffer = device.makeBuffer(
      bytes: initialVertices,
      length: MemoryLayout<StereographicVertex>.stride * initialVertices.count,
      options: [.storageModeShared]
    )!

    let indices = Self.generateMeshIndices(definition: definition, style: style, meshLayout: meshLayout)
    indexCount = indices.count
    indexBuffer = device.makeBuffer(
      bytes: indices,
      length: MemoryLayout<UInt32>.stride * indices.count,
      options: [.storageModeShared]
    )!
  }

  func updateSimulation(_ context: PatternSimulationContext) {
    let deltaTime: Float
    if let lastSimulationTimestamp {
      deltaTime = min(max(context.time - lastSimulationTimestamp, 0), Self.maxDeltaTime)
    } else {
      deltaTime = 0
    }
    lastSimulationTimestamp = context.time
    animationTime += deltaTime * max(0, context.speedMultiplier)

    let vertices = Self.generateMeshVertices(
      definition: definition,
      style: style,
      meshLayout: meshLayout,
      animationTime: animationTime
    )
    vertices.withUnsafeBytes { ptr in
      guard let base = ptr.baseAddress, ptr.count > 0 else { return }
      memcpy(vertexBuffer.contents(), base, ptr.count)
    }
  }

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    encoder.setRenderPipelineState(pipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)
    encoder.setFrontFacing(.counterClockwise)

    context.applyViewConfiguration(on: encoder)

    encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)

    var uniforms = SceneUniforms(time: context.time, layerCount: UInt32(context.viewData.viewCount))
    encoder.setVertexBytes(&uniforms, length: MemoryLayout<SceneUniforms>.stride, index: 2)

    var matrices = context.viewData.viewProjectionMatrices
    if matrices.isEmpty {
      matrices = [matrix_identity_float4x4]
    }
    matrices.withUnsafeBytes { ptr in
      if let base = ptr.baseAddress, ptr.count > 0 {
        encoder.setVertexBytes(base, length: ptr.count, index: 3)
      }
    }

    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: indexCount,
      indexType: .uint32,
      indexBuffer: indexBuffer,
      indexBufferOffset: 0
    )
  }

  func resetToInitialState() {
    animationTime = 0
    lastSimulationTimestamp = nil
    let vertices = Self.generateMeshVertices(
      definition: definition,
      style: style,
      meshLayout: meshLayout,
      animationTime: 0
    )
    vertices.withUnsafeBytes { ptr in
      guard let base = ptr.baseAddress, ptr.count > 0 else { return }
      memcpy(vertexBuffer.contents(), base, ptr.count)
    }
  }

  private static func makePipelineState(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: "stereographicVertexShader")
    descriptor.fragmentFunction = library.makeFunction(name: "stereographicFragmentShader")
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
    vertexDescriptor.attributes[2].format = .float4
    vertexDescriptor.attributes[2].offset = MemoryLayout<SIMD3<Float>>.stride * 2
    vertexDescriptor.attributes[2].bufferIndex = 0
    vertexDescriptor.layouts[0].stride = MemoryLayout<StereographicVertex>.stride
    descriptor.vertexDescriptor = vertexDescriptor
    descriptor.maxVertexAmplificationCount = maxViewCount

    return try device.makeRenderPipelineState(descriptor: descriptor)
  }

  private static func generateMeshVertices(
    definition: PolychoronDefinition,
    style: PolychoronStyle,
    meshLayout: MeshLayout,
    animationTime: Float
  ) -> [StereographicVertex] {
    var vertices: [StereographicVertex] = []
    vertices.reserveCapacity(meshLayout.totalVertexCount)

    for (edgeIndex, edge) in definition.edges.enumerated() {
      let start4D = rotate4D(point: definition.vertices4D[edge.start], time: animationTime)
      let end4D = rotate4D(point: definition.vertices4D[edge.end], time: animationTime)

      var centers = Array(repeating: SIMD3<Float>(repeating: 0), count: style.edgeSegments + 1)
      var tangents = Array(repeating: SIMD3<Float>(repeating: 0), count: style.edgeSegments + 1)
      var radii = Array(repeating: Float(0), count: style.edgeSegments + 1)

      for segmentIndex in 0...style.edgeSegments {
        let interpolation = Float(segmentIndex) / Float(style.edgeSegments)
        let point = sphericalInterpolate(from: start4D, to: end4D, t: interpolation)
        let projected = projectedPoint(for: point, style: style)
        centers[segmentIndex] = projected.position
        radii[segmentIndex] = projected.radius
      }

      for segmentIndex in 0...style.edgeSegments {
        let previous = centers[max(0, segmentIndex - 1)]
        let next = centers[min(style.edgeSegments, segmentIndex + 1)]
        let delta = next - previous
        tangents[segmentIndex] =
          simd_length_squared(delta) > 1e-8 ? simd_normalize(delta) : SIMD3<Float>(1, 0, 0)
      }

      let frames = makeTransportFrames(centers: centers, tangents: tangents)
      let color = definition.edgeColors[edgeIndex]

      for segmentIndex in 0...style.edgeSegments {
        let frame = frames[segmentIndex]
        for radialIndex in 0..<style.radialSegments {
          let angle = (Float(radialIndex) / Float(style.radialSegments)) * (2.0 * .pi)
          let ringDirection = frame.normal * cos(angle) + frame.binormal * sin(angle)
          let position = centers[segmentIndex] + ringDirection * radii[segmentIndex]
          vertices.append(
            StereographicVertex(position: position, normal: ringDirection, color: color)
          )
        }
      }
    }

    for point4D in definition.vertices4D {
      let rotated = rotate4D(point: point4D, time: animationTime)
      let projected = projectedPoint(for: rotated, style: style)
      let radius = min(projected.radius * style.junctionRadiusScale, style.maximumRadius * 1.3)

      for latitude in 0...style.sphereLatitudeSegments {
        let v = Float(latitude) / Float(style.sphereLatitudeSegments)
        let phi = v * .pi
        let sinPhi = sin(phi)
        let cosPhi = cos(phi)

        for longitude in 0...style.sphereLongitudeSegments {
          let u = Float(longitude) / Float(style.sphereLongitudeSegments)
          let theta = u * (2.0 * .pi)
          let sinTheta = sin(theta)
          let cosTheta = cos(theta)

          let normal = SIMD3<Float>(sinPhi * cosTheta, cosPhi, sinPhi * sinTheta)
          let position = projected.position + normal * radius
          vertices.append(
            StereographicVertex(position: position, normal: normal, color: style.junctionColor)
          )
        }
      }
    }

    return vertices
  }

  private static func generateMeshIndices(
    definition: PolychoronDefinition,
    style: PolychoronStyle,
    meshLayout: MeshLayout
  ) -> [UInt32] {
    var indices: [UInt32] = []
    let edgeIndexCount = definition.edges.count * style.edgeSegments * style.radialSegments * 6
    let sphereIndexCount =
      definition.vertices4D.count * style.sphereLatitudeSegments * style.sphereLongitudeSegments * 6
    indices.reserveCapacity(edgeIndexCount + sphereIndexCount)

    for edgeIndex in definition.edges.indices {
      let base = edgeIndex * meshLayout.edgeStride
      for segmentIndex in 0..<style.edgeSegments {
        for radialIndex in 0..<style.radialSegments {
          let nextRadial = (radialIndex + 1) % style.radialSegments

          let a = UInt32(base + segmentIndex * style.radialSegments + radialIndex)
          let b = UInt32(base + (segmentIndex + 1) * style.radialSegments + radialIndex)
          let c = UInt32(base + (segmentIndex + 1) * style.radialSegments + nextRadial)
          let d = UInt32(base + segmentIndex * style.radialSegments + nextRadial)

          indices.append(contentsOf: [a, b, c, a, c, d])
        }
      }
    }

    let sphereBase = definition.edges.count * meshLayout.edgeStride
    let sphereRowStride = style.sphereLongitudeSegments + 1
    for vertexIndex in definition.vertices4D.indices {
      let base = sphereBase + vertexIndex * meshLayout.sphereStride
      for latitude in 0..<style.sphereLatitudeSegments {
        for longitude in 0..<style.sphereLongitudeSegments {
          let a = UInt32(base + latitude * sphereRowStride + longitude)
          let b = UInt32(base + (latitude + 1) * sphereRowStride + longitude)
          let c = UInt32(base + (latitude + 1) * sphereRowStride + longitude + 1)
          let d = UInt32(base + latitude * sphereRowStride + longitude + 1)

          indices.append(contentsOf: [a, b, c, a, c, d])
        }
      }
    }

    return indices
  }

  private static func meshLayout(for definition: PolychoronDefinition, style: PolychoronStyle) -> MeshLayout {
    let edgeStride = (style.edgeSegments + 1) * style.radialSegments
    let sphereStride = (style.sphereLatitudeSegments + 1) * (style.sphereLongitudeSegments + 1)
    let totalVertexCount = definition.edges.count * edgeStride + definition.vertices4D.count * sphereStride
    return MeshLayout(edgeStride: edgeStride, sphereStride: sphereStride, totalVertexCount: totalVertexCount)
  }

  private static func style(for kind: RegularPolychoronKind) -> PolychoronStyle {
    switch kind {
    case .sixteenCell:
      return PolychoronStyle(
        edgeSegments: 36,
        radialSegments: 12,
        sphereLatitudeSegments: 6,
        sphereLongitudeSegments: 10,
        worldScale: 0.26,
        worldOffset: SIMD3<Float>(0, 0, -1.2),
        baseRadius: 0.018,
        minimumRadius: 0.003,
        maximumRadius: 0.024,
        junctionRadiusScale: 1.15,
        junctionColor: junctionHighlight
      )
    case .twentyFourCell:
      return PolychoronStyle(
        edgeSegments: 20,
        radialSegments: 10,
        sphereLatitudeSegments: 5,
        sphereLongitudeSegments: 8,
        worldScale: 0.24,
        worldOffset: SIMD3<Float>(0, 0, -1.2),
        baseRadius: 0.013,
        minimumRadius: 0.0022,
        maximumRadius: 0.016,
        junctionRadiusScale: 1.12,
        junctionColor: junctionHighlight
      )
    case .oneHundredTwentyCell:
      return PolychoronStyle(
        edgeSegments: 10,
        radialSegments: 5,
        sphereLatitudeSegments: 3,
        sphereLongitudeSegments: 5,
        worldScale: 0.15,
        worldOffset: SIMD3<Float>(0, 0, -1.25),
        baseRadius: 0.007,
        minimumRadius: 0.0012,
        maximumRadius: 0.007,
        junctionRadiusScale: 1.05,
        junctionColor: junctionHighlight
      )
    case .sixHundredCell:
      return PolychoronStyle(
        edgeSegments: 10,
        radialSegments: 6,
        sphereLatitudeSegments: 4,
        sphereLongitudeSegments: 6,
        worldScale: 0.18,
        worldOffset: SIMD3<Float>(0, 0, -1.25),
        baseRadius: 0.008,
        minimumRadius: 0.0015,
        maximumRadius: 0.010,
        junctionRadiusScale: 1.08,
        junctionColor: junctionHighlight
      )
    }
  }

  private static func buildPolychoronDefinitions() -> [RegularPolychoronKind: PolychoronDefinition] {
    let sixteenCellVertices = buildSixteenCellVertices()
    let twentyFourCellVertices = buildTwentyFourCellVertices()
    let sixHundredCellVertices = buildSixHundredCellVertices()

    let sixteenCellEdges = buildEdgePairs(
      vertices: sixteenCellVertices,
      expectedValence: 6,
      expectedEdgeCount: 24
    )
    let twentyFourCellEdges = buildEdgePairs(
      vertices: twentyFourCellVertices,
      expectedValence: 8,
      expectedEdgeCount: 96
    )
    let sixHundredCellEdges = buildEdgePairs(
      vertices: sixHundredCellVertices,
      expectedValence: 12,
      expectedEdgeCount: 720
    )
    let sixHundredCellTetrahedra = buildTetrahedralCells(
      vertexCount: sixHundredCellVertices.count,
      edges: sixHundredCellEdges
    )
    precondition(sixHundredCellTetrahedra.count == 600, "Expected 600 tetrahedra for the 600-cell")

    let oneHundredTwentyCellVertices = buildDualVertices(
      cells: sixHundredCellTetrahedra,
      sourceVertices: sixHundredCellVertices,
      expectedVertexCount: 600
    )
    let oneHundredTwentyCellEdges = buildEdgePairs(
      vertices: oneHundredTwentyCellVertices,
      expectedValence: 4,
      expectedEdgeCount: 1200
    )

    return [
      .sixteenCell: makeDefinition(
        kind: .sixteenCell,
        vertices: sixteenCellVertices,
        edges: sixteenCellEdges
      ),
      .twentyFourCell: makeDefinition(
        kind: .twentyFourCell,
        vertices: twentyFourCellVertices,
        edges: twentyFourCellEdges
      ),
      .oneHundredTwentyCell: makeDefinition(
        kind: .oneHundredTwentyCell,
        vertices: oneHundredTwentyCellVertices,
        edges: oneHundredTwentyCellEdges
      ),
      .sixHundredCell: makeDefinition(
        kind: .sixHundredCell,
        vertices: sixHundredCellVertices,
        edges: sixHundredCellEdges
      ),
    ]
  }

  private static func makeDefinition(
    kind: RegularPolychoronKind,
    vertices: [SIMD4<Double>],
    edges: [PolychoronEdge]
  ) -> PolychoronDefinition {
    let floatVertices = vertices.map(floatVector)
    let edgeColors = edges.enumerated().map { edgePalette[$0.offset % edgePalette.count] }
    return PolychoronDefinition(
      kind: kind,
      vertices4D: floatVertices,
      edges: edges,
      edgeColors: edgeColors
    )
  }

  private static func buildSixteenCellVertices() -> [SIMD4<Double>] {
    var vertices: [SIMD4<Double>] = []
    vertices.reserveCapacity(8)

    for axis in 0..<4 {
      for sign in [-1.0, 1.0] {
        var point = SIMD4<Double>(repeating: 0)
        point[axis] = sign
        vertices.append(point)
      }
    }

    return vertices
  }

  private static func buildTwentyFourCellVertices() -> [SIMD4<Double>] {
    let scale = 1.0 / sqrt(2.0)
    var vertices: [SIMD4<Double>] = []
    vertices.reserveCapacity(24)

    for axisA in 0..<4 {
      for axisB in (axisA + 1)..<4 {
        for signA in [-1.0, 1.0] {
          for signB in [-1.0, 1.0] {
            var point = SIMD4<Double>(repeating: 0)
            point[axisA] = signA * scale
            point[axisB] = signB * scale
            vertices.append(point)
          }
        }
      }
    }

    return vertices
  }

  private static func buildSixHundredCellVertices() -> [SIMD4<Double>] {
    let phi = (1.0 + sqrt(5.0)) * 0.5
    let inversePhi = 1.0 / phi
    var vertices: [SIMD4<Double>] = []
    vertices.reserveCapacity(120)

    for axis in 0..<4 {
      for sign in [-1.0, 1.0] {
        var point = SIMD4<Double>(repeating: 0)
        point[axis] = sign
        vertices.append(point)
      }
    }

    for signX in [-0.5, 0.5] {
      for signY in [-0.5, 0.5] {
        for signZ in [-0.5, 0.5] {
          for signW in [-0.5, 0.5] {
            vertices.append(SIMD4<Double>(signX, signY, signZ, signW))
          }
        }
      }
    }

    let base = SIMD4<Double>(0, 0.5, phi * 0.5, inversePhi * 0.5)
    for permutation in coordinatePermutations(evenOnly: true) {
      let permuted = permute(base, by: permutation)
      let nonZeroAxes = (0..<4).filter { abs(permuted[$0]) > 1e-10 }
      let signCombinations = 1 << nonZeroAxes.count
      for signMask in 0..<signCombinations {
        var point = permuted
        for (bitIndex, axis) in nonZeroAxes.enumerated() {
          let sign = ((signMask >> bitIndex) & 1) == 0 ? -1.0 : 1.0
          point[axis] *= sign
        }
        appendUnique(point, to: &vertices)
      }
    }

    precondition(vertices.count == 120, "Expected 120 vertices for the 600-cell")
    return vertices
  }

  private static func buildEdgePairs(
    vertices: [SIMD4<Double>],
    expectedValence: Int,
    expectedEdgeCount: Int
  ) -> [PolychoronEdge] {
    let threshold = adjacencyThreshold(vertices: vertices, expectedValence: expectedValence)
    var edges: [PolychoronEdge] = []
    edges.reserveCapacity(expectedEdgeCount)
    var degrees = Array(repeating: 0, count: vertices.count)

    for start in vertices.indices {
      for end in (start + 1)..<vertices.count {
        let distance = squaredDistance(vertices[start], vertices[end])
        if distance <= threshold {
          edges.append(PolychoronEdge(start: start, end: end))
          degrees[start] += 1
          degrees[end] += 1
        }
      }
    }

    precondition(edges.count == expectedEdgeCount, "Unexpected edge count for \(expectedEdgeCount)-edge polychoron")
    precondition(degrees.allSatisfy { $0 == expectedValence }, "Unexpected valence distribution")
    return edges
  }

  private static func buildTetrahedralCells(
    vertexCount: Int,
    edges: [PolychoronEdge]
  ) -> [[Int]] {
    var adjacency = Array(repeating: Set<Int>(), count: vertexCount)
    for edge in edges {
      adjacency[edge.start].insert(edge.end)
      adjacency[edge.end].insert(edge.start)
    }

    var tetrahedra: [[Int]] = []
    tetrahedra.reserveCapacity(600)

    for a in 0..<vertexCount {
      let neighborsA = adjacency[a].filter { $0 > a }.sorted()
      let neighborSetA = Set(neighborsA)

      for b in neighborsA {
        let commonAB = neighborSetA.intersection(adjacency[b].filter { $0 > b })
        for c in commonAB.sorted() {
          let commonABC = commonAB.intersection(adjacency[c].filter { $0 > c })
          for d in commonABC.sorted() {
            tetrahedra.append([a, b, c, d])
          }
        }
      }
    }

    return tetrahedra
  }

  private static func buildDualVertices(
    cells: [[Int]],
    sourceVertices: [SIMD4<Double>],
    expectedVertexCount: Int
  ) -> [SIMD4<Double>] {
    var dualVertices: [SIMD4<Double>] = []
    dualVertices.reserveCapacity(expectedVertexCount)

    for cell in cells {
      var sum = SIMD4<Double>(repeating: 0)
      for index in cell {
        sum += sourceVertices[index]
      }
      appendUnique(normalized(sum), to: &dualVertices)
    }

    precondition(dualVertices.count == expectedVertexCount, "Unexpected dual vertex count")
    return dualVertices
  }

  private static func adjacencyThreshold(vertices: [SIMD4<Double>], expectedValence: Int) -> Double {
    var threshold = 0.0

    for index in vertices.indices {
      var distances: [Double] = []
      distances.reserveCapacity(vertices.count - 1)
      for neighborIndex in vertices.indices where neighborIndex != index {
        distances.append(squaredDistance(vertices[index], vertices[neighborIndex]))
      }
      distances.sort()
      threshold = max(threshold, distances[expectedValence - 1])
    }

    return threshold + 1e-9
  }

  private static func rotate4D(point: SIMD4<Float>, time: Float) -> SIMD4<Float> {
    var rotated = point
    rotated = rotateInPlane(point: rotated, i: 0, j: 3, angle: baseOrientationAngles.x)
    rotated = rotateInPlane(point: rotated, i: 1, j: 2, angle: baseOrientationAngles.y)
    rotated = rotateInPlane(point: rotated, i: 0, j: 1, angle: baseOrientationAngles.z)
    rotated = rotateInPlane(point: rotated, i: 0, j: 1, angle: time * primaryRotationSpeed)
    rotated = rotateInPlane(point: rotated, i: 2, j: 3, angle: time * secondaryRotationSpeed)
    return rotated
  }

  private static func rotateInPlane(point: SIMD4<Float>, i: Int, j: Int, angle: Float) -> SIMD4<Float> {
    var rotated = point
    let cosine = cos(angle)
    let sine = sin(angle)
    let componentI = point[i]
    let componentJ = point[j]
    rotated[i] = cosine * componentI - sine * componentJ
    rotated[j] = sine * componentI + cosine * componentJ
    return rotated
  }

  private static func sphericalInterpolate(
    from start: SIMD4<Float>,
    to end: SIMD4<Float>,
    t: Float
  ) -> SIMD4<Float> {
    let clampedDot = max(-1.0, min(1.0, simd_dot(start, end)))
    if clampedDot > 0.9995 {
      let linear = start + (end - start) * t
      return simd_normalize(linear)
    }

    let theta = acos(clampedDot)
    let sinTheta = sin(theta)
    let startWeight = sin((1.0 - t) * theta) / sinTheta
    let endWeight = sin(t * theta) / sinTheta
    return simd_normalize(start * startWeight + end * endWeight)
  }

  private static func makeTransportFrames(centers: [SIMD3<Float>], tangents: [SIMD3<Float>])
    -> [(normal: SIMD3<Float>, binormal: SIMD3<Float>)]
  {
    guard !centers.isEmpty else { return [] }

    var frames = Array(
      repeating: (normal: SIMD3<Float>(1, 0, 0), binormal: SIMD3<Float>(0, 1, 0)),
      count: centers.count
    )

    let initialUp =
      abs(simd_dot(tangents[0], SIMD3<Float>(0, 1, 0))) > 0.92
      ? SIMD3<Float>(1, 0, 0)
      : SIMD3<Float>(0, 1, 0)
    let initialNormal = simd_normalize(simd_cross(tangents[0], initialUp))
    let initialBinormal = simd_normalize(simd_cross(tangents[0], initialNormal))
    frames[0] = (initialNormal, initialBinormal)

    for index in 1..<centers.count {
      let previousTangent = tangents[index - 1]
      let tangent = tangents[index]
      let previousNormal = frames[index - 1].normal

      let axis = simd_cross(previousTangent, tangent)
      let axisLength = simd_length(axis)

      let transportedNormal: SIMD3<Float>
      if axisLength > 1e-5 {
        let unitAxis = axis / axisLength
        let angle = atan2(axisLength, simd_dot(previousTangent, tangent))
        transportedNormal = rotate(vector: previousNormal, around: unitAxis, angle: angle)
      } else {
        transportedNormal = previousNormal
      }

      let orthogonalNormal = simd_normalize(
        transportedNormal - tangent * simd_dot(transportedNormal, tangent)
      )
      let binormal = simd_normalize(simd_cross(tangent, orthogonalNormal))
      frames[index] = (orthogonalNormal, binormal)
    }

    return frames
  }

  private static func projectedPoint(for point: SIMD4<Float>, style: PolychoronStyle)
    -> (position: SIMD3<Float>, radius: Float)
  {
    let dotWithPole = simd_dot(point, nPole)
    let denominator = max(1e-4, 1.0 - dotWithPole)
    let projection = (point - dotWithPole * nPole) / denominator

    let u = simd_dot(projection, e1)
    let v = simd_dot(projection, e2)
    let w = simd_dot(projection, e3)
    let position = SIMD3<Float>(u, v, w) * style.worldScale + style.worldOffset

    let perspectiveRadius = style.baseRadius * style.worldScale / denominator
    let radius = min(max(perspectiveRadius, style.minimumRadius), style.maximumRadius)
    return (position, radius)
  }

  private static func rotate(vector: SIMD3<Float>, around axis: SIMD3<Float>, angle: Float)
    -> SIMD3<Float>
  {
    let cosine = cos(angle)
    let sine = sin(angle)
    return vector * cosine
      + simd_cross(axis, vector) * sine
      + axis * simd_dot(axis, vector) * (1 - cosine)
  }

  private static func appendUnique(_ candidate: SIMD4<Double>, to vertices: inout [SIMD4<Double>]) {
    if vertices.contains(where: { simd_length_squared($0 - candidate) < 1e-16 }) {
      return
    }
    vertices.append(candidate)
  }

  private static func normalized(_ vector: SIMD4<Double>) -> SIMD4<Double> {
    let length = sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z + vector.w * vector.w)
    return vector / length
  }

  private static func squaredDistance(_ left: SIMD4<Double>, _ right: SIMD4<Double>) -> Double {
    let delta = left - right
    return delta.x * delta.x + delta.y * delta.y + delta.z * delta.z + delta.w * delta.w
  }

  private static func floatVector(_ vector: SIMD4<Double>) -> SIMD4<Float> {
    SIMD4<Float>(Float(vector.x), Float(vector.y), Float(vector.z), Float(vector.w))
  }

  private static func permute(_ vector: SIMD4<Double>, by permutation: [Int]) -> SIMD4<Double> {
    SIMD4<Double>(
      vector[permutation[0]],
      vector[permutation[1]],
      vector[permutation[2]],
      vector[permutation[3]]
    )
  }

  private static func coordinatePermutations(evenOnly: Bool) -> [[Int]] {
    var permutations: [[Int]] = []
    let values = [0, 1, 2, 3]

    func recurse(_ prefix: [Int], _ remaining: [Int]) {
      if remaining.isEmpty {
        if !evenOnly || inversionCount(of: prefix).isMultiple(of: 2) {
          permutations.append(prefix)
        }
        return
      }

      for index in remaining.indices {
        var nextPrefix = prefix
        nextPrefix.append(remaining[index])
        var nextRemaining = remaining
        nextRemaining.remove(at: index)
        recurse(nextPrefix, nextRemaining)
      }
    }

    recurse([], values)
    return permutations
  }

  private static func inversionCount(of values: [Int]) -> Int {
    var inversions = 0
    for i in values.indices {
      for j in (i + 1)..<values.count where values[i] > values[j] {
        inversions += 1
      }
    }
    return inversions
  }
}
