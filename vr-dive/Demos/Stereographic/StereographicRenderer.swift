// Based on work of https://x.com/txyyss/status/2042565821281837307

import Metal
import simd

final class StereographicRenderer: VisualPatternController {
  private struct EdgeSpec {
    let axisA: Int
    let signA: Float
    let axisB: Int
    let signB: Float
    let color: SIMD4<Float>
  }

  let identifier: VisualPatternKind = .stereographicProjection
  let preferredClearColor = MTLClearColor(red: 0.01, green: 0.01, blue: 0.015, alpha: 1)

  private static let edgeSegments = 56
  private static let radialSegments = 12
  private static let sphereLatitudeSegments = 8
  private static let sphereLongitudeSegments = 12
  private static let maxDeltaTime: Float = 1.0 / 45.0
  private static let primaryRotationSpeed: Float = 0.22
  private static let secondaryRotationSpeed: Float = 0.11
  private static let worldScale: Float = 0.24
  private static let baseRadius: Float = 0.015
  private static let minimumRadius: Float = 0.0025
  private static let maximumRadius: Float = 0.024
  private static let junctionRadiusScale: Float = 1.22
  private static let junctionColor = SIMD4<Float>(0.93, 0.94, 0.97, 1.0)
  private static let e1 = SIMD4<Float>(1.0 / sqrt(2.0), -1.0 / sqrt(2.0), 0.0, 0.0)
  private static let e2 = SIMD4<Float>(0.5, 0.5, -0.5, -0.5)
  private static let e3 = SIMD4<Float>(0.0, 0.0, 1.0 / sqrt(2.0), -1.0 / sqrt(2.0))
  private static let nPole = SIMD4<Float>(repeating: 0.5)
  private static let circlePairs: [(Int, Int)] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
  ]
  private static let colors: [SIMD4<Float>] = [
    SIMD4<Float>(0.90, 0.28, 0.34, 1.0),
    SIMD4<Float>(0.96, 0.66, 0.22, 1.0),
    SIMD4<Float>(0.95, 0.90, 0.27, 1.0),
    SIMD4<Float>(0.23, 0.78, 0.50, 1.0),
    SIMD4<Float>(0.27, 0.62, 0.96, 1.0),
    SIMD4<Float>(0.72, 0.42, 0.96, 1.0),
  ]

  private let pipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  private let vertexBuffer: MTLBuffer
  private let indexBuffer: MTLBuffer
  private let indexCount: Int
  private var animationTime: Float = 0
  private var lastSimulationTimestamp: Float?

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    pipelineState = try StereographicRenderer.makePipelineState(
      device: device,
      library: library,
      maxViewCount: max(1, maxViewCount)
    )

    let depthDescriptor = MTLDepthStencilDescriptor()
    depthDescriptor.depthCompareFunction = .greater
    depthDescriptor.isDepthWriteEnabled = true
    depthStencilState = device.makeDepthStencilState(descriptor: depthDescriptor)!

    let initialVertices = Self.generateSkeletonVertices(animationTime: 0)
    vertexBuffer = device.makeBuffer(
      bytes: initialVertices,
      length: MemoryLayout<StereographicVertex>.stride * initialVertices.count,
      options: [.storageModeShared]
    )!

    let indices = Self.generateSkeletonIndices()
    indexCount = indices.count
    indexBuffer = device.makeBuffer(
      bytes: indices,
      length: MemoryLayout<UInt16>.stride * indices.count,
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
    self.lastSimulationTimestamp = context.time
    animationTime += deltaTime * 0.5 * max(0.01, context.speedMultiplier)

    let vertices = Self.generateSkeletonVertices(animationTime: animationTime)
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
      indexType: .uint16,
      indexBuffer: indexBuffer,
      indexBufferOffset: 0
    )
  }

  func resetToInitialState() {
    animationTime = 0
    lastSimulationTimestamp = nil
    let vertices = Self.generateSkeletonVertices(animationTime: 0)
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

  private static func generateSkeletonVertices(animationTime: Float) -> [StereographicVertex] {
    let edges = edgeSpecifications()
    let vertexPoints = vertexPoints4D()
    let edgeVertexCount = edges.count * (edgeSegments + 1) * radialSegments
    let sphereVertexCount = vertexPoints.count * (sphereLatitudeSegments + 1) * (sphereLongitudeSegments + 1)
    var vertices: [StereographicVertex] = []
    vertices.reserveCapacity(edgeVertexCount + sphereVertexCount)

    for edge in edges {
      var centers = Array(repeating: SIMD3<Float>(repeating: 0), count: edgeSegments + 1)
      var tangents = Array(repeating: SIMD3<Float>(repeating: 0), count: edgeSegments + 1)
      var radii = Array(repeating: Float(0), count: edgeSegments + 1)

      for i in 0...edgeSegments {
        let t = (Float(i) / Float(edgeSegments)) * (.pi * 0.5)
        var p4 = SIMD4<Float>(repeating: 0)
        p4[edge.axisA] = edge.signA * cos(t)
        p4[edge.axisB] = edge.signB * sin(t)

        let rotated = rotate4D(point: p4, time: animationTime)
        let projected = projectedPoint(for: rotated)
        centers[i] = projected.position
        radii[i] = projected.radius
      }

      for i in 0...edgeSegments {
        let prev = centers[max(0, i - 1)]
        let next = centers[min(edgeSegments, i + 1)]
        let delta = next - prev
        tangents[i] =
          simd_length_squared(delta) > 1e-8 ? simd_normalize(delta) : SIMD3<Float>(1, 0, 0)
      }

      let frames = makeTransportFrames(centers: centers, tangents: tangents)

      for i in 0...edgeSegments {
        let frame = frames[i]
        let normal = frame.normal
        let binormal = frame.binormal

        for j in 0..<radialSegments {
          let angle = (Float(j) / Float(radialSegments)) * (2.0 * .pi)
          let ringDir = normal * cos(angle) + binormal * sin(angle)
          let position = centers[i] + ringDir * radii[i]
          vertices.append(StereographicVertex(position: position, normal: ringDir, color: edge.color))
        }
      }
    }

    for point4D in vertexPoints {
      let rotated = rotate4D(point: point4D, time: animationTime)
      let projected = projectedPoint(for: rotated)
      let radius = min(projected.radius * junctionRadiusScale, maximumRadius * 1.3)

      for latitude in 0...sphereLatitudeSegments {
        let v = Float(latitude) / Float(sphereLatitudeSegments)
        let phi = v * .pi
        let sinPhi = sin(phi)
        let cosPhi = cos(phi)

        for longitude in 0...sphereLongitudeSegments {
          let u = Float(longitude) / Float(sphereLongitudeSegments)
          let theta = u * (2.0 * .pi)
          let sinTheta = sin(theta)
          let cosTheta = cos(theta)

          let normal = SIMD3<Float>(sinPhi * cosTheta, cosPhi, sinPhi * sinTheta)
          let position = projected.position + normal * radius
          vertices.append(
            StereographicVertex(position: position, normal: normal, color: junctionColor)
          )
        }
      }
    }

    return vertices
  }

  private static func generateSkeletonIndices() -> [UInt16] {
    let edgeCount = edgeSpecifications().count
    let vertexCount = vertexPoints4D().count
    let edgeStride = (edgeSegments + 1) * radialSegments
    let sphereStride = (sphereLatitudeSegments + 1) * (sphereLongitudeSegments + 1)

    var indices: [UInt16] = []
    let edgeIndexCount = edgeCount * edgeSegments * radialSegments * 6
    let sphereIndexCount = vertexCount * sphereLatitudeSegments * sphereLongitudeSegments * 6
    indices.reserveCapacity(edgeIndexCount + sphereIndexCount)

    for edgeIndex in 0..<edgeCount {
      let base = edgeIndex * edgeStride
      for i in 0..<edgeSegments {
        for j in 0..<radialSegments {
          let nj = (j + 1) % radialSegments

          let a = UInt16(base + i * radialSegments + j)
          let b = UInt16(base + (i + 1) * radialSegments + j)
          let c1 = UInt16(base + (i + 1) * radialSegments + nj)
          let d = UInt16(base + i * radialSegments + nj)

          indices.append(contentsOf: [a, b, c1, a, c1, d])
        }
      }
    }

    let sphereBase = edgeCount * edgeStride
    for sphereIndex in 0..<vertexCount {
      let base = sphereBase + sphereIndex * sphereStride
      for latitude in 0..<sphereLatitudeSegments {
        for longitude in 0..<sphereLongitudeSegments {
          let rowStride = sphereLongitudeSegments + 1
          let a = UInt16(base + latitude * rowStride + longitude)
          let b = UInt16(base + (latitude + 1) * rowStride + longitude)
          let c1 = UInt16(base + (latitude + 1) * rowStride + longitude + 1)
          let d = UInt16(base + latitude * rowStride + longitude + 1)

          indices.append(contentsOf: [a, b, c1, a, c1, d])
        }
      }
    }

    return indices
  }

  private static func rotate4D(point: SIMD4<Float>, time: Float) -> SIMD4<Float> {
    var p = point
    // Standard double rotation in two completely orthogonal invariant planes.
    p = rotateInPlane(point: p, i: 0, j: 1, angle: time * primaryRotationSpeed)
    p = rotateInPlane(point: p, i: 2, j: 3, angle: time * secondaryRotationSpeed)
    return p
  }

  private static func rotateInPlane(point: SIMD4<Float>, i: Int, j: Int, angle: Float)
    -> SIMD4<Float>
  {
    var p = point
    let c = cos(angle)
    let s = sin(angle)
    let pi = point[i]
    let pj = point[j]
    p[i] = c * pi - s * pj
    p[j] = s * pi + c * pj
    return p
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
      let prevTangent = tangents[index - 1]
      let tangent = tangents[index]
      let prevNormal = frames[index - 1].normal

      let axis = simd_cross(prevTangent, tangent)
      let axisLength = simd_length(axis)

      let transportedNormal: SIMD3<Float>
      if axisLength > 1e-5 {
        let unitAxis = axis / axisLength
        let angle = atan2(axisLength, simd_dot(prevTangent, tangent))
        transportedNormal = rotate(vector: prevNormal, around: unitAxis, angle: angle)
      } else {
        transportedNormal = prevNormal
      }

      let orthogonalNormal = simd_normalize(
        transportedNormal - tangent * simd_dot(transportedNormal, tangent))
      let binormal = simd_normalize(simd_cross(tangent, orthogonalNormal))
      frames[index] = (orthogonalNormal, binormal)
    }

    if centers.count <= 2 {
      return frames
    }

    let startNormal = frames[0].normal
    let endNormal = frames[centers.count - 1].normal
    let endTangent = tangents[0]
    let closureAngle = signedAngle(from: endNormal, to: startNormal, around: endTangent)

    if abs(closureAngle) > 1e-4 {
      for index in 0..<frames.count {
        let correction = closureAngle * (Float(index) / Float(frames.count))
        let tangent = tangents[index]
        let normal = rotate(vector: frames[index].normal, around: tangent, angle: correction)
        let binormal = simd_normalize(simd_cross(tangent, normal))
        frames[index] = (simd_normalize(normal), binormal)
      }
    }

    return frames
  }

  private static func projectedPoint(for point: SIMD4<Float>) -> (position: SIMD3<Float>, radius: Float) {
    let dotNPole = simd_dot(point, nPole)
    let denom = max(1e-4, 1.0 - dotNPole)
    let projection = (point - dotNPole * nPole) / denom

    let u = simd_dot(projection, e1)
    let v = simd_dot(projection, e2)
    let w = simd_dot(projection, e3)
    let position = SIMD3<Float>(u, v, w) * worldScale

    let perspectiveRadius = baseRadius * worldScale / denom
    let radius = min(max(perspectiveRadius, minimumRadius), maximumRadius)
    return (position, radius)
  }

  private static func edgeSpecifications() -> [EdgeSpec] {
    var edges: [EdgeSpec] = []
    edges.reserveCapacity(circlePairs.count * 4)

    let signs: [Float] = [-1.0, 1.0]
    for (pairIndex, pair) in circlePairs.enumerated() {
      let color = colors[pairIndex]
      for signA in signs {
        for signB in signs {
          edges.append(
            EdgeSpec(
              axisA: pair.0,
              signA: signA,
              axisB: pair.1,
              signB: signB,
              color: color
            )
          )
        }
      }
    }

    return edges
  }

  private static func vertexPoints4D() -> [SIMD4<Float>] {
    var points: [SIMD4<Float>] = []
    points.reserveCapacity(8)

    for axis in 0..<4 {
      for sign: Float in [-1.0, 1.0] {
        var point = SIMD4<Float>(repeating: 0)
        point[axis] = sign
        points.append(point)
      }
    }

    return points
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

  private static func signedAngle(
    from start: SIMD3<Float>,
    to end: SIMD3<Float>,
    around axis: SIMD3<Float>
  ) -> Float {
    let crossValue = simd_cross(start, end)
    let sine = simd_dot(axis, crossValue)
    let cosine = simd_dot(start, end)
    return atan2(sine, cosine)
  }

}
