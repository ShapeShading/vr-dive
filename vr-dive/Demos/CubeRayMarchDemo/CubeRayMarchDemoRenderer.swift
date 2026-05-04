import Metal
import simd

private struct CRMVertex {
  var position: SIMD3<Float>
  var normal: SIMD3<Float>
  var color: SIMD3<Float>
}

private struct CRMGeometry {
  var vertices: [CRMVertex] = []
  var indices: [UInt32] = []

  mutating func append(vertices newVertices: [CRMVertex], indices newIndices: [UInt32]) {
    let base = UInt32(vertices.count)
    vertices += newVertices
    indices += newIndices.map { $0 + base }
  }
}

private struct CRMTraceStep {
  var distanceAlongRay: Float
  var radius: Float
}

private struct CRMTraceResult {
  var steps: [CRMTraceStep]
  var hitDistance: Float
}

private let crmTraceEpsilon: Float = 0.005

private func crmBasis(for axis: SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) {
  let tangent = simd_normalize(axis)
  let helper = abs(tangent.x) > 0.9 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0)
  let u = simd_normalize(simd_cross(tangent, helper))
  return (u, simd_cross(tangent, u))
}

private func crmAppendSphere(
  _ geometry: inout CRMGeometry,
  center: SIMD3<Float>,
  radius: Float,
  color: SIMD3<Float>,
  latSegments: Int = 10,
  lonSegments: Int = 20
) {
  var vertices: [CRMVertex] = []
  var indices: [UInt32] = []
  vertices.reserveCapacity((latSegments + 1) * (lonSegments + 1))
  indices.reserveCapacity(latSegments * lonSegments * 6)

  for lat in 0...latSegments {
    let theta = Float(lat) * .pi / Float(latSegments)
    let sinTheta = sin(theta)
    let cosTheta = cos(theta)
    for lon in 0...lonSegments {
      let phi = Float(lon) * 2 * .pi / Float(lonSegments)
      let normal = SIMD3<Float>(cos(phi) * sinTheta, cosTheta, sin(phi) * sinTheta)
      vertices.append(CRMVertex(position: center + normal * radius, normal: normal, color: color))
    }
  }

  let ring = lonSegments + 1
  for lat in 0..<latSegments {
    for lon in 0..<lonSegments {
      let i0 = UInt32(lat * ring + lon)
      let i1 = UInt32((lat + 1) * ring + lon)
      let i2 = i0 + 1
      let i3 = i1 + 1
      indices += [i0, i1, i2, i2, i1, i3]
    }
  }

  geometry.append(vertices: vertices, indices: indices)
}

private func crmAppendCylinder(
  _ geometry: inout CRMGeometry,
  from start: SIMD3<Float>,
  to end: SIMD3<Float>,
  radius: Float,
  color: SIMD3<Float>,
  radialSegments: Int = 12
) {
  let axis = end - start
  let length = simd_length(axis)
  guard length > 1e-5 else { return }
  let tangent = axis / length
  let (u, v) = crmBasis(for: tangent)

  var vertices: [CRMVertex] = []
  var indices: [UInt32] = []
  vertices.reserveCapacity((radialSegments + 1) * 2)
  indices.reserveCapacity(radialSegments * 6)

  for ringIndex in 0...1 {
    let center = ringIndex == 0 ? start : end
    for segment in 0...radialSegments {
      let angle = Float(segment) * 2 * .pi / Float(radialSegments)
      let normal = simd_normalize(cos(angle) * u + sin(angle) * v)
      vertices.append(CRMVertex(position: center + normal * radius, normal: normal, color: color))
    }
  }

  let ring = radialSegments + 1
  for segment in 0..<radialSegments {
    let i0 = UInt32(segment)
    let i1 = UInt32(segment + ring)
    let i2 = i0 + 1
    let i3 = i1 + 1
    indices += [i0, i1, i2, i2, i1, i3]
  }

  geometry.append(vertices: vertices, indices: indices)
}

private func crmAppendCircle(
  _ geometry: inout CRMGeometry,
  center: SIMD3<Float>,
  axis: SIMD3<Float>,
  radius: Float,
  tubeRadius: Float,
  color: SIMD3<Float>,
  segments: Int = 40
) {
  let (u, v) = crmBasis(for: axis)
  var lastPoint = center + u * radius
  for segment in 1...segments {
    let angle = Float(segment) * 2 * .pi / Float(segments)
    let point = center + (cos(angle) * u + sin(angle) * v) * radius
    crmAppendCylinder(
      &geometry, from: lastPoint, to: point, radius: tubeRadius, color: color, radialSegments: 8)
    lastPoint = point
  }
}

private func crmAppendWireSphere(
  _ geometry: inout CRMGeometry,
  center: SIMD3<Float>,
  radius: Float,
  wireRadius: Float,
  color: SIMD3<Float>,
  guideDirection: SIMD3<Float>
) {
  let ray = simd_normalize(guideDirection)
  let (u, v) = crmBasis(for: ray)
  crmAppendCircle(
    &geometry, center: center, axis: u, radius: radius, tubeRadius: wireRadius, color: color)
  crmAppendCircle(
    &geometry, center: center, axis: v, radius: radius, tubeRadius: wireRadius, color: color)
  crmAppendCircle(
    &geometry, center: center, axis: ray, radius: radius, tubeRadius: wireRadius, color: color)
}

private func crmAppendTorus(
  _ geometry: inout CRMGeometry,
  center: SIMD3<Float>,
  majorRadius: Float,
  minorRadius: Float,
  color: SIMD3<Float>,
  ringSegments: Int = 28,
  tubeSegments: Int = 14
) {
  var vertices: [CRMVertex] = []
  var indices: [UInt32] = []
  vertices.reserveCapacity((ringSegments + 1) * (tubeSegments + 1))
  indices.reserveCapacity(ringSegments * tubeSegments * 6)

  for ring in 0...ringSegments {
    let u = Float(ring) * 2 * .pi / Float(ringSegments)
    let cu = cos(u)
    let su = sin(u)
    let ringCenter = center + SIMD3<Float>(majorRadius * cu, 0, majorRadius * su)
    let ringNormal = SIMD3<Float>(cu, 0, su)
    for tube in 0...tubeSegments {
      let v = Float(tube) * 2 * .pi / Float(tubeSegments)
      let cv = cos(v)
      let sv = sin(v)
      let normal = simd_normalize(SIMD3<Float>(ringNormal.x * cv, sv, ringNormal.z * cv))
      let position = ringCenter + normal * minorRadius
      vertices.append(CRMVertex(position: position, normal: normal, color: color))
    }
  }

  let stride = tubeSegments + 1
  for ring in 0..<ringSegments {
    for tube in 0..<tubeSegments {
      let i0 = UInt32(ring * stride + tube)
      let i1 = UInt32((ring + 1) * stride + tube)
      let i2 = i0 + 1
      let i3 = i1 + 1
      indices += [i0, i1, i2, i2, i1, i3]
    }
  }

  geometry.append(vertices: vertices, indices: indices)
}

private func crmAppendBoxEdges(
  _ geometry: inout CRMGeometry,
  center: SIMD3<Float>,
  halfExtents: SIMD3<Float>,
  radius: Float,
  color: SIMD3<Float>
) {
  let corners: [SIMD3<Float>] = [
    center + SIMD3<Float>(-halfExtents.x, -halfExtents.y, -halfExtents.z),
    center + SIMD3<Float>(halfExtents.x, -halfExtents.y, -halfExtents.z),
    center + SIMD3<Float>(halfExtents.x, halfExtents.y, -halfExtents.z),
    center + SIMD3<Float>(-halfExtents.x, halfExtents.y, -halfExtents.z),
    center + SIMD3<Float>(-halfExtents.x, -halfExtents.y, halfExtents.z),
    center + SIMD3<Float>(halfExtents.x, -halfExtents.y, halfExtents.z),
    center + SIMD3<Float>(halfExtents.x, halfExtents.y, halfExtents.z),
    center + SIMD3<Float>(-halfExtents.x, halfExtents.y, halfExtents.z),
  ]
  let edges = [
    (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7),
  ]
  for (a, b) in edges {
    crmAppendCylinder(&geometry, from: corners[a], to: corners[b], radius: radius, color: color)
  }
}

private func crmSdTorus(
  _ p: SIMD3<Float>, _ center: SIMD3<Float>, _ majorRadius: Float, _ minorRadius: Float
) -> Float {
  let q = p - center
  return simd_length(SIMD2<Float>(simd_length(SIMD2<Float>(q.x, q.z)) - majorRadius, q.y))
    - minorRadius
}

private func crmTrace(
  from start: SIMD3<Float>,
  rayDir: SIMD3<Float>,
  maxDistance: Float,
  maxSteps: Int,
  distance: (SIMD3<Float>) -> Float
) -> CRMTraceResult {
  var t: Float = 0
  var steps: [CRMTraceStep] = []
  steps.reserveCapacity(maxSteps)
  for _ in 0..<maxSteps {
    let d = distance(start + rayDir * t)
    steps.append(CRMTraceStep(distanceAlongRay: t, radius: d))
    if d < crmTraceEpsilon {
      let hitDistance = crmRefineHitDistance(
        from: start,
        rayDir: rayDir,
        nearDistance: t,
        nearSdf: d,
        maxDistance: maxDistance,
        distance: distance)
      return CRMTraceResult(steps: steps, hitDistance: hitDistance)
    }
    t += d
    if t >= maxDistance { break }
  }
  return CRMTraceResult(steps: steps, hitDistance: min(t, maxDistance))
}

private func crmRefineHitDistance(
  from start: SIMD3<Float>,
  rayDir: SIMD3<Float>,
  nearDistance: Float,
  nearSdf: Float,
  maxDistance: Float,
  distance: (SIMD3<Float>) -> Float
) -> Float {
  var outsideT = nearDistance
  var highT = nearDistance + max(nearSdf * 1.5, 0.002)

  for _ in 0..<32 {
    if highT >= maxDistance { return min(outsideT, maxDistance) }
    let highD = distance(start + rayDir * highT)
    if highD <= 0 {
      var low = outsideT
      var high = highT
      for _ in 0..<24 {
        let mid = 0.5 * (low + high)
        let midD = distance(start + rayDir * mid)
        if midD > 0 {
          low = mid
        } else {
          high = mid
        }
      }
      return 0.5 * (low + high)
    }
    outsideT = highT
    highT += max(highD * 1.25, 0.002)
  }

  return min(outsideT, maxDistance)
}

private func crmProbeColor(_ base: SIMD3<Float>, stepIndex: Int) -> SIMD3<Float> {
  if stepIndex.isMultiple(of: 2) { return base }
  return simd_clamp(
    base * 0.45 + SIMD3<Float>(base.z, base.x, base.y) * 0.35 + SIMD3<Float>(0.20, 0.20, 0.20),
    .zero,
    SIMD3<Float>(repeating: 1.0)
  )
}

private func crmAppendProbeSpheres(
  _ geometry: inout CRMGeometry,
  start: SIMD3<Float>,
  rayDir: SIMD3<Float>,
  steps: [CRMTraceStep],
  color: SIMD3<Float>,
  maxProbes: Int
) {
  for (count, step) in steps.prefix(maxProbes).enumerated() {
    let p = start + rayDir * step.distanceAlongRay
    let probeColor = crmProbeColor(color, stepIndex: count)
    crmAppendWireSphere(
      &geometry, center: p, radius: step.radius, wireRadius: 0.00125, color: probeColor,
      guideDirection: rayDir)
  }
}

final class CubeRayMarchDemoRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .cubeRayMarchDemo
  let preferredClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

  private let pipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  private let vertexBuffer: MTLBuffer
  private let indexBuffer: MTLBuffer
  private let indexCount: Int
  private let maxViewCount: Int

  private let objectCenter = SIMD3<Float>(0.0, 0.0, -1.9)
  private let cubeCenter = SIMD3<Float>(0.0, 0.0, 0.0)
  private let cubeHalfExtents = SIMD3<Float>(0.85, 0.85, 0.85)
  private let lightLocal = SIMD3<Float>(-0.84, 0.54, -0.35)
  private let torusCenter = SIMD3<Float>(0.30, 0.0, 0.0)
  private let torusMajorRadius: Float = 0.35
  private let torusMinorRadius: Float = 0.09

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.maxViewCount = max(1, maxViewCount)

    let lightLocal = SIMD3<Float>(-0.84, 0.54, -0.35)
    let torusCenter = SIMD3<Float>(0.30, 0.0, 0.0)
    let torusMajorRadius: Float = 0.35
    let torusMinorRadius: Float = 0.09
    // Chosen from the side face so the exact sphere-tracing sequence first hits the farther ring.
    let rayAim = SIMD3<Float>(0.80, -0.16, -0.15)
    let rayDir = simd_normalize(rayAim - lightLocal)
    let trace = crmTrace(
      from: lightLocal,
      rayDir: rayDir,
      maxDistance: 4.0,
      maxSteps: 96
    ) { point in
      crmSdTorus(point, torusCenter, torusMajorRadius, torusMinorRadius)
    }
    let rayEnd = lightLocal + rayDir * trace.hitDistance

    let cCube = SIMD3<Float>(0.28, 0.90, 0.45)
    let cLight = SIMD3<Float>(1.0, 0.95, 0.5)
    let cTorus = SIMD3<Float>(1.0, 0.60, 0.16)
    let cRay = SIMD3<Float>(0.40, 0.78, 1.0)
    let cProbe = SIMD3<Float>(0.92, 0.86, 0.36)

    var geometry = CRMGeometry()
    crmAppendBoxEdges(
      &geometry, center: cubeCenter, halfExtents: cubeHalfExtents, radius: 0.016, color: cCube)
    crmAppendTorus(
      &geometry, center: torusCenter, majorRadius: torusMajorRadius, minorRadius: torusMinorRadius,
      color: cTorus)
    crmAppendSphere(
      &geometry, center: lightLocal, radius: 0.07, color: cLight, latSegments: 10, lonSegments: 20)
    crmAppendCylinder(
      &geometry, from: lightLocal, to: rayEnd, radius: 0.003, color: cRay, radialSegments: 10)
    crmAppendProbeSpheres(
      &geometry, start: lightLocal, rayDir: rayDir, steps: trace.steps, color: cProbe,
      maxProbes: 24)

    vertexBuffer = device.makeBuffer(
      bytes: geometry.vertices,
      length: MemoryLayout<CRMVertex>.stride * geometry.vertices.count,
      options: .storageModeShared
    )!
    indexBuffer = device.makeBuffer(
      bytes: geometry.indices,
      length: MemoryLayout<UInt32>.stride * geometry.indices.count,
      options: .storageModeShared
    )!
    indexCount = geometry.indices.count

    pipelineState = try CubeRayMarchDemoRenderer.makePipelineState(
      device: device, library: library, maxViewCount: self.maxViewCount)
    depthStencilState = CubeRayMarchDemoRenderer.makeDepthStencilState(device: device)
  }

  func updateSimulation(_ context: PatternSimulationContext) {}
  func resetToInitialState() {}

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    encoder.setRenderPipelineState(pipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.back)
    context.applyViewConfiguration(on: encoder)

    encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)

    var uniforms = CRMMeshUniforms(
      time: context.time,
      viewCount: UInt32(context.viewData.viewCount),
      objectCenter: SIMD4<Float>(objectCenter.x, objectCenter.y, objectCenter.z, 0),
      lightPosition: SIMD4<Float>(lightLocal.x, lightLocal.y, lightLocal.z, 0)
    )

    encoder.setVertexBytes(&uniforms, length: MemoryLayout<CRMMeshUniforms>.stride, index: 1)

    var vpMatrices = context.viewData.viewProjectionMatrices
    if vpMatrices.isEmpty { vpMatrices = [matrix_identity_float4x4] }
    vpMatrices.withUnsafeBytes {
      if let base = $0.baseAddress, $0.count > 0 {
        encoder.setVertexBytes(base, length: $0.count, index: 2)
      }
    }

    encoder.setFragmentBytes(&uniforms, length: MemoryLayout<CRMMeshUniforms>.stride, index: 0)

    var viewToWorld = context.viewData.viewToWorldTransforms
    if viewToWorld.isEmpty { viewToWorld = [matrix_identity_float4x4] }
    viewToWorld.withUnsafeBytes {
      if let base = $0.baseAddress, $0.count > 0 {
        encoder.setFragmentBytes(base, length: $0.count, index: 1)
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
}

extension CubeRayMarchDemoRenderer {
  fileprivate static func makePipelineState(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let desc = MTLRenderPipelineDescriptor()
    desc.vertexFunction = library.makeFunction(name: "cubeRayMarchVertex")
    desc.fragmentFunction = library.makeFunction(name: "cubeRayMarchFragment")
    desc.colorAttachments[0].pixelFormat = .rgba16Float
    desc.depthAttachmentPixelFormat = .depth32Float

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
    vertexDescriptor.layouts[0].stride = MemoryLayout<CRMVertex>.stride
    desc.vertexDescriptor = vertexDescriptor

    desc.maxVertexAmplificationCount = max(maxViewCount, 1)
    return try device.makeRenderPipelineState(descriptor: desc)
  }

  fileprivate static func makeDepthStencilState(device: MTLDevice) -> MTLDepthStencilState {
    let desc = MTLDepthStencilDescriptor()
    desc.depthCompareFunction = .greater
    desc.isDepthWriteEnabled = true
    return device.makeDepthStencilState(descriptor: desc)!
  }
}
