import Metal
import simd

private struct RMDVertex {
  var position: SIMD3<Float>
  var normal: SIMD3<Float>
  var color: SIMD3<Float>
}

private struct RMDGeometry {
  var vertices: [RMDVertex] = []
  var indices: [UInt32] = []

  mutating func append(vertices newVertices: [RMDVertex], indices newIndices: [UInt32]) {
    let base = UInt32(vertices.count)
    vertices += newVertices
    indices += newIndices.map { $0 + base }
  }
}

private struct RMDMeshBuffers {
  var vertexBuffer: MTLBuffer
  var indexBuffer: MTLBuffer
  var indexCount: Int
}

private struct RMDTraceStep {
  var distanceAlongRay: Float
  var radius: Float
}

private struct RMDTraceResult {
  var steps: [RMDTraceStep]
  var hitDistance: Float
}

private let rmdTraceEpsilon: Float = 0.003

private func rmdBasis(for axis: SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) {
  let tangent = simd_normalize(axis)
  let helper = abs(tangent.x) > 0.9 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0)
  let u = simd_normalize(simd_cross(tangent, helper))
  return (u, simd_cross(tangent, u))
}

private func rmdAppendSphere(
  _ geometry: inout RMDGeometry,
  center: SIMD3<Float>,
  radius: Float,
  color: SIMD3<Float>,
  latSegments: Int = 10,
  lonSegments: Int = 20
) {
  var vertices: [RMDVertex] = []
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
      vertices.append(RMDVertex(position: center + normal * radius, normal: normal, color: color))
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

private func rmdAppendCylinder(
  _ geometry: inout RMDGeometry,
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
  let (u, v) = rmdBasis(for: tangent)

  var vertices: [RMDVertex] = []
  var indices: [UInt32] = []
  vertices.reserveCapacity((radialSegments + 1) * 2)
  indices.reserveCapacity(radialSegments * 6)

  for ringIndex in 0...1 {
    let center = ringIndex == 0 ? start : end
    for segment in 0...radialSegments {
      let angle = Float(segment) * 2 * .pi / Float(radialSegments)
      let normal = simd_normalize(cos(angle) * u + sin(angle) * v)
      vertices.append(RMDVertex(position: center + normal * radius, normal: normal, color: color))
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

private func rmdAppendCircle(
  _ geometry: inout RMDGeometry,
  center: SIMD3<Float>,
  axis: SIMD3<Float>,
  radius: Float,
  tubeRadius: Float,
  color: SIMD3<Float>,
  segments: Int = 40
) {
  let (u, v) = rmdBasis(for: axis)
  var lastPoint = center + u * radius
  for segment in 1...segments {
    let angle = Float(segment) * 2 * .pi / Float(segments)
    let point = center + (cos(angle) * u + sin(angle) * v) * radius
    rmdAppendCylinder(
      &geometry, from: lastPoint, to: point, radius: tubeRadius, color: color, radialSegments: 8)
    lastPoint = point
  }
}

private func rmdAppendWireSphere(
  _ geometry: inout RMDGeometry,
  center: SIMD3<Float>,
  radius: Float,
  wireRadius: Float,
  color: SIMD3<Float>,
  guideDirection: SIMD3<Float>
) {
  let ray = simd_normalize(guideDirection)
  let (u, v) = rmdBasis(for: ray)
  rmdAppendCircle(
    &geometry, center: center, axis: u, radius: radius, tubeRadius: wireRadius, color: color)
  rmdAppendCircle(
    &geometry, center: center, axis: v, radius: radius, tubeRadius: wireRadius, color: color)
  rmdAppendCircle(
    &geometry, center: center, axis: ray, radius: radius, tubeRadius: wireRadius, color: color)
}

private func rmdAppendTorus(
  _ geometry: inout RMDGeometry,
  center: SIMD3<Float>,
  majorRadius: Float,
  minorRadius: Float,
  color: SIMD3<Float>,
  ringSegments: Int = 28,
  tubeSegments: Int = 14
) {
  var vertices: [RMDVertex] = []
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
      vertices.append(RMDVertex(position: position, normal: normal, color: color))
    }
  }

  let ringStride = tubeSegments + 1
  for ring in 0..<ringSegments {
    for tube in 0..<tubeSegments {
      let i0 = UInt32(ring * ringStride + tube)
      let i1 = UInt32((ring + 1) * ringStride + tube)
      let i2 = i0 + 1
      let i3 = i1 + 1
      indices += [i0, i1, i2, i2, i1, i3]
    }
  }

  geometry.append(vertices: vertices, indices: indices)
}

private func rmdAppendBoxEdges(
  _ geometry: inout RMDGeometry,
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
    rmdAppendCylinder(&geometry, from: corners[a], to: corners[b], radius: radius, color: color)
  }
}

private func rmdAppendRectFrame(
  _ geometry: inout RMDGeometry,
  center: SIMD3<Float>,
  halfWidth: Float,
  halfHeight: Float,
  radius: Float,
  color: SIMD3<Float>
) {
  let tl = center + SIMD3<Float>(0, halfHeight, -halfWidth)
  let tr = center + SIMD3<Float>(0, halfHeight, halfWidth)
  let bl = center + SIMD3<Float>(0, -halfHeight, -halfWidth)
  let br = center + SIMD3<Float>(0, -halfHeight, halfWidth)
  rmdAppendCylinder(&geometry, from: tl, to: tr, radius: radius, color: color)
  rmdAppendCylinder(&geometry, from: tr, to: br, radius: radius, color: color)
  rmdAppendCylinder(&geometry, from: br, to: bl, radius: radius, color: color)
  rmdAppendCylinder(&geometry, from: bl, to: tl, radius: radius, color: color)
}

private func rmdSdSphere(_ p: SIMD3<Float>, center: SIMD3<Float>, radius: Float) -> Float {
  simd_length(p - center) - radius
}

private func rmdSdTorus(
  _ p: SIMD3<Float>,
  center: SIMD3<Float>,
  majorRadius: Float,
  minorRadius: Float
) -> Float {
  let q = p - center
  return simd_length(SIMD2<Float>(simd_length(SIMD2<Float>(q.x, q.z)) - majorRadius, q.y))
    - minorRadius
}

private func rmdSceneSdf(
  _ p: SIMD3<Float>,
  sphereCenter: SIMD3<Float>,
  sphereRadius: Float,
  torusCenter: SIMD3<Float>,
  torusMajorRadius: Float,
  torusMinorRadius: Float
) -> Float {
  min(
    rmdSdSphere(p, center: sphereCenter, radius: sphereRadius),
    rmdSdTorus(p, center: torusCenter, majorRadius: torusMajorRadius, minorRadius: torusMinorRadius)
  )
}

private func rmdTrace(
  from start: SIMD3<Float>,
  rayDir: SIMD3<Float>,
  maxDistance: Float,
  maxSteps: Int,
  distance: (SIMD3<Float>) -> Float
) -> RMDTraceResult {
  var t: Float = 0
  var steps: [RMDTraceStep] = []
  steps.reserveCapacity(maxSteps)
  for _ in 0..<maxSteps {
    let d = distance(start + rayDir * t)
    steps.append(RMDTraceStep(distanceAlongRay: t, radius: d))
    if d < rmdTraceEpsilon {
      let hitDistance = rmdRefineHitDistance(
        from: start,
        rayDir: rayDir,
        nearDistance: t,
        nearSdf: d,
        maxDistance: maxDistance,
        distance: distance)
      return RMDTraceResult(steps: steps, hitDistance: hitDistance)
    }
    t += d
    if t >= maxDistance { break }
  }
  return RMDTraceResult(steps: steps, hitDistance: min(t, maxDistance))
}

private func rmdRefineHitDistance(
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

private func rmdProbeColor(_ base: SIMD3<Float>, stepIndex: Int) -> SIMD3<Float> {
  if stepIndex.isMultiple(of: 2) { return base }
  return simd_clamp(
    base * 0.45 + SIMD3<Float>(base.z, base.x, base.y) * 0.35 + SIMD3<Float>(0.20, 0.20, 0.20),
    .zero,
    SIMD3<Float>(repeating: 1.0)
  )
}

private func rmdDimmedProbeColor() -> SIMD3<Float> {
  SIMD3<Float>(repeating: 0.22)
}

private func rmdAppendProbeSpheres(
  _ geometry: inout RMDGeometry,
  from start: SIMD3<Float>,
  rayDir: SIMD3<Float>,
  steps: [RMDTraceStep],
  color: SIMD3<Float>,
  maxProbes: Int
) {
  for (count, step) in steps.prefix(maxProbes).enumerated() {
    let p = start + rayDir * step.distanceAlongRay
    let probeColor = rmdProbeColor(color, stepIndex: count)
    rmdAppendWireSphere(
      &geometry, center: p, radius: step.radius, wireRadius: 0.00125, color: probeColor,
      guideDirection: rayDir)
  }
}

private func rmdMakeMeshBuffers(device: MTLDevice, geometry: RMDGeometry) -> RMDMeshBuffers {
  let vertexBuffer = device.makeBuffer(
    bytes: geometry.vertices,
    length: MemoryLayout<RMDVertex>.stride * geometry.vertices.count,
    options: .storageModeShared
  )!
  let indexBuffer = device.makeBuffer(
    bytes: geometry.indices,
    length: MemoryLayout<UInt32>.stride * geometry.indices.count,
    options: .storageModeShared
  )!
  return RMDMeshBuffers(
    vertexBuffer: vertexBuffer,
    indexBuffer: indexBuffer,
    indexCount: geometry.indices.count)
}

private func rmdBuildSceneGeometry(probeDimTarget: RayMarchingProbeDimTarget) -> RMDGeometry {
  let lightLocal = SIMD3<Float>(-1.0, 0.3, 0.0)
  let screen = SIMD3<Float>(-0.6, 0.3, 0.0)
  let sphereCenter = SIMD3<Float>(0.15, 0.18, -0.58)
  let sphereRadius: Float = 0.26
  let torusCenter = SIMD3<Float>(1.25, -0.14, 0.82)
  let torusMajorRadius: Float = 0.31
  let torusMinorRadius: Float = 0.075
  let sphereAim = SIMD3<Float>(-0.05, -0.02, -0.55)
  let torusAim = SIMD3<Float>(1.25, -0.10, 0.80)
  let sphereRayDir = simd_normalize(sphereAim - lightLocal)
  let torusRayDir = simd_normalize(torusAim - lightLocal)
  let sceneSdf: (SIMD3<Float>) -> Float = { point in
    rmdSceneSdf(
      point,
      sphereCenter: sphereCenter,
      sphereRadius: sphereRadius,
      torusCenter: torusCenter,
      torusMajorRadius: torusMajorRadius,
      torusMinorRadius: torusMinorRadius
    )
  }
  let sphereTrace = rmdTrace(
    from: lightLocal,
    rayDir: sphereRayDir,
    maxDistance: 4.0,
    maxSteps: 64,
    distance: sceneSdf)
  let torusTrace = rmdTrace(
    from: lightLocal,
    rayDir: torusRayDir,
    maxDistance: 4.0,
    maxSteps: 96,
    distance: sceneSdf)
  let sphereRayEnd = lightLocal + sphereRayDir * sphereTrace.hitDistance
  let torusRayEnd = lightLocal + torusRayDir * torusTrace.hitDistance

  let cLight = SIMD3<Float>(1.0, 0.95, 0.5)
  let cScreen = SIMD3<Float>(0.5, 0.65, 1.0)
  let cSphere = SIMD3<Float>(0.3, 0.7, 1.0)
  let cTorus = SIMD3<Float>(1.0, 0.6, 0.15)
  let cDimmed = rmdDimmedProbeColor()
  let cSphereProbe =
    probeDimTarget == .sphere ? cDimmed : cSphere * 0.72 + SIMD3<Float>(0.28, 0.28, 0.28)
  let cTorusProbe =
    probeDimTarget == .torus ? cDimmed : cTorus * 0.72 + SIMD3<Float>(0.28, 0.28, 0.28)

  var geometry = RMDGeometry()
  rmdAppendSphere(
    &geometry, center: lightLocal, radius: 0.08, color: cLight, latSegments: 10, lonSegments: 20)
  rmdAppendRectFrame(
    &geometry, center: screen, halfWidth: 0.38, halfHeight: 0.26, radius: 0.015, color: cScreen)
  rmdAppendSphere(
    &geometry, center: sphereCenter, radius: sphereRadius, color: cSphere, latSegments: 12,
    lonSegments: 24)
  rmdAppendTorus(
    &geometry, center: torusCenter, majorRadius: torusMajorRadius, minorRadius: torusMinorRadius,
    color: cTorus)
  rmdAppendCylinder(
    &geometry, from: lightLocal, to: sphereRayEnd, radius: 0.003, color: cSphere,
    radialSegments: 10)
  rmdAppendProbeSpheres(
    &geometry, from: lightLocal, rayDir: sphereRayDir, steps: sphereTrace.steps,
    color: cSphereProbe,
    maxProbes: sphereTrace.steps.count)
  rmdAppendCylinder(
    &geometry, from: lightLocal, to: torusRayEnd, radius: 0.003, color: cTorus,
    radialSegments: 10)
  rmdAppendProbeSpheres(
    &geometry, from: lightLocal, rayDir: torusRayDir, steps: torusTrace.steps, color: cTorusProbe,
    maxProbes: torusTrace.steps.count)
  return geometry
}

final class RayMarchingDemoRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .rayMarchingDemo
  let preferredClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

  private let pipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  private let meshBuffersByDimTarget: [RayMarchingProbeDimTarget: RMDMeshBuffers]
  private var vertexBuffer: MTLBuffer
  private var indexBuffer: MTLBuffer
  private var indexCount: Int
  private let maxViewCount: Int
  private var activeProbeDimTarget: RayMarchingProbeDimTarget = .none

  private let objectCenter = SIMD3<Float>(0.0, 0.0, -1.9)
  private let lightLocal = SIMD3<Float>(-1.0, 0.3, 0.0)

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.maxViewCount = max(1, maxViewCount)

    let noneBuffers = rmdMakeMeshBuffers(
      device: device,
      geometry: rmdBuildSceneGeometry(probeDimTarget: .none))
    let sphereBuffers = rmdMakeMeshBuffers(
      device: device,
      geometry: rmdBuildSceneGeometry(probeDimTarget: .sphere))
    let torusBuffers = rmdMakeMeshBuffers(
      device: device,
      geometry: rmdBuildSceneGeometry(probeDimTarget: .torus))
    meshBuffersByDimTarget = [
      .none: noneBuffers,
      .sphere: sphereBuffers,
      .torus: torusBuffers,
    ]
    vertexBuffer = noneBuffers.vertexBuffer
    indexBuffer = noneBuffers.indexBuffer
    indexCount = noneBuffers.indexCount

    pipelineState = try RayMarchingDemoRenderer.makePipelineState(
      device: device, library: library, maxViewCount: self.maxViewCount)
    depthStencilState = RayMarchingDemoRenderer.makeDepthStencilState(device: device)
  }

  func synchronizeState(_ context: PatternSimulationContext) {
    let target = context.rayMarchingProbeDimTarget
    guard target != activeProbeDimTarget, let buffers = meshBuffersByDimTarget[target] else {
      return
    }
    activeProbeDimTarget = target
    vertexBuffer = buffers.vertexBuffer
    indexBuffer = buffers.indexBuffer
    indexCount = buffers.indexCount
  }

  func updateSimulation(_ context: PatternSimulationContext) {}
  func resetToInitialState() {}

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    encoder.setRenderPipelineState(pipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.back)
    context.applyViewConfiguration(on: encoder)

    encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)

    var uniforms = RMDMeshUniforms(
      time: context.time,
      viewCount: UInt32(context.viewData.viewCount),
      objectCenter: SIMD4<Float>(objectCenter.x, objectCenter.y, objectCenter.z, 0),
      lightPosition: SIMD4<Float>(lightLocal.x, lightLocal.y, lightLocal.z, 0)
    )

    encoder.setVertexBytes(&uniforms, length: MemoryLayout<RMDMeshUniforms>.stride, index: 1)

    var vpMatrices = context.viewData.viewProjectionMatrices
    if vpMatrices.isEmpty { vpMatrices = [matrix_identity_float4x4] }
    vpMatrices.withUnsafeBytes {
      if let base = $0.baseAddress, $0.count > 0 {
        encoder.setVertexBytes(base, length: $0.count, index: 2)
      }
    }

    encoder.setFragmentBytes(&uniforms, length: MemoryLayout<RMDMeshUniforms>.stride, index: 0)

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

extension RayMarchingDemoRenderer {
  fileprivate static func makePipelineState(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let desc = MTLRenderPipelineDescriptor()
    desc.vertexFunction = library.makeFunction(name: "rayMarchingDemoVertex")
    desc.fragmentFunction = library.makeFunction(name: "rayMarchingDemoFragment")
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
    vertexDescriptor.layouts[0].stride = MemoryLayout<RMDVertex>.stride
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
