import Metal
import simd

// DynamicBoxRenderer.swift
//
// A dynamic version of GlassBox: the outer bounding-box mesh and vertex shader
// are fixed, but the fragment shader can be hot-reloaded at runtime.
//
// Architecture:
//   - Embedded default shader ("3D grid of light points") ships in DynamicBoxShaders.metal
//   - A companion Node.js server (vr-dive/Demos/DynamicBox/shader-server.js) serves .metal
//     files from its shaders/ subdirectory on port 8888
//   - The "Load Shader" button fetches a named shader, compiles it with Metal, and
//     swaps the fragment function in the render pipeline
//   - Compilation errors are POSTed back to the server and written to
//     shader-compiling-error.log in the DynamicBox directory

final class DynamicBoxRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .dynamicBox
  let preferredClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

  private let device: MTLDevice
  private let library: MTLLibrary
  private let maxViewCount: Int

  private var pipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  private let vertexBuffer: MTLBuffer
  private let indexBuffer: MTLBuffer
  private let indexCount: Int

  // The currently-loaded shader name (empty string = default embedded shader)
  private(set) var currentShaderName: String = ""

  // The local box is 1.9 m wide on X/Y at a scale of 1.0.
  private var boxScale: Float = 2.0 / 1.9
  private let objectCenter = SIMD3<Float>(0.0, -0.05, -1.1)
  private var animationTime: Float = 0
  private var lastSimulationTime: Float?

  /// Base URL of the shader server. Keep in sync with PatternMenuModel.shaderServerBaseURL.
  private let serverBaseURL = "http://192.168.31.49:8888"

  // MARK: - Performance sampling
  //
  // Each time a shader is (re)loaded, we get a fresh budget of at most
  // `perfMaxReportsPerLoad` performance reports. Frames whose real wall-clock
  // delta exceeds `perfSlowFrameThresholdSeconds` are candidates; only a random
  // sample of those actually get POSTed, so the reports are spread out over the
  // shader's lifetime instead of firing all at once during the first hitch.
  private static let perfMaxReportsPerLoad = 10
  private static let perfSlowFrameThresholdSeconds: Double = 0.03  // ~33ms (well below 90Hz/60Hz budget)
  private static let perfSampleProbability: Double = 0.2
  private var perfReportsRemaining = DynamicBoxRenderer.perfMaxReportsPerLoad

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.device = device
    self.library = library
    self.maxViewCount = max(1, maxViewCount)

    // Same bounding-box mesh as GlassBox
    let geo = DynamicBoxRenderer.makeBox(
      device: device, localHalfExtents: SIMD3<Float>(0.95, 0.95, 1.25) * 1.02)
    vertexBuffer = geo.vertexBuffer
    indexBuffer = geo.indexBuffer
    indexCount = geo.indexCount

    // Build initial pipeline from embedded default shader
    let vertFn = library.makeFunction(name: "dynamicBoxVertex")!
    let fragFn = library.makeFunction(name: "dynamicBoxFragment")!
    pipelineState = try DynamicBoxRenderer.makePipeline(
      device: device, vertexFn: vertFn, fragmentFn: fragFn, maxViewCount: self.maxViewCount)

    depthStencilState = DynamicBoxRenderer.makeDepthStencilState(device: device)
  }

  // MARK: - Simulation

  func updateSimulation(_ context: PatternSimulationContext) {
    defer { lastSimulationTime = context.time }
    guard let lastSimulationTime else { return }
    let rawDelta = context.time - lastSimulationTime
    let deltaTime = max(0, min(rawDelta, 1.0 / 20.0))
    animationTime += deltaTime * max(context.speedMultiplier, 0)

    maybeSampleSlowFrame(rawDelta: Double(rawDelta))
  }

  func resetToInitialState() {
    animationTime = 0
    lastSimulationTime = nil
  }

  func setBoxSize(meters: Float) {
    boxScale = max(meters, 0.1) / 1.9
  }

  // MARK: - Performance sampling

  /// Called once per real frame with the (uncapped) wall-clock delta since the
  /// previous frame. If the frame was noticeably slow, randomly samples up to
  /// `perfMaxReportsPerLoad` reports for the currently-loaded shader.
  private func maybeSampleSlowFrame(rawDelta: Double) {
    guard perfReportsRemaining > 0 else { return }
    guard rawDelta > Self.perfSlowFrameThresholdSeconds else { return }
    guard Double.random(in: 0..<1) < Self.perfSampleProbability else { return }

    perfReportsRemaining -= 1
    let sampleIndex = Self.perfMaxReportsPerLoad - perfReportsRemaining
    let shaderName = currentShaderName.isEmpty ? "default" : currentShaderName
    let frameMS = rawDelta * 1000.0
    let fps = rawDelta > 0 ? 1.0 / rawDelta : 0
    let message = String(
      format: "[perf][%@] slow frame: %.1f ms (~%.1f fps) — sample %d/%d",
      shaderName, frameMS, fps, sampleIndex, Self.perfMaxReportsPerLoad)

    Task { await reportPerfToServer(message) }
  }

  // MARK: - Shader hot-reload

  /// Fetches a shader named `name` from the shader server, compiles it,
  /// and swaps it into the render pipeline.
  ///
  /// The server-side .metal file must define a fragment function named
  /// `dynamicBoxFragment` with the standard signature (see the default shader
  /// for reference).  Helper structs (DynamicBoxUniforms, DynamicBoxVertexOut)
  /// are prepended automatically; the user only writes the fragment body + helpers.
  ///
  /// - Parameter name: shader file name (without `.metal` extension), e.g. `"waves"`
  /// - Returns: `nil` on success, or an error description string on failure.
  func reloadShader(named name: String) async -> String? {
    // Any load attempt starts a fresh performance-sampling budget for the
    // (possibly new) shader that ends up active.
    perfReportsRemaining = Self.perfMaxReportsPerLoad

    // "default" uses the embedded shader compiled into the app – no server fetch.
    if name == "default" {
      guard let vertFn = library.makeFunction(name: "dynamicBoxVertex"),
        let fragFn = library.makeFunction(name: "dynamicBoxFragment")
      else {
        let msg = "Embedded default shader functions not found."
        await reportToServer(msg)
        return msg
      }
      do {
        pipelineState = try DynamicBoxRenderer.makePipeline(
          device: device, vertexFn: vertFn, fragmentFn: fragFn,
          maxViewCount: maxViewCount)
        currentShaderName = "default"
        return nil
      } catch {
        let msg = "Pipeline creation error: \(error.localizedDescription)"
        await reportToServer(msg)
        return msg
      }
    }

    let rawSource: String
    do {
      rawSource = try await fetchShaderSource(named: name)
    } catch {
      let msg = "Failed to fetch shader \"\(name)\": \(error.localizedDescription)"
      await reportToServer(msg)
      return msg
    }

    // Wrap the user's source with the struct definitions it depends on.
    let wrappedSource = DynamicBoxRenderer.wrapShaderSource(rawSource)

    // Compile
    let newLibrary: MTLLibrary
    do {
      newLibrary = try await device.makeLibrary(source: wrappedSource, options: nil)
    } catch {
      let msg = "Metal compilation error: \(error.localizedDescription)"
      await reportToServer("[\(name)] \(msg)")
      return msg
    }

    guard let newFragFn = newLibrary.makeFunction(name: "dynamicBoxFragment") else {
      let msg = "Compiled library is missing \"dynamicBoxFragment\" function."
      await reportToServer("[\(name)] \(msg)")
      return msg
    }

    // Build a new pipeline state
    let vertFn = library.makeFunction(name: "dynamicBoxVertex")!
    let newPS: MTLRenderPipelineState
    do {
      newPS = try DynamicBoxRenderer.makePipeline(
        device: device, vertexFn: vertFn, fragmentFn: newFragFn,
        maxViewCount: maxViewCount)
    } catch {
      let msg = "Pipeline creation error: \(error.localizedDescription)"
      await reportToServer("[\(name)] \(msg)")
      return msg
    }

    // Swap
    pipelineState = newPS
    currentShaderName = name
    await reportToServer("[\(name)] Shader compiled and loaded successfully.")
    return nil  // success
  }

  // MARK: - Rendering

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    encoder.setRenderPipelineState(pipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)
    context.applyViewConfiguration(on: encoder)

    encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)

    var uniforms = DynamicBoxUniforms(
      time: animationTime,
      viewCount: UInt32(context.viewData.viewCount),
      boxScale: boxScale,
      _pad: 0,
      objectCenter: SIMD4<Float>(objectCenter.x, objectCenter.y, objectCenter.z, 0),
      patternTransform: context.patternNavigationTransform)

    encoder.setVertexBytes(&uniforms, length: MemoryLayout<DynamicBoxUniforms>.stride, index: 1)

    var vpMatrices = context.viewData.viewProjectionMatrices
    if vpMatrices.isEmpty { vpMatrices = [matrix_identity_float4x4] }
    vpMatrices.withUnsafeBytes {
      if let base = $0.baseAddress, $0.count > 0 {
        encoder.setVertexBytes(base, length: $0.count, index: 2)
      }
    }

    // Fragment buffers
    encoder.setFragmentBytes(&uniforms, length: MemoryLayout<DynamicBoxUniforms>.stride, index: 0)

    var viewToWorld = context.viewData.viewToWorldTransforms
    if viewToWorld.isEmpty { viewToWorld = [matrix_identity_float4x4] }
    viewToWorld.withUnsafeBytes {
      if let base = $0.baseAddress, $0.count > 0 {
        encoder.setFragmentBytes(base, length: $0.count, index: 1)
      }
    }
    vpMatrices.withUnsafeBytes {
      if let base = $0.baseAddress, $0.count > 0 {
        encoder.setFragmentBytes(base, length: $0.count, index: 2)
      }
    }

    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: indexCount,
      indexType: .uint16,
      indexBuffer: indexBuffer,
      indexBufferOffset: 0)
  }

  // MARK: - Networking

  private func fetchShaderSource(named name: String) async throws -> String {
    let url = URL(string: "\(serverBaseURL)/shaders/\(name).metal")!
    let (data, response) = try await URLSession.shared.data(from: url)
    guard let httpResp = response as? HTTPURLResponse, httpResp.statusCode == 200 else {
      throw URLError(.badServerResponse)
    }
    guard let source = String(data: data, encoding: .utf8) else {
      throw URLError(.cannotDecodeContentData)
    }
    return source
  }

  private func reportToServer(_ message: String) async {
    let url = URL(string: "\(serverBaseURL)/report-error")!
    var req = URLRequest(url: url)
    req.httpMethod = "POST"
    req.httpBody = message.data(using: .utf8)
    req.setValue("text/plain", forHTTPHeaderField: "Content-Type")
    _ = try? await URLSession.shared.data(for: req)
  }

  /// Posts a sampled performance report (e.g. a noticeably slow frame) to the
  /// shader server, which appends it to `shader-performance.log`.
  private func reportPerfToServer(_ message: String) async {
    let url = URL(string: "\(serverBaseURL)/report-perf")!
    var req = URLRequest(url: url)
    req.httpMethod = "POST"
    req.httpBody = message.data(using: .utf8)
    req.setValue("text/plain", forHTTPHeaderField: "Content-Type")
    _ = try? await URLSession.shared.data(for: req)
  }

  // MARK: - Helpers

  /// Wraps raw user shader source with the Metal boilerplate it needs
  /// (struct definitions that the fragment shader can reference).
  static func wrapShaderSource(_ userSource: String) -> String {
    return """
      #include <metal_stdlib>
      using namespace metal;

      // ─── Auto-generated structs (must match Swift side) ─────────────────────
      struct DynamicBoxUniforms {
          float  time;
          uint   viewCount;
          float  boxScale;
          float  _pad;
          float4 objectCenter;
          float4x4 patternTransform;
      };

      struct DynamicBoxVertexOut {
          float4 clipPos   [[position]];
          float3 worldPos;
          uint   viewIndex [[flat]];
      };

      // ─── Shared helpers ─────────────────────────────────────────────────────
      #define DB_PI      3.14159265f
      #define DB_BOXDIMS float3(0.95f, 0.95f, 1.25f)

      static float db_boxHit(float3 ro, float3 rd, float3 r, thread float3 &nn, bool entering) {
          rd += 0.0001f * (1.0f - abs(sign(rd)));
          float3 dr = 1.0f / rd;
          float3 n  = ro * dr;
          float3 k  = r  * abs(dr);
          float3 pin  = -k - n;
          float3 pout =  k - n;
          float tin  = max(pin.x,  max(pin.y,  pin.z));
          float tout = min(pout.x, min(pout.y, pout.z));
          if (tin > tout) return -1.0f;
          if (entering) {
              nn = -sign(rd) * step(pin.zxy,  pin.xyz)  * step(pin.yzx,  pin.xyz);
              return tin;
          } else {
              nn =  sign(rd) * step(pout.xyz, pout.zxy) * step(pout.xyz, pout.yzx);
              return tout;
          }
      }

      // ─── User shader ────────────────────────────────────────────────────────
      \(userSource)
      """
  }
}

// MARK: - Geometry & Pipeline factory

extension DynamicBoxRenderer {

  fileprivate static func makeBox(
    device: MTLDevice, localHalfExtents e: SIMD3<Float>
  ) -> (vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, indexCount: Int) {
    typealias V = MeshVertex
    let (x, y, z) = (e.x, e.y, e.z)
    let faces: [(positions: [SIMD3<Float>], normal: SIMD3<Float>)] = [
      ([[-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]], [0, 0, 1]),
      ([[x, -y, -z], [-x, -y, -z], [-x, y, -z], [x, y, -z]], [0, 0, -1]),
      ([[x, -y, z], [x, -y, -z], [x, y, -z], [x, y, z]], [1, 0, 0]),
      ([[-x, -y, -z], [-x, -y, z], [-x, y, z], [-x, y, -z]], [-1, 0, 0]),
      ([[-x, y, z], [x, y, z], [x, y, -z], [-x, y, -z]], [0, 1, 0]),
      ([[-x, -y, -z], [x, -y, -z], [x, -y, z], [-x, -y, z]], [0, -1, 0]),
    ]
    var vertices: [V] = []
    vertices.reserveCapacity(24)
    var indices: [UInt16] = []
    indices.reserveCapacity(36)
    for face in faces {
      let base = UInt16(vertices.count)
      for p in face.positions { vertices.append(V(position: p, normal: face.normal)) }
      indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
    }
    let vBuf = device.makeBuffer(
      bytes: vertices, length: MemoryLayout<V>.stride * vertices.count,
      options: .storageModeShared)!
    let iBuf = device.makeBuffer(
      bytes: indices, length: MemoryLayout<UInt16>.stride * indices.count,
      options: .storageModeShared)!
    return (vBuf, iBuf, indices.count)
  }

  fileprivate static func makePipeline(
    device: MTLDevice, vertexFn: MTLFunction, fragmentFn: MTLFunction, maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let desc = MTLRenderPipelineDescriptor()
    desc.vertexFunction = vertexFn
    desc.fragmentFunction = fragmentFn
    desc.colorAttachments[0].pixelFormat = .rgba16Float
    desc.depthAttachmentPixelFormat = .depth32Float

    let vd = MTLVertexDescriptor()
    vd.attributes[0].format = .float3
    vd.attributes[0].offset = 0
    vd.attributes[0].bufferIndex = 0
    vd.attributes[1].format = .float3
    vd.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
    vd.attributes[1].bufferIndex = 0
    vd.layouts[0].stride = MemoryLayout<MeshVertex>.stride
    desc.vertexDescriptor = vd

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
