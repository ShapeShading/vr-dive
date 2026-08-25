import Metal
import MetalKit
import simd

struct SceneUniforms {
  var time: Float
  var layerCount: UInt32
  var padding: SIMD2<Float> = .zero
}

struct MeshVertex {
  var position: SIMD3<Float>
  var normal: SIMD3<Float>
}

struct ViewRenderingData {
  var viewProjectionMatrices: [simd_float4x4]
  var viewports: [MTLViewport]
  var renderTargetLayers: [UInt32]
  var viewToWorldTransforms: [simd_float4x4]
  /// Compositor clipping distances in reverse-Z order: far, then near.
  var depthRange: SIMD2<Float>
  var viewCount: Int
  var supportsVertexAmplification: Bool
}

struct PatternSimulationContext {
  let commandQueue: MTLCommandQueue
  let time: Float
  let speedMultiplier: Float
  let isPaused: Bool
  let originCellInspectionEnabled: Bool
  let rayMarchingProbeDimTarget: RayMarchingProbeDimTarget
  let huashanSampleRatio: Float
  let simoneOrbit3DPreset: SimoneOrbit3DPreset
  let infiniteZoomRate: Float
  let infiniteZoomDirection: Float
  let infiniteZoomQuality: InfiniteZoomQuality
}

struct PatternRenderContext {
  let viewData: ViewRenderingData
  let time: Float
  let renderTargetWidth: Int  // actual color texture width (not display viewport)
  let renderTargetHeight: Int  // actual color texture height
  /// Virtual-viewpoint transform for pattern-space navigation (identity in normal mode).
  /// Applied in box/scene-local space by renderers that support it (e.g. GlassBox).
  let patternNavigationTransform: simd_float4x4

  func applyViewConfiguration(on encoder: MTLRenderCommandEncoder) {
    if !viewData.viewports.isEmpty {
      encoder.setViewports(viewData.viewports)
    }

    guard viewData.supportsVertexAmplification else {
      return
    }

    if viewData.viewCount > 1 {
      var viewMappings = (0..<viewData.viewCount).map {
        MTLVertexAmplificationViewMapping(
          viewportArrayIndexOffset: UInt32($0),
          renderTargetArrayIndexOffset: viewData.renderTargetLayers[$0]
        )
      }
      encoder.setVertexAmplificationCount(viewData.viewCount, viewMappings: &viewMappings)
    } else {
      // ⚠️ 必须保留此 else 分支，即使 viewCount == 1。
      // foveation 开启时，Metal 需要显式调用 setVertexAmplificationCount 才能
      // 正确将虚拟 viewport 坐标（如 4338×3478）映射到物理 texture（如 1888×1792）。
      // 缺少此调用会导致未被几何体覆盖的 foveation tile 以 clear color 颜色暴露，
      // 形成可见的彩色瓦片伪影（tile artifacts）。见 notes/05-08-tile-artifacts-and-stereo-bugs.md
      encoder.setVertexAmplificationCount(1, viewMappings: nil)
    }
  }
}

protocol VisualPatternController: AnyObject {
  var identifier: VisualPatternKind { get }
  var preferredClearColor: MTLClearColor { get }

  func synchronizeState(_ context: PatternSimulationContext)
  func updateSimulation(_ context: PatternSimulationContext)
  /// Optional compute pre-pass (runs before the render encoder is created).
  /// Use this to dispatch compute kernels that produce data consumed by encodeFrame.
  func encodeComputePrepass(commandBuffer: MTLCommandBuffer, context: PatternRenderContext)
  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext)
  /// Optional post-pass (runs after the render encoder ends).
  /// Intended for low-frequency diagnostics that need to inspect the final color target.
  func encodePostpass(
    commandBuffer: MTLCommandBuffer,
    context: PatternRenderContext,
    colorTexture: MTLTexture,
    depthTexture: MTLTexture?)
  func resetToInitialState()

  /// Optional pre-warm hook. Called once at startup on the CPU thread to force
  /// Metal's JIT pipeline compilation before the first real frame is submitted.
  /// Implement this for any pattern whose shader compilation takes >100 ms.
  func warmupPipeline(device: MTLDevice, commandQueue: MTLCommandQueue)
}

extension VisualPatternController {
  func synchronizeState(_ context: PatternSimulationContext) {}
  func encodeComputePrepass(commandBuffer: MTLCommandBuffer, context: PatternRenderContext) {}
  func encodePostpass(
    commandBuffer: MTLCommandBuffer,
    context: PatternRenderContext,
    colorTexture: MTLTexture,
    depthTexture: MTLTexture?
  ) {}
  func warmupPipeline(device: MTLDevice, commandQueue: MTLCommandQueue) {}
}
