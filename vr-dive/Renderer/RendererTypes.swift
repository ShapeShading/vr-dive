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
  var viewCount: Int
}

struct PatternSimulationContext {
  let commandQueue: MTLCommandQueue
  let time: Float
  let speedMultiplier: Float
  let isPaused: Bool
  let originCellInspectionEnabled: Bool
  let rayMarchingProbeDimTarget: RayMarchingProbeDimTarget
}

struct PatternRenderContext {
  let viewData: ViewRenderingData
  let time: Float
  let renderTargetWidth: Int  // actual color texture width (not display viewport)
  let renderTargetHeight: Int  // actual color texture height

  func applyViewConfiguration(on encoder: MTLRenderCommandEncoder) {
    if !viewData.viewports.isEmpty {
      encoder.setViewports(viewData.viewports)
    }

    if viewData.viewCount > 1 {
      var viewMappings = (0..<viewData.viewCount).map {
        MTLVertexAmplificationViewMapping(
          viewportArrayIndexOffset: UInt32($0),
          renderTargetArrayIndexOffset: viewData.renderTargetLayers[$0]
        )
      }
      encoder.setVertexAmplificationCount(viewData.viewCount, viewMappings: &viewMappings)
    }
    // Single-view: no amplification call needed (default count is 1)
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
  func resetToInitialState()

  /// Optional pre-warm hook. Called once at startup on the CPU thread to force
  /// Metal's JIT pipeline compilation before the first real frame is submitted.
  /// Implement this for any pattern whose shader compilation takes >100 ms.
  func warmupPipeline(device: MTLDevice, commandQueue: MTLCommandQueue)
}

extension VisualPatternController {
  func synchronizeState(_ context: PatternSimulationContext) {}
  func encodeComputePrepass(commandBuffer: MTLCommandBuffer, context: PatternRenderContext) {}
  func warmupPipeline(device: MTLDevice, commandQueue: MTLCommandQueue) {}
}
