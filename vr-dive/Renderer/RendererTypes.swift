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
}

struct PatternRenderContext {
  let viewData: ViewRenderingData
  let time: Float

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
    } else {
      encoder.setVertexAmplificationCount(1, viewMappings: nil)
    }
  }
}

protocol VisualPatternController: AnyObject {
  var identifier: VisualPatternKind { get }
  var preferredClearColor: MTLClearColor { get }

  func synchronizeState(_ context: PatternSimulationContext)
  func updateSimulation(_ context: PatternSimulationContext)
  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext)
  func resetToInitialState()
}

extension VisualPatternController {
  func synchronizeState(_ context: PatternSimulationContext) {}
}
