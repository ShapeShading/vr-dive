import Foundation
import Metal
import MetalKit
import simd

final class PagodaSolidRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .pagoda
  let preferredClearColor = MTLClearColor(red: 0.05, green: 0.06, blue: 0.08, alpha: 1)

  private let pipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  private var submeshes: [(v: MTLBuffer, i: MTLBuffer, c: Int, color: SIMD4<Float>)] = []

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: "pagodaSolidVertexShader")
    descriptor.fragmentFunction = library.makeFunction(name: "pagodaSolidFragmentShader")
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
    vertexDescriptor.layouts[0].stride = MemoryLayout<MeshVertex>.stride
    descriptor.vertexDescriptor = vertexDescriptor
    descriptor.maxVertexAmplificationCount = max(maxViewCount, 1)

    pipelineState = try device.makeRenderPipelineState(descriptor: descriptor)

    let depthDescriptor = MTLDepthStencilDescriptor()
    depthDescriptor.depthCompareFunction = .greater
    depthDescriptor.isDepthWriteEnabled = true
    depthStencilState = device.makeDepthStencilState(descriptor: depthDescriptor)!

    buildGeometry(device: device)
  }

  func updateSimulation(_ context: PatternSimulationContext) {}

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    encoder.setRenderPipelineState(pipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)
    encoder.setFrontFacing(.counterClockwise)
    context.applyViewConfiguration(on: encoder)

    var uniforms = SceneUniforms(time: context.time, layerCount: UInt32(context.viewData.viewCount))
    encoder.setVertexBytes(&uniforms, length: MemoryLayout<SceneUniforms>.stride, index: 2)

    var matrices = context.viewData.viewProjectionMatrices
    if matrices.isEmpty { matrices = [matrix_identity_float4x4] }
    matrices.withUnsafeBytes { ptr in
      if let base = ptr.baseAddress, ptr.count > 0 {
        encoder.setVertexBytes(base, length: ptr.count, index: 3)
      }
    }

    for m in submeshes {
      var color = m.color
      encoder.setFragmentBytes(&uniforms, length: MemoryLayout<SceneUniforms>.stride, index: 0)
      encoder.setFragmentBytes(&color, length: MemoryLayout<SIMD4<Float>>.stride, index: 1)
      encoder.setVertexBuffer(m.v, offset: 0, index: 0)
      encoder.drawIndexedPrimitives(
        type: .triangle, indexCount: m.c, indexType: .uint16, indexBuffer: m.i, indexBufferOffset: 0
      )
    }
  }

  func resetToInitialState() {}

  private func buildGeometry(device: MTLDevice) {
    let colBody = SIMD4<Float>(0.745, 0.666, 0.588, 1)
    let colRoof = SIMD4<Float>(0.275, 0.245, 0.215, 1)
    let colBase = SIMD4<Float>(0.59, 0.59, 0.59, 1)
    let colWood = SIMD4<Float>(0.43, 0.32, 0.24, 1)
    let colTile = SIMD4<Float>(0.27, 0.28, 0.29, 1)
    let colGround = SIMD4<Float>(0.66, 0.66, 0.66, 1)

    var v: [MeshVertex] = []
    var ix: [UInt16] = []

    let baseWidth: Float = 32
    let baseHeight: Float = 4
    let baseLevels = 7
    for i in 0..<baseLevels {
      let w = baseWidth * (1 - Float(i) * 0.06)
      let y = -2 + Float(i) * (baseHeight / Float(baseLevels))
      appendBox(
        center: SIMD3<Float>(0, y + (baseHeight / Float(baseLevels)) * 0.5, 0),
        size: SIMD3<Float>(w, baseHeight / Float(baseLevels), w), color: colBase, vertices: &v,
        indices: &ix)
    }
    finalize(device: device, vertices: v, indices: ix, color: colBase)
    v.removeAll(keepingCapacity: true)
    ix.removeAll(keepingCapacity: true)

    var currentY: Float = 2
    var currentWidth: Float = 26
    let topWidth: Float = 11
    let layers = 7
    let h0: Float = 5
    let h1: Float = 3

    var windowsV: [MeshVertex] = []
    var windowsIx: [UInt16] = []
    var bracketsV: [MeshVertex] = []
    var bracketsIx: [UInt16] = []
    var beamsV: [MeshVertex] = []
    var beamsIx: [UInt16] = []
    var tilesV: [MeshVertex] = []
    var tilesIx: [UInt16] = []

    for layer in 0..<layers {
      let h = h0 - (h0 - h1) * Float(layer) / Float(layers - 1)
      let bodyCenter = SIMD3<Float>(0, currentY + h * 0.5, 0)
      appendBox(
        center: bodyCenter, size: SIMD3<Float>(currentWidth, h, currentWidth), color: colBody,
        vertices: &v, indices: &ix)

      let eaveOut: Float = 1.8
      let roofT: Float = 0.25
      let innerW = currentWidth
      let outerW = currentWidth + eaveOut * 2
      appendRoof(
        centerY: currentY + h, inner: innerW, outer: outerW, thickness: roofT,
        slope: 12 * (.pi / 180), color: colRoof, vertices: &v, indices: &ix)

      let halfW = currentWidth * 0.5
      let inset: Float = currentWidth * 0.08
      let frameDepth: Float = 0.06
      let frameThick: Float = 0.12
      let windowRows = 2
      let windowCols = max(2, Int(currentWidth / 6))
      for c in 0..<windowCols {
        for r in 0..<windowRows {
          let colT = (Float(c) + 1) / (Float(windowCols) + 1)
          let rowT = (Float(r) + 1) / (Float(windowRows) + 1)
          let px = -halfW + colT * currentWidth
          let py = currentY + rowT * h
          let w = currentWidth / Float(windowCols) * 0.6
          let ht = h / Float(windowRows) * 0.5
          appendWindowFrameFront(
            px: px, py: py, pz: halfW - inset, w: w, h: ht, thick: frameThick, depth: frameDepth,
            vertices: &windowsV, indices: &windowsIx)
          appendWindowFrameBack(
            px: px, py: py, pz: -halfW + inset, w: w, h: ht, thick: frameThick, depth: frameDepth,
            vertices: &windowsV, indices: &windowsIx)
        }
      }

      let eaveY = currentY + h
      let beamsPerSide = 40
      let drop: Float = 0.22
      let outExtra: Float = 0.45
      let beamThick: Float = 0.18
      for i in 0...beamsPerSide {
        let t = Float(i) / Float(beamsPerSide)
        let xo = -outerW * 0.5 + t * outerW
        let zo = -outerW * 0.5 + t * outerW
        appendBeamOutX(
          x: outerW * 0.5, z: zo, y: eaveY, length: outExtra + 0.2, drop: drop, thick: beamThick,
          vertices: &beamsV, indices: &beamsIx)
        appendBeamOutX(
          x: -outerW * 0.5, z: zo, y: eaveY, length: outExtra + 0.2, drop: drop, thick: beamThick,
          vertices: &beamsV, indices: &beamsIx)
        appendBeamOutZ(
          x: xo, z: outerW * 0.5, y: eaveY, length: outExtra + 0.2, drop: drop, thick: beamThick,
          vertices: &beamsV, indices: &beamsIx)
        appendBeamOutZ(
          x: xo, z: -outerW * 0.5, y: eaveY, length: outExtra + 0.2, drop: drop, thick: beamThick,
          vertices: &beamsV, indices: &beamsIx)
      }

      let bracketCols = max(6, Int(currentWidth / 4))
      for c in 0..<bracketCols {
        let t = (Float(c) + 0.5) / Float(bracketCols)
        let x = -halfW + t * currentWidth
        let y0 = eaveY
        appendBracketSetFront(
          x: x, y: y0, z: halfW, out: eaveOut, vertices: &bracketsV, indices: &bracketsIx)
        appendBracketSetBack(
          x: x, y: y0, z: -halfW, out: eaveOut, vertices: &bracketsV, indices: &bracketsIx)
      }

      let tileCount = Int(outerW / 0.6)
      appendTileStripFront(
        y: eaveY + 0.05, inner: innerW, outer: outerW, strips: tileCount, vertices: &tilesV,
        indices: &tilesIx)
      appendTileStripBack(
        y: eaveY + 0.05, inner: innerW, outer: outerW, strips: tileCount, vertices: &tilesV,
        indices: &tilesIx)
      appendTileStripLeft(
        y: eaveY + 0.05, inner: innerW, outer: outerW, strips: tileCount, vertices: &tilesV,
        indices: &tilesIx)
      appendTileStripRight(
        y: eaveY + 0.05, inner: innerW, outer: outerW, strips: tileCount, vertices: &tilesV,
        indices: &tilesIx)

      currentY += h + roofT
      let nextW = currentWidth - (currentWidth - topWidth) / Float(layers - 1)
      currentWidth = max(nextW, topWidth)
    }

    finalize(device: device, vertices: v, indices: ix, color: colBody)
    v.removeAll(keepingCapacity: true)
    ix.removeAll(keepingCapacity: true)
    finalize(device: device, vertices: windowsV, indices: windowsIx, color: colWood)
    windowsV.removeAll(keepingCapacity: true)
    windowsIx.removeAll(keepingCapacity: true)
    finalize(device: device, vertices: bracketsV, indices: bracketsIx, color: colWood)
    bracketsV.removeAll(keepingCapacity: true)
    bracketsIx.removeAll(keepingCapacity: true)
    finalize(device: device, vertices: beamsV, indices: beamsIx, color: colWood)
    beamsV.removeAll(keepingCapacity: true)
    beamsIx.removeAll(keepingCapacity: true)
    finalize(device: device, vertices: tilesV, indices: tilesIx, color: colTile)
    tilesV.removeAll(keepingCapacity: true)
    tilesIx.removeAll(keepingCapacity: true)

    let finialH: Float = 3
    appendPyramid(
      center: SIMD3<Float>(0, currentY + finialH * 0.5, 0),
      size: SIMD3<Float>(currentWidth * 0.7, finialH, currentWidth * 0.7), color: colRoof,
      vertices: &v, indices: &ix)
    finalize(device: device, vertices: v, indices: ix, color: colRoof)
    v.removeAll(keepingCapacity: true)
    ix.removeAll(keepingCapacity: true)

    appendGroundPlaza(
      centerY: -2.5, sizeX: 60, sizeZ: 60, tile: 1.2, borderSteps: 3, vertices: &v, indices: &ix)
    finalize(device: device, vertices: v, indices: ix, color: colGround)
  }

  private func finalize(
    device: MTLDevice, vertices: [MeshVertex], indices: [UInt16], color: SIMD4<Float>
  ) {
    let vbuf = device.makeBuffer(
      bytes: vertices, length: MemoryLayout<MeshVertex>.stride * vertices.count,
      options: .storageModeShared)!
    let ibuf = device.makeBuffer(
      bytes: indices, length: MemoryLayout<UInt16>.stride * indices.count,
      options: .storageModeShared)!
    submeshes.append((v: vbuf, i: ibuf, c: indices.count, color: color))
  }

  private func appendBox(
    center: SIMD3<Float>, size: SIMD3<Float>, color: SIMD4<Float>, vertices: inout [MeshVertex],
    indices: inout [UInt16]
  ) {
    let hw = size.x * 0.5
    let hh = size.y * 0.5
    let hd = size.z * 0.5
    let corners = [
      SIMD3<Float>(center.x - hw, center.y - hh, center.z + hd),
      SIMD3<Float>(center.x + hw, center.y - hh, center.z + hd),
      SIMD3<Float>(center.x + hw, center.y + hh, center.z + hd),
      SIMD3<Float>(center.x - hw, center.y + hh, center.z + hd),
      SIMD3<Float>(center.x + hw, center.y - hh, center.z - hd),
      SIMD3<Float>(center.x - hw, center.y - hh, center.z - hd),
      SIMD3<Float>(center.x - hw, center.y + hh, center.z - hd),
      SIMD3<Float>(center.x + hw, center.y + hh, center.z - hd),
    ]
    let base = UInt16(vertices.count)
    let norms: [SIMD3<Float>] = [
      [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1],
    ]
    for i in 0..<8 { vertices.append(MeshVertex(position: corners[i], normal: norms[i])) }
    indices.append(contentsOf: [base + 0, base + 1, base + 2, base + 0, base + 2, base + 3])
    indices.append(contentsOf: [base + 4, base + 5, base + 6, base + 4, base + 6, base + 7])
    let l = UInt16(vertices.count)
    let left = [
      MeshVertex(
        position: SIMD3<Float>(center.x - hw, center.y - hh, center.z - hd), normal: [-1, 0, 0]),
      MeshVertex(
        position: SIMD3<Float>(center.x - hw, center.y - hh, center.z + hd), normal: [-1, 0, 0]),
      MeshVertex(
        position: SIMD3<Float>(center.x - hw, center.y + hh, center.z + hd), normal: [-1, 0, 0]),
      MeshVertex(
        position: SIMD3<Float>(center.x - hw, center.y + hh, center.z - hd), normal: [-1, 0, 0]),
    ]
    vertices.append(contentsOf: left)
    indices.append(contentsOf: [l + 0, l + 1, l + 2, l + 0, l + 2, l + 3])
    let r = UInt16(vertices.count)
    let right = [
      MeshVertex(
        position: SIMD3<Float>(center.x + hw, center.y - hh, center.z + hd), normal: [1, 0, 0]),
      MeshVertex(
        position: SIMD3<Float>(center.x + hw, center.y - hh, center.z - hd), normal: [1, 0, 0]),
      MeshVertex(
        position: SIMD3<Float>(center.x + hw, center.y + hh, center.z - hd), normal: [1, 0, 0]),
      MeshVertex(
        position: SIMD3<Float>(center.x + hw, center.y + hh, center.z + hd), normal: [1, 0, 0]),
    ]
    vertices.append(contentsOf: right)
    indices.append(contentsOf: [r + 0, r + 1, r + 2, r + 0, r + 2, r + 3])
    let t = UInt16(vertices.count)
    let top = [
      MeshVertex(
        position: SIMD3<Float>(center.x - hw, center.y + hh, center.z + hd), normal: [0, 1, 0]),
      MeshVertex(
        position: SIMD3<Float>(center.x + hw, center.y + hh, center.z + hd), normal: [0, 1, 0]),
      MeshVertex(
        position: SIMD3<Float>(center.x + hw, center.y + hh, center.z - hd), normal: [0, 1, 0]),
      MeshVertex(
        position: SIMD3<Float>(center.x - hw, center.y + hh, center.z - hd), normal: [0, 1, 0]),
    ]
    vertices.append(contentsOf: top)
    indices.append(contentsOf: [t + 0, t + 1, t + 2, t + 0, t + 2, t + 3])
    let b = UInt16(vertices.count)
    let bottom = [
      MeshVertex(
        position: SIMD3<Float>(center.x - hw, center.y - hh, center.z - hd), normal: [0, -1, 0]),
      MeshVertex(
        position: SIMD3<Float>(center.x + hw, center.y - hh, center.z - hd), normal: [0, -1, 0]),
      MeshVertex(
        position: SIMD3<Float>(center.x + hw, center.y - hh, center.z + hd), normal: [0, -1, 0]),
      MeshVertex(
        position: SIMD3<Float>(center.x - hw, center.y - hh, center.z + hd), normal: [0, -1, 0]),
    ]
    vertices.append(contentsOf: bottom)
    indices.append(contentsOf: [b + 0, b + 1, b + 2, b + 0, b + 2, b + 3])
  }

  private func appendRoof(
    centerY: Float, inner: Float, outer: Float, thickness: Float, slope: Float, color: SIMD4<Float>,
    vertices: inout [MeshVertex], indices: inout [UInt16]
  ) {
    let ih = inner * 0.5
    let oh = outer * 0.5
    let rise: Float = (oh - ih) * tan(slope)
    let yTop = centerY + rise
    let yBottom = centerY - thickness
    func face(
      _ nx: Float, _ nz: Float, p1: SIMD3<Float>, p2: SIMD3<Float>, p3: SIMD3<Float>,
      p4: SIMD3<Float>
    ) {
      let base = UInt16(vertices.count)
      let n = simd_normalize(SIMD3<Float>(nx, rise, nz))
      vertices.append(contentsOf: [
        MeshVertex(position: p1, normal: n), MeshVertex(position: p2, normal: n),
        MeshVertex(position: p3, normal: n), MeshVertex(position: p4, normal: n),
      ])
      indices.append(contentsOf: [base + 0, base + 1, base + 2, base + 0, base + 2, base + 3])
    }
    face(
      0, 1, p1: SIMD3<Float>(-ih, yBottom, ih), p2: SIMD3<Float>(ih, yBottom, ih),
      p3: SIMD3<Float>(oh, yTop, oh), p4: SIMD3<Float>(-oh, yTop, oh))
    face(
      1, 0, p1: SIMD3<Float>(ih, yBottom, -ih), p2: SIMD3<Float>(ih, yBottom, ih),
      p3: SIMD3<Float>(oh, yTop, oh), p4: SIMD3<Float>(oh, yTop, -oh))
    face(
      0, -1, p1: SIMD3<Float>(-ih, yBottom, -ih), p2: SIMD3<Float>(ih, yBottom, -ih),
      p3: SIMD3<Float>(oh, yTop, -oh), p4: SIMD3<Float>(-oh, yTop, -oh))
    face(
      -1, 0, p1: SIMD3<Float>(-ih, yBottom, -ih), p2: SIMD3<Float>(-ih, yBottom, ih),
      p3: SIMD3<Float>(-oh, yTop, oh), p4: SIMD3<Float>(-oh, yTop, -oh))
  }

  private func appendPyramid(
    center: SIMD3<Float>, size: SIMD3<Float>, color: SIMD4<Float>, vertices: inout [MeshVertex],
    indices: inout [UInt16]
  ) {
    let hw = size.x * 0.5
    let hh = size.y * 0.5
    let hd = size.z * 0.5
    let top = SIMD3<Float>(center.x, center.y + hh, center.z)
    let b1 = SIMD3<Float>(center.x - hw, center.y - hh, center.z + hd)
    let b2 = SIMD3<Float>(center.x + hw, center.y - hh, center.z + hd)
    let b3 = SIMD3<Float>(center.x + hw, center.y - hh, center.z - hd)
    let b4 = SIMD3<Float>(center.x - hw, center.y - hh, center.z - hd)
    let base = UInt16(vertices.count)
    func tri(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>) {
      let n = simd_normalize(simd_cross(b - a, c - a))
      vertices.append(contentsOf: [
        MeshVertex(position: a, normal: n), MeshVertex(position: b, normal: n),
        MeshVertex(position: c, normal: n),
      ])
      indices.append(contentsOf: [
        UInt16(vertices.count - 3), UInt16(vertices.count - 2), UInt16(vertices.count - 1),
      ])
    }
    tri(top, b1, b2)
    tri(top, b2, b3)
    tri(top, b3, b4)
    tri(top, b4, b1)
    vertices.append(contentsOf: [
      MeshVertex(position: b1, normal: [0, -1, 0]), MeshVertex(position: b2, normal: [0, -1, 0]),
      MeshVertex(position: b3, normal: [0, -1, 0]), MeshVertex(position: b4, normal: [0, -1, 0]),
    ])
    indices.append(contentsOf: [base + 12, base + 13, base + 14, base + 12, base + 14, base + 15])
  }

  private func appendWindowFrameFront(
    px: Float, py: Float, pz: Float, w: Float, h: Float, thick: Float, depth: Float,
    vertices: inout [MeshVertex], indices: inout [UInt16]
  ) {
    appendBox(
      center: SIMD3<Float>(px, py + h * 0.5, pz), size: SIMD3<Float>(w, thick, depth), color: .zero,
      vertices: &vertices, indices: &indices)
    appendBox(
      center: SIMD3<Float>(px, py - h * 0.5, pz), size: SIMD3<Float>(w, thick, depth), color: .zero,
      vertices: &vertices, indices: &indices)
    appendBox(
      center: SIMD3<Float>(px - w * 0.5, py, pz), size: SIMD3<Float>(thick, h, depth), color: .zero,
      vertices: &vertices, indices: &indices)
    appendBox(
      center: SIMD3<Float>(px + w * 0.5, py, pz), size: SIMD3<Float>(thick, h, depth), color: .zero,
      vertices: &vertices, indices: &indices)
    appendBox(
      center: SIMD3<Float>(px, py, pz), size: SIMD3<Float>(w * 0.02, h, depth), color: .zero,
      vertices: &vertices, indices: &indices)
    appendBox(
      center: SIMD3<Float>(px, py, pz), size: SIMD3<Float>(w, h * 0.02, depth), color: .zero,
      vertices: &vertices, indices: &indices)
  }

  private func appendWindowFrameBack(
    px: Float, py: Float, pz: Float, w: Float, h: Float, thick: Float, depth: Float,
    vertices: inout [MeshVertex], indices: inout [UInt16]
  ) {
    appendBox(
      center: SIMD3<Float>(px, py + h * 0.5, pz), size: SIMD3<Float>(w, thick, depth), color: .zero,
      vertices: &vertices, indices: &indices)
    appendBox(
      center: SIMD3<Float>(px, py - h * 0.5, pz), size: SIMD3<Float>(w, thick, depth), color: .zero,
      vertices: &vertices, indices: &indices)
    appendBox(
      center: SIMD3<Float>(px - w * 0.5, py, pz), size: SIMD3<Float>(thick, h, depth), color: .zero,
      vertices: &vertices, indices: &indices)
    appendBox(
      center: SIMD3<Float>(px + w * 0.5, py, pz), size: SIMD3<Float>(thick, h, depth), color: .zero,
      vertices: &vertices, indices: &indices)
    appendBox(
      center: SIMD3<Float>(px, py, pz), size: SIMD3<Float>(w * 0.02, h, depth), color: .zero,
      vertices: &vertices, indices: &indices)
    appendBox(
      center: SIMD3<Float>(px, py, pz), size: SIMD3<Float>(w, h * 0.02, depth), color: .zero,
      vertices: &vertices, indices: &indices)
  }

  private func appendBracketSetFront(
    x: Float, y: Float, z: Float, out: Float, vertices: inout [MeshVertex], indices: inout [UInt16]
  ) {
    let s: Float = 0.4
    appendBox(
      center: SIMD3<Float>(x, y - s * 0.3, z + out * 0.35),
      size: SIMD3<Float>(s * 0.6, s * 0.2, s * 0.6), color: .zero, vertices: &vertices,
      indices: &indices)
    appendBox(
      center: SIMD3<Float>(x, y - s * 0.6, z + out * 0.15),
      size: SIMD3<Float>(s * 0.5, s * 0.2, s * 0.5), color: .zero, vertices: &vertices,
      indices: &indices)
  }

  private func appendBracketSetBack(
    x: Float, y: Float, z: Float, out: Float, vertices: inout [MeshVertex], indices: inout [UInt16]
  ) {
    let s: Float = 0.4
    appendBox(
      center: SIMD3<Float>(x, y - s * 0.3, z - out * 0.35),
      size: SIMD3<Float>(s * 0.6, s * 0.2, s * 0.6), color: .zero, vertices: &vertices,
      indices: &indices)
    appendBox(
      center: SIMD3<Float>(x, y - s * 0.6, z - out * 0.15),
      size: SIMD3<Float>(s * 0.5, s * 0.2, s * 0.5), color: .zero, vertices: &vertices,
      indices: &indices)
  }

  private func appendBeamOutZ(
    x: Float, z: Float, y: Float, length: Float, drop: Float, thick: Float,
    vertices: inout [MeshVertex], indices: inout [UInt16]
  ) {
    appendBox(
      center: SIMD3<Float>(x, y - drop, z + length * 0.5), size: SIMD3<Float>(thick, thick, length),
      color: .zero, vertices: &vertices, indices: &indices)
  }

  private func appendBeamOutX(
    x: Float, z: Float, y: Float, length: Float, drop: Float, thick: Float,
    vertices: inout [MeshVertex], indices: inout [UInt16]
  ) {
    appendBox(
      center: SIMD3<Float>(x + (x > 0 ? length * 0.5 : -length * 0.5), y - drop, z),
      size: SIMD3<Float>(length, thick, thick), color: .zero, vertices: &vertices, indices: &indices
    )
  }

  private func appendTileStripFront(
    y: Float, inner: Float, outer: Float, strips: Int, vertices: inout [MeshVertex],
    indices: inout [UInt16]
  ) {
    let ih = inner * 0.5
    let pitch = (outer - inner) / Float(strips)
    for i in 0..<strips {
      let z = ih + pitch * Float(i)
      appendBox(
        center: SIMD3<Float>(0, y, z), size: SIMD3<Float>(outer, 0.05, 0.08), color: .zero,
        vertices: &vertices, indices: &indices)
    }
  }

  private func appendTileStripBack(
    y: Float, inner: Float, outer: Float, strips: Int, vertices: inout [MeshVertex],
    indices: inout [UInt16]
  ) {
    let ih = inner * 0.5
    let pitch = (outer - inner) / Float(strips)
    for i in 0..<strips {
      let z = -ih - pitch * Float(i)
      appendBox(
        center: SIMD3<Float>(0, y, z), size: SIMD3<Float>(outer, 0.05, 0.08), color: .zero,
        vertices: &vertices, indices: &indices)
    }
  }

  private func appendTileStripLeft(
    y: Float, inner: Float, outer: Float, strips: Int, vertices: inout [MeshVertex],
    indices: inout [UInt16]
  ) {
    let ih = inner * 0.5
    let pitch = (outer - inner) / Float(strips)
    for i in 0..<strips {
      let x = -ih - pitch * Float(i)
      appendBox(
        center: SIMD3<Float>(x, y, 0), size: SIMD3<Float>(0.08, 0.05, outer), color: .zero,
        vertices: &vertices, indices: &indices)
    }
  }

  private func appendTileStripRight(
    y: Float, inner: Float, outer: Float, strips: Int, vertices: inout [MeshVertex],
    indices: inout [UInt16]
  ) {
    let ih = inner * 0.5
    let pitch = (outer - inner) / Float(strips)
    for i in 0..<strips {
      let x = ih + pitch * Float(i)
      appendBox(
        center: SIMD3<Float>(x, y, 0), size: SIMD3<Float>(0.08, 0.05, outer), color: .zero,
        vertices: &vertices, indices: &indices)
    }
  }

  private func appendGroundPlaza(
    centerY: Float, sizeX: Float, sizeZ: Float, tile: Float, borderSteps: Int,
    vertices: inout [MeshVertex], indices: inout [UInt16]
  ) {
    let halfX = sizeX * 0.5
    let halfZ = sizeZ * 0.5
    var x: Float = -halfX
    while x < halfX {
      var z: Float = -halfZ
      while z < halfZ {
        let cx = x + tile * 0.5
        let cz = z + tile * 0.5
        appendBox(
          center: SIMD3<Float>(cx, centerY, cz), size: SIMD3<Float>(tile * 0.98, 0.05, tile * 0.98),
          color: .zero, vertices: &vertices, indices: &indices)
        z += tile
      }
      x += tile
    }
    for s in 0..<borderSteps {
      let wX = sizeX + Float(s + 1) * 2
      let wZ = sizeZ + Float(s + 1) * 2
      appendBox(
        center: SIMD3<Float>(0, centerY - 0.05 * Float(s + 1), 0), size: SIMD3<Float>(wX, 0.08, wZ),
        color: .zero, vertices: &vertices, indices: &indices)
    }
  }
}
