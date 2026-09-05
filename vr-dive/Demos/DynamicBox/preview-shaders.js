#!/usr/bin/env node
// Render real Metal frames on the Mac, not a JavaScript approximation.
// Usage: node preview-shaders.js /absolute/output/directory [shader names...]
// PREVIEW_TIMES=0,17,43 PREVIEW_SIZE=512 node preview-shaders.js ...
// PREVIEW_ASSERT_DYNAMIC=1 also rejects unchanged frames at later times.
// Includes front and oblique views. This is visual QA, not a visionOS benchmark.
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const { PRELUDE } = require("./check-shaders.js");

const output = process.argv[2];
if (!output || !path.isAbsolute(output)) {
  console.error("Usage: node preview-shaders.js /absolute/output/directory [shader names...]");
  process.exit(1);
}
const names = process.argv.slice(3);
if (!names.length) names.push("clifford-lantern", "hopf-fibration", "tesseract-jewel", "cell24-prism", "s3-trefoil");
for (const name of names) {
  if (!/^[a-z0-9-]+$/.test(name) || !fs.existsSync(path.join(__dirname, "shaders", name + ".metal"))) {
    throw new Error("Unknown shader: " + name);
  }
}
const times = process.env.PREVIEW_TIMES || "0,17,43";
if (!times.split(",").every(t => t.trim() !== "" && Number.isFinite(Number(t)) && Number(t) >= 0 && Number(t) <= 86400)) {
  throw new Error("PREVIEW_TIMES must contain seconds in 0...86400, separated by commas");
}
const size = Number(process.env.PREVIEW_SIZE || 512);
if (!Number.isInteger(size) || size < 64 || size > 2048) throw new Error("PREVIEW_SIZE must be 64...2048");
fs.mkdirSync(output, { recursive: true });
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "dynbox-preview-"));
const swift = String.raw`
import Foundation
import Metal
import simd
import CoreGraphics
import CoreText
import ImageIO
import UniformTypeIdentifiers

struct Uniforms {
    var time: Float
    var viewCount: UInt32 = 1
    var boxScale: Float = 1
    var pad: Float = 0
    var objectCenter = SIMD4<Float>(repeating: 0)
    var patternTransform = matrix_identity_float4x4
}
enum PreviewError: Error { case failure(String) }
func require<T>(_ value: T?, _ message: String) throws -> T {
    guard let value else { throw PreviewError.failure(message) }
    return value
}
let args = CommandLine.arguments
let shaderDirectory = args[1], preludePath = args[2], outputDirectory = args[3]
let times = args[4].split(separator: ",").map { Float($0.trimmingCharacters(in: .whitespaces))! }
let size = Int(args[5])!
let device = try require(MTLCreateSystemDefaultDevice(), "No Metal GPU is available")
let queue = try require(device.makeCommandQueue(), "No command queue")
let prelude = try String(contentsOfFile: preludePath, encoding: .utf8)
let vertexSource = """
vertex DynamicBoxVertexOut previewVertex(uint id [[vertex_id]],
                                         constant float4x4 *v2w [[buffer(1)]]) {
    float2 xy = id==0 ? float2(-1,-1) : (id==1 ? float2(3,-1) : float2(-1,3));
    DynamicBoxVertexOut out;
    out.clipPos=float4(xy,0,1);
    out.worldPos=v2w[0][3].xyz-v2w[0][2].xyz
               +0.5f*(xy.x*v2w[0][0].xyz+xy.y*v2w[0][1].xyz);
    out.viewIndex=0;
    return out;
}
"""
let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: size, height: size, mipmapped: false)
td.usage = [.renderTarget]
td.storageMode = .shared
let texture = try require(device.makeTexture(descriptor: td), "No output texture")
let cameras: [(String, SIMD3<Float>)] = [
    ("front", SIMD3<Float>(0,0.16,3.45)),
    ("oblique", SIMD3<Float>(2.5,1.25,2.8))
]
func stamp(_ time: Float) -> String {
    time.rounded() == time ? String(Int(time)) : String(time).replacingOccurrences(of: ".",with: "p")
}
func srgb(_ v: Float) -> UInt8 {
    let x = max(0,min(1,v))
    let encoded = x <= 0.0031308 ? x*12.92 : 1.055*pow(x,1/2.4)-0.055
    return UInt8(max(0,min(255,Int((encoded*255).rounded()))))
}
print("Metal preview GPU: \(device.name); \(size)x\(size), mono, offscreen")
for name in args.dropFirst(6) {
    let source = try String(contentsOfFile: shaderDirectory + "/" + name + ".metal", encoding: .utf8)
    let library = try device.makeLibrary(source: prelude + "\n" + source + "\n" + vertexSource, options: nil)
    let pd = MTLRenderPipelineDescriptor()
    pd.vertexFunction = library.makeFunction(name: "previewVertex")
    pd.fragmentFunction = library.makeFunction(name: "dynamicBoxFragment")
    pd.colorAttachments[0].pixelFormat = .rgba16Float
    let pipeline = try device.makeRenderPipelineState(descriptor: pd)
    var baselineByView: [String: [UInt16]] = [:]
    for time in times {
        for (view, eye) in cameras {
            let forward = normalize(-eye)
            let right = normalize(cross(forward,SIMD3<Float>(0,1,0)))
            let up = cross(right,forward)
            var camera = simd_float4x4(SIMD4<Float>(right,0),SIMD4<Float>(up,0),
                                       SIMD4<Float>(-forward,0),SIMD4<Float>(eye,1))
            var uniform = Uniforms(time: time)
            var gpuMS = 0.0
            // Warm the pipeline before recording an indicative GPU duration.
            for _ in 0..<3 {
                let pass = MTLRenderPassDescriptor()
                pass.colorAttachments[0].texture = texture
                pass.colorAttachments[0].loadAction = .clear
                pass.colorAttachments[0].storeAction = .store
                let command = try require(queue.makeCommandBuffer(), "No command buffer")
                let encoder = try require(command.makeRenderCommandEncoder(descriptor: pass), "No render encoder")
                encoder.setRenderPipelineState(pipeline)
                encoder.setVertexBytes(&camera,length: MemoryLayout<simd_float4x4>.stride,index: 1)
                encoder.setFragmentBytes(&uniform,length: MemoryLayout<Uniforms>.stride,index: 0)
                encoder.setFragmentBytes(&camera,length: MemoryLayout<simd_float4x4>.stride,index: 1)
                encoder.setFragmentBytes(&camera,length: MemoryLayout<simd_float4x4>.stride,index: 2)
                encoder.drawPrimitives(type: .triangle,vertexStart: 0,vertexCount: 3)
                encoder.endEncoding()
                command.commit()
                command.waitUntilCompleted()
                if let error = command.error { throw error }
                gpuMS = (command.gpuEndTime-command.gpuStartTime)*1000
            }
            var raw = [UInt16](repeating: 0,count: size*size*4)
            raw.withUnsafeMutableBytes {
                texture.getBytes($0.baseAddress!,bytesPerRow: size*8,
                                 from: MTLRegionMake2D(0,0,size,size),mipmapLevel: 0)
            }
            var rgba = [UInt8](repeating: 255,count: size*size*4)
            var invalid = 0, visible = 0
            for pixel in 0..<size*size {
                var peak: Float = 0
                for channel in 0..<3 {
                    let value = Float(Float16(bitPattern: raw[pixel*4+channel]))
                    if !value.isFinite { invalid += 1 }
                    peak = max(peak,value.isFinite ? value : 0)
                    rgba[pixel*4+channel] = value.isFinite ? srgb(value) : 0
                }
                if peak > 0.08 { visible += 1 }
            }
            if invalid > 0 { throw PreviewError.failure("\(name): \(invalid) non-finite channels") }
            if visible == 0 { throw PreviewError.failure("\(name): empty frame at \(time), \(view)") }
            if ProcessInfo.processInfo.environment["PREVIEW_ASSERT_DYNAMIC"] == "1" {
                if let baseline = baselineByView[view] {
                    var changed = 0
                    for pixel in 0..<size*size {
                        var difference: Float = 0
                        for channel in 0..<3 {
                            let index = pixel*4+channel
                            difference += abs(Float(Float16(bitPattern: raw[index]))-Float(Float16(bitPattern: baseline[index])))
                        }
                        if difference > 0.015 { changed += 1 }
                    }
                    if changed < size*size/500 {
                        throw PreviewError.failure("\(name): no meaningful animation at \(time), \(view)")
                    }
                    print("\(name) t=\(time) \(view): \(changed) pixels changed from first time")
                } else {
                    baselineByView[view] = raw
                }
            }
            let data = Data(rgba)
            let provider = try require(CGDataProvider(data: data as CFData), "No image provider")
            let space = try require(CGColorSpace(name: CGColorSpace.sRGB), "No color space")
            let image = try require(CGImage(width: size,height: size,bitsPerComponent: 8,bitsPerPixel: 32,
                              bytesPerRow: size*4,space: space,bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                              provider: provider,decode: nil,shouldInterpolate: false,intent: .defaultIntent), "No image")
            let filename = "\(name)-\(stamp(time))-\(view).png"
            let url = URL(fileURLWithPath: outputDirectory).appendingPathComponent(filename)
            let destination = try require(CGImageDestinationCreateWithURL(url as CFURL,UTType.png.identifier as CFString,1,nil),
                                          "No PNG destination")
            CGImageDestinationAddImage(destination,image,nil)
            guard CGImageDestinationFinalize(destination) else { throw PreviewError.failure("Could not write PNG") }
            print("\(filename): GPU \(String(format: "%.2f",gpuMS)) ms, visible \(visible) pixels, finite OK")
        }
    }
}
// Contact sheets make visual review at several times/views inexpensive.
for time in times {
    for (view, _) in cameras {
        let names = Array(args.dropFirst(6))
        let columns = names.count == 4 ? 2 : min(3,names.count), rows = (names.count+columns-1)/columns
        let cellHeight = size+36, width = columns*size, height = rows*cellHeight
        let space = try require(CGColorSpace(name: CGColorSpace.sRGB),"No sheet color space")
        let context = try require(CGContext(data: nil,width: width,height: height,bitsPerComponent: 8,bytesPerRow: width*4,
                                   space: space,bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),"No sheet context")
        context.setFillColor(CGColor(red: 0.05,green: 0.065,blue: 0.1,alpha: 1))
        context.fill(CGRect(x: 0,y: 0,width: width,height: height))
        for (index, name) in names.enumerated() {
            let x = (index%columns)*size, y = height-(index/columns+1)*cellHeight
            let url = URL(fileURLWithPath: outputDirectory).appendingPathComponent("\(name)-\(stamp(time))-\(view).png")
            let source = try require(CGImageSourceCreateWithURL(url as CFURL,nil),"No sheet input")
            let image = try require(CGImageSourceCreateImageAtIndex(source,0,nil),"No sheet image")
            context.draw(image,in: CGRect(x: x,y: y+36,width: size,height: size))
            let attributes: [NSAttributedString.Key: Any] = [
                NSAttributedString.Key(kCTFontAttributeName as String): CTFontCreateWithName("Menlo" as CFString,18,nil),
                NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(gray: 0.9,alpha: 1)
            ]
            let line = CTLineCreateWithAttributedString(NSAttributedString(string: name,attributes: attributes))
            context.textMatrix = .identity
            context.setTextDrawingMode(.fill)
            context.textPosition = CGPoint(x: x+16,y: y+12)
            CTLineDraw(line,context)
        }
        let image = try require(context.makeImage(),"No sheet image")
        let url = URL(fileURLWithPath: outputDirectory).appendingPathComponent("overview-\(stamp(time))-\(view).png")
        let destination = try require(CGImageDestinationCreateWithURL(url as CFURL,UTType.png.identifier as CFString,1,nil),
                                      "No sheet destination")
        CGImageDestinationAddImage(destination,image,nil)
        guard CGImageDestinationFinalize(destination) else { throw PreviewError.failure("Could not write sheet") }
    }
}
`;
try {
  fs.writeFileSync(path.join(tmp, "prelude.metal"), PRELUDE);
  fs.writeFileSync(path.join(tmp, "preview.swift"), swift);
  execFileSync("xcrun", ["swift", path.join(tmp, "preview.swift"), path.join(__dirname, "shaders"),
    path.join(tmp, "prelude.metal"), output, times, String(size), ...names], { stdio: "inherit" });
} finally {
  fs.rmSync(tmp, { recursive: true, force: true });
}
