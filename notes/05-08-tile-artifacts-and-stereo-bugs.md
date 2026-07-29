# 05-08 瓦片伪影与立体渲染 Bug 排查记录

## 症状

多个 demo（PongWar、Snake3D、Stereographic 多胞体等）在 visionOS Simulator 上，
几何体周围出现明显的彩色方块（瓦片 / tile artifacts）。同时发现过右眼黑屏的问题。

---

## 根本原因 1：瓦片伪影（tile artifacts）

### 机制

visionOS 的 foveation（注视渲染）通过 `rasterizationRateMap` 将渲染目标分成密度
不均的 tile 块。启用 foveation 后，Metal 要求渲染器在每个 draw call 之前**显式**调
用 `setVertexAmplificationCount`，即使 viewCount == 1 时也必须调用，否则 GPU 不
知道如何将虚拟 viewport 坐标（如 4338×3478）映射到物理 texture（如 1888×1792），
未覆盖的 tile 会以 clear color 的颜色暴露出来。

### 直接触发条件

`RendererTypes.swift` 的 `applyViewConfiguration` 的 `else` 分支被误删：

```swift
// 错误（会导致 tile artifacts）：
if viewData.viewCount > 1 {
    encoder.setVertexAmplificationCount(...)
}
// ← 缺少 else，单视图时 GPU 拿不到 amplification count

// 正确：
if viewData.viewCount > 1 {
    encoder.setVertexAmplificationCount(viewData.viewCount, viewMappings: &viewMappings)
} else {
    encoder.setVertexAmplificationCount(1, viewMappings: nil)  // ← 必须保留！
}
```

### 次要暴露条件：clearColor 不是纯黑

Foveation tile 被 clear 到 clearColor。如果 clearColor 是纯黑 `(0,0,0,1)`，tile
在黑色背景中不可见。如果 clearColor 有任何颜色分量，tile 就会以彩色方块出现。

**当前解决方案**：Renderer.swift 中 clearColor 硬编码为纯黑，与 `main` 分支保持
一致，完全忽略各 demo 的 `preferredClearColor` 属性。

```swift
// Renderer.swift - encodeDrawable 内
// ⚠️ 必须是纯黑。foveation 的 rasterizationRateMap 会把渲染目标分成 tile，
// 未被几何体覆盖的区域会被 clear 到此颜色。任何非黑色都会造成可见的彩色瓦片。
// 各 demo 的 preferredClearColor 属性目前被忽略（见各 renderer 文件）。
let clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
```

---

## 根本原因 2：右眼黑屏（两眼画面叠到左眼）

### 机制

visionOS layered layout 下，左右眼在**同一张 texture** 的不同 array slice（左眼
slice=0，右眼 slice=1）。

- `textureMap.textureIndex`：该 view 对应 drawable 的哪个 texture（两眼都是 0）
- `textureMap.sliceIndex`：该 texture 内的哪个 array slice（左眼 0，右眼 1）

在 `makeViewRenderingData` 中必须用 `sliceIndex`：

```swift
// 错误（导致两眼都渲染到 layer 0，右眼黑屏）：
renderTargetLayers.append(UInt32(textureMap.textureIndex))

// 正确：
renderTargetLayers.append(UInt32(textureMap.sliceIndex))
```

---

## 根本原因 3：`cp_frame_end_submission` before `encodePresent` 崩溃

### 机制

CompositorServices 规定：调用 `queryDrawables()` 后，每个 drawable 都必须经过
`encodePresent(commandBuffer:)` 才能调用 `endSubmission()`。

当 world tracking 尚未就绪（无 deviceAnchor）时，原代码走 `completeEmptySubmission`
路径，只调用 `startSubmission/endSubmission`，跳过了 `encodePresent`，导致崩溃。

**修复**：`completeEmptySubmissionIfPossible` 接受 `drawables` 参数，对每个 drawable
提交一个空的命令 buffer：

```swift
private func completeEmptySubmissionIfPossible(
    for frame: LayerRenderer.Frame,
    drawables: [LayerRenderer.Drawable] = []
) {
    guard layerRenderer.state == .running else { return }
    frame.startSubmission()
    for drawable in drawables {
        guard let cmdBuf = commandQueue.makeCommandBuffer() else { continue }
        drawable.encodePresent(commandBuffer: cmdBuf)
        cmdBuf.commit()
    }
    frame.endSubmission()
}
```

---

## 排查过程中的误操作记录

以下操作**不是解决方案**，记录以避免重蹈：

1. **`isFoveationEnabled = false`**：不能消除 tile。即使禁用 foveation，
   simulator 仍可能返回非 trivial 的 rate map。
2. **`rasterizationRateMap = nil`**：单独设为 nil 不能解决问题，且会破坏
   foveation 的虚拟坐标到物理坐标的映射。
3. **用物理 texture 尺寸覆盖 viewport**：`textureMap.viewport` 是虚拟 foveation
   坐标，不应替换为物理尺寸，这会破坏 foveation 的工作方式。
4. **`renderTargetArrayLength = max(min(...), 1)` 而非 `colorTexture.arrayLength`**：
   clamp 版本会导致 render target 层数不足。

---

## 关键文件位置

| 文件                           | 关键位置                            | 说明                                                             |
| ------------------------------ | ----------------------------------- | ---------------------------------------------------------------- |
| `Renderer/RendererTypes.swift` | `applyViewConfiguration`            | **必须**保留 else 分支调用 `setVertexAmplificationCount(1, nil)` |
| `Renderer/Renderer.swift`      | `encodeDrawable` clearColor         | 硬编码黑色，不得改为读取 `preferredClearColor`                   |
| `Renderer/Renderer.swift`      | `makeViewRenderingData`             | 用 `sliceIndex` 不是 `textureIndex`                              |
| `Renderer/Renderer.swift`      | `completeEmptySubmissionIfPossible` | 必须传入 drawables 并 encodePresent                              |
