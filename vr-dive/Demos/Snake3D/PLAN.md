# 3D 贪食蛇 (Snake3D) - 开发计划

## 项目概述

在 VisionOS 沉浸式空间中实现的 3D 贪食蛇游戏。核心视觉机制：**蛇头始终朝向玩家正前方**，玩家按转向键时，游戏世界做反向旋转，造成"蛇一直向前冲、空间在翻滚"的沉浸感。

---

## 核心机制：世界旋转视角

### 原理

- 蛇在逻辑 3D 网格中记录真实方向，但渲染时蛇头坐标系始终对齐摄像机前方（-Z 轴）
- 每次转向，维护一个**累积世界旋转四元数** `worldOrientation: simd_quatf`
- 渲染所有几何体（蛇身、食物、边界框）时，先用 `worldOrientation` 旋转世界坐标，再做 ViewProjection 变换
- 转向时，用 `simd_slerp` 做平滑插值，动画时长约 0.3 秒

### 转向与世界旋转对应关系

| PS 手柄按键 | 蛇的逻辑转向（世界坐标） | 世界渲染反向旋转     |
| ----------- | ------------------------ | -------------------- |
| Dpad Up     | 蛇向当前相对"上"（+Y）转 | 绕局部 X 轴旋转 −90° |
| Dpad Down   | 蛇向当前相对"下"（−Y）转 | 绕局部 X 轴旋转 +90° |
| Dpad Left   | 蛇向当前相对"左"（−X）转 | 绕局部 Y 轴旋转 +90° |
| Dpad Right  | 蛇向当前相对"右"（+X）转 | 绕局部 Y 轴旋转 −90° |
| □ (Square)  | 绕前进轴向左翻滚         | 绕局部 Z 轴旋转 −90° |
| ○ (Circle)  | 绕前进轴向右翻滚         | 绕局部 Z 轴旋转 +90° |

> 注意：每次转向都是相对于蛇当前的朝向坐标系，而非世界绝对轴。旋转轴需在蛇的局部坐标系中计算后转换到世界空间。

### 蛇头位置的锚定

- 蛇头在渲染空间始终位于固定坐标（如略微偏前，距相机约 0.6m 处）
- 世界中所有坐标 = `worldRotation * (worldPos - snakeHeadWorldPos)` + 渲染锚点
- 这样旋转轴心固定在蛇头位置，转向时空间绕蛇头翻转

---

## 文件结构

```
vr-dive/Demos/Snake3D/
├── PLAN.md                  ← 本文件
├── Snake3DTypes.swift        # 数据类型 & GPU 结构体
├── Snake3DGameLogic.swift    # 游戏逻辑（网格、移动、碰撞）
├── Snake3DRenderer.swift     # Metal 渲染器（VisualPatternController）
└── Snake3DShaders.metal      # 顶点/片段着色器
```

---

## 各文件职责详述

### Snake3DTypes.swift

```swift
// 6 个方向枚举
enum SnakeDirection { case posX, negX, posY, negY, posZ, negZ }

// 游戏状态（纯值类型，线程安全拷贝）
struct Snake3DState {
    var segments: [SIMD3<Int>]   // 蛇体，index 0 = 头部
    var direction: SnakeDirection
    var foods: [SIMD3<Int>]      // 同时存在的食物列表
    var score: Int
    var isGameOver: Bool
    var gridSize: Int            // 立方体网格边长（默认 20）
}

// GPU 实例缓冲区（蛇身节段）
struct SnakeSegmentInstance {
    var position: SIMD3<Float>
    var color: SIMD4<Float>      // 头部亮绿，尾部渐暗
    var scale: Float
}

// GPU 食物实例
struct FoodInstance {
    var position: SIMD3<Float>
    var phase: Float             // 动画相位（脉冲）
}

// CPU→GPU Scene Uniforms
struct Snake3DSceneUniforms {
    var worldRotation: float4x4  // 世界旋转矩阵（累积）
    var anchorOffset: SIMD3<Float> // 蛇头锚定偏移
    var time: Float
    var layerCount: UInt32
}
```

### Snake3DGameLogic.swift

- **网格**：20×20×20，坐标范围 [0, 19]³
- **蛇体存储**：`[SIMD3<Int>]`，每次前进 prepend 新头，pop 尾（若未吃食物）
- **食物**：随机生成，最多 3 个同时存在，不与蛇身重叠
- **碰撞检测**：
  - 撞自身：新头坐标在 `segments` 中命中 → isGameOver
  - 撞墙：是否环绕由 `wallMode` 参数控制（计划支持环绕 / 死亡两种）
- **移动计时**：每 `moveInterval` 秒前进一格，随分数加速
- **转向处理**：不允许与当前方向相反（180° 调头）

### Snake3DRenderer.swift

继承 `VisualPatternController`，职责：

1. **世界旋转状态**
   - `currentWorldQuat: simd_quatf`（当前渲染四元数）
   - `targetWorldQuat: simd_quatf`（目标四元数，转向时更新）
   - 每帧用 `simd_slerp` 插值，旋转动画 0.3s 完成
   - 旋转轴心偏移：将蛇头锚定在屏幕前方固定点

2. **输入处理**（读取 `GameManager` D-pad 状态）
   - Dpad → `gameLogic.handleInput(...)` 改变蛇方向
   - 同步更新 `targetWorldQuat`（反向旋转）
   - 按键防抖：`buttonDelay = 0.3s`（≥ 移动间隔）

3. **渲染帧**（`draw(drawable:commandBuffer:uniforms:)` 协议方法）
   - 更新 `Snake3DSceneUniforms`（worldRotation, time, layerCount）
   - 上传蛇身 Instance Buffer（最多 200 节）
   - 上传食物 Instance Buffer（最多 3 个）
   - Draw call：蛇身（instanced cube）→ 食物（instanced sphere/cube）→ 边界框（line strip）

4. **边界框**
   - 20×20×20 半透明线框，帮助玩家感知空间
   - 随世界旋转一同变换

### Snake3DShaders.metal

```metal
// 顶点着色器（蛇身 + 食物共用）
vertex SnakeVertexOut snake3DVertexShader(...)
// 应用 worldRotation * (pos - headAnchor) + screenAnchor
// 再乘 ViewProjection

// 片段着色器（蛇身）
fragment float4 snake3DBodyFragment(...)
// 头部 = 明亮绿色，尾部颜色根据实例 index 衰减至深色

// 片段着色器（食物）
fragment float4 snake3DFoodFragment(...)
// 发光脉冲：sin(time * 3 + phase) * 0.3 + 0.7 调制亮度

// 边界框着色器
vertex/fragment snake3DBorderShader(...)
// 低透明度白色线框
```

---

## 与现有系统集成

### PatternSelection.swift

在 `VisualPatternKind` 枚举中添加：

```swift
case snake3D
// displayName: "3D 贪食蛇"
```

### Renderer.swift

仿照 `addTetris3D` 方法，添加 `addSnake3D(...)` 静态方法：

```swift
static func addSnake3D(
    to controllers: inout [VisualPatternKind: VisualPatternController],
    device: MTLDevice, library: MTLLibrary,
    maxViewCount: Int, gameManager: GameManager
) {
    controllers[.snake3D] = Snake3DRenderer(...)
}
```

---

## 游戏参数（默认值）

| 参数         | 值                               | 说明                           |
| ------------ | -------------------------------- | ------------------------------ |
| 网格大小     | 20×20×20                         | 约 2m³ 的物理空间              |
| 格子步长     | 0.1m                             | 每格 10cm                      |
| 方块大小     | 0.09m                            | 方块不占满格子，留 1cm 间隙    |
| 蛇头锚点     | (0, 0, −0.6m)                    | 相对相机的固定渲染位置         |
| 初始蛇长     | 5 节                             |                                |
| 初始移动间隔 | 0.4s/格                          |                                |
| 加速规则     | 每 5 分提速 5%                   | 上限 0.15s/格                  |
| 同时食物数   | 3 个                             |                                |
| 旋转动画时长 | 0.3s                             | slerp 插值                     |
| 边界模式     | 无限空间，食物只出现在原始区域内 | 蛇出界不死，食物限制在 [0,19]³ |
| 蛇身渲染     | 立方体（方的）                   |                                |
| 翻滚控制     | □ 左翻 / ○ 右翻                  | 绕前进轴 Z 轴旋转              |
| 重开按键     | Options                          |                                |

---

## 待确认问题

1. **边界模式**：撞墙死亡 vs 环绕穿越（Pac-Man 风格）？
2. **旋转动画时长**：0.3s 是否舒适？是否需要防晕动病（降低旋转速度/增加参考线）？
3. **游戏空间大小**：20³ 格 × 4cm = 80cm 边长，是否合适？
4. **蛇速**：初始 0.4s/格，是否合适？
5. **食物数量**：同时 3 个，是否合适？
6. **蛇身渲染**：纯立方体节段 vs 带圆角/连接处的光滑外观？
7. **游戏重置**：按哪个按键重开游戏（如 Options 键）？

---

## 开发顺序

- [ ] 1. `Snake3DTypes.swift` — 类型定义 & GPU 结构体
- [ ] 2. `Snake3DGameLogic.swift` — 游戏逻辑（不含渲染）
- [ ] 3. `Snake3DShaders.metal` — Metal 着色器
- [ ] 4. `Snake3DRenderer.swift` — 渲染器 & 输入处理
- [ ] 5. `PatternSelection.swift` — 注册新模式
- [ ] 6. `Renderer.swift` — 集成到主渲染循环
