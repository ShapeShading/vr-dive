// maze3d.metal – 3D Procedural Maze (randomized 3D spanning-tree redesign)
//
// Complete redesign replacing the earlier deterministic serpentine corridor:
// this version generates a genuinely random, fully-connected 3D maze bounded
// to a finite grid, using a "growing binary tree" spanning-tree algorithm
// generalized from 2D (north/east) to 3D (north/east/up). Every cell picks
// exactly one outgoing passage among the directions that stay inside the
// grid; boundary cells have fewer choices, and the single far corner cell
// has none (it is the tree's root). Because every non-root cell contributes
// exactly one edge and no cell can ever point at a smaller coordinate, the
// result is mathematically a single connected tree with zero cycles and zero
// isolated rooms — real branching, real dead ends, and (new) vertical shafts
// connecting multiple stacked floors, unlike the old flat single-level snake.
//
// ─── 设计方案 ────────────────────────────────────────────────────────────
// 思路: 每个格子 (ix,iy,iz) 只有一条"出边"，从{北(+z)/东(+x)/上(+y)}中按
//       hash 随机选择一个仍在网格范围内的方向；到达网格最大边界的格子
//       会失去对应方向的选项，最远角落格子（ix=iz=MAX,iy=MAX）三个方向都
//       耗尽，作为生成树的根，不需要出边。某格子的北/东/上墙是否开启，
//       只取决于它自己的选择；南/西/下墙是否开启，则取决于对应邻居格子
//       的选择是否指向自己。由于「出边只能指向更大坐标」，不可能出现环
//       路，也不可能有格子被孤立——这是与旧版本「独立随机四面墙」在数学
//       上的根本区别（旧版本不保证连通，新版本用生成树结构保证连通）。
//       网格范围外的格子整体渲染成实心块，构成迷宫的外壳边界。
// 关键参数:
//   - cellSize = 2.2、floorH = 2.2、passageWidth = 1.5、wallHalf = 0.04：
//     房间、门洞、竖直井口净宽均在摄像机可轻松通过的 1–2m 范围内。
//   - 网格范围 ix,iz ∈ [-11,11]（半径 ≈24.2m，略小于 maxD=25，保证可见
//     范围内几乎处处是真实迷宫结构而非过早撞到外壳）；iy ∈ [0,2]
//     （3 层楼），让迷宫第一次真正利用竖直方向。
//   - 墙体/楼板洞口统一用「实体盒子 - CSG 减去中心洞口盒子」实现
//     （boxWithHole），比旧版本的双段拼接写法更简单、更不容易出现接缝。
// 性能特征: 每次 map() 用 hash31 做 4 次「出边方向」查询（自身+南/西/下
//           三个邻居），再评估 6 个面（4 面墙 + 上下楼板），每面最多 1 次
//           boxSDF + 1 次 CSG 减法；无迭代循环，march 110 步/maxD=25，
//           开销与旧版本相近（仅多了 hash 查询，未新增循环）。
// 已知限制/优化方向:
//   - 当前只有 3 层楼、11×11 格的有限区域，超出范围会撞到实心外壳；如
//     需更大迷宫可直接放宽网格边界常量。
//   - 生成树只保证「存在且仅存在一条路径」连接任意两个房间（树结构，无
//     环路），如果想要有环路（多条路径）的迷宫，需要在树的基础上额外
//     随机打通少量墙（不破坏连通性，只增加环）。

// ─── Pseudo-random ────────────────────────────────────────────────────────────
static float hash21(float2 c) {
    float3 p = fract(float3(c.x, c.y, c.x + c.y) * 0.1031f);
    p += dot(p, p.yxz + 33.33f);
    return fract((p.x + p.y) * p.z);
}

static float hash31(float3 c) {
    float3 p = fract(c * 0.1031f);
    p += dot(p, p.yxz + 33.33f);
    return fract((p.x + p.y) * p.z);
}

static float noise(float3 p) {
    float3 i = floor(p);
    float3 f = fract(p);
    f = f * f * (3.0f - 2.0f * f);
    return mix(mix(mix(hash31(i), hash31(i + float3(1,0,0)), f.x),
                   mix(hash31(i + float3(0,1,0)), hash31(i + float3(1,1,0)), f.x), f.y),
               mix(mix(hash31(i + float3(0,0,1)), hash31(i + float3(1,0,1)), f.x),
                   mix(hash31(i + float3(0,1,1)), hash31(i + float3(1,1,1)), f.x), f.y), f.z);
}

// ─── 3D growing-tree maze: per-cell single outgoing direction ────────────────
// 0 = none (root cell, no outgoing edge), 1 = North (+z), 2 = East (+x), 3 = Up (+y)
static int cellParentDir(float3 cell) {
    const float maxIX = 11.0f;
    const float maxIZ = 11.0f;
    const float maxIY = 2.0f;

    bool canNorth = cell.z < maxIZ;
    bool canEast  = cell.x < maxIX;
    bool canUp    = cell.y < maxIY;

    int count = (canNorth ? 1 : 0) + (canEast ? 1 : 0) + (canUp ? 1 : 0);
    if (count == 0) return 0;

    float h = hash31(cell);
    int pick = int(floor(h * float(count)));
    pick = min(pick, count - 1);

    int idx = 0;
    if (canNorth) { if (idx == pick) return 1; idx++; }
    if (canEast)  { if (idx == pick) return 2; idx++; }
    if (canUp)    { if (idx == pick) return 3; idx++; }
    return 0;
}

// ─── Maze SDF ────────────────────────────────────────────────────────────────
static float boxSDF(float3 p, float3 halfSize) {
    float3 q = abs(p) - halfSize;
    return length(max(q, 0.0f)) + min(max(q.x, max(q.y, q.z)), 0.0f);
}

// Solid box with an optional rectangular hole subtracted from its center.
static float boxWithHole(float3 p, float3 fullHalf, float3 holeHalf, bool open) {
    float outer = boxSDF(p, fullHalf);
    if (!open) return outer;
    float hole = boxSDF(p, holeHalf);
    return max(outer, -hole);
}

static float wallX(float3 p, float boundaryX, float cellIY, float cellIZ,
                   float cellSize, float floorH, float passageWidth,
                   float wallHalf, bool open) {
    float centerY = (cellIY + 0.5f) * floorH;
    float centerZ = (cellIZ + 0.5f) * cellSize;
    float3 local = p - float3(boundaryX, centerY, centerZ);
    float3 fullHalf = float3(wallHalf, floorH * 0.5f, cellSize * 0.5f);
    float3 holeHalf = float3(wallHalf + 0.5f, floorH * 0.5f + 0.01f, passageWidth * 0.5f);
    return boxWithHole(local, fullHalf, holeHalf, open);
}

static float wallZ(float3 p, float boundaryZ, float cellIX, float cellIY,
                   float cellSize, float floorH, float passageWidth,
                   float wallHalf, bool open) {
    float centerX = (cellIX + 0.5f) * cellSize;
    float centerY = (cellIY + 0.5f) * floorH;
    float3 local = p - float3(centerX, centerY, boundaryZ);
    float3 fullHalf = float3(cellSize * 0.5f, floorH * 0.5f, wallHalf);
    float3 holeHalf = float3(passageWidth * 0.5f, floorH * 0.5f + 0.01f, wallHalf + 0.5f);
    return boxWithHole(local, fullHalf, holeHalf, open);
}

static float slabY(float3 p, float boundaryY, float cellIX, float cellIZ,
                   float cellSize, float slabHalf, float passageWidth,
                   bool open) {
    float centerX = (cellIX + 0.5f) * cellSize;
    float centerZ = (cellIZ + 0.5f) * cellSize;
    float3 local = p - float3(centerX, boundaryY, centerZ);
    float3 fullHalf = float3(cellSize * 0.5f, slabHalf, cellSize * 0.5f);
    float3 holeHalf = float3(passageWidth * 0.5f, slabHalf + 0.5f, passageWidth * 0.5f);
    return boxWithHole(local, fullHalf, holeHalf, open);
}

static float mazeSDF(float3 p) {
    float cellSize = 2.2f;
    float floorH = 2.2f;
    float passageWidth = 1.5f;
    float wallHalf = 0.04f;
    float slabHalf = 0.05f;

    const float minIX = -11.0f, maxIX = 11.0f;
    const float minIZ = -11.0f, maxIZ = 11.0f;
    const float minIY = 0.0f,   maxIY = 2.0f;

    float3 cellF = floor(float3(p.x / cellSize, p.y / floorH, p.z / cellSize));
    float ix = cellF.x, iy = cellF.y, iz = cellF.z;

    // Outside the finite maze grid: render a solid enclosing shell.
    if (ix < minIX || ix > maxIX || iz < minIZ || iz > maxIZ || iy < minIY || iy > maxIY) {
        float3 centerLocal = p - (cellF + 0.5f) * float3(cellSize, floorH, cellSize);
        return boxSDF(centerLocal, float3(cellSize * 0.5f, floorH * 0.5f, cellSize * 0.5f));
    }

    int myDir = cellParentDir(cellF);
    bool northOpen = (myDir == 1);
    bool eastOpen  = (myDir == 2);
    bool upOpen    = (myDir == 3);

    float3 southNeighbor = cellF - float3(0.0f, 0.0f, 1.0f);
    float3 westNeighbor  = cellF - float3(1.0f, 0.0f, 0.0f);
    float3 downNeighbor  = cellF - float3(0.0f, 1.0f, 0.0f);

    bool southOpen = (southNeighbor.z >= minIZ) && (cellParentDir(southNeighbor) == 1);
    bool westOpen  = (westNeighbor.x >= minIX)  && (cellParentDir(westNeighbor) == 2);
    bool downOpen  = (downNeighbor.y >= minIY)  && (cellParentDir(downNeighbor) == 3);

    float d = 1e10f;
    d = min(d, wallX(p, (ix + 1.0f) * cellSize, iy, iz, cellSize, floorH, passageWidth, wallHalf, eastOpen));
    d = min(d, wallX(p, ix * cellSize, iy, iz, cellSize, floorH, passageWidth, wallHalf, westOpen));
    d = min(d, wallZ(p, (iz + 1.0f) * cellSize, ix, iy, cellSize, floorH, passageWidth, wallHalf, northOpen));
    d = min(d, wallZ(p, iz * cellSize, ix, iy, cellSize, floorH, passageWidth, wallHalf, southOpen));
    d = min(d, slabY(p, (iy + 1.0f) * floorH, ix, iz, cellSize, slabHalf, passageWidth, upOpen));
    d = min(d, slabY(p, iy * floorH, ix, iz, cellSize, slabHalf, passageWidth, downOpen));

    return d;
}

// ─── Normal ───────────────────────────────────────────────────────────────────
static float3 calcNormal(float3 p) {
    float2 e = float2(0.003f, 0.0f);
    return normalize(float3(
        mazeSDF(p + e.xyy) - mazeSDF(p - e.xyy),
        mazeSDF(p + e.yxy) - mazeSDF(p - e.yxy),
        mazeSDF(p + e.yyx) - mazeSDF(p - e.yyx)
    ));
}

// ─── Fragment shader ──────────────────────────────────────────────────────────
fragment float4 dynamicBoxFragment(
    DynamicBoxVertexOut        in         [[stage_in]],
    constant DynamicBoxUniforms &uniforms [[buffer(0)]],
    constant float4x4         *v2wMats    [[buffer(1)]],
    constant float4x4         *vpMatrices [[buffer(2)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);

    float4x4 v2w    = v2wMats[vi];
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

    float3 center  = uniforms.objectCenter.xyz;
    float  sc      = uniforms.boxScale;
    float3 boxEye  = (camWorld - center) / sc;
    float3 boxRd   = normalize(in.worldPos - camWorld);

    bool insideBox = all(abs(boxEye) < (DB_BOXDIMS - 1e-3f));
    float3 marchOrigin;
    float3 bgColor = float3(0.02f, 0.02f, 0.03f);

    if (!insideBox) {
        float3 entryNormal;
        float  tEnter = db_boxHit(boxEye, boxRd, DB_BOXDIMS, entryNormal, true);
        if (tEnter < 0.0f) return float4(bgColor, 1.0f);
        marchOrigin = boxEye + boxRd * (tEnter + 1e-3f);
    } else {
        marchOrigin = boxEye;
    }

    float3 exitNormal;
    float  tExit = db_boxHit(marchOrigin, boxRd, DB_BOXDIMS, exitNormal, false);
    if (tExit <= 0.0f) discard_fragment();

    float3 ro = (uniforms.patternTransform * float4(marchOrigin, 1.0f)).xyz;
    float3 rd = normalize(float3(uniforms.patternTransform * float4(boxRd, 0.0f)));

    float march = 0.0f;
    float maxD = 25.0f;

    for (int i = 0; i < 120; i++) {
        float3 p = ro + rd * march;
        float d = mazeSDF(p);
        if (d < 0.005f) {
            float3 n = calcNormal(p);
            float3 light = normalize(float3(0.3f, 0.6f, 0.5f));

            float dif = max(dot(n, light), 0.0f);
            float amb = 0.2f + 0.8f * max(dot(n, float3(0,1,0)), 0.0f);
            float rim = 1.0f - max(dot(-rd, n), 0.0f);
            rim = pow(rim, 3.0f) * 0.4f;

            // Determine if floor, wall, or ceiling using the *local* height
            // within this cell's floor level (multiple stacked floors mean a
            // global p.y test no longer identifies floor vs. ceiling).
            float cellSize = 2.2f;
            float floorH = 2.2f;
            float ix = floor(p.x / cellSize);
            float iy = floor(p.y / floorH);
            float iz = floor(p.z / cellSize);
            float localY = p.y - iy * floorH;
            float floorMask = step(localY, 0.07f);
            float ceilMask = step(floorH - 0.07f, localY);
            float isWall = 1.0f - floorMask - ceilMask;

            // Wall color: brick-like with per-cell-and-floor variation
            float cellHue = hash21(float2(ix + iy * 37.0f, iz));
            float3 wallColor;
            {
                float4 K = float4(1,2/3.f,1/3.f,3);
                float3 pp = abs(fract(cellHue + K.xyz) * 6.0f - K.www);
                wallColor = clamp(pp - K.xxx, 0.0f, 1.0f);
            }
            wallColor = mix(wallColor, float3(0.8f, 0.7f, 0.5f), 0.5f);

            // Brick texture on walls
            float brick = noise(p * 4.0f) * 0.15f;
            wallColor += brick;

            // Floor: dark with subtle grid
            float3 floorColor = float3(0.05f, 0.05f, 0.08f);
            float2 grid = abs(fract(p.xz * 0.5f) - 0.5f);
            float gridLine = max(grid.x, grid.y);
            floorColor += float3(0.03f, 0.03f, 0.05f) * (1.0f - smoothstep(0.0f, 0.05f, gridLine));

            // Ceiling: dark with small glowing dots
            float3 ceilColor = float3(0.03f, 0.03f, 0.05f);
            float star = hash21(floor(p.xz * 8.0f) + iy * 5.0f);
            if (star > 0.995f) {
                ceilColor += float3(0.5f, 0.7f, 1.0f) * 3.0f;
            }

            // Highlight vertical shaft openings near this cell so the player
            // notices the maze now connects floors, not just a flat plane.
            int hereDir = cellParentDir(float3(ix, iy, iz));
            bool hereUpOpen = (hereDir == 3);
            float3 downNeighbor = float3(ix, iy - 1.0f, iz);
            bool hereDownOpen = (downNeighbor.y >= 0.0f) && (cellParentDir(downNeighbor) == 3);
            float shaftGlow = 0.0f;
            if (hereUpOpen || hereDownOpen) {
                float2 cellCenter = (float2(ix, iz) + 0.5f) * cellSize;
                float distToShaft = length(p.xz - cellCenter);
                shaftGlow = (1.0f - smoothstep(0.5f, 1.1f, distToShaft)) * 0.6f;
            }

            // Composite
            float3 col = wallColor * isWall + floorColor * floorMask + ceilColor * ceilMask;
            col = col * (dif * 0.8f + amb * 0.5f) + float3(0.1f, 0.2f, 0.4f) * rim;
            col += float3(0.3f, 0.8f, 1.0f) * shaftGlow * (floorMask + ceilMask);

            // Fog
            col *= exp(-march * 0.12f);
            return float4(col, 1.0f);
        }
        march += max(d, 0.01f);
        if (march > maxD) break;
    }
    return float4(bgColor, 1.0f);
}
