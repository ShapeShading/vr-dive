// maze3d.metal – 3D Procedural Maze
//
// Generates a traversable maze using hash-based wall placement.
// Each 2m × 2m cell randomly has walls on its north/east borders,
// creating corridors wide enough (~1.9m) to walk through.
// Walls have decorative brick texture and colored accents.

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

// ─── Maze SDF ────────────────────────────────────────────────────────────────
// 2m cells, walls on cell borders determined by hash.
static float mazeSDF(float3 p) {
    float cellSize = 2.0f;
    float wallHalf = 0.04f;   // wall thickness / 2
    float wallH    = 1.0f;    // wall height (floor to ceiling = 2m)

    float2 cell = floor(p.xz / cellSize);
    float2 local = (p.xz / cellSize) - cell; // [0, 1) within cell

    float d = 1e10f;

    // East wall (+x border of this cell)
    float hE = hash21(cell);
    if (hE < 0.45f) {
        float dx = (local.x - 1.0f) * cellSize; // distance to east border
        float dz = min(local.y, 1.0f - local.y) * cellSize;
        float dy = abs(p.y);
        // Span-containment term: very negative at the wall's z-center (deep
        // inside its length), rising to 0 at the wall's z-ends.
        d = min(d, max(max(abs(dx) - wallHalf, -dz), dy - wallH));
    }

    // North wall (+z border of this cell)
    float hN = hash21(cell + float2(100, 0));
    if (hN < 0.45f) {
        float dz = (local.y - 1.0f) * cellSize;
        float dx = min(local.x, 1.0f - local.x) * cellSize;
        float dy = abs(p.y);
        d = min(d, max(max(abs(dz) - wallHalf, -dx), dy - wallH));
    }

    // West wall (-x border = east wall of cell (ix-1, iz))
    float hW = hash21(cell + float2(-1, 0));
    if (hW < 0.45f) {
        float dx = local.x * cellSize;
        float dz = min(local.y, 1.0f - local.y) * cellSize;
        float dy = abs(p.y);
        d = min(d, max(max(abs(dx) - wallHalf, -dz), dy - wallH));
    }

    // South wall (-z border = north wall of cell (ix, iz-1))
    float hS = hash21(cell + float2(100, -1));
    if (hS < 0.45f) {
        float dz = local.y * cellSize;
        float dx = min(local.x, 1.0f - local.x) * cellSize;
        float dy = abs(p.y);
        d = min(d, max(max(abs(dz) - wallHalf, -dx), dy - wallH));
    }

    // Floor and ceiling
    float floorCeil = abs(p.y) - wallH;
    d = min(d, floorCeil);

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

            // Determine if floor, wall, or ceiling
            float2 cell = floor(p.xz / 2.0f);
            float floorMask = step(p.y, -0.95f);
            float ceilMask = step(0.95f, p.y);
            float isWall = 1.0f - floorMask - ceilMask;

            // Wall color: brick-like with cell-based variation
            float cellHue = hash21(cell);
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
            float star = hash21(floor(p.xz * 8.0f));
            if (star > 0.995f) {
                ceilColor += float3(0.5f, 0.7f, 1.0f) * 3.0f;
            }

            // Composite
            float3 col = wallColor * isWall + floorColor * floorMask + ceilColor * ceilMask;
            col = col * (dif * 0.8f + amb * 0.5f) + float3(0.1f, 0.2f, 0.4f) * rim;

            // Fog
            col *= exp(-march * 0.12f);
            return float4(col, 1.0f);
        }
        march += max(d, 0.01f);
        if (march > maxD) break;
    }
    return float4(bgColor, 1.0f);
}
