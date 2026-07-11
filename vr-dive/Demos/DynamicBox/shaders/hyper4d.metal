// hyper4d.metal – 4D Hypersphere Projection
//
// Renders 4D geometric shapes projected into 3D via ray marching.
// The 4D scene rotates continuously in 4D space (XY, XZ, YZ, XW, YW, ZW
// planes), and the 3D ray-march samples the 4D distance field.
//
// This creates morphing 3D shapes that smoothly transform as 4D rotation
// brings different cross-sections into view.

// ─── 4D rotation helpers ──────────────────────────────────────────────────────
static float4 rotXY(float4 p, float a) {
    float s = sin(a), c = cos(a);
    return float4(p.x*c - p.y*s, p.x*s + p.y*c, p.z, p.w);
}
static float4 rotXZ(float4 p, float a) {
    float s = sin(a), c = cos(a);
    return float4(p.x*c - p.z*s, p.y, p.x*s + p.z*c, p.w);
}
static float4 rotXW(float4 p, float a) {
    float s = sin(a), c = cos(a);
    return float4(p.x*c - p.w*s, p.y, p.z, p.x*s + p.w*c);
}
static float4 rotYZ(float4 p, float a) {
    float s = sin(a), c = cos(a);
    return float4(p.x, p.y*c - p.z*s, p.y*s + p.z*c, p.w);
}
static float4 rotYW(float4 p, float a) {
    float s = sin(a), c = cos(a);
    return float4(p.x, p.y*c - p.w*s, p.z, p.y*s + p.w*c);
}
static float4 rotZW(float4 p, float a) {
    float s = sin(a), c = cos(a);
    return float4(p.x, p.y, p.z*c - p.w*s, p.z*s + p.w*c);
}

// ─── 4D SDF: distance from a 4D point to a 4D hyper-torus ────────────────────
// A 3-torus in 4D: (sqrt(x²+y²) - R1)² + (sqrt(z²+w²) - R2)² = r²
static float sdHyperTorus(float4 p, float t) {
    float R1 = 0.6f;
    float R2 = 0.35f;
    float r  = 0.08f;

    float d1 = length(p.xy) - R1;
    float d2 = length(p.zw) - R2;
    return length(float2(d1, d2)) - r;
}

// ─── 4D SDF: 4D sphere packing (hyper-balls at lattice points) ───────────────
static float sdSpherePack4D(float4 p, float t) {
    float4 gp = round(p / 0.5f) * 0.5f;
    float  r  = 0.12f + 0.03f * sin(t * 0.5f + dot(gp, float4(1.3f, 2.7f, 3.1f, 4.9f)));
    return length(p - gp) - r;
}

// ─── Combined 4D SDF ─────────────────────────────────────────────────────────
static float map(float4 p, float t) {
    float d1 = sdHyperTorus(p, t);
    float d2 = sdSpherePack4D(p, t);
    return min(d1, d2);
}

// ─── 4D → 3D perspective projection ──────────────────────────────────────────
// Projects a 4D point into 3D using perspective divide by (1 + w * fovFactor).
// Points with large |w| are pushed toward the center (perspective foreshortening).
static float3 project4D(float4 p, float fov) {
    float wScale = 1.0f / (1.0f + p.w * fov);
    return p.xyz * wScale;
}

// ─── 3D normal via 4D distance gradient ──────────────────────────────────────
static float3 calcNormal(float3 p3, float4 p4, float t) {
    float2 e = float2(0.002f, 0.0f);
    float4 dx = float4(e.x, 0, 0, 0);
    float4 dy = float4(0, e.x, 0, 0);
    float4 dz = float4(0, 0, e.x, 0);
    return normalize(float3(
        map(p4 + dx, t) - map(p4 - dx, t),
        map(p4 + dy, t) - map(p4 - dy, t),
        map(p4 + dz, t) - map(p4 - dz, t)
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
    float3 bgColor = float3(0.0f, 0.0f, 0.03f);

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

    // ─── 4D rotation (animates over time) ──────────────────────────────────
    float t = uniforms.time;
    float4 p4;

    // ─── Ray march ─────────────────────────────────────────────────────────
    float march = 0.0f;
    float maxD  = 25.0f;

    for (int i = 0; i < 100; i++) {
        float3 p3 = ro + rd * march;

        // Build 4D coordinate from 3D position + implicit w from projection
        // For perspective projection, w is derived from the 3D distance
        float fov = 0.45f;
        float w = (1.0f - fov * length(p3)) / fov;
        // Alternative: start with w=0 and use 4D rotation to mix it in

        p4 = float4(p3, 0.0f);

        // Apply 4D rotations
        p4 = rotXY(p4, t * 0.17f);
        p4 = rotZW(p4, t * 0.23f);
        p4 = rotXW(p4, t * 0.11f);
        p4 = rotYZ(p4, t * 0.13f);

        // Perspective projection from 4D → 3D
        float3 q3 = project4D(p4, 0.4f);

        // The 4D SDF evaluated at the 4D point
        float d = map(p4, t);

        // Scale SDF by the projection Jacobian for correct step size
        float projScale = 1.0f / (1.0f + p4.w * 0.4f);
        d *= projScale;

        if (d < 0.004f) {
            float3 n = calcNormal(q3, p4, t);
            float3 light = normalize(float3(0.5f, 1.0f, 0.3f));

            float dif = max(dot(n, light), 0.0f);
            float amb = 0.3f + 0.7f * abs(n.y);
            float rim = 1.0f - max(dot(-rd, n), 0.0f);
            rim = pow(rim, 4.0f) * 0.7f;

            // Iridescent color shifting with 4D position
            float hue = fract(length(p4) * 0.3f + t * 0.04f + p4.w * 0.1f);
            float3 col;
            {
                float4 K = float4(1.0f, 2.0f/3.0f, 1.0f/3.0f, 3.0f);
                float3 pp = abs(fract(hue + K.xyz) * 6.0f - K.www);
                col = clamp(pp - K.xxx, 0.0f, 1.0f);
            }
            col = col * (dif * 1.0f + amb * 0.5f) + float3(0.3f, 0.6f, 1.0f) * rim;
            col *= exp(-march * 0.3f);

            return float4(col, 1.0f);
        }
        march += d;
        if (march > maxD) break;
    }

    return float4(bgColor, 1.0f);
}
