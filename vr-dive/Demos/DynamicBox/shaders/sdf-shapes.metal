// sdf-shapes.metal – Complex SDF shapes in the box
//
// Renders a smooth blend of geometric SDF primitives (sphere, torus, box,
// rounded cylinder) inside the bounding box with dynamic colors.
//
// The function MUST be named `dynamicBoxFragment`.

// ─── Basic SDF primitives ─────────────────────────────────────────────────────
static float sdSphere(float3 p, float r) {
    return length(p) - r;
}

static float sdBox(float3 p, float3 b) {
    float3 q = abs(p) - b;
    return length(max(q, 0.0f)) + min(max(q.x, max(q.y, q.z)), 0.0f);
}

static float sdTorus(float3 p, float2 t) {
    float2 q = float2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

static float sdCylinder(float3 p, float3 c) {
    return length(max(abs(p) - c, 0.0f));
}

static float opSmoothUnion(float d1, float d2, float k) {
    float h = clamp(0.5f + 0.5f * (d2 - d1) / k, 0.0f, 1.0f);
    return mix(d2, d1, h) - k * h * (1.0f - h);
}

static float opSmoothSubtraction(float d1, float d2, float k) {
    float h = clamp(0.5f - 0.5f * (d2 + d1) / k, 0.0f, 1.0f);
    return mix(d2, -d1, h) + k * h * (1.0f - h);
}

// ─── Scene SDF ────────────────────────────────────────────────────────────────
static float map(float3 p, float t_time) {
    float t = t_time * 0.3f;

    // Central sphere
    float3 p1 = p;
    p1.y += 0.15f * sin(t + p.x * 0.5f);
    float d1 = sdSphere(p1 - float3(0.3f * cos(t * 0.7f), 0.0f, 0.2f * sin(t * 0.5f)), 0.35f);

    // Torus (rotating)
    float3 p2 = p;
    float ca = cos(t * 0.6f), sa = sin(t * 0.6f);
    p2.xz = float2(p2.x * ca - p2.z * sa, p2.x * sa + p2.z * ca);
    float d2 = sdTorus(p2 - float3(-0.2f * cos(t * 0.4f), -0.1f, 0.0f), float2(0.35f, 0.08f));

    // Rounded box
    float3 p3 = p;
    p3.yz = float2(p3.y * ca - p3.z * sa, p3.y * sa + p3.z * ca);
    float d3 = sdBox(p3 - float3(0.0f, 0.25f, 0.0f), float3(0.25f, 0.1f, 0.25f));

    // Combine
    float d = opSmoothUnion(d1, d2, 0.15f);
    d = opSmoothUnion(d, d3, 0.12f);

    // Subtractive sphere to create a cavity
    float3 p4 = p;
    p4.z += 0.3f * sin(t * 0.8f);
    float d4 = sdSphere(p4, 0.2f);
    d = opSmoothSubtraction(d, d4, 0.08f);

    return d;
}

// ─── Normal from gradient ─────────────────────────────────────────────────────
static float3 calcNormal(float3 p, float t) {
    float2 e = float2(0.005f, 0.0f);
    return normalize(float3(
        map(p + e.xyy, t) - map(p - e.xyy, t),
        map(p + e.yxy, t) - map(p - e.yxy, t),
        map(p + e.yyx, t) - map(p - e.yyx, t)
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

    if (!insideBox) {
        float3 entryNormal;
        float  tEnter = db_boxHit(boxEye, boxRd, DB_BOXDIMS, entryNormal, true);
        if (tEnter < 0.0f) {
            return float4(0.01f, 0.01f, 0.03f, 1.0f);
        }
        marchOrigin = boxEye + boxRd * (tEnter + 1e-3f);
    } else {
        marchOrigin = boxEye;
    }

    float3 exitNormal;
    float  tExit = db_boxHit(marchOrigin, boxRd, DB_BOXDIMS, exitNormal, false);
    if (tExit <= 0.0f) discard_fragment();

    float3 ro = (uniforms.patternTransform * float4(marchOrigin, 1.0f)).xyz;
    float3 rd = normalize(float3(uniforms.patternTransform * float4(boxRd, 0.0f)));

    // ─── Ray march ─────────────────────────────────────────────────────────
    float t = 0.0f;
    float tmax = 25.0f;
    int   i;

    for (i = 0; i < 80; i++) {
        float3 p = ro + rd * t;
        float d = map(p, uniforms.time);
        if (d < 0.002f) break;
        t += d;
        if (t > tmax) break;
    }

    if (t >= tmax || i >= 80) {
        // Subtle fog
        return float4(0.02f, 0.02f, 0.06f, 1.0f);
    }

    // ─── Shade ─────────────────────────────────────────────────────────────
    float3 p = ro + rd * t;
    float3 n = calcNormal(p, uniforms.time);

    // Lighting
    float3 lightDir = normalize(float3(0.5f, 1.0f, 0.3f));
    float  dif = max(dot(n, lightDir), 0.0f);
    float  amb = 0.3f + 0.7f * max(dot(n, float3(0.0f, 1.0f, 0.0f)), 0.0f);
    float  rim = 1.0f - max(dot(-rd, n), 0.0f);
    rim = pow(rim, 3.0f) * 0.6f;

    // Color based on position + normal
    float hue = fract(length(p) * 0.4f + uniforms.time * 0.02f);
    float3 albedo;
    {
        float4 K = float4(1.0f, 2.0f / 3.0f, 1.0f / 3.0f, 3.0f);
        float3 pp = abs(fract(hue + K.xyz) * 6.0f - K.www);
        albedo = clamp(pp - K.xxx, 0.0f, 1.0f);
    }
    albedo = mix(albedo, float3(1.0f), 0.2f);

    float3 col = albedo * (dif * 1.2f + amb * 0.4f) + float3(0.4f, 0.6f, 1.0f) * rim;

    // Fog falloff
    col *= exp(-t * 0.3f);

    return float4(col, 1.0f);
}
