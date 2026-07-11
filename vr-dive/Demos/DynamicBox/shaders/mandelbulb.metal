// mandelbulb.metal – 3D Mandelbulb Fractal
//
// A classic 3D fractal rendered with ray marching inside the DynamicBox.
// The Mandelbulb is a 3D generalization of the Mandelbrot set using
// spherical coordinates. Colors cycle with position and time.

// ─── Rotation helpers ─────────────────────────────────────────────────────────
static float3x3 rotX(float a) {
    float s = sin(a), c = cos(a);
    return float3x3(float3(1,0,0), float3(0,c,s), float3(0,-s,c));
}
static float3x3 rotY(float a) {
    float s = sin(a), c = cos(a);
    return float3x3(float3(c,0,s), float3(0,1,0), float3(-s,0,c));
}

// ─── Mandelbulb DE (Distance Estimator) ───────────────────────────────────────
// Standard power-8 Mandelbulb.
static float mandelbulbDE(float3 p, float t) {
    float3 w = p;
    float  m = dot(w, w);
    float  dz = 1.0f;
    float  r  = 1.0f;
    float  power = 8.0f + 0.5f * sin(t * 0.15f);

    for (int i = 0; i < 10; i++) {
        if (m > 256.0f) break;

        // Convert to spherical coordinates
        r   = length(w);
        float th  = acos(w.y / r);
        float ph  = atan2(w.z, w.x);

        // Apply power
        float rp = pow(r, power - 1.0f);
        dz = rp * dz * power + 1.0f;
        rp *= r;

        th *= power;
        ph *= power;

        w = rp * float3(sin(th) * cos(ph), cos(th), sin(th) * sin(ph)) + p;

        m = dot(w, w);
    }
    return 0.5f * log(m) * r / dz;
}

// ─── Color palette ────────────────────────────────────────────────────────────
static float3 palette(float t) {
    float3 a = float3(0.5f, 0.5f, 0.5f);
    float3 b = float3(0.5f, 0.5f, 0.5f);
    float3 c = float3(1.0f, 0.7f, 0.4f);
    float3 d = float3(0.00f, 0.15f, 0.20f);
    return a + b * cos(6.28318f * (c * t + d));
}

// ─── Normal ───────────────────────────────────────────────────────────────────
static float3 calcNormal(float3 p, float t) {
    float2 e = float2(0.001f, 0.0f);
    return normalize(float3(
        mandelbulbDE(p + e.xyy, t) - mandelbulbDE(p - e.xyy, t),
        mandelbulbDE(p + e.yxy, t) - mandelbulbDE(p - e.yxy, t),
        mandelbulbDE(p + e.yyx, t) - mandelbulbDE(p - e.yyx, t)
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
    float3 bgColor = float3(0.02f, 0.0f, 0.04f);

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

    // Rotate scene for visual interest
    float t = uniforms.time;
    ro *= rotY(t * 0.08f);
    rd *= rotY(t * 0.08f);
    ro *= rotX(t * 0.05f);
    rd *= rotX(t * 0.05f);

    // ─── Ray march ─────────────────────────────────────────────────────────
    float marchDist = 0.0f;
    float maxDist = min(tExit, 3.5f);

    for (int i = 0; i < 100; i++) {
        float3 p = ro + rd * marchDist;
        float d = mandelbulbDE(p, t);
        if (d < 0.002f) {
            // Color based on iteration count and position
            float3 n = calcNormal(p, t);
            float3 light = normalize(float3(0.5f, 1.0f, 0.3f));
            float dif = max(dot(n, light), 0.0f);
            float amb = 0.3f + 0.7f * max(dot(n, float3(0,1,0)), 0.0f);

            float colT = length(p) * 0.5f + t * 0.05f;
            float3 col = palette(colT) * (dif * 1.2f + amb * 0.4f);

            // Fog
            col *= exp(-marchDist * 0.4f);
            return float4(col, 1.0f);
        }
        marchDist += d;
        if (marchDist > maxDist) break;
    }

    return float4(bgColor, 1.0f);
}
