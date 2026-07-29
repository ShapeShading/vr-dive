// nebula.metal – Volumetric Nebula with Stars
//
// Renders a colorful nebula cloud with embedded stars using
// volumetric ray marching. The nebula density is generated with
// layered procedural noise, producing wispy organic shapes.

// ─── Pseudo-random ────────────────────────────────────────────────────────────
static float hash(float3 p) {
    p = fract(p * 0.3183099f + 0.1f);
    p *= 17.0f;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

static float noise(float3 p) {
    float3 i = floor(p);
    float3 f = fract(p);
    f = f * f * (3.0f - 2.0f * f);
    float a = hash(i);
    float b = hash(i + float3(1,0,0));
    float c = hash(i + float3(0,1,0));
    float d = hash(i + float3(1,1,0));
    float e = hash(i + float3(0,0,1));
    float f_ = hash(i + float3(1,0,1));
    float g = hash(i + float3(0,1,1));
    float h = hash(i + float3(1,1,1));
    return mix(mix(mix(a,b,f.x), mix(c,d,f.x), f.y),
               mix(mix(e,f_,f.x), mix(g,h,f.x), f.y), f.z);
}

static float fbm(float3 p) {
    float v = 0.0f;
    float a = 0.5f;
    float3 shift = float3(100, 200, 300);
    for (int i = 0; i < 3; i++) {
        v += a * noise(p);
        p = p * 2.0f + shift;
        a *= 0.5f;
    }
    return v;
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
    float3 bgColor = float3(0.0f, 0.0f, 0.01f);

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

    float t = uniforms.time;

    // ─── Volumetric ray march ──────────────────────────────────────────────
    float march = 0.0f;
    float maxD  = 25.0f;
    float step  = 0.08f;

    float3 accumC = float3(0.0f);
    float  accumA = 1.0f;

    for (int i = 0; i < 100; i++) {
        float3 p = ro + rd * march;
        float  d = fbm(p * 0.8f + t * 0.05f);

        // Nebula density
        float density = max(0.0f, d * 1.2f - 0.35f);

        if (density > 0.01f) {
            // Nebula color: blue-purple-pink gradient based on position + noise
            float3 col = mix(
                float3(0.1f, 0.2f, 0.8f),  // blue
                float3(0.8f, 0.2f, 0.6f),  // pink
                fbm(p + float3(50, 50, 50))
            );
            col = mix(col, float3(1.0f, 0.5f, 0.2f),  // orange glow
                      smoothstep(0.3f, 0.7f, fbm(p * 0.5f + t * 0.03f)));

            float alpha = density * 0.3f * step * 20.0f;
            accumC += col * alpha * accumA;
            accumA *= 1.0f - alpha;
            if (accumA < 0.02f) break;
        }

        // Tiny stars: bright pinpricks at random positions (check every other step)
        if ((i & 1) == 0) {
            float3 cell = floor(p * 15.0f);
            float3 rnd  = float3(hash(cell), hash(cell + 100.0f), hash(cell + 200.0f));
            float star = smoothstep(0.998f, 1.0f, rnd.x);
            if (star > 0.0f) {
                float3 starColor = mix(float3(1.0f, 1.0f, 1.0f),
                                       float3(0.5f, 0.8f, 1.0f), rnd.y);
                float twinkle = 0.7f + 0.3f * sin(t * 3.0f + rnd.z * 100.0f);
                accumC += starColor * star * 2.0f * twinkle * accumA;
            }
        }

        march += step * (1.0f + density * 2.0f);
        if (march > maxD) break;
    }

    float3 finalColor = bgColor + accumC * 0.6f;
    return float4(finalColor, 1.0f);
}
