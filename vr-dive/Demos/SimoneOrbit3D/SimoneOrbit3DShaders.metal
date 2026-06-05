#include <metal_stdlib>
using namespace metal;

struct SimoneOrbit3DUniforms {
    float  time;
    uint   viewCount;
    float  cubeScale;
    float  padding;
    float4 simoneParameters;
    float4 objectCenter;
};

struct MeshVertex {
    float3 position;
    float3 normal;
};

struct SimoneOrbit3DVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct SimoneOrbit3DFieldSample {
    float density;
    float glow;
    float minDistance;
    float3 closestOrbit;
};

struct SimoneOrbit3DRenderSample {
    float3 color;
    float opacity;
};

static constant float3 SO_BOX_HALF = float3(1.0f);
static constant float SO_SCENE_SCALE = 0.92f;
static constant int SO_VOLUME_STEPS = 10;
static constant int SO_ORBIT_STEPS = 10;
static constant int SO_WARMUP_STEPS = 10;
static constant int SO_SEED_COUNT = 1;

vertex SimoneOrbit3DVertexOut simoneOrbit3DVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant SimoneOrbit3DUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    SimoneOrbit3DVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 soRotate(float2 p, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return float2(c * p.x + s * p.y, -s * p.x + c * p.y);
}

static float2 soBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / max(abs(rd), 1.0e-4f) * sign(rd);
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float3 soMap(float3 p, float3 params, float time, float seedPhase) {
    float3 q = p;
    q.xy = soRotate(q.xy, 0.03f * time + 0.35f * seedPhase);
    q.yz = soRotate(q.yz, -0.02f * time + 0.22f * seedPhase);

    float cDrift = params.z + 0.04f * sin(0.11f * time + seedPhase);
    return float3(
        sin(q.x * q.x - q.y * q.y - q.z * q.z + params.x),
        cos(2.0f * q.x * q.y + params.y),
        sin(2.0f * q.x * q.z + cDrift));
}

static float soDistanceToSegment(float3 p, float3 a, float3 b) {
    float3 ab = b - a;
    float denom = max(dot(ab, ab), 1.0e-5f);
    float t = clamp(dot(p - a, ab) / denom, 0.0f, 1.0f);
    float3 closest = a + t * ab;
    return length(p - closest);
}

static SimoneOrbit3DFieldSample soField(float3 p, float time, float3 params) {
    const float3 seeds[SO_SEED_COUNT] = {
        float3(0.14f, -0.11f, 0.07f)
    };

    float density = 0.0f;
    float minDistance = 1000.0f;
    float3 closestOrbit = float3(0.0f);

    for (int seedIndex = 0; seedIndex < SO_SEED_COUNT; ++seedIndex) {
        float seedPhase = float(seedIndex) * 1.9f;
        float3 state = seeds[seedIndex] + 0.03f * float3(
            sin(params.x + seedPhase),
            cos(params.y - 0.5f * seedPhase),
            sin(params.z + 0.7f * seedPhase));

        for (int i = 0; i < SO_WARMUP_STEPS; ++i) {
            state = soMap(state, params, time, seedPhase);
        }

        float3 previousOrbit = state * 0.22f;

        for (int i = 0; i < SO_ORBIT_STEPS; ++i) {
            state = soMap(state, params, time + 0.05f * float(i), seedPhase);
            float3 orbit = state * 0.22f;
            float pointDistance = length(p - orbit);
            float segmentDistance = soDistanceToSegment(p, previousOrbit, orbit);
            float d = min(pointDistance, segmentDistance);
            float weight = mix(1.0f, 0.62f, float(i) / float(SO_ORBIT_STEPS));
            density += weight * exp(-26.0f * d * d);
            density += 0.24f * weight * exp(-64.0f * pointDistance * pointDistance);

            if (d < minDistance) {
                minDistance = d;
                closestOrbit = orbit;
            }

            previousOrbit = orbit;
        }
    }

    density = density / float(SO_SEED_COUNT * SO_ORBIT_STEPS);

    SimoneOrbit3DFieldSample sample;
    sample.density = density;
    sample.glow = exp(-18.0f * minDistance * minDistance);
    sample.minDistance = minDistance;
    sample.closestOrbit = closestOrbit;
    return sample;
}

static float3 soPalette(float3 params, float3 orbit, float glow) {
    float3 phase = 0.5f + 0.5f * sin(params + float3(0.0f, 1.7f, 4.1f));
    float3 cool = mix(float3(0.10f, 0.32f, 0.70f), float3(0.14f, 0.72f, 0.76f), phase.x);
    float3 warm = mix(float3(0.74f, 0.26f, 0.38f), float3(0.96f, 0.68f, 0.18f), phase.y);
    float orbitMix = clamp(0.5f + 0.5f * orbit.y, 0.0f, 1.0f);
    float3 base = mix(cool, warm, orbitMix);
    float3 rim = mix(float3(0.85f, 0.94f, 1.0f), float3(1.0f, 0.96f, 0.84f), phase.z);
    return mix(base, rim, clamp(glow, 0.0f, 1.0f));
}

static SimoneOrbit3DRenderSample soRender(float3 ro, float3 rd, float tStart, float tEnd, float time, float3 params) {
    float span = max(tEnd - tStart, 0.0f);
    float dt = span / float(SO_VOLUME_STEPS);
    float transmittance = 1.0f;
    float3 color = float3(0.0f);
    float accumulatedOpacity = 0.0f;

    for (int step = 0; step < SO_VOLUME_STEPS; ++step) {
        float t = tStart + (float(step) + 0.5f) * dt;
        float3 localPos = ro + rd * t;
        float radiusFade = 1.0f - smoothstep(0.24f, 0.78f, length(localPos));
        if (radiusFade <= 0.0f) {
            continue;
        }

        float3 scenePos = localPos * SO_SCENE_SCALE;
        SimoneOrbit3DFieldSample sample = soField(scenePos, time, params);
        float density = max(sample.density * radiusFade - 0.010f, 0.0f);
        if (density <= 0.0f) {
            continue;
        }

        float absorption = density * (0.10f + 0.34f * sample.glow);
        float alpha = 1.0f - exp(-absorption * dt * 5.8f);
        float3 emission = soPalette(params, sample.closestOrbit, sample.glow);
        emission *= density * (0.70f + 1.10f * sample.glow);
        emission *= 0.42f + 0.58f * (1.0f - clamp(abs(dot(rd, normalize(sample.closestOrbit + 1.0e-4f))), 0.0f, 1.0f));
        color += transmittance * alpha * emission;
        transmittance *= (1.0f - alpha);
        accumulatedOpacity = max(accumulatedOpacity, 1.0f - transmittance);
        if (transmittance < 0.02f) {
            break;
        }
    }

    SimoneOrbit3DRenderSample result;
    result.color = sqrt(max(color, 0.0f));
    result.opacity = accumulatedOpacity;
    return result;
}

fragment float4 simoneOrbit3DFragment(
    SimoneOrbit3DVertexOut in [[stage_in]],
    constant SimoneOrbit3DUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center = uniforms.objectCenter.xyz;
    float3 cameraWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float scale = max(uniforms.cubeScale, 1.0e-4f);
    float3 eye = (cameraWorld - center) / scale;
    float3 surfacePos = (in.worldPos - center) / scale;
    float3 viewDir = normalize(surfacePos - eye);

    bool insideBox = all(abs(eye) < SO_BOX_HALF - 1.0e-3f);
    float2 hit = soBoxIntersect(eye, viewDir, SO_BOX_HALF);
    if (!insideBox && hit.x > hit.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(hit.x, 0.0f);
    float tEnd = hit.y;
    if (tEnd <= tStart) {
        discard_fragment();
    }

    SimoneOrbit3DRenderSample result = soRender(
        eye,
        viewDir,
        tStart,
        tEnd,
        uniforms.time,
        uniforms.simoneParameters.xyz);
    if (result.opacity < 0.003f) {
        discard_fragment();
    }
    return float4(clamp(result.color, 0.0f, 1.0f), 1.0f);
}