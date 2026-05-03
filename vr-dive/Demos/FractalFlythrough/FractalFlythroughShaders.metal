// FractalFlythroughShaders.metal
//
// Source reference requested by user:
// https://www.shadertoy.com/view/4s3SRN
// This is an original Metal adaptation for vr-dive. It preserves the source
// shader's Catmull-Rom flythrough path and layered lattice ideas, but renders
// them as a view-independent 3D volume inside a 2 meter cube container.

#include <metal_stdlib>
using namespace metal;

#define FFT_FAR           40.0f
#define FFT_MAX_STEPS     96
#define FFT_HIT_EPS       0.001f
// Uniform scene scale. 4.0 → 2*4=8 scene units per side → 8/4=2 repetitions (3x larger than 12.0).
#define FFT_SCENE_SCALE   4.0f

struct FractalFlythroughUniforms {
  float time;
  uint viewCount;
  float cubeScale;
  float travelSpeed;
  float4 objectCenter;
};

struct MeshVertex {
  float3 position;
  float3 normal;
};

struct FractalFlythroughVertexOut {
  float4 clipPos [[position]];
  float3 worldPos;
  uint viewIndex [[flat]];
};

struct FractalSample {
  float distance;
  float material;
};

vertex FractalFlythroughVertexOut fractalFlythroughVertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant FractalFlythroughUniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex vtx = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
  // Non-uniform scale: XY from cubeScale, Z from objectCenter.w
  float3 scale3 = float3(uniforms.cubeScale, uniforms.cubeScale, uniforms.objectCenter.w);
  float3 worldPos = vtx.position * scale3 + uniforms.objectCenter.xyz;

  FractalFlythroughVertexOut out;
  out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

static constant float3 FFT_CP[16] = {
  float3(0.0f, 0.0f, 0.0f),
  float3(0.0f, 0.0f, 3.84f),
  float3(1.92f, 0.0f, 3.84f),
  float3(1.92f, 0.0f, 1.92f),
  float3(1.92f, 1.92f, 1.92f),
  float3(-1.92f, 1.92f, 1.92f),
  float3(-1.92f, 0.0f, 1.92f),
  float3(-1.92f, 0.0f, 0.0f),
  float3(0.0f, 0.0f, 0.0f),
  float3(0.0f, 0.0f, -3.84f),
  float3(0.0f, 3.84f, -3.84f),
  float3(-1.92f, 3.84f, -3.84f),
  float3(-1.92f, 0.0f, -3.84f),
  float3(-1.92f, 0.0f, 0.0f),
  float3(-1.92f, -1.92f, 0.0f),
  float3(0.0f, -1.92f, 0.0f),
};

static float fft_hash(float n) {
  return fract(cos(n) * 45758.5453f);
}

static float fft_smin(float a, float b, float s) {
  float h = clamp(0.5f + 0.5f * (b - a) / s, 0.0f, 1.0f);
  return mix(b, a, h) - s * h * (1.0f - h);
}

static float3 fft_catmull(float3 p0, float3 p1, float3 p2, float3 p3, float t) {
  return (((-p0 + p1 * 3.0f - p2 * 3.0f + p3) * t * t * t
    + (p0 * 2.0f - p1 * 5.0f + p2 * 4.0f - p3) * t * t
    + (-p0 + p2) * t
    + p1 * 2.0f) * 0.5f);
}

static float3 fft_camPath(float t) {
  const int count = 16;
  float wrapped = fract(t / float(count)) * float(count);
  int seg = int(floor(wrapped));
  float localT = wrapped - float(seg);

  int i0 = (seg + count - 1) % count;
  int i1 = seg % count;
  int i2 = (seg + 1) % count;
  int i3 = (seg + 2) % count;
  return fft_catmull(FFT_CP[i0], FFT_CP[i1], FFT_CP[i2], FFT_CP[i3], localT);
}

static FractalSample fft_map(float3 q) {
  float3 p = abs(fract(q / 4.0f) * 4.0f - 2.0f);
  float tube = min(max(p.x, p.y), min(max(p.y, p.z), max(p.x, p.z))) - 4.0f / 3.0f - 0.015f;

  p = abs(fract(q / 2.0f) * 2.0f - 1.0f);
  tube = max(tube, fft_smin(max(p.x, p.y), fft_smin(max(p.y, p.z), max(p.x, p.z), 0.05f), 0.05f) - 2.0f / 3.0f);

  float panel = fft_smin(max(p.x, p.y), fft_smin(max(p.y, p.z), max(p.x, p.z), 0.125f), 0.125f) - 0.5f;
  float strip = step(p.x, 0.75f) * step(p.y, 0.75f) * step(p.z, 0.75f);
  panel -= strip * 0.025f;

  p = abs(fract(q * 2.0f) * 0.5f - 0.25f);
  float pan2 = min(p.x, min(p.y, p.z)) - 0.05f;
  panel = max(abs(panel), abs(pan2)) - 0.0425f;

  p = abs(fract(q * 1.5f) / 1.5f - 1.0f / 3.0f);
  tube = max(tube, min(max(p.x, p.y), min(max(p.y, p.z), max(p.x, p.z))) - 2.0f / 9.0f + 0.025f);

  p = abs(fract(q * 3.0f) / 3.0f - 1.0f / 6.0f);
  tube = max(tube, min(max(p.x, p.y), min(max(p.y, p.z), max(p.x, p.z))) - 1.0f / 9.0f - 0.035f);

  FractalSample sample;
  sample.distance = min(panel, tube);
  sample.material = 1.0f + step(tube, panel) + step(panel, tube) * strip * 2.0f;
  return sample;
}

static float fft_trace(float3 ro, float3 rd, float tMax, thread FractalSample &hitSample) {
  float t = 0.01f;
  hitSample = fft_map(ro);
  for (int i = 0; i < FFT_MAX_STEPS; ++i) {
    float3 pos = ro + rd * t;
    hitSample = fft_map(pos);
    float h = hitSample.distance;
    if (abs(h) < FFT_HIT_EPS * (t * 0.25f + 1.0f) || t > tMax) {
      break;
    }
    t += h * 0.8f;
  }
  return t;
}

static float3 fft_normal(float3 p) {
  float2 e = float2(0.005f, 0.0f);
  return normalize(float3(
    fft_map(p + e.xyy).distance - fft_map(p - e.xyy).distance,
    fft_map(p + e.yxy).distance - fft_map(p - e.yxy).distance,
    fft_map(p + e.yyx).distance - fft_map(p - e.yyx).distance));
}

static float fft_ao(float3 pos, float3 nor) {
  float scale = 2.0f;
  float occ = 0.0f;
  for (int i = 0; i < 5; ++i) {
    float hr = 0.01f + float(i) * 0.5f / 4.0f;
    float dd = fft_map(pos + nor * hr).distance;
    occ += (hr - dd) * scale;
    scale *= 0.7f;
  }
  return clamp(1.0f - occ, 0.0f, 1.0f);
}

static bool fft_boxHit(
  float3 ro, float3 rd, float3 bmin, float3 bmax,
  thread float &tNear, thread float &tFar)
{
  float3 t0 = (bmin - ro) / rd;
  float3 t1 = (bmax - ro) / rd;
  float3 lo = min(t0, t1);
  float3 hi = max(t0, t1);
  tNear = max(max(lo.x, lo.y), lo.z);
  tFar = min(min(hi.x, hi.y), hi.z);
  return tFar >= max(tNear, 0.0f);
}

static float fft_edgeDistance(float3 p) {
  float3 a = abs(p);
  if (a.x > a.y && a.x > a.z) {
    return min(1.0f - a.y, 1.0f - a.z);
  }
  if (a.y > a.z) {
    return min(1.0f - a.x, 1.0f - a.z);
  }
  return min(1.0f - a.x, 1.0f - a.y);
}

static float3 fft_faceNormal(float3 p) {
  float3 a = abs(p);
  if (a.x > a.y && a.x > a.z) {
    return float3(sign(p.x), 0.0f, 0.0f);
  }
  if (a.y > a.z) {
    return float3(0.0f, sign(p.y), 0.0f);
  }
  return float3(0.0f, 0.0f, sign(p.z));
}

static float3 fft_woodColor(float3 p) {
  float rings = 0.5f + 0.5f * sin(p.x * 5.0f + sin(p.y * 2.0f) * 1.2f + p.z * 0.7f);
  return mix(float3(0.25f, 0.16f, 0.08f), float3(0.47f, 0.30f, 0.16f), rings);
}

static float3 fft_metalColor(float3 p) {
  float brushed = 0.5f + 0.5f * sin(p.z * 8.0f + p.x * 2.0f);
  return mix(float3(0.34f, 0.36f, 0.39f), float3(0.55f, 0.58f, 0.62f), brushed);
}

static float3 fft_goldColor(float3 p) {
  float glint = 0.5f + 0.5f * sin(p.x * 10.0f + p.y * 7.0f + p.z * 3.0f);
  return mix(float3(0.68f, 0.44f, 0.12f), float3(0.98f, 0.84f, 0.32f), glint);
}

fragment float4 fractalFlythroughFragment(
  FractalFlythroughVertexOut in [[stage_in]],
  constant FractalFlythroughUniforms &uniforms [[buffer(0)]],
  constant float4x4 *v2wMats [[buffer(1)]])
{
  uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
  float4x4 v2w = v2wMats[vi];
  float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

  float3 center = uniforms.objectCenter.xyz;
  // Non-uniform box scale: XY from cubeScale, Z from objectCenter.w
  float3 scale3 = float3(uniforms.cubeScale, uniforms.cubeScale, uniforms.objectCenter.w);
  float3 roLocal = (camWorld - center) / scale3;
  float3 rdWorld = normalize(in.worldPos - camWorld);
  float3 rdLocal = rdWorld / scale3;

  float tEntry, tExit;
  if (!fft_boxHit(roLocal, rdLocal, float3(-1.0f), float3(1.0f), tEntry, tExit)) {
    discard_fragment();
  }

  // Virtual camera — exact port of mainImage() from ShaderToy 4s3SRN:
  //   speed = iTime*0.35 + 8; ro = camPath(speed); lk = camPath(speed+.5);
  //   fwd = normalize(lk-ro); rgt = normalize(vec3(fwd.z,0,-fwd.x)); up = cross(fwd,rgt);
  float time  = uniforms.time * uniforms.travelSpeed;
  float speed = time * 0.35f + 8.0f;
  float3 ro = fft_camPath(speed);         // camera position
  float3 lk = fft_camPath(speed + 0.5f); // look-at
  float3 lp = lk + float3(0.0f, 0.25f, 0.0f); // light

  float3 fwd = normalize(lk - ro);
  float3 rgt = normalize(float3(fwd.z, 0.0f, -fwd.x));
  float3 up  = cross(fwd, rgt);
  // Camera basis: columns are right, up, forward
  float3x3 basis = float3x3(rgt, up, fwd);

  // Entry point in box-local space
  float3 entryLocal = roLocal + rdLocal * max(tEntry, 0.0f);

  // Map box entry to scene space.
  // Convention: front face center (local z=+1) anchors to the virtual camera (ro).
  // Rays entering the front face travel in direction fwd — into the scene.
  float3 roScene = ro + basis * ((entryLocal - float3(0.0f, 0.0f, 1.0f)) * FFT_SCENE_SCALE);
  // Flip z so that local -z (into the box) maps to scene +fwd.
  float3 rdScene = normalize(basis * float3(rdWorld.x, rdWorld.y, -rdWorld.z));

  FractalSample hitSample;
  float t = fft_trace(roScene, rdScene, FFT_FAR, hitSample);

  float3 col = float3(0.0f);

  if (t < FFT_FAR) {
    float3 pos = roScene + rdScene * t;
    float3 nor = fft_normal(pos);

    float3 li = lp - pos;
    float lDist = max(length(li), 0.001f);
    li /= lDist;
    float atten = 1.0f / (1.0f + lDist * 0.125f + lDist * lDist * 0.05f);

    float occ = fft_ao(pos, nor);
    // Diffuse and specular from reference:
    // dif = pow(clamp(dot(nor,li),0,1), 4)*2; spe = pow(max(dot(reflect(-li,nor),-rd),0),8)
    float dif  = pow(clamp(dot(nor, li), 0.0f, 1.0f), 4.0f) * 2.0f;
    float spe  = pow(max(dot(reflect(-li, nor), -rdScene), 0.0f), 8.0f);
    float spe2 = spe * spe;

    // Procedural material colors (replacing iChannel0 texture from reference)
    float3 baseColor;
    if (hitSample.material > 2.5f) {
      baseColor = fft_goldColor(pos);
      // Gold fire tint from reference shading block
      float3 fire = pow(float3(1.5f, 1.0f, 1.0f) * baseColor, float3(8.0f, 2.0f, 1.5f));
      baseColor = baseColor + min(mix(float3(1.0f, 0.9f, 0.375f), float3(0.75f, 0.375f, 0.3f), fire), 2.0f) * 0.5f;
    } else if (hitSample.material > 1.5f) {
      baseColor = fft_metalColor(pos);
      float grey = dot(baseColor, float3(0.299f, 0.587f, 0.114f));
      baseColor = float3(grey) * 0.7f + baseColor * 0.15f;
    } else {
      baseColor = fft_woodColor(pos);
    }

    // Lighting from reference:
    // col = col*(dif + .25 + vec3(.35,.45,.5)*spe) + vec3(.7,.9,1)*spe2
    col = baseColor * (dif + 0.25f + float3(0.35f, 0.45f, 0.5f) * spe)
        + float3(0.7f, 0.9f, 1.0f) * spe2;
    col *= occ * atten;

    // Fog from reference: col = mix(col, vec3(0), 1-exp(-t*t/FAR/FAR*20))
    float fogK = t * t * 20.0f / (FFT_FAR * FFT_FAR);
    col = mix(max(col, 0.0f), float3(0.0f), 1.0f - exp(-fogK));
  }

  // Gamma from reference: sqrt(max(col, 0))
  return float4(sqrt(max(col, 0.0f)), 1.0f);
}