#include <metal_stdlib>
using namespace metal;

struct InfiniteMandelbulbZoomUniforms {
  float zoomPhase;
  float zoomDirection;
  uint viewCount;
  uint generation;
  uint maxRaySteps;
  uint fractalIterations;
  float surfaceEpsilon;
  float padding;
  float4 objectCenterAndScale;
};

struct InfiniteMandelbulbZoomViewUniform {
  float4x4 viewToWorld;
  float4x4 projectionInverse;
};

struct InfiniteMandelbulbZoomVertexOut {
  float4 position [[position]];
  float2 uv;
  uint viewIndex [[flat]];
};

struct IMZSample {
  float distance;
  float orbit;
  float level;
};

vertex InfiniteMandelbulbZoomVertexOut infiniteMandelbulbZoomVertex(
  ushort amplificationID [[amplification_id]],
  uint vertexID [[vertex_id]],
  constant InfiniteMandelbulbZoomUniforms &uniforms [[buffer(0)]])
{
  const float2 positions[3] = {
    float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)
  };
  InfiniteMandelbulbZoomVertexOut out;
  out.position = float4(positions[vertexID], 0.0f, 1.0f);
  out.uv = out.position.xy * 0.5f + 0.5f;
  out.viewIndex = min((uint)amplificationID, max(uniforms.viewCount, 1u) - 1u);
  return out;
}

static float3 imzRotateY(float3 p, float angle) {
  float c = cos(angle), s = sin(angle);
  return float3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

static float3 imzRotateX(float3 p, float angle) {
  float c = cos(angle), s = sin(angle);
  return float3(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}

static float3 imzInverseChildRotation(float3 p) {
  return imzRotateY(imzRotateX(p, 0.46f), -0.72f);
}

/// Standard power-eight Mandelbulb distance estimator. The orbit trap is reused
/// for coloring, so the ray marcher does not need a second full fractal pass.
static IMZSample imzBulb(float3 p, uint iterations) {
  float3 z = p;
  float dr = 1.0f;
  float r = 0.0f;
  float trap = 10.0f;
  uint escapedAt = iterations;

  for (uint i = 0; i < iterations; ++i) {
    r = length(z);
    trap = min(trap, abs(length(z.xy) - 0.42f) + abs(z.z) * 0.12f);
    if (r > 2.25f) {
      escapedAt = i;
      break;
    }
    r = max(r, 1.0e-6f);
    float theta = acos(clamp(z.z / r, -1.0f, 1.0f));
    float phi = atan2(z.y, z.x);
    float r7 = pow(r, 7.0f);
    float zr = r7 * r;
    dr = max(8.0f * r7 * dr + 1.0f, 1.0e-5f);
    theta *= 8.0f;
    phi *= 8.0f;
    z = zr * float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)) + p;
  }

  r = max(length(z), 1.0e-6f);
  IMZSample result;
  result.distance = max(0.5f * log(r) * r / dr, 1.0e-6f);
  result.orbit = clamp(trap * 1.8f + float(escapedAt) / max(float(iterations), 1.0f), 0.0f, 1.0f);
  result.level = 0.0f;
  return result;
}

/// A Mandelbulb whose selected surface feature contains a smaller, rotated copy
/// of the entire construction. Three local levels are enough because the CPU
/// rebases onto the child after every zoom cycle, making depth independent of time.
static IMZSample imzHierarchy(float3 p, uint iterations) {
  const float childScale = 0.215f;
  const float3 anchor = float3(0.70f, 0.44f, 0.50f);
  IMZSample best = imzBulb(p, iterations);
  float accumulatedScale = 1.0f;
  float3 q = p;

  for (uint level = 1; level < 3; ++level) {
    // A cheap bound skips most expensive bulb iterations away from the branch.
    float bound = length(q - anchor) - childScale * 1.28f;
    if (bound > best.distance / accumulatedScale + 0.08f) {
      break;
    }
    q = imzInverseChildRotation(q - anchor) / childScale;
    accumulatedScale *= childScale;
    IMZSample child = imzBulb(q, iterations);
    child.distance *= accumulatedScale;
    child.level = float(level);
    if (child.distance < best.distance) {
      best = child;
    }
  }
  return best;
}

static void imzZoomTransform(float3 p, float phase, thread float3 &sceneP, thread float &scale) {
  float eased = phase * phase * (3.0f - 2.0f * phase);
  scale = pow(0.215f, eased);
  float3 rotated = imzRotateX(imzRotateY(p, 0.72f * eased), -0.46f * eased);
  sceneP = mix(float3(0.0f), float3(0.70f, 0.44f, 0.50f), eased) + rotated * scale;
}

static IMZSample imzMap(float3 p, constant InfiniteMandelbulbZoomUniforms &u) {
  float3 sceneP;
  float zoomScale;
  imzZoomTransform(p, u.zoomPhase, sceneP, zoomScale);
  IMZSample sample = imzHierarchy(sceneP, u.fractalIterations);
  sample.distance /= zoomScale;
  return sample;
}

static float3 imzNormal(float3 p, constant InfiniteMandelbulbZoomUniforms &u) {
  float e = max(u.surfaceEpsilon * 1.8f, 0.0012f);
  const float2 k = float2(1.0f, -1.0f);
  return normalize(
    k.xyy * imzMap(p + k.xyy * e, u).distance +
    k.yyx * imzMap(p + k.yyx * e, u).distance +
    k.yxy * imzMap(p + k.yxy * e, u).distance +
    k.xxx * imzMap(p + k.xxx * e, u).distance);
}

static bool imzBoxHit(float3 ro, float3 rd, thread float &nearT, thread float &farT) {
  const float3 halfSize = float3(1.46f);
  float3 safeRD = float3(
    abs(rd.x) < 1.0e-6f ? copysign(1.0e-6f, rd.x) : rd.x,
    abs(rd.y) < 1.0e-6f ? copysign(1.0e-6f, rd.y) : rd.y,
    abs(rd.z) < 1.0e-6f ? copysign(1.0e-6f, rd.z) : rd.z);
  float3 a = (-halfSize - ro) / safeRD;
  float3 b = ( halfSize - ro) / safeRD;
  float3 lo = min(a, b), hi = max(a, b);
  nearT = max(max(lo.x, lo.y), lo.z);
  farT = min(min(hi.x, hi.y), hi.z);
  return farT >= max(nearT, 0.0f);
}

static float3 imzPalette(float orbit) {
  // Color only by the canonical orbit trap so a layer rebase is visually seamless.
  float3 phase = float3(0.04f, 0.31f, 0.63f) + orbit * 0.34f;
  return 0.48f + 0.48f * cos(6.2831853f * phase);
}

fragment float4 infiniteMandelbulbZoomFragment(
  InfiniteMandelbulbZoomVertexOut in [[stage_in]],
  constant InfiniteMandelbulbZoomUniforms &u [[buffer(0)]],
  constant InfiniteMandelbulbZoomViewUniform *views [[buffer(1)]])
{
  uint vi = min(in.viewIndex, max(u.viewCount, 1u) - 1u);
  InfiniteMandelbulbZoomViewUniform view = views[vi];
  float2 ndc = in.uv * 2.0f - 1.0f;
  float4 viewTarget = view.projectionInverse * float4(ndc, 1.0f, 1.0f);
  float safeW = abs(viewTarget.w) < 1.0e-6f ? 1.0e-6f : viewTarget.w;
  viewTarget /= safeW;
  float3 cameraWorld = view.viewToWorld[3].xyz;
  float3 targetWorld = (view.viewToWorld * float4(viewTarget.xyz, 1.0f)).xyz;

  float3 center = u.objectCenterAndScale.xyz;
  float worldScale = u.objectCenterAndScale.w;
  float3 ro = (cameraWorld - center) / worldScale;
  float3 rd = normalize(targetWorld - cameraWorld);

  float nearT, farT;
  if (!imzBoxHit(ro, rd, nearT, farT)) {
    return float4(0.001f, 0.003f, 0.01f, 1.0f);
  }

  float travel = max(nearT, 0.0f);
  bool hit = false;
  IMZSample sample;
  sample.distance = 1.0f;
  sample.orbit = 0.0f;
  sample.level = 0.0f;
  float minDistance = 10.0f;

  for (uint step = 0; step < u.maxRaySteps; ++step) {
    float3 p = ro + rd * travel;
    sample = imzMap(p, u);
    minDistance = min(minDistance, sample.distance);
    float epsilon = u.surfaceEpsilon * (1.0f + travel * 0.12f);
    if (sample.distance < epsilon) {
      hit = true;
      break;
    }
    travel += max(sample.distance * 0.72f, epsilon * 0.55f);
    if (travel > farT) break;
  }

  float rayGlow = exp(-minDistance * 22.0f);
  float3 background = float3(0.002f, 0.005f, 0.016f)
    + float3(0.018f, 0.028f, 0.055f) * pow(max(0.0f, 1.0f - length(ndc) * 0.58f), 3.0f);
  if (!hit) {
    return float4(background + rayGlow * float3(0.025f, 0.045f, 0.09f), 1.0f);
  }

  float3 p = ro + rd * travel;
  float3 n = imzNormal(p, u);
  float3 lightDir = normalize(float3(-0.48f, 0.76f, 0.42f));
  float diffuse = max(dot(n, lightDir), 0.0f);
  float rim = pow(1.0f - max(dot(n, -rd), 0.0f), 2.4f);
  float specular = pow(max(dot(reflect(-lightDir, n), -rd), 0.0f), 36.0f);
  float3 base = imzPalette(sample.orbit);
  float detailBands = 0.78f + 0.22f * cos(sample.orbit * 42.0f);
  float3 color = base * detailBands * (0.16f + 0.92f * diffuse);
  color += rim * mix(float3(0.06f, 0.28f, 0.55f), base, 0.35f);
  color += specular * float3(1.0f, 0.82f, 0.55f);
  color += rayGlow * base * 0.08f;
  color = color / (1.0f + color);
  color = pow(max(color, 0.0f), float3(0.82f));
  return float4(color, 1.0f);
}
