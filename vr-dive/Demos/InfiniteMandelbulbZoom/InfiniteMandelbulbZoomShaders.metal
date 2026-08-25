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
  float4 cameraAndScale;
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
    float r2 = r * r;
    float r4 = r2 * r2;
    float r7 = r4 * r2 * r;
    float zr = r7 * r;
    dr = max(8.0f * r7 * dr + 1.0f, 1.0e-5f);
    theta *= 8.0f;
    phi *= 8.0f;
    z = zr * float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)) + p;
  }

  r = max(length(z), 1.0e-6f);
  IMZSample result;
  float distance = 0.5f * log(r) * r / dr;
  result.distance = isfinite(distance) ? max(distance, 1.0e-6f) : 0.05f;
  result.orbit = clamp(trap * 1.8f + float(escapedAt) / max(float(iterations), 1.0f), 0.0f, 1.0f);
  result.level = 0.0f;
  return result;
}

/// A Mandelbulb whose selected surface feature contains a smaller, rotated copy
/// of the entire construction. One parent and one child are sufficient because
/// the CPU rebases onto that child after every zoom cycle.
static IMZSample imzHierarchy(float3 p, uint iterations) {
  const float childScale = 0.215f;
  const float3 anchor = float3(0.70f, 0.44f, 0.50f);
  IMZSample best = imzBulb(p, iterations);
  float accumulatedScale = 1.0f;
  float3 q = p;

  for (uint level = 1; level < 2; ++level) {
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
  float3 gradient =
    k.xyy * imzMap(p + k.xyy * e, u).distance +
    k.yyx * imzMap(p + k.yyx * e, u).distance +
    k.yxy * imzMap(p + k.yxy * e, u).distance +
    k.xxx * imzMap(p + k.xxx * e, u).distance;
  float gradient2 = dot(gradient, gradient);
  // Interior DE samples can all clamp to the same epsilon. Avoid normalize(0),
  // which produces NaNs and commonly appears as a completely black surface.
  float position2 = dot(p, p);
  float3 fallback = position2 > 1.0e-12f
    ? p * rsqrt(position2) : float3(0.0f, 0.0f, 1.0f);
  return gradient2 > 1.0e-12f ? gradient * rsqrt(gradient2) : fallback;
}

static bool imzSphereHit(float3 ro, float3 rd, thread float &nearT, thread float &farT) {
  const float radius = 1.48f;
  float b = dot(ro, rd);
  float h = b * b - (dot(ro, ro) - radius * radius);
  if (h < 0.0f) return false;
  h = sqrt(h);
  nearT = -b - h;
  farT = -b + h;
  return farT >= max(nearT, 0.0f);
}

static float3 imzPalette(float orbit) {
  // Color only by the canonical orbit trap so a layer rebase is visually seamless.
  float3 phase = float3(0.04f, 0.31f, 0.63f) + orbit * 0.34f;
  return 0.48f + 0.48f * cos(6.2831853f * phase);
}

static float4 imzRender(
  float2 ndc,
  constant InfiniteMandelbulbZoomUniforms &u)
{
  // This pattern is a full-screen zoom rather than a world-anchored object. Use
  // a canonical screen-space camera instead of unprojecting a compositor matrix:
  // reverse-Z/infinite-far projections can produce w == 0 on device and turn the
  // entire image into near-black sphere misses even though the compute pass ran.
  float scale = max(u.cameraAndScale.w, 0.01f);
  float3 ro = u.cameraAndScale.xyz / scale;
  float3 rd = normalize(float3(ndc.x * 0.80f, ndc.y * 0.70f, -1.0f));

  float nearT, farT;
  if (!imzSphereHit(ro, rd, nearT, farT)) {
    float vignette = pow(max(0.0f, 1.0f - length(ndc) * 0.52f), 3.0f);
    return float4(float3(0.006f, 0.012f, 0.03f) + vignette * float3(0.025f, 0.035f, 0.07f), 1.0f);
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
    travel += max(sample.distance * 0.82f, epsilon * 0.6f);
    if (travel > farT) break;
  }

  float rayGlow = exp(-minDistance * 22.0f);
  float3 background = float3(0.006f, 0.012f, 0.03f)
    + float3(0.025f, 0.035f, 0.07f) * pow(max(0.0f, 1.0f - length(ndc) * 0.52f), 3.0f);
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

/// The fractal is intentionally evaluated into a modest offscreen texture.
/// Running this DE directly at the compositor's full stereo resolution can
/// enqueue frames much faster than the GPU completes them, leaving only the
/// black clear color visible and eventually risking a watchdog reset.
kernel void infiniteMandelbulbZoomCompute(
  constant InfiniteMandelbulbZoomUniforms &u [[buffer(0)]],
  texture2d_array<float, access::write> output [[texture(0)]],
  uint3 gid [[thread_position_in_grid]])
{
  if (gid.x >= output.get_width() || gid.y >= output.get_height()
      || gid.z >= min(u.viewCount, output.get_array_size())) return;
  float2 uv = (float2(gid.xy) + 0.5f) / float2(output.get_width(), output.get_height());
  float2 ndc = uv * 2.0f - 1.0f;
  output.write(imzRender(ndc, u), gid.xy, gid.z);
}

fragment float4 infiniteMandelbulbZoomFragment(
  InfiniteMandelbulbZoomVertexOut in [[stage_in]],
  texture2d_array<float, access::sample> raymarchTexture [[texture(0)]])
{
  constexpr sampler linearSampler(
    coord::normalized, address::clamp_to_edge, filter::linear);
  return raymarchTexture.sample(linearSampler, in.uv, in.viewIndex);
}
