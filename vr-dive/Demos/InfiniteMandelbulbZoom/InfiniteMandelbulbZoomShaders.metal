#include <metal_stdlib>
using namespace metal;

struct InfiniteMandelbulbZoomUniforms {
  float zoomPhase;
  uint viewCount;
  uint generation;
  uint maxRaySteps;

  uint fractalIterations;
  float surfaceEpsilon;
  float boxScale;
  float padding;

  float4 objectCenter;
  float4x4 patternTransform;
};

struct MeshVertex {
  float3 position;
  float3 normal;
};

struct InfiniteMandelbulbZoomVertexOut {
  float4 clipPosition [[position]];
  float3 worldPosition;
  uint viewIndex [[flat]];
};

struct IMZSample {
  float distance;
  float orbit;
  float level;
};

constant float3 IMZ_BOX_DIMS = float3(0.95f, 0.95f, 1.25f);
constant float IMZ_CONTENT_SCALE = 0.58f;

vertex InfiniteMandelbulbZoomVertexOut infiniteMandelbulbZoomVertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant InfiniteMandelbulbZoomUniforms &uniforms [[buffer(1)]],
  constant float4x4 *viewProjectionMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex meshVertex = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, max(uniforms.viewCount, 1u) - 1u);
  float3 worldPosition = meshVertex.position * uniforms.boxScale + uniforms.objectCenter.xyz;

  InfiniteMandelbulbZoomVertexOut out;
  out.clipPosition = viewProjectionMatrices[viewIndex] * float4(worldPosition, 1.0f);
  out.worldPosition = worldPosition;
  out.viewIndex = viewIndex;
  return out;
}

static float imzBoxHit(
  float3 ro,
  float3 rd,
  float3 halfExtents,
  thread float3 &normal,
  bool entering)
{
  rd += 0.0001f * (1.0f - abs(sign(rd)));
  float3 reciprocal = 1.0f / rd;
  float3 center = ro * reciprocal;
  float3 radius = halfExtents * abs(reciprocal);
  float3 nearPlane = -radius - center;
  float3 farPlane = radius - center;
  float nearT = max(nearPlane.x, max(nearPlane.y, nearPlane.z));
  float farT = min(farPlane.x, min(farPlane.y, farPlane.z));
  if (nearT > farT) return -1.0f;

  if (entering) {
    normal = -sign(rd) * step(nearPlane.zxy, nearPlane.xyz)
      * step(nearPlane.yzx, nearPlane.xyz);
    return nearT;
  }
  normal = sign(rd) * step(farPlane.xyz, farPlane.zxy)
    * step(farPlane.xyz, farPlane.yzx);
  return farT;
}

static float3 imzRotateY(float3 p, float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return float3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

static float3 imzRotateX(float3 p, float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return float3(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}

static float3 imzInverseChildRotation(float3 p) {
  return imzRotateY(imzRotateX(p, 0.46f), -0.72f);
}

/// Standard power-eight Mandelbulb distance estimator. The orbit trap drives
/// color without requiring another fractal evaluation after a hit.
static IMZSample imzBulb(float3 p, uint iterations) {
  float3 z = p;
  float derivative = 1.0f;
  float radius = 0.0f;
  float trap = 10.0f;
  uint escapedAt = iterations;

  for (uint i = 0; i < iterations; ++i) {
    radius = length(z);
    trap = min(trap, abs(length(z.xy) - 0.42f) + abs(z.z) * 0.12f);
    if (radius > 2.25f) {
      escapedAt = i;
      break;
    }

    radius = max(radius, 1.0e-6f);
    float theta = acos(clamp(z.z / radius, -1.0f, 1.0f));
    float phi = atan2(z.y, z.x);
    float radius2 = radius * radius;
    float radius4 = radius2 * radius2;
    float radius7 = radius4 * radius2 * radius;
    float radius8 = radius7 * radius;
    derivative = max(8.0f * radius7 * derivative + 1.0f, 1.0e-5f);
    theta *= 8.0f;
    phi *= 8.0f;
    z = radius8 * float3(
      sin(theta) * cos(phi),
      sin(theta) * sin(phi),
      cos(theta)) + p;
  }

  radius = max(length(z), 1.0e-6f);
  float distance = 0.5f * log(radius) * radius / derivative;

  IMZSample result;
  result.distance = isfinite(distance) ? max(distance, 1.0e-6f) : 0.05f;
  result.orbit = clamp(
    trap * 1.8f + float(escapedAt) / max(float(iterations), 1.0f),
    0.0f,
    1.0f);
  result.level = 0.0f;
  return result;
}

/// The selected feature of the parent contains a rotated, smaller copy of the
/// same bulb. At phase one this child maps exactly back to canonical coordinates.
static IMZSample imzHierarchy(float3 p, uint iterations) {
  const float childScale = 0.215f;
  const float3 childAnchor = float3(0.70f, 0.44f, 0.50f);
  IMZSample best = imzBulb(p, iterations);

  float childBound = length(p - childAnchor) - childScale * 1.28f;
  if (childBound <= best.distance + 0.08f) {
    float3 childPoint = imzInverseChildRotation(p - childAnchor) / childScale;
    IMZSample child = imzBulb(childPoint, iterations);
    child.distance *= childScale;
    child.level = 1.0f;
    if (child.distance < best.distance) {
      best = child;
    }
  }
  return best;
}

static void imzZoomTransform(
  float3 p,
  float phase,
  thread float3 &scenePoint,
  thread float &zoomScale)
{
  float eased = phase * phase * (3.0f - 2.0f * phase);
  zoomScale = pow(0.215f, eased);
  float3 rotated = imzRotateX(imzRotateY(p, 0.72f * eased), -0.46f * eased);
  scenePoint = mix(float3(0.0f), float3(0.70f, 0.44f, 0.50f), eased)
    + rotated * zoomScale;
}

static IMZSample imzMap(
  float3 boxPoint,
  constant InfiniteMandelbulbZoomUniforms &uniforms)
{
  float3 canonicalPoint = boxPoint / IMZ_CONTENT_SCALE;
  float3 scenePoint;
  float zoomScale;
  imzZoomTransform(canonicalPoint, uniforms.zoomPhase, scenePoint, zoomScale);
  IMZSample sample = imzHierarchy(scenePoint, uniforms.fractalIterations);
  sample.distance *= IMZ_CONTENT_SCALE / zoomScale;
  return sample;
}

static float3 imzNormal(
  float3 p,
  constant InfiniteMandelbulbZoomUniforms &uniforms)
{
  float epsilon = max(uniforms.surfaceEpsilon * 1.7f, 0.0012f);
  const float2 k = float2(1.0f, -1.0f);
  float3 gradient =
    k.xyy * imzMap(p + k.xyy * epsilon, uniforms).distance
    + k.yyx * imzMap(p + k.yyx * epsilon, uniforms).distance
    + k.yxy * imzMap(p + k.yxy * epsilon, uniforms).distance
    + k.xxx * imzMap(p + k.xxx * epsilon, uniforms).distance;

  float gradientSquared = dot(gradient, gradient);
  float positionSquared = dot(p, p);
  float3 fallback = positionSquared > 1.0e-12f
    ? p * rsqrt(positionSquared)
    : float3(0.0f, 0.0f, 1.0f);
  return gradientSquared > 1.0e-12f
    ? gradient * rsqrt(gradientSquared)
    : fallback;
}

static bool imzSphereInterval(
  float3 ro,
  float3 rd,
  float radius,
  thread float &nearT,
  thread float &farT)
{
  float projection = dot(ro, rd);
  float discriminant = projection * projection - (dot(ro, ro) - radius * radius);
  if (discriminant < 0.0f) return false;
  float root = sqrt(discriminant);
  nearT = -projection - root;
  farT = -projection + root;
  return farT >= max(nearT, 0.0f);
}

static float3 imzPalette(float orbit, float level) {
  float3 phase = float3(0.04f, 0.31f, 0.63f) + orbit * 0.34f + level * 0.07f;
  return 0.48f + 0.48f * cos(6.2831853f * phase);
}

static float imzBoxEdge(float3 localSurfacePoint) {
  float3 normalized = abs(localSurfacePoint) / IMZ_BOX_DIMS;
  float largest = max(normalized.x, max(normalized.y, normalized.z));
  float smallest = min(normalized.x, min(normalized.y, normalized.z));
  float secondLargest = normalized.x + normalized.y + normalized.z - largest - smallest;
  return smoothstep(0.88f, 0.985f, secondLargest);
}

fragment float4 infiniteMandelbulbZoomFragment(
  InfiniteMandelbulbZoomVertexOut in [[stage_in]],
  constant InfiniteMandelbulbZoomUniforms &uniforms [[buffer(0)]],
  constant float4x4 *viewToWorldTransforms [[buffer(1)]])
{
  uint viewIndex = min(in.viewIndex, max(uniforms.viewCount, 1u) - 1u);
  float4x4 viewToWorld = viewToWorldTransforms[viewIndex];
  float3 cameraWorld = viewToWorld[3].xyz;

  // This is the same world-to-box ray construction used by Dynamic Box.
  float3 boxEye = (cameraWorld - uniforms.objectCenter.xyz) / uniforms.boxScale;
  float3 boxDirection = normalize(in.worldPosition - cameraWorld);
  bool cameraInsideBox = all(abs(boxEye) < (IMZ_BOX_DIMS - 1.0e-3f));

  float3 realOrigin;
  if (cameraInsideBox) {
    realOrigin = boxEye;
  } else {
    float3 entryNormal;
    float entryT = imzBoxHit(boxEye, boxDirection, IMZ_BOX_DIMS, entryNormal, true);
    if (entryT < 0.0f) discard_fragment();
    realOrigin = boxEye + boxDirection * (entryT + 1.0e-3f);
  }

  float3 exitNormal;
  float boxExitT = imzBoxHit(realOrigin, boxDirection, IMZ_BOX_DIMS, exitNormal, false);
  if (boxExitT <= 0.0f) discard_fragment();

  // Square-button pattern navigation changes only the virtual contents. The
  // rasterized cube remains fixed in world space and keeps normal stereo depth.
  float3 rayOrigin = (uniforms.patternTransform * float4(realOrigin, 1.0f)).xyz;
  float3 rayDirection = normalize(float3(
    uniforms.patternTransform * float4(boxDirection, 0.0f)));

  float sphereNear;
  float sphereFar;
  bool intersectsFractalBounds = imzSphereInterval(
    rayOrigin,
    rayDirection,
    1.48f * IMZ_CONTENT_SCALE,
    sphereNear,
    sphereFar);

  float3 background = float3(0.003f, 0.006f, 0.016f);
  float minDistance = 10.0f;
  bool hit = false;
  float travel = 0.0f;
  float maxTravel = 0.0f;
  IMZSample sample;
  sample.distance = 1.0f;
  sample.orbit = 0.0f;
  sample.level = 0.0f;

  if (intersectsFractalBounds) {
    travel = max(sphereNear, 0.0f);
    maxTravel = min(sphereFar, boxExitT);
    for (uint step = 0; step < uniforms.maxRaySteps && travel <= maxTravel; ++step) {
      float3 p = rayOrigin + rayDirection * travel;
      sample = imzMap(p, uniforms);
      minDistance = min(minDistance, sample.distance);
      float epsilon = uniforms.surfaceEpsilon * (1.0f + travel * 0.10f);
      if (sample.distance < epsilon) {
        hit = true;
        break;
      }
      travel += max(sample.distance * 0.78f, epsilon * 0.55f);
    }
  }

  float3 color = background;
  if (hit) {
    float3 p = rayOrigin + rayDirection * travel;
    float3 normal = imzNormal(p, uniforms);
    float3 lightDirection = normalize(float3(-0.48f, 0.76f, 0.42f));
    float diffuse = max(dot(normal, lightDirection), 0.0f);
    float rim = pow(1.0f - max(dot(normal, -rayDirection), 0.0f), 2.4f);
    float specular = pow(
      max(dot(reflect(-lightDirection, normal), -rayDirection), 0.0f),
      34.0f);
    float3 base = imzPalette(sample.orbit, sample.level);
    float bands = 0.76f + 0.24f * cos(sample.orbit * 42.0f);
    color = base * bands * (0.14f + 0.96f * diffuse);
    color += rim * mix(float3(0.06f, 0.30f, 0.62f), base, 0.35f);
    color += specular * float3(1.0f, 0.84f, 0.58f);
    color = color / (1.0f + color);
    color = pow(max(color, 0.0f), float3(0.82f));
  } else if (minDistance < 10.0f) {
    float glow = exp(-minDistance * 30.0f);
    color += glow * float3(0.025f, 0.085f, 0.22f);
  }

  // A restrained luminous frame makes the physical cube boundary readable
  // without replacing the fractal with an opaque box surface.
  float3 localSurfacePoint = (in.worldPosition - uniforms.objectCenter.xyz)
    / uniforms.boxScale;
  float edge = imzBoxEdge(localSurfacePoint);
  float pulse = 0.82f + 0.18f * sin(uniforms.zoomPhase * 6.2831853f);
  color += edge * pulse * float3(0.055f, 0.20f, 0.38f);

  return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}
