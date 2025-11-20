#include <metal_stdlib>
using namespace metal;

struct Uniforms {
  float4x4 viewMatrix[2];
  float4x4 projectionMatrix[2];
  float4 time;               // x = time
  float4 cameraPosition;     // xyz = pos
  float4 projectileData[10]; // xyz: pos, w: active/radius
  float4 projectileCount;    // x = count
};

struct VertexOut {
  float4 position [[position]];
  float2 uv;
  float3 rayDir;
  uint layer [[render_target_array_index]];
};

// ...existing code...
// --- SDF Functions ---

float sdSphere(float3 p, float s) { return length(p) - s; }

float sdBox(float3 p, float3 b) {
  float3 q = abs(p) - b;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdTorus(float3 p, float2 t) {
  float2 q = float2(length(p.xz) - t.x, p.y);
  return length(q) - t.y;
}

float smin(float a, float b, float k) {
  float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

// --- Scene ---

// Returns float2(distance, materialID)
float2 map(float3 p, constant Uniforms &uniforms) {
  float2 res = float2(1000.0, 0.0);

  // Ground/Seabed (ID 1)
  float ground = p.y + 2.0 + sin(p.x * 0.5) * 0.2 * sin(p.z * 0.5);
  if (ground < res.x)
    res = float2(ground, 1.0);

  // Corals (Instanced) (ID 2)
  float3 q = p;
  q.xz = fmod(q.xz + 5.0, 10.0) - 5.0; // Repeat every 10 units
  float coral = sdSphere(q - float3(0, -1.5, 0),
                         0.8 + sin(p.y * 10.0 + uniforms.time.x) * 0.1);
  // Smooth blend coral with ground, but keep ID if coral is dominant
  // For simplicity in this demo, just min
  if (coral < res.x)
    res = float2(coral, 2.0);

  // Floating Objects (Fish) (ID 3)
  float3 fishPos =
      float3(sin(uniforms.time.x) * 3.0, 0.0, cos(uniforms.time.x) * 3.0 + 5.0);
  float fish = sdSphere(p - fishPos, 0.5);
  fish += sin(p.x * 10.0 + uniforms.time.x * 5.0) * 0.02;
  if (fish < res.x)
    res = float2(fish, 3.0);

  // Ancient Ruins (Box) (ID 5)
  float3 boxPos = float3(-3.0, -1.0, 8.0);
  float box = sdBox(p - boxPos, float3(0.5, 0.5, 0.5));
  if (box < res.x)
    res = float2(box, 5.0);

  // Strange Ring (Torus) (ID 6)
  float3 torusPos = float3(3.0, 1.0, 8.0);
  // Rotate torus
  float3 tp = p - torusPos;
  float angle = uniforms.time.x;
  float c = cos(angle);
  float s = sin(angle);
  float3x3 rot = float3x3(float3(c, 0, s), float3(0, 1, 0), float3(-s, 0, c));
  tp = rot * tp;
  float torus = sdTorus(tp, float2(0.6, 0.2));
  if (torus < res.x)
    res = float2(torus, 6.0);

  // Projectiles (ID 4)
  int count = int(uniforms.projectileCount.x);
  for (int i = 0; i < count; i++) {
    if (uniforms.projectileData[i].w > 0.0) {
      float3 projPos = uniforms.projectileData[i].xyz;
      float proj = sdSphere(p - projPos, 0.1);
      if (proj < res.x)
        res = float2(proj, 4.0);
    }
  }

  return res;
}

float3 getNormal(float3 p, constant Uniforms &uniforms) {
  float2 e = float2(0.001, 0.0);
  return normalize(
      float3(map(p + e.xyy, uniforms).x - map(p - e.xyy, uniforms).x,
             map(p + e.yxy, uniforms).x - map(p - e.yxy, uniforms).x,
             map(p + e.yyx, uniforms).x - map(p - e.yyx, uniforms).x));
}

// --- Raymarching ---
// ...existing code...

// --- Raymarching ---

vertex VertexOut vertexShader(uint vertexID [[vertex_id]],
                              uint instanceID [[instance_id]],
                              constant Uniforms &uniforms [[buffer(0)]]) {
  VertexOut out;
  // Full screen triangle
  float2 positions[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
  out.position = float4(positions[vertexID], 0.0, 1.0);
  out.uv = positions[vertexID];
  out.layer = instanceID;

  // Calculate ray direction
  // We use the view/proj for the current eye (instanceID)
  // float4x4 view = uniforms.viewMatrix[instanceID];
  // float4x4 proj = uniforms.projectionMatrix[instanceID];

  // Simple ray dir calculation (same as before but using indexed matrices)
  // Note: This logic is still "look at" approximation.
  // Ideally we use inverse(proj * view) * clipPos.

  // Let's try to do it properly-ish:
  // Clip space pos: (out.uv.x, out.uv.y, 1.0, 1.0) (at far plane)
  // World space = Inverse(ViewProj) * Clip

  // Since we don't have inverse passed in, we stick to the basis vector
  // approach using the indexed view matrix.

  // We pass the index to fragment shader via 'layer' (implicit) or just
  // recompute in fragment? Fragment shader doesn't get instance_id
  // automatically unless passed. But we are rendering to a layer, so we are
  // fine? Wait, fragment shader runs for a specific pixel. We need to know
  // which eye we are rendering to calculate the ray. We can pass 'instanceID'
  // as a flat int.

  return out;
}

fragment float4 fragmentShader(VertexOut in [[stage_in]],
                               constant Uniforms &uniforms [[buffer(0)]]) {

  uint eyeIndex = in.layer;

  // Screen coords to NDC
  // ...

  float3 ro = uniforms.cameraPosition.xyz;

  // Use the matrix for the current eye
  float4x4 view = uniforms.viewMatrix[eyeIndex];
  float4x4 proj = uniforms.projectionMatrix[eyeIndex];

  float3 camRight = float3(view[0][0], view[1][0], view[2][0]);
  float3 camUp = float3(view[0][1], view[1][1], view[2][1]);
  float3 camFwd = float3(view[0][2], view[1][2], view[2][2]);

  float tanHalfFovY = 1.0 / proj[1][1];
  float tanHalfFovX = 1.0 / proj[0][0];

  float3 rd = normalize(-camFwd + (in.uv.x * tanHalfFovX * camRight) +
                        (in.uv.y * tanHalfFovY * camUp));
  // Note: Sign of camFwd might need flipping depending on LH/RH. Metal is
  // usually LH? View matrix usually transforms to camera space where camera is
  // at 0, looking down -Z (or +Z). If looking down -Z, then forward is -Z.

  // ...existing code...
  // Raymarching
  float t = 0.0;
  float id = 0.0;
  for (int i = 0; i < 64; i++) {
    float3 p = ro + rd * t;
    float2 res = map(p, uniforms);
    if (res.x < 0.01) {
      id = res.y;
      break;
    }
    t += res.x;
    if (t > 50.0)
      break;
  }

  // Color
  float3 col = float3(0.0, 0.1, 0.2); // Deep blue background
  if (t < 50.0) {
    float3 p = ro + rd * t;
    float3 n = getNormal(p, uniforms);
    float3 lightDir = normalize(float3(0.5, 1.0, -0.5));
    float diff = max(dot(n, lightDir), 0.0);

    // Material Colors
    if (id == 1.0)
      col = float3(0.76, 0.7, 0.5); // Sand
    else if (id == 2.0)
      col = float3(0.8, 0.4, 0.6); // Coral (Pink)
    else if (id == 3.0)
      col = float3(1.0, 0.5, 0.0); // Fish (Orange)
    else if (id == 4.0)
      col = float3(1.0, 0.1, 0.1); // Projectile (Red)
    else if (id == 5.0)
      col = float3(0.6, 0.4, 0.2); // Box (Wood)
    else if (id == 6.0)
      col = float3(0.1, 0.8, 0.2); // Torus (Green)
    else
      col = float3(0.5, 0.5, 0.5); // Default

    col *= diff;

    // Fog
    col = mix(col, float3(0.0, 0.1, 0.2), 1.0 - exp(-0.05 * t));
  }

  return float4(col, 1.0);
}
