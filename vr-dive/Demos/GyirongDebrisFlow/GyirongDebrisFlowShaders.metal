#include <metal_stdlib>
using namespace metal;

struct GyirongDebrisFlowUniforms {
  uint viewCount;
  uint gridWidth;
  uint gridHeight;
  uint flags;

  float simulationTime;
  float simulationDelta;
  float navigationSpeedScale;
  uint particleCount;

  float4 terrainSizeAndDatum;
  float4 portSourceUV;
  float4 sceneOrigin;
  float4 floodScenario;
  float4 floodHydrology;
  float4 facilityUVAndElevation;
  float4 facilityFrame;
  float4x4 navigationInverse;
};

struct GyirongMeshVertex {
  float3 position;
  float3 normal;
  float4 color;
};

struct GyirongWaterVertex {
  float2 uv;
};

struct GyirongFlowParticle {
  float4 uvHeightLife;
  float4 velocitySeed;
};

struct GyirongMeshOut {
  float4 clipPosition [[position]];
  float3 normal;
  float4 color;
  float3 scenePosition;
  uint viewIndex [[flat]];
};

struct GyirongWaterOut {
  float4 clipPosition [[position]];
  float3 normal;
  float depth;
  float speed;
  float sediment;
  uint viewIndex [[flat]];
};

struct GyirongParticleOut {
  float4 clipPosition [[position]];
  float3 normal;
  float depth;
  float speed;
  float sediment;
  float frontness;
  float particleKind;
  uint viewIndex [[flat]];
};

struct GyirongSkyOut {
  float4 clipPosition [[position]];
  float3 direction;
};

vertex GyirongSkyOut gyirongSkyVertex(
  ushort amplificationID [[amplification_id]],
  const device float3 *vertices [[buffer(0)]],
  constant GyirongDebrisFlowUniforms &uniforms [[buffer(1)]],
  constant float4x4 *viewProjectionMatrices [[buffer(2)]],
  constant float4x4 *viewToWorldTransforms [[buffer(3)]],
  uint vertexID [[vertex_id]])
{
  uint viewIndex = min((uint)amplificationID, max(uniforms.viewCount, 1u) - 1u);
  float3 skyOffset = vertices[vertexID];
  float3 cameraPosition = viewToWorldTransforms[viewIndex][3].xyz;
  float3 worldPosition = cameraPosition + skyOffset;
  GyirongSkyOut out;
  out.clipPosition = viewProjectionMatrices[viewIndex] * float4(worldPosition, 1.0f);
  out.direction = skyOffset;
  return out;
}

fragment float4 gyirongSkyFragment(GyirongSkyOut in [[stage_in]]) {
  float3 direction = normalize(in.direction);
  float vertical = clamp(direction.y * 0.5f + 0.5f, 0.0f, 1.0f);
  float azimuth = atan2(direction.z, direction.x);
  float3 horizon = float3(0.49f, 0.50f, 0.51f);
  float3 zenith = float3(0.34f, 0.35f, 0.36f);
  float cloudBand = sin(azimuth * 3.0f + vertical * 5.0f) * 0.014f
    + sin(azimuth * 7.0f - vertical * 9.0f) * 0.008f
    + sin((direction.x + direction.z) * 17.0f) * 0.005f;
  float3 color = mix(horizon, zenith, smoothstep(0.47f, 0.98f, vertical));
  color += cloudBand * smoothstep(0.25f, 0.82f, vertical);
  return float4(color, 1.0f);
}

static uint2 gyirongClampCoord(int2 value, uint2 size) {
  return uint2(clamp(value, int2(0), int2(size) - 1));
}

static float4 gyirongSafeWaterState(float4 state) {
  if (!all(isfinite(state))) return float4(0.0f);
  state.x = clamp(state.x, 0.0f, 18.0f);
  float speed = length(state.yz);
  if (!isfinite(speed)) {
    state.yz = float2(0.0f);
  } else if (speed > 48.0f) {
    state.yz *= 48.0f / speed;
  }
  state.w = clamp(state.w, 0.0f, 1.0f);
  return state;
}

static float3 gyirongScenePosition(
  float2 uv,
  float elevation,
  float waterDepth,
  constant GyirongDebrisFlowUniforms &uniforms)
{
  float north = (uv.y - uniforms.portSourceUV.y) * uniforms.terrainSizeAndDatum.x;
  float east = (uv.x - uniforms.portSourceUV.x) * uniforms.terrainSizeAndDatum.y;
  return float3(
    north + uniforms.sceneOrigin.x,
    elevation - uniforms.terrainSizeAndDatum.z + uniforms.sceneOrigin.y + waterDepth,
    -east + uniforms.sceneOrigin.z);
}

static float3 gyirongNavigatePosition(
  float3 scenePosition,
  constant GyirongDebrisFlowUniforms &uniforms)
{
  return (uniforms.navigationInverse * float4(scenePosition, 1.0f)).xyz;
}

static float3 gyirongNavigateDirection(
  float3 direction,
  constant GyirongDebrisFlowUniforms &uniforms)
{
  return normalize(float3(uniforms.navigationInverse * float4(direction, 0.0f)));
}

vertex GyirongMeshOut gyirongMeshVertex(
  ushort amplificationID [[amplification_id]],
  const device GyirongMeshVertex *vertices [[buffer(0)]],
  constant GyirongDebrisFlowUniforms &uniforms [[buffer(1)]],
  constant float4x4 *viewProjectionMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  GyirongMeshVertex meshVertex = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, max(uniforms.viewCount, 1u) - 1u);
  float3 worldPosition = gyirongNavigatePosition(meshVertex.position, uniforms);

  GyirongMeshOut out;
  out.clipPosition = viewProjectionMatrices[viewIndex] * float4(worldPosition, 1.0f);
  out.normal = gyirongNavigateDirection(meshVertex.normal, uniforms);
  out.color = meshVertex.color;
  out.scenePosition = meshVertex.position;
  out.viewIndex = viewIndex;
  return out;
}

static float gyirongTerrainHash(float2 p) {
  return fract(sin(dot(p, float2(127.1f, 311.7f))) * 43758.5453f);
}

static float gyirongTerrainNoise(float2 p) {
  float2 cell = floor(p);
  float2 f = fract(p);
  f = f * f * (3.0f - 2.0f * f);
  float a = gyirongTerrainHash(cell);
  float b = gyirongTerrainHash(cell + float2(1.0f, 0.0f));
  float c = gyirongTerrainHash(cell + float2(0.0f, 1.0f));
  float d = gyirongTerrainHash(cell + 1.0f);
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

static float gyirongTerrainFBM(float2 p) {
  float value = 0.0f;
  float amplitude = 0.56f;
  for (uint octave = 0u; octave < 4u; ++octave) {
    value += amplitude * gyirongTerrainNoise(p);
    p = float2(1.73f * p.x - 1.11f * p.y, 1.11f * p.x + 1.73f * p.y);
    amplitude *= 0.48f;
  }
  return value;
}

fragment float4 gyirongMeshFragment(GyirongMeshOut in [[stage_in]]) {
  const float3 lightDirection = normalize(float3(-0.34f, 0.82f, 0.46f));
  float highAltitudeHaze = smoothstep(2600.0f, 5200.0f, in.scenePosition.y + 1848.0f);
  float3 shadingNormal = normalize(in.normal);
  float3 baseColor = in.color.rgb;

  // Terrain vertices are roughly 90 m apart, which made CPU vertex colors
  // look like large uniform triangles. Alpha zero is a terrain material tag;
  // evaluate several world-space frequencies per fragment for contiguous
  // forest, scrub and exposed-rock patches across tile boundaries.
  if (in.color.a < 0.5f) {
    float2 ground = in.scenePosition.xz;
    float broad = gyirongTerrainFBM(ground / 620.0f);
    float medium = gyirongTerrainFBM(ground / 165.0f + 17.3f);
    float fine = gyirongTerrainFBM(ground / 48.0f - 9.7f);
    float mottling = clamp(broad * 0.48f + medium * 0.34f + fine * 0.18f, 0.0f, 1.0f);
    float greenDominance = baseColor.g - 0.5f * (baseColor.r + baseColor.b);
    float vegetation = smoothstep(0.012f, 0.105f, greenDominance);
    float rockPatch = smoothstep(0.58f, 0.78f, medium) * (1.0f - vegetation * 0.55f);
    float3 forest = mix(
      float3(0.025f, 0.105f, 0.035f),
      float3(0.25f, 0.34f, 0.105f),
      mottling);
    baseColor = mix(baseColor * (0.78f + mottling * 0.42f), forest, vegetation * 0.68f);
    baseColor = mix(baseColor, float3(0.31f, 0.285f, 0.25f), rockPatch * 0.34f);
    float detailX = gyirongTerrainNoise((ground + float2(7.0f, 0.0f)) / 34.0f)
      - gyirongTerrainNoise((ground - float2(7.0f, 0.0f)) / 34.0f);
    float detailZ = gyirongTerrainNoise((ground + float2(0.0f, 7.0f)) / 34.0f)
      - gyirongTerrainNoise((ground - float2(0.0f, 7.0f)) / 34.0f);
    shadingNormal = normalize(shadingNormal + float3(-detailX, 0.14f, -detailZ) * 0.24f);
  } else if (in.color.a > 1.8f) {
    // Recessed window geometry receives a cool reflective variation and thin
    // mullion-like bands instead of reading as a featureless black rectangle.
    float reflection = 0.5f + 0.5f * sin(
      in.scenePosition.y * 0.74f + in.scenePosition.x * 0.09f - in.scenePosition.z * 0.07f);
    float horizontalFrame = 1.0f - smoothstep(
      0.035f, 0.10f, abs(fract(in.scenePosition.y / 1.4f) - 0.5f));
    baseColor = mix(
      float3(0.025f, 0.07f, 0.095f),
      float3(0.16f, 0.31f, 0.38f),
      reflection * 0.58f + horizontalFrame * 0.18f);
  } else if (in.color.a > 1.1f) {
    // Metre-scale stone/concrete variation and recessed horizontal joints make
    // large walls readable without relying on one flat triangle or color.
    float facadeNoise = gyirongTerrainFBM(
      float2(in.scenePosition.x - in.scenePosition.z, in.scenePosition.y) / 3.8f);
    float floorJoint = 1.0f - smoothstep(
      0.015f, 0.055f, abs(fract(in.scenePosition.y / 3.2f) - 0.5f));
    float blockJoint = 1.0f - smoothstep(
      0.018f, 0.07f,
      min(
        abs(fract((in.scenePosition.x + in.scenePosition.z) / 1.9f) - 0.5f),
        abs(fract(in.scenePosition.y / 0.95f) - 0.5f)));
    baseColor *= 0.88f + facadeNoise * 0.24f;
    baseColor *= 1.0f - floorJoint * 0.13f - blockJoint * 0.055f;
  }

  float diffuse = max(dot(shadingNormal, lightDirection), 0.0f);
  float lighting = 0.28f + 0.82f * diffuse;
  float3 color = baseColor * lighting;
  float terrainMaterial = in.color.a < 0.5f ? 1.0f : 0.0f;
  color = mix(
    color,
    float3(0.52f, 0.61f, 0.66f),
    highAltitudeHaze * 0.13f * terrainMaterial);
  return float4(color, 1.0f);
}

vertex GyirongWaterOut gyirongWaterVertex(
  ushort amplificationID [[amplification_id]],
  const device GyirongWaterVertex *vertices [[buffer(0)]],
  constant GyirongDebrisFlowUniforms &uniforms [[buffer(1)]],
  constant float4x4 *viewProjectionMatrices [[buffer(2)]],
  texture2d<float, access::read> terrainHeight [[texture(0)]],
  texture2d<half, access::read> waterState [[texture(1)]],
  uint vertexID [[vertex_id]])
{
  float2 uv = vertices[vertexID].uv;
  uint2 size = uint2(uniforms.gridWidth, uniforms.gridHeight);
  uint2 coordinate = uint2(round(uv * float2(size - 1u)));
  float elevation = terrainHeight.read(coordinate).x;
  float4 state = gyirongSafeWaterState(float4(waterState.read(coordinate)));

  uint2 left = gyirongClampCoord(int2(coordinate) + int2(-1, 0), size);
  uint2 right = gyirongClampCoord(int2(coordinate) + int2(1, 0), size);
  uint2 down = gyirongClampCoord(int2(coordinate) + int2(0, -1), size);
  uint2 up = gyirongClampCoord(int2(coordinate) + int2(0, 1), size);
  float leftSurface = terrainHeight.read(left).x
    + gyirongSafeWaterState(float4(waterState.read(left))).x;
  float rightSurface = terrainHeight.read(right).x
    + gyirongSafeWaterState(float4(waterState.read(right))).x;
  float downSurface = terrainHeight.read(down).x
    + gyirongSafeWaterState(float4(waterState.read(down))).x;
  float upSurface = terrainHeight.read(up).x
    + gyirongSafeWaterState(float4(waterState.read(up))).x;
  float eastCell = uniforms.terrainSizeAndDatum.y / max(float(uniforms.gridWidth - 1u), 1.0f);
  float northCell = uniforms.terrainSizeAndDatum.x / max(float(uniforms.gridHeight - 1u), 1.0f);
  float eastSlope = (rightSurface - leftSurface) / max(2.0f * eastCell, 0.001f);
  float northSlope = (upSurface - downSurface) / max(2.0f * northCell, 0.001f);
  float3 normal = normalize(float3(-northSlope, 1.0f, eastSlope));

  float visibleDepth = max(state.x, 0.025f);
  float3 scenePosition = gyirongScenePosition(uv, elevation, visibleDepth, uniforms);
  float3 worldPosition = gyirongNavigatePosition(scenePosition, uniforms);
  uint viewIndex = min((uint)amplificationID, max(uniforms.viewCount, 1u) - 1u);

  GyirongWaterOut out;
  out.clipPosition = viewProjectionMatrices[viewIndex] * float4(worldPosition, 1.0f);
  out.normal = gyirongNavigateDirection(normal, uniforms);
  out.depth = state.x;
  out.speed = length(state.yz);
  out.sediment = state.w;
  out.viewIndex = viewIndex;
  return out;
}

fragment float4 gyirongWaterFragment(GyirongWaterOut in [[stage_in]]) {
  if (!isfinite(in.depth) || !isfinite(in.speed) || !isfinite(in.sediment)
    || in.depth < 0.025f) discard_fragment();
  float sediment = clamp(in.sediment, 0.0f, 1.0f);
  float3 clearWater = float3(0.05f, 0.25f, 0.34f);
  float3 debrisWater = float3(0.31f, 0.20f, 0.105f);
  float3 color = mix(clearWater, debrisWater, 0.30f + sediment * 0.70f);
  float diffuse = 0.35f + 0.65f * max(dot(normalize(in.normal), normalize(float3(-0.3f, 0.9f, 0.2f))), 0.0f);
  float foam = smoothstep(9.0f, 28.0f, in.speed) * (0.35f + 0.65f * sediment);
  color = color * diffuse + foam * float3(0.60f, 0.58f, 0.50f);
  float alpha = min(0.84f, 0.32f + smoothstep(0.1f, 2.5f, in.depth) * 0.48f);
  return float4(color, alpha);
}

vertex GyirongParticleOut gyirongParticleVertex(
  ushort amplificationID [[amplification_id]],
  const device GyirongFlowParticle *particles [[buffer(0)]],
  constant GyirongDebrisFlowUniforms &uniforms [[buffer(1)]],
  constant float4x4 *viewProjectionMatrices [[buffer(2)]],
  texture2d<float, access::read> terrainHeight [[texture(0)]],
  texture2d<half, access::read> waterState [[texture(1)]],
  texture2d<float, access::read> flowGuide [[texture(2)]],
  uint vertexID [[vertex_id]])
{
  // Render four decorrelated tetrahedral micro-clasts per simulated carrier.
  // This gives 786,432 visible opaque fragments (4x density) with half as many
  // vertices per fragment as the former octahedron, retaining stereo volume,
  // hard faces and depth occlusion at a manageable cost on device.
  const float3 polyhedron[12] = {
    float3(1, 1, 1), float3(-1, -1, 1), float3(-1, 1, -1),
    float3(1, 1, 1), float3(1, -1, -1), float3(-1, -1, 1),
    float3(1, 1, 1), float3(-1, 1, -1), float3(1, -1, -1),
    float3(-1, -1, 1), float3(1, -1, -1), float3(-1, 1, -1)
  };
  const uint verticesPerReplica = 12u;
  const uint replicaCount = 4u;
  uint particleID = vertexID / (verticesPerReplica * replicaCount);
  uint replicaID = (vertexID / verticesPerReplica) % replicaCount;
  uint localVertexID = vertexID % verticesPerReplica;
  uint faceStart = (localVertexID / 3u) * 3u;
  GyirongFlowParticle particle = particles[particleID];
  float replicaSeed = fract(
    particle.velocitySeed.w * 13.713f + float(replicaID + 1u) * 0.23837f);
  float2 uv = particle.uvHeightLife.xy;
  uint2 size = uint2(uniforms.gridWidth, uniforms.gridHeight);
  uint2 coordinate = uint2(clamp(uv, 0.0f, 1.0f) * float2(size - 1u));
  float elevation = terrainHeight.read(coordinate).x;
  float4 water = float4(waterState.read(coordinate));
  float4 guide = flowGuide.read(coordinate);
  float speed = length(water.yz);
  bool avalancheParticle =
    uniforms.simulationTime < uniforms.floodScenario.x
    && particle.velocitySeed.z < -50.0f;
  float breachElapsed = uniforms.simulationTime - uniforms.floodScenario.x;
  float routedFront = clamp(breachElapsed / uniforms.floodScenario.y, 0.0f, 1.12f);
  float frontDelta = (guide.y - routedFront) / 0.042f;
  float frontness = avalancheParticle
    ? 0.0f
    : exp(-frontDelta * frontDelta) * smoothstep(0.05f, 0.62f, guide.x);
  float particleKind = fract(replicaSeed * 19.371f);
  float debrisRadius = mix(
    0.18f + particle.velocitySeed.w * 0.32f,
    0.65f + particle.velocitySeed.w * 0.70f,
    smoothstep(0.38f, 0.82f, particleKind));
  float radius = particle.uvHeightLife.w > 0.0f
    ? (avalancheParticle
      ? 0.45f + particle.velocitySeed.w * 1.15f
      : debrisRadius * (1.0f + frontness * 0.22f))
    : 0.0f;
  radius *= mix(0.42f, 0.70f, replicaSeed);
  float3 dimensions = radius * float3(
    0.56f + fract(replicaSeed * 7.13f) * 0.50f,
    0.48f + fract(replicaSeed * 11.71f) * 0.62f,
    0.60f + fract(replicaSeed * 5.37f) * 0.56f);

  float tumble = uniforms.simulationTime * (0.35f + particleKind * 1.6f)
    + replicaSeed * 31.0f;
  float2 rotation = float2(cos(tumble), sin(tumble));
  float3 localPosition = polyhedron[localVertexID] * dimensions * 0.68f;
  localPosition.xz = float2(
    rotation.x * localPosition.x - rotation.y * localPosition.z,
    rotation.y * localPosition.x + rotation.x * localPosition.z);
  float3 localA = polyhedron[faceStart] * dimensions * 0.68f;
  float3 localB = polyhedron[faceStart + 1u] * dimensions * 0.68f;
  float3 localC = polyhedron[faceStart + 2u] * dimensions * 0.68f;
  localA.xz = float2(
    rotation.x * localA.x - rotation.y * localA.z,
    rotation.y * localA.x + rotation.x * localA.z);
  localB.xz = float2(
    rotation.x * localB.x - rotation.y * localB.z,
    rotation.y * localB.x + rotation.x * localB.z);
  localC.xz = float2(
    rotation.x * localC.x - rotation.y * localC.z,
    rotation.y * localC.x + rotation.x * localC.z);
  float3 localNormal = normalize(cross(localB - localA, localC - localA));

  float2 flowDirection2D = length(water.yz) > 0.2f ? normalize(water.yz) : guide.zw;
  float3 forward = normalize(float3(flowDirection2D.y, 0.0f, -flowDirection2D.x));
  float3 up = float3(0.0f, 1.0f, 0.0f);
  float3 side = normalize(cross(up, forward));
  float replicaAngle = replicaSeed * 6.2831853f + float(replicaID) * 1.5707963f;
  float replicaSpread = mix(0.65f, 2.8f, fract(replicaSeed * 8.19f))
    * max(radius, 0.12f);
  float3 replicaOffset = float3(
    cos(replicaAngle) * replicaSpread,
    fract(replicaSeed * 17.41f) * max(radius, 0.12f) * 1.3f,
    sin(replicaAngle) * replicaSpread * 1.25f);
  float3 sceneOffset = side * (localPosition.x + replicaOffset.x)
    + up * (localPosition.y + replicaOffset.y)
    + forward * (localPosition.z + replicaOffset.z);
  float3 sceneNormal = normalize(
    side * localNormal.x + up * localNormal.y + forward * localNormal.z);
  float3 scenePosition = gyirongScenePosition(
    uv,
    elevation,
    water.x + max(particle.uvHeightLife.z, 0.0f) + dimensions.y * 0.58f,
    uniforms) + sceneOffset;
  float3 worldPosition = gyirongNavigatePosition(scenePosition, uniforms);
  uint viewIndex = min((uint)amplificationID, max(uniforms.viewCount, 1u) - 1u);

  GyirongParticleOut out;
  out.clipPosition = viewProjectionMatrices[viewIndex] * float4(worldPosition, 1.0f);
  out.normal = gyirongNavigateDirection(sceneNormal, uniforms);
  out.depth = avalancheParticle ? 1.0f : water.x;
  out.speed = avalancheParticle
    ? clamp(uniforms.simulationTime / 72.0f, 0.0f, 1.0f)
    : speed;
  out.sediment = avalancheParticle ? -1.0f : water.w;
  out.frontness = frontness;
  out.particleKind = particleKind;
  out.viewIndex = viewIndex;
  return out;
}

fragment float4 gyirongParticleFragment(GyirongParticleOut in [[stage_in]]) {
  if (in.depth < 0.02f) discard_fragment();
  float3 lightDirection = normalize(float3(-0.28f, 0.88f, 0.38f));
  float diffuse = max(dot(normalize(in.normal), lightDirection), 0.0f);
  float lighting = 0.24f + diffuse * 0.88f;
  if (in.sediment < 0.0f) {
    float3 snowAndRock = mix(
      float3(0.86f, 0.91f, 0.93f),
      float3(0.31f, 0.30f, 0.28f),
      clamp(in.particleKind * 0.72f + in.speed * 0.18f, 0.0f, 1.0f));
    return float4(snowAndRock * lighting, 1.0f);
  }
  float3 debrisColor = mix(
    float3(0.20f, 0.145f, 0.085f),
    float3(0.48f, 0.315f, 0.14f),
    in.particleKind);
  debrisColor = mix(
    debrisColor,
    float3(0.28f, 0.17f, 0.075f),
    clamp(in.sediment, 0.0f, 1.0f) * 0.62f);
  float foam = smoothstep(10.0f, 32.0f, in.speed);
  float frontWetness = in.frontness * smoothstep(0.48f, 0.95f, in.particleKind);
  float3 color = mix(
    debrisColor,
    float3(0.62f, 0.55f, 0.42f),
    max(foam * 0.22f, frontWetness * 0.36f));
  return float4(color * lighting, 1.0f);
}

static uint gyirongHash(uint value) {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  value ^= value >> 16u;
  return value;
}

static float gyirongRandom(uint value) {
  return float(gyirongHash(value)) * (1.0f / 4294967296.0f);
}

static float2 gyirongSampleFlowPath(
  device const float2 *flowPath,
  uint pathPointCount,
  float progress)
{
  uint count = max(pathPointCount, 2u);
  float pathIndex = clamp(progress, 0.0f, 1.0f) * float(count - 1u);
  uint lower = min(uint(floor(pathIndex)), count - 2u);
  return mix(flowPath[lower], flowPath[lower + 1u], fract(pathIndex));
}

kernel void gyirongResetWater(
  texture2d<half, access::write> waterA [[texture(0)]],
  texture2d<half, access::write> waterB [[texture(1)]],
  texture2d<float, access::read> flowGuide [[texture(2)]],
  constant GyirongDebrisFlowUniforms &uniforms [[buffer(0)]],
  uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= uniforms.gridWidth || gid.y >= uniforms.gridHeight) return;
  float4 guide = flowGuide.read(gid);
  float corridor = smoothstep(0.18f, 0.82f, guide.x);
  float baseDepth = corridor * (0.08f + 0.12f * (1.0f - guide.y));
  float2 baseVelocity = guide.zw * corridor * 1.5f;
  half4 state = half4(float4(baseDepth, baseVelocity, corridor * 0.08f));
  waterA.write(state, gid);
  waterB.write(state, gid);
}

kernel void gyirongResetParticles(
  device GyirongFlowParticle *particles [[buffer(0)]],
  constant GyirongDebrisFlowUniforms &uniforms [[buffer(1)]],
  uint gid [[thread_position_in_grid]])
{
  if (gid >= uniforms.particleCount) return;
  float seed = gyirongRandom(gid * 5u + 1u);
  float angle = gyirongRandom(gid * 5u + 2u) * 6.2831853f;
  float radius = sqrt(gyirongRandom(gid * 5u + 3u)) * 0.007f;
  float2 uv = uniforms.portSourceUV.zw + float2(cos(angle), sin(angle)) * radius;
  GyirongFlowParticle particle;
  particle.uvHeightLife = float4(uv, 0.0f, -gyirongRandom(gid * 5u + 4u) * 5.0f);
  particle.velocitySeed = float4(0.0f, 0.0f, 0.0f, seed);
  particles[gid] = particle;
}

kernel void gyirongStepWater(
  texture2d<half, access::read> previousWater [[texture(0)]],
  texture2d<half, access::write> nextWater [[texture(1)]],
  texture2d<float, access::read> terrainHeight [[texture(2)]],
  texture2d<float, access::read> flowGuide [[texture(3)]],
  constant GyirongDebrisFlowUniforms &uniforms [[buffer(0)]],
  uint2 gid [[thread_position_in_grid]])
{
  uint2 size = uint2(uniforms.gridWidth, uniforms.gridHeight);
  if (gid.x >= size.x || gid.y >= size.y) return;
  if (gid.x == 0u || gid.y == 0u || gid.x + 1u == size.x || gid.y + 1u == size.y) {
    nextWater.write(half4(0.0h), gid);
    return;
  }

  uint2 left = uint2(gid.x - 1u, gid.y);
  uint2 right = uint2(gid.x + 1u, gid.y);
  uint2 down = uint2(gid.x, gid.y - 1u);
  uint2 up = uint2(gid.x, gid.y + 1u);
  float4 centerState = gyirongSafeWaterState(float4(previousWater.read(gid)));
  float4 leftState = gyirongSafeWaterState(float4(previousWater.read(left)));
  float4 rightState = gyirongSafeWaterState(float4(previousWater.read(right)));
  float4 downState = gyirongSafeWaterState(float4(previousWater.read(down)));
  float4 upState = gyirongSafeWaterState(float4(previousWater.read(up)));

  float eastCell = uniforms.terrainSizeAndDatum.y / max(float(size.x - 1u), 1.0f);
  float northCell = uniforms.terrainSizeAndDatum.x / max(float(size.y - 1u), 1.0f);
  float leftSurface = terrainHeight.read(left).x + leftState.x;
  float rightSurface = terrainHeight.read(right).x + rightState.x;
  float downSurface = terrainHeight.read(down).x + downState.x;
  float upSurface = terrainHeight.read(up).x + upState.x;
  float2 surfaceGradient = float2(
    (rightSurface - leftSurface) / max(2.0f * eastCell, 0.001f),
    (upSurface - downSurface) / max(2.0f * northCell, 0.001f));

  float dt = uniforms.simulationDelta;
  float depth = max(centerState.x, 0.0f);
  float2 velocity = centerState.yz - 9.81f * surfaceGradient * dt;
  velocity *= 1.0f / (1.0f + (0.035f + centerState.w * 0.05f) * dt);
  float speed = length(velocity);
  if (speed > 48.0f) velocity *= 48.0f / speed;

  float eastFluxDerivative =
    (rightState.x * rightState.y - leftState.x * leftState.y)
    / max(2.0f * eastCell, 0.001f);
  float northFluxDerivative =
    (upState.x * upState.z - downState.x * downState.z)
    / max(2.0f * northCell, 0.001f);
  depth = max(depth - dt * (eastFluxDerivative + northFluxDerivative), 0.0f);

  float2 uv = float2(gid) / float2(size - 1u);
  float4 guide = flowGuide.read(gid);
  float2 physicalOffset = (uv - uniforms.portSourceUV.zw)
    * float2(uniforms.terrainSizeAndDatum.y, uniforms.terrainSizeAndDatum.x);
  float sourceDistance = length(physicalOffset);
  float breachStart = uniforms.floodScenario.x;
  float breachElapsed = uniforms.simulationTime - breachStart;
  float releaseDuration = max(
    2.0f * uniforms.floodHydrology.y / max(uniforms.floodHydrology.x, 1.0f),
    300.0f);
  float releasePhase = clamp(breachElapsed / releaseDuration, 0.0f, 1.0f);
  float pulse = sin(releasePhase * 3.14159265f);
  float peakSectionVelocity = uniforms.floodHydrology.x
    / max(
      uniforms.floodHydrology.z * uniforms.floodScenario.z * 0.68f,
      1.0f);
  if (breachElapsed < 0.0f && sourceDistance < 440.0f) {
    float accumulation = smoothstep(54.0f, breachStart - 12.0f, uniforms.simulationTime);
    float falloff = smoothstep(440.0f, 0.0f, sourceDistance);
    float targetDepth = uniforms.terrainSizeAndDatum.w * 0.92f * accumulation * falloff;
    depth = max(depth, targetDepth);
    velocity *= 1.0f - accumulation * falloff;
  } else if (breachElapsed <= releaseDuration && sourceDistance < 440.0f) {
    float falloff = smoothstep(440.0f, 0.0f, sourceDistance);
    float targetDepth = uniforms.floodScenario.z * (0.62f + pulse * 0.58f) * falloff;
    depth = max(depth, targetDepth);
    velocity = mix(
      velocity,
      guide.zw * peakSectionVelocity * (0.62f + pulse * 0.50f),
      falloff);
  }

  // The ~90 m physics grid cannot resolve the narrow Lhende/Bhotekoshi
  // channel continuously. A DEM-derived corridor preserves a visible routed
  // flood front while gravity, surface gradients, friction and continuity
  // still evolve the surrounding cells.
  float corridor = smoothstep(0.12f, 0.78f, guide.x);
  float routedTravelDuration = uniforms.floodScenario.y;
  float routedFront = clamp(
    breachElapsed / routedTravelDuration,
    0.0f,
    1.12f);
  float frontDelta = (guide.y - routedFront) / 0.030f;
  float routedSurge = exp(-frontDelta * frontDelta) * corridor;
  float routedWake = smoothstep(guide.y - 0.015f, guide.y + 0.22f, routedFront) * corridor;
  if (breachElapsed >= 0.0f
    && breachElapsed < routedTravelDuration * 1.55f
    && corridor > 0.001f)
  {
    float routedDepth =
      0.10f * corridor
      + uniforms.floodScenario.z * (0.96f * routedSurge + 0.18f * routedWake);
    depth = max(depth, routedDepth);
    float2 routedVelocity = guide.zw * (
      2.0f
      + peakSectionVelocity * (1.05f * routedSurge + 0.42f * routedWake));
    velocity = mix(velocity, routedVelocity, clamp(0.18f + routedSurge, 0.0f, 1.0f));
  }

  // OSM gate way 904894059 and the supplied satellite/aerial views place the
  // river on the viewer-right side of the Chinese approach. Resolve the final
  // coarse-grid surge in gate-local coordinates: a dominant right-side branch,
  // a smaller left-side branch and a central portal spill all strike the
  // building zone, then continue downstream toward China.
  float2 facilityMetric = (uv - uniforms.facilityUVAndElevation.xy)
    * float2(uniforms.terrainSizeAndDatum.y, uniforms.terrainSizeAndDatum.x);
  float2 facilityAcross = uniforms.facilityFrame.xy;
  float2 facilityForward = uniforms.facilityFrame.zw;
  float localAcross = dot(facilityMetric, facilityAcross);
  float localForward = dot(facilityMetric, facilityForward);
  float overflowArrival = smoothstep(0.965f, 0.995f, routedFront);
  float overflowPulse = exp(-pow((routedFront - 0.998f) / 0.034f, 2.0f));
  float overflowWake = smoothstep(0.982f, 1.012f, routedFront);
  float portRoute = smoothstep(0.955f, 0.992f, guide.y);
  float portLengthMask = smoothstep(-260.0f, -205.0f, localForward)
    * (1.0f - smoothstep(95.0f, 155.0f, localForward));
  float rightBranch = exp(-pow((localAcross + 55.0f) / 34.0f, 2.0f));
  float leftBranch = exp(-pow((localAcross - 55.0f) / 38.0f, 2.0f));
  float portalBranch = exp(-pow(localAcross / 25.0f, 2.0f));
  float branchMask = clamp(
    rightBranch + leftBranch * 0.56f + portalBranch * 0.34f,
    0.0f,
    1.25f);
  float overflowStrength = branchMask * portLengthMask * overflowArrival
    * max(portRoute, 0.72f);
  if (breachElapsed >= 0.0f && overflowStrength > 0.001f) {
    float targetDepth = uniforms.floodScenario.z
      * (0.96f * overflowPulse + 0.30f * overflowWake)
      * overflowStrength;
    depth = max(depth, targetDepth);
    float sideSign = rightBranch >= leftBranch ? -1.0f : 1.0f;
    float2 spillVector = -facilityForward + facilityAcross * sideSign * 0.10f;
    float2 spillDirection = length(spillVector) > 0.001f
      ? normalize(spillVector)
      : -facilityForward;
    float spillSpeed = 3.5f + peakSectionVelocity
      * (0.94f * overflowPulse + 0.38f * overflowWake);
    velocity = mix(
      velocity,
      spillDirection * spillSpeed,
      clamp(overflowStrength * (0.48f + overflowPulse * 0.52f), 0.0f, 0.94f));
  }

  if (!isfinite(depth) || !all(isfinite(velocity))) {
    depth = 0.0f;
    velocity = float2(0.0f);
  } else if (depth < 0.012f) {
    depth = 0.0f;
    velocity = float2(0.0f);
  }
  depth = min(depth, 18.0f);
  speed = length(velocity);
  float slopeEnergy = clamp(length(surfaceGradient) * 2.5f, 0.0f, 1.0f);
  float sediment = clamp(
    centerState.w + dt * (speed * 0.0022f + slopeEnergy * 0.035f - 0.008f),
    0.0f,
    1.0f);
  nextWater.write(half4(gyirongSafeWaterState(float4(depth, velocity, sediment))), gid);
}

kernel void gyirongUpdateParticles(
  device GyirongFlowParticle *particles [[buffer(0)]],
  texture2d<float, access::read> terrainHeight [[texture(0)]],
  texture2d<half, access::read> waterState [[texture(1)]],
  texture2d<float, access::read> flowGuide [[texture(2)]],
  constant GyirongDebrisFlowUniforms &uniforms [[buffer(1)]],
  device const float2 *flowPath [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  if (gid >= uniforms.particleCount) return;
  GyirongFlowParticle particle = particles[gid];
  float seed = particle.velocitySeed.w;
  float2 uv = particle.uvHeightLife.xy;
  float life = particle.uvHeightLife.w - uniforms.simulationDelta;

  float breachStart = uniforms.floodScenario.x;
  if (uniforms.simulationTime < breachStart) {
    float avalanchePhase = clamp(uniforms.simulationTime / 72.0f, 0.0f, 1.0f);
    float settlingPhase = smoothstep(72.0f, breachStart, uniforms.simulationTime);
    uint avalancheCycle = gid * 17u + uint(uniforms.simulationTime * 0.7f);
    float seedA = gyirongRandom(gid * 7u + 1u);
    float seedB = gyirongRandom(gid * 7u + 2u);
    float angle = gyirongRandom(avalancheCycle + 3u) * 6.2831853f;
    float pathProgress = avalanchePhase * (0.018f + seedA * 0.050f);
    float2 pathUV = gyirongSampleFlowPath(flowPath, uniforms.flags, pathProgress);
    float cloudRadius = (35.0f + seedB * 190.0f) * (1.0f - settlingPhase * 0.62f);
    float2 metricJitter = float2(cos(angle), sin(angle)) * cloudRadius;
    uv = pathUV + metricJitter
      / float2(uniforms.terrainSizeAndDatum.y, uniforms.terrainSizeAndDatum.x);
    float loft = (18.0f + seedA * 115.0f)
      * sin(avalanchePhase * 3.14159265f)
      * (1.0f - settlingPhase * 0.85f);
    particle.uvHeightLife = float4(uv, max(loft, 0.0f), 1.0f);
    particle.velocitySeed = float4(0.0f, 0.0f, -100.0f, seedB);
    particles[gid] = particle;
    return;
  }

  bool outside = any(uv <= 0.0f) || any(uv >= 1.0f);
  uint2 size = uint2(uniforms.gridWidth, uniforms.gridHeight);
  uint2 coordinate = uint2(clamp(uv, 0.0f, 1.0f) * float2(size - 1u));
  float4 water = outside
    ? float4(0.0f)
    : gyirongSafeWaterState(float4(waterState.read(coordinate)));
  if (life <= 0.0f || outside || water.x < 0.025f) {
    uint cycle = uint(uniforms.simulationTime * 3.0f) + gid * 13u;
    float angle = gyirongRandom(cycle + 1u) * 6.2831853f;
    float spawnRadiusSeed = sqrt(gyirongRandom(cycle + 2u));
    float activeProgress = clamp(
      (uniforms.simulationTime - breachStart) / uniforms.floodScenario.y,
      0.0f,
      1.0f);
    // Most particles form a dense, tall moving bore at the routed front. The
    // rest remain in the turbulent wake, making the river position readable
    // at mountain scale instead of looking like uniformly scattered dots.
    float distribution = gyirongRandom(cycle + 5u);
    float progressRandom = gyirongRandom(cycle + 7u);
    float relativeProgress = distribution < 0.72f
      ? mix(0.82f, 1.0f, pow(progressRandom, 2.2f))
      : pow(progressRandom, 0.52f);
    float particleProgress = activeProgress * relativeProgress;
    float2 pathUV = gyirongSampleFlowPath(flowPath, uniforms.flags, particleProgress);
    float portSpawn = smoothstep(0.955f, 0.995f, particleProgress);
    float branchSeed = gyirongRandom(cycle + 11u);
    float branchAcross = branchSeed < 0.62f
      ? -55.0f
      : (branchSeed < 0.88f ? 55.0f : 0.0f);
    float branchForward = mix(
      38.0f,
      -185.0f,
      gyirongRandom(cycle + 12u));
    float2 branchMetric = uniforms.facilityFrame.xy * branchAcross
      + uniforms.facilityFrame.zw * branchForward;
    float2 branchUV = uniforms.facilityUVAndElevation.xy
      + branchMetric
        / float2(uniforms.terrainSizeAndDatum.y, uniforms.terrainSizeAndDatum.x);
    pathUV = mix(pathUV, branchUV, portSpawn * 0.94f);
    float radiusMeters = spawnRadiusSeed * mix(42.0f, 28.0f, portSpawn);
    float2 metricJitter = float2(cos(angle), sin(angle)) * radiusMeters;
    uv = pathUV + metricJitter
      / float2(uniforms.terrainSizeAndDatum.y, uniforms.terrainSizeAndDatum.x);
    uint2 spawnCoordinate = uint2(clamp(uv, 0.0f, 1.0f) * float2(size - 1u));
    float2 spawnDirection = flowGuide.read(spawnCoordinate).zw;
    spawnDirection = normalize(mix(
      spawnDirection,
      -uniforms.facilityFrame.zw,
      portSpawn * 0.90f));
    particle.uvHeightLife = float4(
      uv,
      0.0f,
      18.0f + gyirongRandom(cycle + 3u) * 34.0f);
    float frontRatio = particleProgress / max(activeProgress, 0.001f);
    float boreLift = smoothstep(0.76f, 0.98f, frontRatio);
    particle.velocitySeed = float4(
      spawnDirection * (5.5f + gyirongRandom(cycle + 9u) * 5.0f),
      2.5f + gyirongRandom(cycle + 4u)
        * mix(6.5f, uniforms.floodScenario.w * 1.08f, boreLift),
      seed);
    particles[gid] = particle;
    return;
  }

  float2 velocity = particle.velocitySeed.xy;
  float verticalVelocity = particle.velocitySeed.z;
  float sprayHeight = particle.uvHeightLife.z;
  float remainingTime = min(uniforms.simulationDelta, 1.35f);
  float particleKind = fract(seed * 19.371f);
  float eastCell = uniforms.terrainSizeAndDatum.y / max(float(size.x - 1u), 1.0f);
  float northCell = uniforms.terrainSizeAndDatum.x / max(float(size.y - 1u), 1.0f);

  // Integrate rigid debris in short fixed substeps. Drag transfers momentum
  // from the shallow-water velocity, gravity accelerates material down the
  // DEM gradient, and terrain/water-surface contact applies restitution and
  // friction. This remains stable when one rendered frame advances more than
  // a second of the accelerated event timeline.
  for (uint substep = 0u; substep < 9u && remainingTime > 0.0001f; ++substep) {
    float dt = min(remainingTime, 0.15f);
    coordinate = uint2(clamp(uv, 0.0f, 1.0f) * float2(size - 1u));
    water = gyirongSafeWaterState(float4(waterState.read(coordinate)));
    uint2 left = gyirongClampCoord(int2(coordinate) + int2(-1, 0), size);
    uint2 right = gyirongClampCoord(int2(coordinate) + int2(1, 0), size);
    uint2 down = gyirongClampCoord(int2(coordinate) + int2(0, -1), size);
    uint2 up = gyirongClampCoord(int2(coordinate) + int2(0, 1), size);
    float2 terrainGradient = float2(
      (terrainHeight.read(right).x - terrainHeight.read(left).x) / max(2.0f * eastCell, 0.001f),
      (terrainHeight.read(up).x - terrainHeight.read(down).x) / max(2.0f * northCell, 0.001f));

    float2 flowVelocity = water.yz;
    float4 localGuide = flowGuide.read(coordinate);
    float2 corridorGradient = float2(
      flowGuide.read(right).x - flowGuide.read(left).x,
      flowGuide.read(up).x - flowGuide.read(down).x) * 0.5f;
    float dragRate = mix(5.2f, 1.35f, particleKind);
    float dragBlend = 1.0f - exp(-dragRate * dt);
    velocity = mix(velocity, flowVelocity, dragBlend);
    // The mapped river tangent is a hard large-scale constraint while local
    // water velocity and DEM gravity retain turbulence and individual clast
    // motion.  The corridor gradient gently returns material toward the
    // centreline at bends instead of allowing a ballistic shortcut through a
    // mountain spur.
    float routeSpeed = max(length(flowVelocity), 4.0f);
    float2 routedVelocity = localGuide.zw * routeSpeed;
    float overflowZone = smoothstep(0.955f, 0.992f, localGuide.y)
      * smoothstep(0.04f, 0.42f, localGuide.x);
    float routeBlend = (1.0f - exp(-2.8f * dt))
      * smoothstep(0.05f, 0.55f, localGuide.x)
      * (1.0f - overflowZone * 0.82f);
    velocity = mix(velocity, routedVelocity, routeBlend);
    float gradientLength = length(corridorGradient);
    if (gradientLength > 0.0001f) {
      velocity += corridorGradient / gradientLength
        * min(routeSpeed * 0.65f, 7.5f) * dt
        * (1.0f - overflowZone * 0.88f);
    }
    velocity -= terrainGradient * (9.81f * dt * mix(0.20f, 0.48f, particleKind));
    float jitter = gyirongRandom(
      gid * 37u + substep * 101u + uint(uniforms.simulationTime * 17.0f)) - 0.5f;
    velocity += float2(jitter, -jitter)
      * min(length(flowVelocity) * 0.045f, 1.25f) * dt;
    uv += velocity * dt
      / float2(uniforms.terrainSizeAndDatum.y, uniforms.terrainSizeAndDatum.x);

    // Lighter fragments are partially supported by turbulent water; dense
    // rocks fall almost ballistically until they strike the moving surface.
    float effectiveGravity = 9.81f * mix(0.38f, 1.0f, particleKind);
    verticalVelocity -= effectiveGravity * dt;
    sprayHeight += verticalVelocity * dt;
    if (sprayHeight < 0.0f) {
      sprayHeight = 0.0f;
      float restitution = mix(0.08f, 0.36f, particleKind);
      float impactVelocity = abs(verticalVelocity) * restitution;
      float turbulenceKick = length(flowVelocity) * mix(0.018f, 0.055f, seed);
      verticalVelocity = max(impactVelocity, turbulenceKick);
      velocity *= mix(0.72f, 0.88f, particleKind);
    }
    remainingTime -= dt;
  }
  particle.uvHeightLife = float4(uv, sprayHeight, life);
  particle.velocitySeed = float4(velocity, verticalVelocity, seed);
  particles[gid] = particle;
}
