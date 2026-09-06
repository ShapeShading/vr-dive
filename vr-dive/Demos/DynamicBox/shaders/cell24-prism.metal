// 24-cell prism — 24 vertices, 96 edges, 24 octahedral cells in R4.
// Unit vertices are permutations of (+/-1,+/-1,0,0)/sqrt(2).
// Adjacency is exact: squared edge length is 1 after normalization.
// Reference: https://mathworld.wolfram.com/24-Cell.html
// Self-contained runtime shader: the DynamicBox wrapper supplies Metal/types.
// Geometry is finite; the Box is a viewing portal, not a ray-exit clipping volume.
static float4 spRotate(float4 q, float3 c, float3 s) {
    q.xw = float2(c.x*q.x-s.x*q.w, s.x*q.x+c.x*q.w);
    q.yz = float2(c.y*q.y-s.y*q.z, s.y*q.y+c.y*q.z);
    q.zw = float2(c.z*q.z-s.z*q.w, s.z*q.z+c.z*q.w);
    return q;
}
static bool spRay(DynamicBoxVertexOut in, constant DynamicBoxUniforms &u,
                  constant float4x4 *v2w, thread float3 &ro, thread float3 &rd) {
    uint vi = min(in.viewIndex, max(u.viewCount, 1u)-1u);
    float3 cam = v2w[vi][3].xyz;
    ro = (cam-u.objectCenter.xyz)/max(u.boxScale, 0.0001f);
    rd = normalize(in.worldPos-cam);
    if (!all(abs(ro)<DB_BOXDIMS-0.001f)) {
        float3 nn;
        float entry = db_boxHit(ro, rd, DB_BOXDIMS, nn, true);
        if (entry < 0.0f) return false;
        // Keep the eye origin. Finite geometry bounds choose the trace start;
        // starting at the Box face would cut projected lobes in front of it.
    }
    ro = (u.patternTransform*float4(ro, 1)).xyz;
    rd = normalize((u.patternTransform*float4(rd, 0)).xyz);
    return true;
}
static bool spBound(float3 ro, float3 rd, float radius, thread float2 &interval) {
    float b = dot(ro, rd), h = b*b-dot(ro, ro)+radius*radius;
    if (h < 0.0f) return false;
    interval = float2(max(0.0f, -b-sqrt(h)), -b+sqrt(h));
    return interval.y > interval.x;
}
static float3 spBackground(float3 rd) {
    return float3(0.004f, 0.007f, 0.016f);
}
static float3 spPalette(float phase) {
    // Positive pearl / petrol / rose-gold palette in linear light.
    float a = 0.5f+0.5f*cos(phase);
    float b = 0.5f+0.5f*sin(phase+0.8f);
    return mix(mix(float3(0.035f,0.22f,0.32f), float3(0.12f,0.63f,0.59f), a),
               float3(0.88f,0.40f,0.22f), b*b);
}
static float3 spShade(float3 n, float3 rd, float3 base, float glow, float occ) {
    n = dot(n, rd)>0.0f ? -n : n;
    float3 key = normalize(float3(-0.5f,0.8f,0.9f));
    float3 rim = normalize(float3(0.8f,0.1f,-0.7f));
    float facing = saturate(dot(n,-rd));
    float spec = pow(saturate(dot(n,normalize(key-rd))), 72.0f);
    float fresnel = pow(1.0f-facing, 4.0f);
    float3 color = base*(0.22f+0.85f*saturate(dot(n,key)))*occ;
    color += float3(0.10f,0.25f,0.38f)*saturate(dot(n,rim))*occ;
    color += mix(float3(0.8f,0.9f,1.0f),base,0.4f)*spec*0.8f;
    color += float3(0.26f,0.48f,0.65f)*fresnel*0.4f + base*glow;
    return color/(1.0f+0.22f*color);
}

// Direct intersections: fixed primitive count, no per-edge raymarch/normal probes.
struct SPHit { float t; float3 normal; float id; float along; };
static void spSphere(float3 ro, float3 rd, float3 center, float radius,
                     float id, thread SPHit &hit) {
    float3 oc = ro-center;
    float b = dot(oc, rd), h = b*b-dot(oc, oc)+radius*radius;
    if (h < 0.0f) return;
    float t = -b-sqrt(h);
    if (t < 0.0f) t = -b+sqrt(h);
    if (t >= 0.0f && t < hit.t) {
        hit.t = t; hit.normal = normalize(ro+t*rd-center);
        hit.id = id; hit.along = 0.0f;
    }
}
static void spCapsule(float3 ro, float3 rd, float3 a, float3 b,
                      float radius, float id, thread SPHit &hit) {
    float3 axis = b-a, oa = ro-a;
    float axisLength = length(axis);
    if (axisLength < 0.00001f) { spSphere(ro,rd,a,radius,id,hit); return; }
    float3 boundOffset = ro-(a+b)*0.5f;
    float boundRadius = axisLength*0.5f+radius;
    float projected = dot(boundOffset,rd);
    if (dot(boundOffset,boundOffset)-projected*projected > boundRadius*boundRadius) return;
    axis /= axisLength;
    float od = dot(oa,axis), dd = dot(rd,axis);
    float3 m = oa-axis*od, n = rd-axis*dd;
    float A = dot(n,n), B = dot(m,n), C = dot(m,m)-radius*radius;
    float discriminant = B*B-A*C;
    if (A > 0.0000001f && discriminant >= 0.0f) {
        float root = sqrt(discriminant);
        for (int side=0; side<2; ++side) {
            float t = (-B+(side==0 ? -root : root))/A;
            float y = od+t*dd;
            if (t>=0.0f && t<hit.t && y>=0.0f && y<=axisLength) {
                hit.t=t; hit.normal=normalize(ro+t*rd-a-axis*y);
                hit.id=id; hit.along=y/axisLength;
            }
        }
    }
    // Test hemispheres, including the exit root when the eye is inside a tube.
    for (int end=0; end<2; ++end) {
        float3 center = end==0 ? a : b, oc=ro-center;
        float sb=dot(oc,rd), sh=sb*sb-dot(oc,oc)+radius*radius;
        if (sh<0.0f) continue;
        float root=sqrt(sh);
        for (int side=0; side<2; ++side) {
            float t=-sb+(side==0 ? -root : root);
            float y=od+t*dd;
            if (t>=0.0f && t<hit.t && (end==0 ? y<=0.0f : y>=axisLength)) {
                hit.t=t; hit.normal=normalize(ro+t*rd-center);
                hit.id=id; hit.along=float(end);
            }
        }
    }
}

static constant float4 CELL24_VERTICES[24] = {
    float4(-0.70710678f,-0.70710678f,0.0f,0.0f),
    float4(-0.70710678f,0.70710678f,0.0f,0.0f),
    float4(0.70710678f,-0.70710678f,0.0f,0.0f),
    float4(0.70710678f,0.70710678f,0.0f,0.0f),
    float4(-0.70710678f,0.0f,-0.70710678f,0.0f),
    float4(-0.70710678f,0.0f,0.70710678f,0.0f),
    float4(0.70710678f,0.0f,-0.70710678f,0.0f),
    float4(0.70710678f,0.0f,0.70710678f,0.0f),
    float4(-0.70710678f,0.0f,0.0f,-0.70710678f),
    float4(-0.70710678f,0.0f,0.0f,0.70710678f),
    float4(0.70710678f,0.0f,0.0f,-0.70710678f),
    float4(0.70710678f,0.0f,0.0f,0.70710678f),
    float4(0.0f,-0.70710678f,-0.70710678f,0.0f),
    float4(0.0f,-0.70710678f,0.70710678f,0.0f),
    float4(0.0f,0.70710678f,-0.70710678f,0.0f),
    float4(0.0f,0.70710678f,0.70710678f,0.0f),
    float4(0.0f,-0.70710678f,0.0f,-0.70710678f),
    float4(0.0f,-0.70710678f,0.0f,0.70710678f),
    float4(0.0f,0.70710678f,0.0f,-0.70710678f),
    float4(0.0f,0.70710678f,0.0f,0.70710678f),
    float4(0.0f,0.0f,-0.70710678f,-0.70710678f),
    float4(0.0f,0.0f,-0.70710678f,0.70710678f),
    float4(0.0f,0.0f,0.70710678f,-0.70710678f),
    float4(0.0f,0.0f,0.70710678f,0.70710678f)
};
static constant ushort2 CELL24_EDGES[96] = {
    ushort2(0,4), ushort2(0,5), ushort2(0,8), ushort2(0,9),
    ushort2(0,12), ushort2(0,13), ushort2(0,16), ushort2(0,17),
    ushort2(1,4), ushort2(1,5), ushort2(1,8), ushort2(1,9),
    ushort2(1,14), ushort2(1,15), ushort2(1,18), ushort2(1,19),
    ushort2(2,6), ushort2(2,7), ushort2(2,10), ushort2(2,11),
    ushort2(2,12), ushort2(2,13), ushort2(2,16), ushort2(2,17),
    ushort2(3,6), ushort2(3,7), ushort2(3,10), ushort2(3,11),
    ushort2(3,14), ushort2(3,15), ushort2(3,18), ushort2(3,19),
    ushort2(4,8), ushort2(4,9), ushort2(4,12), ushort2(4,14),
    ushort2(4,20), ushort2(4,21), ushort2(5,8), ushort2(5,9),
    ushort2(5,13), ushort2(5,15), ushort2(5,22), ushort2(5,23),
    ushort2(6,10), ushort2(6,11), ushort2(6,12), ushort2(6,14),
    ushort2(6,20), ushort2(6,21), ushort2(7,10), ushort2(7,11),
    ushort2(7,13), ushort2(7,15), ushort2(7,22), ushort2(7,23),
    ushort2(8,16), ushort2(8,18), ushort2(8,20), ushort2(8,22),
    ushort2(9,17), ushort2(9,19), ushort2(9,21), ushort2(9,23),
    ushort2(10,16), ushort2(10,18), ushort2(10,20), ushort2(10,22),
    ushort2(11,17), ushort2(11,19), ushort2(11,21), ushort2(11,23),
    ushort2(12,16), ushort2(12,17), ushort2(12,20), ushort2(12,21),
    ushort2(13,16), ushort2(13,17), ushort2(13,22), ushort2(13,23),
    ushort2(14,18), ushort2(14,19), ushort2(14,20), ushort2(14,21),
    ushort2(15,18), ushort2(15,19), ushort2(15,22), ushort2(15,23),
    ushort2(16,20), ushort2(16,22), ushort2(17,21), ushort2(17,23),
    ushort2(18,20), ushort2(18,22), ushort2(19,21), ushort2(19,23)
};

fragment float4 dynamicBoxFragment(
    DynamicBoxVertexOut in [[stage_in]],
    constant DynamicBoxUniforms &u [[buffer(0)]],
    constant float4x4 *v2w [[buffer(1)]],
    constant float4x4 *vp [[buffer(2)]]) {
    float3 ro, rd;
    if (!spRay(in,u,v2w,ro,rd)) return float4(0.004f,0.007f,0.016f,1);
    float3 bg=spBackground(rd);

    float2 interval;
    if (!spBound(ro,rd,1.3f,interval)) return float4(bg,1);
    float3 angles=float3(0.105f*u.time+0.21f,0.083f*u.time+0.57f,0.071f*u.time+0.31f);
    float3 c=cos(angles), s=sin(angles);
    float3 points[24];
    for (int i=0; i<24; ++i) {
        float4 q=spRotate(CELL24_VERTICES[i],c,s);
        points[i]=2.0f*q.xyz/(2.1f-q.w);
    }
    SPHit hit; hit.t=1.0e6f; hit.normal=float3(0,1,0); hit.id=0; hit.along=0;
    for (int i=0; i<96; ++i) {
        ushort2 edge=CELL24_EDGES[i];
        spCapsule(ro,rd,points[edge.x],points[edge.y],0.0125f,float(i),hit);
    }
    for (int i=0; i<24; ++i) spSphere(ro,rd,points[i],0.029f,96.0f+float(i),hit);
    // The warm seed is an artistic accent, not an extra polytope cell.
    spSphere(ro,rd,float3(0),0.09f,120.0f,hit);
    if (hit.t==1.0e6f) return float4(bg,1);
    float3 base=spPalette(hit.id*0.41f+1.1f);
    if (hit.id>=96.0f) base=float3(0.55f,0.77f,0.83f);
    if (hit.id==120.0f) base=float3(0.98f,0.48f,0.16f);
    float pulse=pow(0.5f+0.5f*cos(2.0f*DB_PI*(hit.along-0.14f*u.time)+hit.id*0.61f),32.0f);
    float glow=hit.id==120.0f ? 0.8f : 0.25f*pulse;
    return float4(spShade(hit.normal,rd,base,glow,0.94f),1);
}
