// Hopf fibration — twelve genuine linked S3 great-circle fibers.
// For z=(cos(eta),sin(eta)*exp(i*phi)), a fiber is exp(i*t)*z.
// Stereographic projection maps each great circle to an exact R3 circle.
// Reference: https://math.ucr.edu/home/baez/octonions/node9.html
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

struct HopfCircle { float3 center; float3 normal; float radius; };
static float2 hopfMap(float3 p, thread const HopfCircle *rings) {
    float2 result=float2(1.0e6f,0);
    for (int i=0; i<12; ++i) {
        float3 d=p-rings[i].center;
        float h=dot(d,rings[i].normal);
        float radial=length(d-rings[i].normal*h);
        float distance=length(float2(radial-rings[i].radius,h))-0.022f;
        if (distance<result.x) result=float2(distance,float(i));
    }
    return result;
}

fragment float4 dynamicBoxFragment(
    DynamicBoxVertexOut in [[stage_in]],
    constant DynamicBoxUniforms &u [[buffer(0)]],
    constant float4x4 *v2w [[buffer(1)]],
    constant float4x4 *vp [[buffer(2)]]) {
    float3 ro, rd;
    if (!spRay(in,u,v2w,ro,rd)) return float4(0.004f,0.007f,0.016f,1);
    float3 bg=spBackground(rd);

    float3 angles=float3(0.18f*sin(u.time*0.13f),0.11f*u.time+0.32f,
                        0.12f*sin(u.time*0.09f+0.4f));
    float3 c=cos(angles), s=sin(angles);
    HopfCircle rings[12];
    float bound=0.0f;
    for (int i=0; i<12; ++i) {
        float phi=2.0f*DB_PI*float(i)/12.0f;
        float eta=0.55f+0.13f*sin(3.0f*phi+0.3f);
        float ce=cos(eta), se=sin(eta), cp=cos(phi), sp=sin(phi);
        float4 a=spRotate(float4(ce,0,se*cp,se*sp),c,s);
        float4 b=spRotate(float4(0,ce,-se*sp,se*cp),c,s);
        float denominator=1.0f-a.w*a.w-b.w*b.w;
        // Bounded rotations keep this denominator strictly positive.
        rings[i].center=0.44f*(a.w*a.xyz+b.w*b.xyz)/denominator;
        rings[i].normal=normalize(cross(a.xyz,b.xyz));
        rings[i].radius=0.44f*rsqrt(denominator);
        bound=max(bound,length(rings[i].center)+rings[i].radius+0.024f);
    }
    float2 interval;
    if (!spBound(ro,rd,bound,interval)) return float4(bg,1);
    float travel=interval.x;
    for (int step=0; step<88 && travel<interval.y; ++step) {
        float3 p=ro+rd*travel;
        float2 sample=hopfMap(p,rings);
        float epsilon=0.0008f+0.00012f*travel;
        if (abs(sample.x)<epsilon) {
            int id=int(sample.y);
            float3 offset=p-rings[id].center;
            float h=dot(offset,rings[id].normal);
            float3 radial=offset-rings[id].normal*h;
            float3 n=normalize(radial*(1.0f-rings[id].radius/max(length(radial),0.00001f))
                              +rings[id].normal*h);
            float3 axis=normalize(cross(rings[id].normal,
                                      abs(rings[id].normal.y)<0.9f ? float3(0,1,0):float3(1,0,0)));
            float angle=atan2(dot(radial,cross(rings[id].normal,axis)),dot(radial,axis));
            float3 base=spPalette(float(id)*0.62f+0.4f);
            float winding=pow(0.5f+0.5f*cos(36.0f*angle+float(id)),6.0f);
            base=mix(base,float3(0.86f,0.74f,0.48f),winding*0.45f);
            float pulse=pow(0.5f+0.5f*cos(angle*3.0f-u.time*0.42f+float(id)*0.7f),20.0f);
            float3 outward=dot(n,rd)>0.0f ? -n:n;
            float occ=clamp(1.0f-6.0f*(0.035f-hopfMap(p+outward*0.035f,rings).x),0.5f,1.0f);
            return float4(spShade(n,rd,base,pulse*0.18f,occ),1);
        }
        travel+=max(abs(sample.x)*0.92f,0.0004f);
    }
    return float4(bg,1);
}
