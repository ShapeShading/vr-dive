// Clifford lantern — thickened, perforated Clifford torus in S3, projected to R3.
// In R4 the underlying surface is |q.xy|=|q.zw|=1/sqrt(2).
// Evaluate through INVERSE stereographic projection, not an arbitrary W=0 slice.
// The filigree cut-outs and pearl/copper material are artistic additions.
// Reference: https://www.math.brown.edu/tbanchof/Beyond3d.new/chapter6/s6_8.html
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

static float4 cliffordPoint(float3 p, float4x4 inverseRotation) {
    float3 x=p/0.40f;
    float r2=dot(x,x);
    return inverseRotation*float4(2.0f*x,r2-1.0f)/(1.0f+r2);
}
static float cliffordMap(float3 p, float4x4 inverseRotation) {
    float4 q=cliffordPoint(p,inverseRotation);
    float a=length(q.xy), b=length(q.zw);
    float u=atan2(q.y,q.x), v=atan2(q.w,q.z);
    float shell=abs(a-b)*0.70710678f-0.020f;
    // Angular bands weighted by their circle radius: bounded derivatives even
    // at an angular coordinate's axis; integer frequencies close without seams.
    float ribsU=(abs(sin(10.0f*u))-0.30f)*a/11.0f;
    float ribsV=(abs(sin(7.0f*v))-0.26f)*b/7.7f;
    // Smooth intersection rounds the filigree's cross-section for narrow
    // highlights instead of flat, rectangular bars. It keeps a bounded slope.
    float ribs=min(ribsU,ribsV);
    float blend=saturate(0.5f+0.5f*(shell-ribs)/0.014f);
    float field=mix(ribs,shell,blend)+0.014f*blend*(1.0f-blend);
    float r2=dot(p,p)/(0.40f*0.40f);
    // Conservative chord-to-world step, not simply multiplying by (1+r^2)/2
    // far from the surface. This avoids overstepping under projection stretch.
    return 0.40f*field*(1.0f+r2)/(2.0f+abs(field)*sqrt(1.0f+r2));
}
static float3 cliffordNormal(float3 p, float4x4 inverseRotation) {
    const float e=0.0008f;
    const float3 a=float3(1,-1,-1), b=float3(-1,-1,1);
    const float3 c=float3(-1,1,-1), d=float3(1,1,1);
    return normalize(a*cliffordMap(p+a*e,inverseRotation)+b*cliffordMap(p+b*e,inverseRotation)
                    +c*cliffordMap(p+c*e,inverseRotation)+d*cliffordMap(p+d*e,inverseRotation));
}

fragment float4 dynamicBoxFragment(
    DynamicBoxVertexOut in [[stage_in]],
    constant DynamicBoxUniforms &u [[buffer(0)]],
    constant float4x4 *v2w [[buffer(1)]],
    constant float4x4 *vp [[buffer(2)]]) {
    float3 ro, rd;
    if (!spRay(in,u,v2w,ro,rd)) return float4(0.004f,0.007f,0.016f,1);
    float3 bg=spBackground(rd);

    float2 interval;
    if (!spBound(ro,rd,2.1f,interval)) return float4(bg,1);
    float3 angles=float3(0.26f*sin(0.16f*u.time+0.5f),0.12f*u.time+0.4f,
                        0.08f*sin(0.11f*u.time));
    float3 c=cos(angles), s=sin(angles);
    float4x4 rotation=float4x4(spRotate(float4(1,0,0,0),c,s),spRotate(float4(0,1,0,0),c,s),
                               spRotate(float4(0,0,1,0),c,s),spRotate(float4(0,0,0,1),c,s));
    float4x4 inverseRotation=transpose(rotation);
    float travel=interval.x;
    for (int step=0; step<104 && travel<interval.y; ++step) {
        float3 p=ro+travel*rd;
        float distance=cliffordMap(p,inverseRotation);
        if (abs(distance)<0.0008f+0.0001f*travel) {
            float3 n=cliffordNormal(p,inverseRotation);
            float4 q=cliffordPoint(p,inverseRotation);
            float uAngle=atan2(q.y,q.x), vAngle=atan2(q.w,q.z);
            float warp=abs(sin(10.0f*uAngle)), weft=abs(sin(7.0f*vAngle));
            float3 base=mix(float3(0.045f,0.38f,0.44f),float3(0.82f,0.51f,0.23f),
                            smoothstep(-0.08f,0.08f,weft-warp));
            float pearl=pow(0.5f+0.5f*sin(3.0f*uAngle-2.0f*vAngle),4.0f);
            base=mix(base,float3(0.75f,0.83f,0.78f),pearl*0.75f);
            float3 outward=dot(n,rd)>0.0f ? -n:n;
            float occ=clamp(1.0f-5.0f*(0.04f-cliffordMap(p+outward*0.04f,inverseRotation)),0.5f,1.0f);
            return float4(spShade(n,rd,base,0.025f,occ),1);
        }
        travel+=max(abs(distance)*0.86f,0.00035f);
    }
    return float4(bg,1);
}
