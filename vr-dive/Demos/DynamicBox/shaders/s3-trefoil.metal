// S3 trefoil — (0.8 exp(2it), 0.6 exp(3it)) in C2 = R4.
// Rotate in R4, then stereographically project q.xyz/(1-q.w).
// This is a genuine (2,3) torus knot; the rendered tube uses 128 short capsules.
// Background: https://www.math.brown.edu/tbanchof/Beyond3d.new/chapter6/s6_8.html
// Knot: https://www.mathcurve.com/courbes3d.gb/noeuds/noeuddetrefle.shtml
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

static float3 knotProject(float2 z1, float2 z2, float3 c, float3 s) {
    float4 q=spRotate(float4(z1*0.8f,z2*0.6f),c,s);
    return 0.43f*q.xyz/(1.0f-q.w);
}
static float3 knotCurve(float phase, float3 c, float3 s, thread float3 &tangent) {
    float4 q=spRotate(float4(0.8f*cos(2.0f*phase),0.8f*sin(2.0f*phase),
                            0.6f*cos(3.0f*phase),0.6f*sin(3.0f*phase)),c,s);
    float4 dq=spRotate(float4(-1.6f*sin(2.0f*phase),1.6f*cos(2.0f*phase),
                             -1.8f*sin(3.0f*phase),1.8f*cos(3.0f*phase)),c,s);
    float denominator=1.0f-q.w;
    tangent=0.43f*(dq.xyz*denominator+q.xyz*dq.w)/(denominator*denominator);
    return 0.43f*q.xyz/denominator;
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
    if (!spBound(ro,rd,1.65f,interval)) return float4(bg,1);
    // Bounded XW / ZW angles avoid the stereographic pole at every animation time.
    float3 angles=float3(0.22f*sin(u.time*0.13f),u.time*0.11f+0.38f,
                        0.10f*sin(u.time*0.17f+0.7f));
    float3 c=cos(angles), s=sin(angles);
    SPHit hit; hit.t=1.0e6f; hit.normal=float3(0,1,0); hit.id=0; hit.along=0;
    float2 z1=float2(1,0), z2=float2(1,0);
    float3 first=knotProject(z1,z2,c,s), a=first;
    // Complex recurrence: no trigonometry inside the 128-segment traversal.
    const float2 step1=float2(0.995184727f,0.098017140f);
    const float2 step2=float2(0.989176510f,0.146730474f);
    for (int i=0; i<128; ++i) {
        z1=float2(z1.x*step1.x-z1.y*step1.y,z1.x*step1.y+z1.y*step1.x);
        z2=float2(z2.x*step2.x-z2.y*step2.y,z2.x*step2.y+z2.y*step2.x);
        float3 b=i==127 ? first : knotProject(z1,z2,c,s);
        spCapsule(ro,rd,a,b,0.052f,float(i),hit);
        a=b;
    }
    if (hit.t==1.0e6f) return float4(bg,1);
    float phase=2.0f*DB_PI*(hit.id+hit.along)/128.0f;
    float3 p=ro+rd*hit.t, tangent;
    // Refine only the winning segment against the smooth curve. This removes
    // capsule-joint seams from normals without raising the geometry budget.
    for (int refine=0; refine<2; ++refine) {
        float3 center=knotCurve(phase,c,s,tangent);
        phase-=clamp(dot(center-p,tangent)/max(dot(tangent,tangent),0.00001f),-0.025f,0.025f);
    }
    float3 center=knotCurve(phase,c,s,tangent);
    float3 normal=normalize(p-center);
    float3 base=mix(float3(0.025f,0.11f,0.48f),float3(0.48f,0.055f,0.25f),
                    0.5f+0.5f*sin(phase*2.0f));
    float rib=0.5f+0.5f*cos(phase*64.0f);
    base*=0.78f+0.22f*rib;
    float enamel=smoothstep(0.68f,0.85f,sin(phase*9.0f+3.0f*normal.y));
    base=mix(base,float3(0.83f,0.64f,0.28f),enamel*0.85f);
    float pulse=pow(0.5f+0.5f*cos(phase*3.0f-0.5f*u.time),20.0f);
    return float4(spShade(normal,rd,base,0.12f*pulse,1.0f),1);
}
