// Tesseract jewel — rotating R4 hypercube, perspective-projected into R3.
// Vertices (+/-1,+/-1,+/-1,+/-1), scaled to unit circumradius.
// Reference: https://mathworld.wolfram.com/Tesseract.html
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

// Decorative membranes on three genuine square faces. This is thin-film
// compositing, deliberately not presented as physical multi-bounce glass.
static void jewelTriangle(float3 ro, float3 rd, float3 a, float3 b, float3 c,
                          float id, thread SPHit &hit) {
    float3 ab=b-a, ac=c-a, h=cross(rd,ac);
    float determinant=dot(ab,h);
    if (abs(determinant)<0.000001f) return;
    float3 offset=ro-a;
    float u=dot(offset,h)/determinant;
    float3 q=cross(offset,ab);
    float v=dot(rd,q)/determinant;
    float t=dot(ac,q)/determinant;
    if (u>=0.0f && v>=0.0f && u+v<=1.0f && t>=0.0f && t<hit.t) {
        hit.t=t; hit.normal=normalize(cross(ab,ac)); hit.id=id; hit.along=u;
    }
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
    if (!spBound(ro,rd,1.35f,interval)) return float4(bg,1);
    float3 angles=float3(0.19f*u.time+0.43f,0.13f*u.time+0.62f,0.11f*u.time+0.27f);
    float3 c=cos(angles), s=sin(angles);
    float3 points[16];
    for (int i=0; i<16; ++i) {
        float4 q=float4((i&1)?0.5f:-0.5f,(i&2)?0.5f:-0.5f,
                        (i&4)?0.5f:-0.5f,(i&8)?0.5f:-0.5f);
        q=spRotate(q,c,s);
        // The projection eye stays outside S3: denominator is always >=1.1.
        points[i]=2.2f*q.xyz/(2.1f-q.w);
    }
    SPHit hit; hit.t=1.0e6f; hit.normal=float3(0,1,0); hit.id=0; hit.along=0;
    for (int i=0; i<16; ++i) {
        spSphere(ro,rd,points[i],0.048f,4.0f+float(i),hit);
        for (int axis=0; axis<4; ++axis) {
            int mask=1<<axis;
            if ((i&mask)==0) spCapsule(ro,rd,points[i],points[i|mask],0.021f,float(axis),hit);
        }
    }
    SPHit film; film.t=1.0e6f; film.normal=float3(0,1,0); film.id=0; film.along=0;
    const ushort4 faces[3]={ushort4(0,1,3,2),ushort4(8,9,13,12),ushort4(9,11,15,13)};
    for (int i=0; i<3; ++i) {
        ushort4 f=faces[i];
        jewelTriangle(ro,rd,points[f.x],points[f.y],points[f.z],float(i),film);
        jewelTriangle(ro,rd,points[f.x],points[f.z],points[f.w],float(i),film);
    }
    bool joint=hit.id>=4.0f;
    float3 base=joint ? float3(0.80f,0.76f,0.63f) : spPalette(hit.id*1.7f+0.4f);
    float pulse=pow(0.5f+0.5f*cos(2.0f*DB_PI*(2.0f*hit.along-0.22f*u.time+hit.id*0.17f)),24.0f);
    float cuff=smoothstep(0.94f,0.99f,cos(6.0f*DB_PI*hit.along));
    if (!joint) base=mix(base,float3(0.85f,0.63f,0.29f),cuff*0.75f);
    float3 color=hit.t==1.0e6f ? bg : spShade(hit.normal,rd,base,joint?0.03f:pulse*0.55f,1.0f);
    if (film.t<hit.t) {
        float grazing=pow(1.0f-abs(dot(film.normal,rd)),3.0f);
        float3 tint=spPalette(film.id*1.9f+0.9f);
        float3 sheen=spShade(film.normal,rd,tint,0.06f,1.0f);
        color=mix(color,sheen,0.14f+0.35f*grazing);
    }
    return float4(color,1);
}
