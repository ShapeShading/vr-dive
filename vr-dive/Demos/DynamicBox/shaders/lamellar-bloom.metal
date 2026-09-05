// Lamellar bloom — ten continuous, finite membranes with scalloped margins.
// Phase, curvature and individual margins change with time; the sculpture is
// not animated by rigid rotation. There are no tubes or wireframe primitives.
// Runtime-only Metal: types and standard library come from DynamicBox's wrapper.
// The eye ray is not clipped to the Box entry/exit; only the raster aperture is.
static bool matterRay(DynamicBoxVertexOut in, constant DynamicBoxUniforms &u,
                      constant float4x4 *v2w, thread float3 &ro, thread float3 &rd) {
    uint vi=min(in.viewIndex,max(u.viewCount,1u)-1u);
    float3 eye=v2w[vi][3].xyz;
    ro=(eye-u.objectCenter.xyz)/max(u.boxScale,0.0001f);
    rd=normalize(in.worldPos-eye);
    if (!all(abs(ro)<DB_BOXDIMS-0.001f)) {
        float3 normal;
        if (db_boxHit(ro,rd,DB_BOXDIMS,normal,true)<0.0f) return false;
    }
    ro=(u.patternTransform*float4(ro,1)).xyz;
    rd=normalize((u.patternTransform*float4(rd,0)).xyz);
    return true;
}
static bool matterBound(float3 ro, float3 rd, float radius, thread float2 &span) {
    float b=dot(ro,rd), h=b*b-dot(ro,ro)+radius*radius;
    if (h<0.0f) return false;
    span=float2(max(0.0f,-b-sqrt(h)),-b+sqrt(h));
    return span.y>span.x;
}
static float3 matterShade(float3 normal, float3 rd, float3 base,
                          float occlusion, float translucency) {
    float3 n=dot(normal,rd)>0.0f ? -normal : normal;
    float3 light=normalize(float3(-0.55f,0.85f,0.65f));
    float diffuse=max(dot(n,light),0.0f);
    float back=pow(max(dot(-n,light),0.0f),2.0f)*translucency;
    float spec=pow(max(dot(n,normalize(light-rd)),0.0f),54.0f);
    float fresnel=pow(1.0f-max(dot(n,-rd),0.0f),4.0f);
    float3 color=base*(0.19f+0.95f*diffuse)*occlusion;
    color+=float3(1.0f,0.58f,0.30f)*back*0.32f;
    color+=float3(0.24f,0.52f,0.66f)*fresnel*0.25f;
    color+=float3(0.90f,0.92f,0.84f)*spec*0.5f;
    return color/(1.0f+0.18f*color);
}

static float3 lamellaLocal(float3 p) {
    // A fixed presentation tilt exposes the broad faces of the sheets.
    p.xy=float2(0.95533649f*p.x-0.29552021f*p.y,0.29552021f*p.x+0.95533649f*p.y);
    p.yz=float2(0.90044710f*p.y-0.43496553f*p.z,0.43496553f*p.y+0.90044710f*p.z);
    return p;
}
static float2 lamellaMap(float3 p,float time) {
    float3 q=lamellaLocal(p);
    float r=length(q.xz), a=atan2(q.z,q.x);
    float bend=0.24f*r*r+0.12f*sin(q.x*2.7f+time*0.27f)
               +0.065f*r*sin(3.0f*a+4.0f*r-time*0.32f);
    float2 result=float2(10,0);
    float phaseA=5.0f*a+time*0.21f, phaseB=17.0f*a-time*0.35f;
    float2 waveA=float2(cos(phaseA),sin(phaseA));
    float2 waveB=float2(cos(phaseB),sin(phaseB));
    // Visit every finite membrane so trimmed margins cannot hide another
    // layer. Complex recurrences avoid trigonometry in this ten-layer loop.
    for (int i=0; i<10; ++i) {
        float h=-0.43f+0.105f*float(i);
        float radius=0.82f-0.20f*pow((float(i)-4.5f)/5.0f,2.0f);
        radius+=0.065f*waveA.y+0.018f*waveB.y;
        float sheet=abs(q.y-h-bend)-0.012f;
        float edge=r-radius;
        // Rounded sheet margins; a slope safety factor accounts for the warp.
        float2 d=max(float2(sheet,edge),0.0f);
        float distance=(length(d)+min(max(sheet,edge),0.0f))*0.43f;
        if (distance<result.x) result=float2(distance,float(i));
        waveA=float2(0.89605250f*waveA.x-0.44394811f*waveA.y,
                     0.44394811f*waveA.x+0.89605250f*waveA.y);
        waveB=float2(0.96377090f*waveB.x+0.26673144f*waveB.y,
                    -0.26673144f*waveB.x+0.96377090f*waveB.y);
    }
    return result;
}
static float3 lamellaNormal(float3 p,float t) {
    const float e=0.001f;
    const float3 a=float3(1,-1,-1), b=float3(-1,-1,1), c=float3(-1,1,-1), d=float3(1,1,1);
    float3 n=a*lamellaMap(p+a*e,t).x+b*lamellaMap(p+b*e,t).x
            +c*lamellaMap(p+c*e,t).x+d*lamellaMap(p+d*e,t).x;
    return n*rsqrt(max(dot(n,n),0.00000001f));
}

fragment float4 dynamicBoxFragment(
    DynamicBoxVertexOut in [[stage_in]],
    constant DynamicBoxUniforms &u [[buffer(0)]],
    constant float4x4 *v2w [[buffer(1)]],
    constant float4x4 *vp [[buffer(2)]]) {
    const float3 background=float3(0.003f,0.005f,0.010f);
    float3 ro,rd;
    if (!matterRay(in,u,v2w,ro,rd)) return float4(background,1);

    float2 span;
    if (!matterBound(ro,rd,1.23f,span)) return float4(background,1);
    float travel=span.x;
    for (int step=0; step<112 && travel<span.y; ++step) {
        float3 p=ro+rd*travel;
        float2 sample=lamellaMap(p,u.time);
        if (abs(sample.x)<0.0010f) {
            float3 n=lamellaNormal(p,u.time), q=lamellaLocal(p);
            float3 outward=dot(n,rd)>0.0f?-n:n;
            float occ=clamp(1.0f-7.0f*(0.03f-lamellaMap(p+outward*0.03f,u.time).x)
                                    -2.0f*(0.09f-lamellaMap(p+outward*0.09f,u.time).x),0.38f,1.0f);
            float a=atan2(q.z,q.x), r=length(q.xz);
            float band=0.5f+0.5f*sin(sample.y*0.51f+r*3.0f);
            float3 base=mix(float3(0.065f,0.39f,0.49f),float3(0.81f,0.86f,0.77f),band);
            float lip=pow(0.5f+0.5f*sin(5.0f*a+sample.y*0.46f+u.time*0.21f),8.0f);
            base=mix(base,float3(0.95f,0.57f,0.30f),lip*0.30f);
            float micro=0.95f+0.05f*cos(72.0f*a+18.0f*r);
            return float4(matterShade(n,rd,base*micro,occ,0.7f),1);
        }
        travel+=max(abs(sample.x)*0.83f,0.0005f);
    }
    return float4(background,1);
}
