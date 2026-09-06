// Quaternion reef — a solid Julia isosurface from q -> q*q+c in R4.
// The fourth-dimensional slice and c evolve, opening/closing cavities rather
// than spinning a fixed mesh. Finite 8-iteration detail is an approximation.
// Reference: https://www.paulbourke.net/fractals/quatjulia/
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

struct ReefState { float4 c; float slice; };
static float2 reefMap(float3 p, ReefState state) {
    const float scale=0.80f;
    float3 v=p/scale;
    // Scalar component is w; x/y/z are the quaternion's imaginary components.
    float4 q=float4(v.y,v.z,state.slice,v.x);
    float derivative=1.0f, radius=length(q), trap=10.0f;
    for (int i=0; i<8; ++i) {
        radius=length(q);
        trap=min(trap,length(q.yz)+0.25f*abs(q.x));
        if (radius>4.0f) break;
        derivative*=max(2.0f*radius,0.0001f);
        q=float4(2.0f*q.w*q.xyz,q.w*q.w-dot(q.xyz,q.xyz))+state.c;
    }
    radius=length(q);
    float distance=radius>1.0f ? 0.5f*radius*log(radius)/max(derivative,0.0001f) : 0.0f;
    return float2(max(distance,0.0f)*scale,trap);
}
static float3 reefNormal(float3 p,ReefState state,float3 fallback) {
    const float e=0.0013f;
    const float3 a=float3(1,-1,-1),b=float3(-1,-1,1),c=float3(-1,1,-1),d=float3(1,1,1);
    float3 n=a*reefMap(p+a*e,state).x+b*reefMap(p+b*e,state).x
            +c*reefMap(p+c*e,state).x+d*reefMap(p+d*e,state).x;
    return dot(n,n)>1.0e-14f ? normalize(n) : fallback;
}

fragment float4 dynamicBoxFragment(
    DynamicBoxVertexOut in [[stage_in]],
    constant DynamicBoxUniforms &u [[buffer(0)]],
    constant float4x4 *v2w [[buffer(1)]],
    constant float4x4 *vp [[buffer(2)]]) {
    const float3 background=float3(0.003f,0.005f,0.010f);
    float3 ro,rd;
    if (!matterRay(in,u,v2w,ro,rd)) return float4(background,1);

    ReefState state;
    state.c=float4(0.64f+0.045f*sin(u.time*0.19f),0.075f*sin(u.time*0.17f),
                   0.09f*cos(u.time*0.13f),-0.32f+0.035f*sin(u.time*0.23f));
    state.slice=0.16f*sin(u.time*0.24f);
    float2 span;
    if (!matterBound(ro,rd,1.42f,span)) return float4(background,1);
    float travel=span.x;
    for (int step=0; step<96 && travel<span.y; ++step) {
        float3 p=ro+rd*travel;
        float2 sample=reefMap(p,state);
        if (sample.x<0.0014f) {
            float3 n=reefNormal(p,state,-rd);
            float3 outward=dot(n,rd)>0.0f?-n:n;
            float occ=clamp(1.0f-9.0f*(0.025f-reefMap(p+outward*0.025f,state).x)
                                    -2.5f*(0.08f-reefMap(p+outward*0.08f,state).x),0.35f,1.0f);
            float pigment=smoothstep(0.08f,0.45f,sample.y);
            float3 base=mix(float3(0.07f,0.21f,0.32f),float3(0.72f,0.79f,0.71f),pigment);
            float mineral=0.5f+0.5f*sin(sample.y*16.0f+p.x*3.0f+p.y*2.0f);
            base=mix(base,float3(0.73f,0.27f,0.11f),mineral*mineral*0.45f);
            return float4(matterShade(n,rd,base,occ,0.0f),1);
        }
        travel+=max(sample.x*0.68f,0.00045f);
    }
    return float4(background,1);
}
