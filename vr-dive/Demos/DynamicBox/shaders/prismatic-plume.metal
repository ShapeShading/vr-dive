// Prismatic plume — continuous density, front-to-back emission/absorption.
// Broad folded density sheets are visible inside a finite translucent body.
// This is procedural volume ray casting, NOT a voxel grid or a fluid simulation.
// Optical model: https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-39-volume-rendering-techniques
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

static float plumeDensity(float3 p,float time) {
    float3 q=p/float3(0.86f,0.96f,0.79f);
    float radius=length(q);
    if (radius>=1.0f) return 0.0f;
    float envelope=1.0f-smoothstep(0.68f,1.0f,radius);
    float3 warp=0.75f*sin(p.yzx*3.2f+float3(0.23f,-0.19f,0.17f)*time);
    float3 wave=p*6.2f+warp+float3(0.13f,0.16f,-0.12f)*time;
    float fold=sin(wave.x)+sin(wave.y)+sin(wave.z);
    fold+=0.18f*sin(13.0f*p.x+9.0f*p.z-time*0.27f)*sin(p.y*15.0f+time*0.21f);
    float sheet=saturate(1.0f-abs(fold)*1.65f);
    return envelope*sheet*sheet;
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
    if (!matterBound(ro,rd,0.98f,span)) return float4(background,1);
    float stepSize=(span.y-span.x)/88.0f;
    float transmission=1.0f;
    float3 accumulated=float3(0);
    float3 light=normalize(float3(-0.6f,0.8f,0.5f));
    // Midpoint sampling is deterministic across stereo eyes: no screen-space
    // time jitter. The opacity is corrected for the actual world step length.
    for (int step=0; step<88 && transmission>0.025f; ++step) {
        float3 p=ro+rd*(span.x+(float(step)+0.5f)*stepSize);
        float density=plumeDensity(p,u.time);
        if (density<0.003f) continue;
        float nearby=plumeDensity(p+light*0.12f,u.time);
        // One light probe approximates local self-shadowing at a bounded cost;
        // this is not a multi-scattering or physically exact shadow solution.
        float shadow=exp(-2.6f*nearby);
        float edge=saturate(0.5f+1.8f*(density-nearby));
        float warmth=0.5f+0.5f*sin(p.y*3.5f+p.x*2.0f+u.time*0.12f);
        float3 pigment=mix(float3(0.055f,0.32f,0.58f),float3(0.96f,0.39f,0.16f),warmth);
        float3 scattering=mix(pigment,float3(0.80f,0.87f,0.86f),edge*0.3f);
        scattering*=0.20f+0.95f*shadow;
        scattering+=pigment*0.10f*density;
        float alpha=1.0f-exp(-density*stepSize*14.0f);
        accumulated+=transmission*alpha*scattering;
        transmission*=1.0f-alpha;
    }
    return float4(accumulated+transmission*background,1);
}
