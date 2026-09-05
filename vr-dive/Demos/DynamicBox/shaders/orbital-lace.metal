// Orbital lace: three orthogonal, gently twisted ribbed toroidal loops.
// Analytical torus distances, no fractal iterations or volumetric accumulation.
static float sculptureDE(float3 p,float t){
    float angle=0.32f*sin(p.y*2.0f+t*0.12f);
    float c=cos(angle),s=sin(angle);
    p.xz=float2(c*p.x-s*p.z,s*p.x+c*p.z);
    float3 q=p;
    float result=10.0f;
    for(int i=0;i<3;i++){
        float r=length(q.xz);
        float a=atan2(q.z,q.x);
        float tube=0.047f+0.008f*cos(a*32.0f);
        result=min(result,length(float2(r-0.56f,q.y))-tube);
        q=q.yzx;
    }
    return result*0.55f;
}
static float3 sculptureColor(float3 p){
    float band=0.5f+0.5f*sin(5.0f*p.y+4.0f*p.z);
    return mix(float3(0.08f,0.38f,0.52f),float3(0.88f,0.53f,0.20f),band);
}

static float3 sculptureNormal(float3 p, float t) {
    const float e = 0.0012f;
    const float3 a=float3(1,-1,-1), b=float3(-1,-1,1);
    const float3 c=float3(-1,1,-1), d=float3(1,1,1);
    return normalize(a*sculptureDE(p+a*e,t)+b*sculptureDE(p+b*e,t)
                   +c*sculptureDE(p+c*e,t)+d*sculptureDE(p+d*e,t));
}
fragment float4 dynamicBoxFragment(
    DynamicBoxVertexOut in [[stage_in]],
    constant DynamicBoxUniforms &u [[buffer(0)]],
    constant float4x4 *v2w [[buffer(1)]],
    constant float4x4 *vp [[buffer(2)]]) {
    uint vi=min(in.viewIndex,u.viewCount-1u);
    float3 cam=v2w[vi][3].xyz;
    float3 ro=(cam-u.objectCenter.xyz)/u.boxScale;
    float3 rd=normalize(in.worldPos-cam);
    if (!all(abs(ro)<DB_BOXDIMS-0.001f)) {
        float3 nn;
        float entry=db_boxHit(ro,rd,DB_BOXDIMS,nn,true);
        if(entry<0) return float4(0.003f,0.005f,0.009f,1);
        ro+=rd*(entry+0.001f);
    }
    ro=(u.patternTransform*float4(ro,1)).xyz;
    rd=normalize((u.patternTransform*float4(rd,0)).xyz);
    // Conservative analytic bound, not a clip at the portal exit.
    float b=dot(ro,rd), h=b*b-dot(ro,ro)+1.0f;
    if(h<0) return float4(0.003f,0.005f,0.009f,1);
    float travel=max(0.0f,-b-sqrt(h)), end=-b+sqrt(h);
    for(int i=0;i<112 && travel<end;i++){
        float3 p=ro+rd*travel;
        float distance=sculptureDE(p,u.time);
        if(distance<0.001f){
            float3 n=sculptureNormal(p,u.time);
            n=dot(n,rd)>0?-n:n;
            float3 light=normalize(float3(-0.6f,0.8f,0.7f));
            float diffuse=max(dot(n,light),0.0f);
            float fresnel=pow(1.0f-max(dot(n,-rd),0.0f),4.0f);
            float3 base=sculptureColor(p);
            // Two short normal probes give contact shading at bounded cost.
            float occ=clamp(1.0f-8.0f*(0.025f-sculptureDE(p+n*0.025f,u.time))
                                    -3.0f*(0.07f-sculptureDE(p+n*0.07f,u.time)),0.35f,1.0f);
            float spec=pow(max(dot(n,normalize(light-rd)),0.0f),64.0f);
            float3 color=base*(0.24f+0.85f*diffuse)*occ;
            color+=float3(0.65f,0.84f,1.0f)*fresnel*0.3f;
            color+=float3(1.0f,0.86f,0.68f)*spec*0.65f;
            return float4(color,1);
        }
        travel+=max(distance*0.8f,0.0006f);
    }
    return float4(0.003f,0.005f,0.009f,1);
}
