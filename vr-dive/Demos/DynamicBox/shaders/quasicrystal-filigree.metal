// quasicrystal-filigree.metal — icosahedral quasiperiodic filigree
//
// Six axes from an icosahedral star generate two incommensurate wave fields.
// Their simultaneous near-zero sets form fine, non-periodic strands rather
// than the repeating cells of a TPMS such as the existing gyroid demo.

static float2 qfRotate(float2 v, float a) {
    float s=sin(a), c=cos(a);
    return float2(c*v.x-s*v.y, s*v.x+c*v.y);
}

static float3 qfObjectPoint(float3 p, float t) {
    p.xz=qfRotate(p.xz,t*.075f);
    p.yz=qfRotate(p.yz,.22f*sin(t*.11f));
    return p;
}

static float2 qfWavePair(float3 p, float t) {
    const float a=.5257311f, b=.8506508f, f=8.4f, g=13.591f;
    float3 d0=float3(0,a,b), d1=float3(0,a,-b);
    float3 d2=float3(a,b,0), d3=float3(a,-b,0);
    float3 d4=float3(b,0,a), d5=float3(-b,0,a);
    float u=t*.10f;
    float x=(cos(dot(p,d0)*f+u)+cos(dot(p,d1)*f-u*.7f+.7f)
        +cos(dot(p,d2)*f+u*.5f+1.4f)+cos(dot(p,d3)*f-u*.4f+2.1f)
        +cos(dot(p,d4)*f+u*.3f+2.8f)+cos(dot(p,d5)*f-u*.2f+3.5f))/6.f;
    float y=(sin(dot(p,d0)*g-u*.6f+.3f)+sin(dot(p,d1)*g+u*.4f+1.2f)
        +sin(dot(p,d2)*g-u*.3f+2.f)+sin(dot(p,d3)*g+u*.5f+2.9f)
        +sin(dot(p,d4)*g-u*.2f+3.7f)+sin(dot(p,d5)*g+u*.3f+4.6f))/6.f;
    return float2(x,y);
}

static float qfDistance(float3 p, float t) {
    float3 q=qfObjectPoint(p,t);
    float2 f=qfWavePair(q,t);
    float strands=max(abs(f.x-.055f)-.07f,abs(f.y+.025f)-.07f)*.095f;
    float r=length(q);
    return max(max(strands,r-.78f),.18f-r);
}

static float3 qfNormal(float3 p, float t) {
    float2 e=float2(.0014f,0);
    return normalize(float3(
        qfDistance(p+e.xyy,t)-qfDistance(p-e.xyy,t),
        qfDistance(p+e.yxy,t)-qfDistance(p-e.yxy,t),
        qfDistance(p+e.yyx,t)-qfDistance(p-e.yyx,t)));
}

static float3 qfPalette(float p) {
    return .48f+.48f*cos(6.2831853f*(p+float3(.03f,.30f,.61f)));
}

fragment float4 dynamicBoxFragment(
    DynamicBoxVertexOut in [[stage_in]],
    constant DynamicBoxUniforms &u [[buffer(0)]],
    constant float4x4 *v2w [[buffer(1)]],
    constant float4x4 *vp [[buffer(2)]])
{
    uint vi=min(in.viewIndex,max(u.viewCount,1u)-1u);
    float3 camera=v2w[vi][3].xyz;
    float3 eye=(camera-u.objectCenter.xyz)/u.boxScale;
    float3 ray=normalize(in.worldPos-camera);
    float3 bg=float3(.0015f,.0025f,.010f);
    float3 origin=eye;
    if(!all(abs(eye)<DB_BOXDIMS-1e-3f)){
        float3 n; float entry=db_boxHit(eye,ray,DB_BOXDIMS,n,true);
        if(entry<0.f)return float4(bg,1);
        origin=eye+ray*(entry+1e-3f);
    }
    float3 exitNormal;
    float exitDistance=db_boxHit(origin,ray,DB_BOXDIMS,exitNormal,false);
    if(exitDistance<=0.f)return float4(bg,1);
    float3 ro=(u.patternTransform*float4(origin,1)).xyz;
    float3 rd=normalize(float3(u.patternTransform*float4(ray,0)));
    float travel=0, closest=1, time=u.time;
    for(int step=0;step<128;++step){
        float3 p=ro+rd*travel;
        float d=qfDistance(p,time);
        closest=min(closest,abs(d));
        float eps=.00075f+travel*.00010f;
        if(d<eps){
            float3 n=qfNormal(p,time), light=normalize(float3(-.45f,.80f,.40f));
            float dif=max(dot(n,light),0.f);
            float rim=pow(1.f-max(dot(n,-rd),0.f),3.f);
            float spec=pow(max(dot(reflect(-light,n),-rd),0.f),52.f);
            float2 fields=qfWavePair(qfObjectPoint(p,time),time);
            float3 base=qfPalette(fields.x*.43f+fields.y*.31f+length(p)*.18f);
            float3 color=base*(.18f+1.08f*dif)
                +rim*float3(.12f,.42f,1.f)*.65f
                +spec*float3(1.f,.78f,.48f);
            return float4(color*exp(-travel*.10f),1);
        }
        travel+=clamp(max(d,eps)*.64f,.00065f,.040f);
        if(travel>exitDistance+.01f)break;
    }
    float glow=exp(-closest*110.f)*.045f;
    return float4(bg+glow*float3(.15f,.38f,.90f),1);
}
