// coxeter-rose-window.metal — nested icosahedral mirror-plane tracery
//
// Coxeter-style mirror planes are intersected with two spherical shells and
// joined by tubes along six icosahedral axes. The result is a hollow spatial
// rose window rather than another volumetric kaleidoscope.

static float2 crwRotate(float2 v,float a){
    float s=sin(a),c=cos(a);return float2(c*v.x-s*v.y,s*v.x+c*v.y);
}
static float3 crwObjectPoint(float3 p,float t){
    p.xz=crwRotate(p.xz,t*.07f);
    p.yz=crwRotate(p.yz,.20f*sin(t*.10f));return p;
}
static float crwMirrorBand(float3 p){
    const float a=.5257311f,b=.8506508f;
    float d=min(abs(p.x),min(abs(p.y),abs(p.z)));
    d=min(d,abs(dot(p,float3(0,a,b))));
    d=min(d,abs(dot(p,float3(0,a,-b))));
    d=min(d,abs(dot(p,float3(a,b,0))));
    d=min(d,abs(dot(p,float3(a,-b,0))));
    d=min(d,abs(dot(p,float3(b,0,a))));
    return min(d,abs(dot(p,float3(-b,0,a))));
}
static float crwAxisDistance(float3 p){
    const float a=.5257311f,b=.8506508f;
    float3 d0=float3(0,a,b),d1=float3(0,a,-b);
    float3 d2=float3(a,b,0),d3=float3(a,-b,0);
    float3 d4=float3(b,0,a),d5=float3(-b,0,a);
    float d=length(p-d0*dot(p,d0));
    d=min(d,length(p-d1*dot(p,d1)));d=min(d,length(p-d2*dot(p,d2)));
    d=min(d,length(p-d3*dot(p,d3)));d=min(d,length(p-d4*dot(p,d4)));
    return min(d,length(p-d5*dot(p,d5)));
}
static float crwDistance(float3 p,float t){
    float3 q=crwObjectPoint(p,t);float r=length(q);
    float outer=max(abs(r-.70f)-.014f,crwMirrorBand(q)-.020f);
    float3 inner=q;inner.xy=crwRotate(inner.xy,.314159f);
    inner.yz=crwRotate(inner.yz,-.231f);
    float innerRose=max(abs(r-.49f)-.012f,crwMirrorBand(inner)-.015f);
    float spokes=max(crwAxisDistance(q)-.013f,abs(r-.595f)-.115f);
    float nodes=max(abs(r-.70f)-.024f,crwAxisDistance(q)-.026f);
    return min(min(outer,innerRose),min(spokes,nodes));
}
static float3 crwNormal(float3 p,float t){
    float2 e=float2(.0013f,0);
    return normalize(float3(
        crwDistance(p+e.xyy,t)-crwDistance(p-e.xyy,t),
        crwDistance(p+e.yxy,t)-crwDistance(p-e.yxy,t),
        crwDistance(p+e.yyx,t)-crwDistance(p-e.yyx,t)));
}
fragment float4 dynamicBoxFragment(
    DynamicBoxVertexOut in [[stage_in]],
    constant DynamicBoxUniforms&u [[buffer(0)]],
    constant float4x4*v2w [[buffer(1)]],
    constant float4x4*vp [[buffer(2)]])
{
    uint vi=min(in.viewIndex,max(u.viewCount,1u)-1u);
    float3 camera=v2w[vi][3].xyz;
    float3 eye=(camera-u.objectCenter.xyz)/u.boxScale;
    float3 ray=normalize(in.worldPos-camera),bg=float3(.003f,.0015f,.009f);
    float3 origin=eye;
    if(!all(abs(eye)<DB_BOXDIMS-1e-3f)){
        float3 n;float entry=db_boxHit(eye,ray,DB_BOXDIMS,n,true);
        if(entry<0.f)return float4(bg,1);
        origin=eye+ray*(entry+1e-3f);
    }
    float3 exitNormal;
    float exitDistance=db_boxHit(origin,ray,DB_BOXDIMS,exitNormal,false);
    if(exitDistance<=0.f)return float4(bg,1);
    float3 ro=(u.patternTransform*float4(origin,1)).xyz;
    float3 rd=normalize(float3(u.patternTransform*float4(ray,0)));
    float travel=0,closest=1,time=u.time;
    for(int step=0;step<120;++step){
        float3 p=ro+rd*travel;float d=crwDistance(p,time);
        closest=min(closest,abs(d));float eps=.0007f+travel*.00010f;
        if(d<eps){
            float3 n=crwNormal(p,time),light=normalize(float3(-.50f,.78f,.37f));
            float dif=max(dot(n,light),0.f);
            float rim=pow(1.f-max(dot(n,-rd),0.f),2.7f);
            float spec=pow(max(dot(reflect(-light,n),-rd),0.f),64.f);
            float outerWeight=smoothstep(.56f,.68f,length(p));
            float3 base=mix(float3(.14f,.48f,.82f),float3(.95f,.55f,.16f),outerWeight);
            float3 color=base*(.18f+1.08f*dif)
                +rim*float3(.62f,.20f,.92f)*.58f
                +spec*float3(1.f,.92f,.68f);
            return float4(color*exp(-travel*.09f),1);
        }
        travel+=clamp(max(d,eps)*.72f,.00065f,.045f);
        if(travel>exitDistance+.01f)break;
    }
    float glow=exp(-closest*120.f)*.04f;
    return float4(bg+glow*float3(.32f,.12f,.65f),1);
}
