// sediment-ribbons.metal – layered sediment / folded ribbon strata
static float sedimentDE(float3 p,float t){
 float3 q=p/float3(.72f,.55f,.51f);float envelope=length(q)-.91f+.08f*sin(q.x*3.f-q.z*4.f)+.04f*sin(q.y*10.f);
 q.xy+=.12f*float2(sin(q.y*4.0f+q.z),sin(q.x*5.0f-t*.05f));
 float f=q.x+q.y*.8f+.25f*sin(q.y*5.0f+q.x*2.5f)+.09f*sin(q.x*17.0f+q.z*6.0f);
 return max(abs(sin(f*17.0f+q.z*3.5f))*.025f-.005f,envelope*.27f);
}
static float3 sedimentN(float3 p,float t){float e=.002f;return normalize(float3(sedimentDE(p+float3(e,0,0),t)-sedimentDE(p-float3(e,0,0),t),sedimentDE(p+float3(0,e,0),t)-sedimentDE(p-float3(0,e,0),t),sedimentDE(p+float3(0,0,e),t)-sedimentDE(p-float3(0,0,e),t)));}

// All geometry is inside this sphere, in pattern space. Skip empty space
// analytically; this is independent of the viewing box's exit plane.
static bool foldInterval(float3 ro, float3 rd, thread float &nearT, thread float &farT) {
    float b = dot(ro, rd);
    float h = b*b - dot(ro, ro) + 0.90f*0.90f;
    if (h < 0.0f) return false;
    float root = sqrt(h);
    nearT = max(0.0f, -b-root);
    farT = -b+root;
    return farT > nearT;
}

fragment float4 dynamicBoxFragment(DynamicBoxVertexOut in [[stage_in]],constant DynamicBoxUniforms&u [[buffer(0)]],constant float4x4*v2w [[buffer(1)]],constant float4x4*vp [[buffer(2)]]){
uint vi=min(in.viewIndex,u.viewCount-1u);float3 ro=(v2w[vi][3].xyz-u.objectCenter.xyz)/u.boxScale,rd=normalize(in.worldPos-v2w[vi][3].xyz);float3 bn;if(!all(abs(ro)<DB_BOXDIMS-1e-3f)){float en=db_boxHit(ro,rd,DB_BOXDIMS,bn,true);if(en<0)return float4(.01,.008,.006,1);}ro=(u.patternTransform*float4(ro,1)).xyz;rd=normalize(float3(u.patternTransform*float4(rd,0)));float z, foldFar;
 if (!foldInterval(ro,rd,z,foldFar)) return float4(.004f,.007f,.012f,1.0f);
 float t=u.time;
for(int i=0;i<120;i++){float3 p=ro+rd*z;float d=sedimentDE(p,t);if(d<.00085f){float3 n=sedimentN(p,t),l=normalize(float3(-.5,.8,.35));float dif=max(dot(n,l),0.f),r=pow(1-max(dot(n,-rd),0.f),2.f),sp=pow(max(dot(reflect(-l,n),-rd),0.f),34.f);float bands=.5f+.5f*sin((p.x+p.y)*18.f);float3 c=mix(float3(.035,.20,.29),float3(.88,.48,.19),smoothstep(.18,.82,bands));c=mix(c,float3(.88,.81,.65),smoothstep(.2,.5,p.z)*.35);return float4(c*(.22+dif)+float3(.4,.25,.08)*r*.26f+sp*.45f,1);}z+=clamp(d*.68f,.001f,.05f);if(z>foldFar)break;}return float4(.004,.005,.008,1);}
