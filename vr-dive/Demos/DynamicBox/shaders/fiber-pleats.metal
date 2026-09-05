// fiber-pleats.metal – dense silky fibers with broad organic bends
static float fiberPleatsDE(float3 p,float t){
 float3 q=p/float3(.68f,.60f,.51f);float envelope=length(q)-.90f+.07f*sin(q.x*4.f+q.y*3.f)+.035f*sin(q.z*11.f);
 q.x+=.13f*sin(q.y*3.5f+q.z*2.5f+t*.05f);q.y+=.10f*sin(q.x*5.0f-q.z*3.0f);
 float f=q.x*1.1f+q.y*.55f+.20f*sin(q.y*6.0f+q.z*5.0f)+.08f*sin(q.x*17.0f-q.y*6.0f);return max(abs(sin(f*47.0f))*.012f-.003f,envelope*.25f);
}
static float3 fiberPleatsN(float3 p,float t){float e=.0015f;return normalize(float3(fiberPleatsDE(p+float3(e,0,0),t)-fiberPleatsDE(p-float3(e,0,0),t),fiberPleatsDE(p+float3(0,e,0),t)-fiberPleatsDE(p-float3(0,e,0),t),fiberPleatsDE(p+float3(0,0,e),t)-fiberPleatsDE(p-float3(0,0,e),t)));}

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
uint vi=min(in.viewIndex,u.viewCount-1u);float3 ro=(float3(v2w[vi][3].xyz)-u.objectCenter.xyz)/u.boxScale,rd=normalize(in.worldPos-float3(v2w[vi][3].xyz));float3 bn;if(!all(abs(ro)<DB_BOXDIMS-1e-3f)){float en=db_boxHit(ro,rd,DB_BOXDIMS,bn,true);if(en<0)return float4(.005,.01,.015,1);ro+=rd*(en+.002f);}ro=(u.patternTransform*float4(ro,1)).xyz;rd=normalize(float3(u.patternTransform*float4(rd,0)));float z, foldFar;
 if (!foldInterval(ro,rd,z,foldFar)) return float4(.004f,.007f,.012f,1.0f);
 float t=u.time;
for(int i=0;i<125;i++){float3 p=ro+rd*z;float d=fiberPleatsDE(p,t);if(d<.00065f){float3 n=fiberPleatsN(p,t),l=normalize(float3(-.4,.85,.25));float dif=max(dot(n,l),0.f),rim=pow(1-max(dot(n,-rd),0.f),3.f),sp=pow(max(dot(reflect(-l,n),-rd),0.f),55.f);float3 c=mix(float3(.025,.14,.23),float3(.94,.82,.62),smoothstep(-.4,.55,p.z+p.y*.3));c=mix(c,float3(.36,.16,.47),smoothstep(.3,.55,p.x)*.25);return float4(c*(.18+dif*1.08f)+float3(.2,.55,.7)*rim*.3f+sp*.6f,1);}z+=clamp(d*.6f,.0008f,.04f);if(z>foldFar)break;}return float4(.003,.006,.01,1);}
