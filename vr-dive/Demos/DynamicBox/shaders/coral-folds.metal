// coral-folds.metal – branching petal / coral-like soft folds
static float coralDE(float3 p,float t){
 float3 q=p/float3(.68f,.68f,.57f);float r=length(q.xy),a=atan2(q.y,q.x);
 float envelope=length(q)-.89f+.075f*sin(a*6.f+q.z*4.f)+.04f*sin(a*13.f-r*7.f);
 float petals=cos(a*17.0f+sin(r*8.0f+q.z*4.0f)*.75f);float curl=q.z-.14f*sin(r*6.0f+petals*1.8f);
 return max(max(abs(petals)*.018f-.004f,abs(curl)*.045f-.0055f),envelope*.25f);
}
static float3 coralN(float3 p,float t){float e=.002f;return normalize(float3(coralDE(p+float3(e,0,0),t)-coralDE(p-float3(e,0,0),t),coralDE(p+float3(0,e,0),t)-coralDE(p-float3(0,e,0),t),coralDE(p+float3(0,0,e),t)-coralDE(p-float3(0,0,e),t)));}
fragment float4 dynamicBoxFragment(DynamicBoxVertexOut in [[stage_in]],constant DynamicBoxUniforms&u [[buffer(0)]],constant float4x4*v2w [[buffer(1)]],constant float4x4*vp [[buffer(2)]]){
uint vi=min(in.viewIndex,u.viewCount-1u);float3 ro=(v2w[vi][3].xyz-u.objectCenter.xyz)/u.boxScale,rd=normalize(in.worldPos-v2w[vi][3].xyz);float3 bn;if(!all(abs(ro)<DB_BOXDIMS-1e-3f)){float en=db_boxHit(ro,rd,DB_BOXDIMS,bn,true);if(en<0)return float4(.008,.004,.012,1);ro+=rd*(en+.002f);}ro=(u.patternTransform*float4(ro,1)).xyz;rd=normalize(float3(u.patternTransform*float4(rd,0)));float z=0,t=u.time;
for(int i=0;i<120;i++){float3 p=ro+rd*z;float d=coralDE(p,t);if(d<.0008f){float3 n=coralN(p,t),l=normalize(float3(-.35,.9,.3));float dif=max(dot(n,l),0.f),r=pow(1-max(dot(n,-rd),0.f),3.f),sp=pow(max(dot(reflect(-l,n),-rd),0.f),40.f);float3 c=mix(float3(.10,.025,.16),float3(.96,.40,.43),smoothstep(-.35,.45,p.z+p.y*.25));c=mix(c,float3(.95,.73,.38),smoothstep(.25,.5,p.x)*.35);return float4(c*(.20+dif)+float3(.5,.12,.28)*r*.3f+sp*.48f,1);}z+=clamp(d*.66f,.001f,.05f);if(z>6.f)break;}return float4(.004,.002,.008,1);}
