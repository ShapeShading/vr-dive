// topographic-velvet.metal – soft topographic contour ribbons
static float topoDE(float3 p,float t){
 float3 q=p/float3(.72f,.57f,.51f);float envelope=length(q)-.90f+.08f*sin(q.x*4.f+q.y*3.f)+.035f*sin(q.z*12.f);
 q.xy+=.13f*float2(sin(q.y*4.5f+q.z*2.0f),cos(q.x*4.0f-q.z*3.0f));
 float h=.30f*sin(q.x*2.8f+sin(q.y*4.2f)*.8f)+.13f*sin(q.y*9.0f+q.x*2.5f);
 float contours=abs(sin((q.z-h)*32.0f))*.014f-.0035f;return max(contours,envelope*.26f);
}
static float3 topoN(float3 p,float t){float e=.002f;return normalize(float3(topoDE(p+float3(e,0,0),t)-topoDE(p-float3(e,0,0),t),topoDE(p+float3(0,e,0),t)-topoDE(p-float3(0,e,0),t),topoDE(p+float3(0,0,e),t)-topoDE(p-float3(0,0,e),t)));}
fragment float4 dynamicBoxFragment(DynamicBoxVertexOut in [[stage_in]],constant DynamicBoxUniforms&u [[buffer(0)]],constant float4x4*v2w [[buffer(1)]],constant float4x4*vp [[buffer(2)]]){
uint vi=min(in.viewIndex,u.viewCount-1u);float3 ro=(v2w[vi][3].xyz-u.objectCenter.xyz)/u.boxScale,rd=normalize(in.worldPos-v2w[vi][3].xyz);float3 bn;if(!all(abs(ro)<DB_BOXDIMS-1e-3f)){float en=db_boxHit(ro,rd,DB_BOXDIMS,bn,true);if(en<0)return float4(.005,.01,.006,1);ro+=rd*(en+.002f);}ro=(u.patternTransform*float4(ro,1)).xyz;rd=normalize(float3(u.patternTransform*float4(rd,0)));float z=0,t=u.time;
for(int i=0;i<125;i++){float3 p=ro+rd*z;float d=topoDE(p,t);if(d<.0007f){float3 n=topoN(p,t),l=normalize(float3(-.45,.85,.4));float dif=max(dot(n,l),0.f),r=pow(1-max(dot(n,-rd),0.f),2.5f),sp=pow(max(dot(reflect(-l,n),-rd),0.f),48.f);float3 c=mix(float3(.025,.17,.14),float3(.80,.84,.48),smoothstep(-.35,.48,p.z));c=mix(c,float3(.13,.36,.58),smoothstep(.25,.52,p.y)*.3);return float4(c*(.22+dif)+float3(.2,.6,.3)*r*.26f+sp*.5f,1);}z+=clamp(d*.62f,.0009f,.045f);if(z>6.f)break;}return float4(.003,.007,.005,1);}
