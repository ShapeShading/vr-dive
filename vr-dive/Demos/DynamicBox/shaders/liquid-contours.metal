// liquid-contours.metal – soft folded contour layers / fluid marble
static float liquidContoursDE(float3 p,float t){
 float3 q=p/float3(.70f,.57f,.52f);float envelope=length(q)-.91f+.08f*sin(q.x*4.0f+q.z*3.0f)+.04f*sin(q.y*9.0f);
 q.xy+=.13f*float2(sin(q.y*4.2f+t*.06f),cos(q.x*5.1f-t*.05f));
 float f=q.x*1.4f+q.y*.7f+.28f*sin(q.x*3.5f+q.y*4.8f)+.12f*sin(q.y*13.0f-q.z*4.0f);
 float layer=abs(sin(f*15.0f+q.z*3.0f))*.027f-.0055f;return max(layer,envelope*.27f);
}
static float3 liquidContoursN(float3 p,float t){float e=.002f;return normalize(float3(liquidContoursDE(p+float3(e,0,0),t)-liquidContoursDE(p-float3(e,0,0),t),liquidContoursDE(p+float3(0,e,0),t)-liquidContoursDE(p-float3(0,e,0),t),liquidContoursDE(p+float3(0,0,e),t)-liquidContoursDE(p-float3(0,0,e),t)));}

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
uint vi=min(in.viewIndex,u.viewCount-1u);float3 ro=(float3(v2w[vi][3].xyz)-u.objectCenter.xyz)/u.boxScale,rd=normalize(in.worldPos-float3(v2w[vi][3].xyz));float3 bn;if(!all(abs(ro)<DB_BOXDIMS-1e-3f)){float en=db_boxHit(ro,rd,DB_BOXDIMS,bn,true);if(en<0)return float4(.01,.01,.015,1);}ro=(u.patternTransform*float4(ro,1)).xyz;rd=normalize(float3(u.patternTransform*float4(rd,0)));float z, foldFar;
 if (!foldInterval(ro,rd,z,foldFar)) return float4(.004f,.007f,.012f,1.0f);
 float t=u.time;
for(int i=0;i<120;i++){float3 p=ro+rd*z;float d=liquidContoursDE(p,t);if(d<.0009f){float3 n=liquidContoursN(p,t),l=normalize(float3(-.5,.7,.5));float dif=max(dot(n,l),0.f),rim=pow(1-max(dot(n,-rd),0.f),2.f),sp=pow(max(dot(reflect(-l,n),-rd),0.f),40.f);float band=.5f+.5f*sin((p.x-p.y)*12.f);float3 c=mix(float3(.82,.86,.81),float3(.05,.42,.55),smoothstep(.2,.85,band));c=mix(c,float3(.91,.42,.22),smoothstep(.25,.55,p.y)*.35);return float4(c*(.26+dif)+float3(.12,.5,.7)*rim*.34f+sp*.45f,1);}z+=clamp(d*.68f,.001f,.05f);if(z>foldFar)break;}return float4(.004,.007,.012,1);}
