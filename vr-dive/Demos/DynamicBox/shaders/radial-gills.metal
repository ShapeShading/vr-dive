// radial-gills.metal – mushroom-gill / radial pleat sculpture
static float radialGillsDE(float3 p, float t) {
    float3 q=p/float3(.68f,.68f,.50f);float r=length(q.xy),a=atan2(q.y,q.x);
    float envelope=length(q)-.92f+.07f*sin(a*5.0f+q.z*3.0f)+.04f*sin(a*11.0f-r*5.0f);
    float warp=.10f*sin(q.z*5.0f+t*.06f)+.045f*sin(r*13.0f-a*3.0f);
    float spokes=sin(a*44.0f+warp*14.0f+sin(r*10.0f)*1.1f);
    float crown=q.z-(.10f+.12f*cos(r*4.0f)+.045f*sin(a*9.0f+r*13.0f));
    return max(max(abs(spokes)*.018f-.004f,abs(crown)*.055f-.006f),envelope*.26f);
}
static float3 radialGillsN(float3 p,float t) { float e=.002f; return normalize(float3(
 radialGillsDE(p+float3(e,0,0),t)-radialGillsDE(p-float3(e,0,0),t),
 radialGillsDE(p+float3(0,e,0),t)-radialGillsDE(p-float3(0,e,0),t),
 radialGillsDE(p+float3(0,0,e),t)-radialGillsDE(p-float3(0,0,e),t))); }

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

fragment float4 dynamicBoxFragment(DynamicBoxVertexOut in [[stage_in]],constant DynamicBoxUniforms&u [[buffer(0)]],constant float4x4*v2w [[buffer(1)]],constant float4x4*vp [[buffer(2)]]) {
 uint vi=min(in.viewIndex,u.viewCount-1u); float3 ro=(float3(v2w[vi][3].xyz)-u.objectCenter.xyz)/u.boxScale; float3 rd=normalize(in.worldPos-float3(v2w[vi][3].xyz)); float3 bn;
 if(!all(abs(ro)<DB_BOXDIMS-1e-3f)){float en=db_boxHit(ro,rd,DB_BOXDIMS,bn,true);if(en<0)return float4(.01,.015,.02,1);ro+=rd*(en+.002f);}
 ro=(u.patternTransform*float4(ro,1)).xyz;rd=normalize(float3(u.patternTransform*float4(rd,0)));float d0, foldFar;
 if (!foldInterval(ro,rd,d0,foldFar)) return float4(.004f,.007f,.012f,1.0f);
 float t=u.time;
 for(int i=0;i<120;i++){float3 p=ro+rd*d0;float d=radialGillsDE(p,t);if(d<.0008f){float3 n=radialGillsN(p,t),l=normalize(float3(-.3,.9,.4));float dif=max(dot(n,l),0.f),rim=pow(1-max(dot(n,-rd),0.f),3.f),spec=pow(max(dot(reflect(-l,n),-rd),0.f),36.f);float3 c=mix(float3(.035,.19,.27),float3(.96,.72,.46),smoothstep(-.3,.4,p.z+p.x*.3));c=mix(c,float3(.83,.27,.31),smoothstep(.25,.52,p.y)*.45f);return float4(c*(.22+dif)+float3(.25,.6,.7)*rim*.3+spec*.5,1);}d0+=clamp(d*.68f,.001f,.05f);if(d0>foldFar)break;}
 return float4(.005,.008,.012,1);
}
