// pleated-marble.metal — bounded porcelain fold sculpture
// Fine warped laminae are clipped by an asymmetric organic envelope. Rays may
// continue beyond the DynamicBox boundary; the box is only the viewing portal.

static float pmHash(float3 p) {
    p = fract(p * 0.3183099f + 0.17f); p *= 17.0f;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
static float pmNoise(float3 p) {
    float3 i=floor(p), f=fract(p); f=f*f*(3.0f-2.0f*f);
    return mix(mix(mix(pmHash(i),pmHash(i+float3(1,0,0)),f.x),
                   mix(pmHash(i+float3(0,1,0)),pmHash(i+float3(1,1,0)),f.x),f.y),
               mix(mix(pmHash(i+float3(0,0,1)),pmHash(i+float3(1,0,1)),f.x),
                   mix(pmHash(i+float3(0,1,1)),pmHash(i+float3(1,1,1)),f.x),f.y),f.z);
}
static float pmField(float3 p,float t,thread float &groove,thread float &shell) {
    // Small, off-centre object with a silhouette that is neither spherical nor tiled.
    p -= float3(0.05f,-0.03f,0.02f);
    float3 q=p/float3(0.70f,0.59f,0.52f);
    float n=pmNoise(q*2.3f)+0.45f*pmNoise(q*4.6f);
    shell=length(q)-0.92f+0.11f*(n-0.65f)+0.055f*sin(q.x*3.2f+q.y*2.5f);

    // Curled strata: many narrow sheets, locally pinched and fanned.
    float bend=0.22f*sin(q.x*2.8f)+0.14f*sin(q.z*3.7f-q.x*1.6f);
    float layer=q.y+bend+0.09f*sin(q.x*7.0f+q.z*3.0f)+0.035f*sin(q.z*15.0f);
    float phase=layer*28.0f+2.2f*sin(atan2(q.z,q.x)*3.0f+length(q.xz)*8.0f);
    groove=0.5f+0.5f*cos(phase);
    float sheet=abs(sin(phase))*0.030f-0.0065f;
    return max(sheet,shell*0.30f);
}
static float pmDE(float3 p,float t){float g,s;return pmField(p,t,g,s);}
static float3 pmNormal(float3 p,float t){float e=.0015f;return normalize(float3(
 pmDE(p+float3(e,0,0),t)-pmDE(p-float3(e,0,0),t),
 pmDE(p+float3(0,e,0),t)-pmDE(p-float3(0,e,0),t),
 pmDE(p+float3(0,0,e),t)-pmDE(p-float3(0,0,e),t)));}


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

fragment float4 dynamicBoxFragment(
 DynamicBoxVertexOut in [[stage_in]], constant DynamicBoxUniforms &u [[buffer(0)]],
 constant float4x4 *v2w [[buffer(1)]], constant float4x4 *vp [[buffer(2)]]) {
 uint vi=min(in.viewIndex,u.viewCount-1u); float3 cam=v2w[vi][3].xyz;
 float3 ro=(cam-u.objectCenter.xyz)/u.boxScale, rd=normalize(in.worldPos-cam);
 if(!all(abs(ro)<DB_BOXDIMS-1e-3f)){float3 bn;float en=db_boxHit(ro,rd,DB_BOXDIMS,bn,true);if(en<0)return float4(.008,.012,.018,1);ro+=rd*(en+.001f);}
 ro=(u.patternTransform*float4(ro,1)).xyz; rd=normalize(float3(u.patternTransform*float4(rd,0)));
 float dist, foldFar;
 if (!foldInterval(ro,rd,dist,foldFar)) return float4(.004f,.007f,.012f,1.0f);
 float t=u.time;
 for(int i=0;i<120;i++){float3 p=ro+rd*dist;float groove,shell;float d=pmField(p,t,groove,shell);
  if(d<.0009f){float3 n=pmNormal(p,t),l=normalize(float3(-.45,.78,.43));float dif=max(dot(n,l),0.0f);float wrap=.5f+.5f*dot(n,l);float rim=pow(1.0f-max(dot(n,-rd),0.0f),3.0f);
   float valley=pow(1.0f-groove,2.0f);float3 ivory=float3(.92,.88,.80),cyan=float3(.08,.45,.58),peach=float3(.95,.43,.25);
   float3 base=mix(ivory,cyan,smoothstep(-.42,.30,p.z+p.x*.35));base=mix(base,peach,smoothstep(.25,.62,p.y-p.x*.25)*.38f);
   base*=1.0f-valley*.42f;float spec=pow(max(dot(reflect(-l,n),-rd),0.0f),42.0f);
  float3 col=base*(.18f+.82f*dif+.18f*wrap)*.82f+float3(1.0,.92,.82)*spec*.55f+float3(.18,.55,.72)*rim*.30f;
   return float4(col,1);}
  dist+=clamp(d*.72f,.0012f,.06f);if(dist>foldFar)break;}
 float vignette=exp(-.14f*dist);return float4(float3(.006,.011,.017)*vignette,1);
}
