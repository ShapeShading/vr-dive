// voronoi-cellular-foam.metal — 3D cellular membranes and struts
//
// F2-F1 forms thin Voronoi membranes. The simultaneous F2-F1/F3-F2
// near-zero set reinforces triple-cell edges into a delicate foam cage.
// This is fully 3D, unlike the 2D Worley texture used by Galaxy.

static float2 vcfRotate(float2 v,float a){
    float s=sin(a),c=cos(a);return float2(c*v.x-s*v.y,s*v.x+c*v.y);
}
static float3 vcfHash33(float3 p){
    // Arithmetic hash: substantially cheaper than three transcendental hashes
    // for every candidate at every ray-march step.
    p=fract(p*float3(.1031f,.1030f,.0973f));
    p+=dot(p,p.yxz+33.33f);
    return fract((p.xxy+p.yxx)*p.zyx);
}
static float3 vcfObjectPoint(float3 p,float t){
    p.xz=vcfRotate(p.xz,t*.055f);
    p.xy=vcfRotate(p.xy,.18f*sin(t*.09f));
    return p+.035f*sin(p.zxy*5.f+t*.16f);
}
static float2 vcfEdges(float3 p,float t,thread float3& nearestCell){
    const float scale=4.0f;
    float3 sample=vcfObjectPoint(p,t)*scale;
    // Feature points stay inside the middle 56% of each cell. Centering the
    // search on sample-.5 means the nearest candidates lie in this 2x2x2
    // neighborhood, reducing 27 evaluations to eight.
    float3 base=floor(sample-.5f);
    float first=1e6f,second=1e6f,third=1e6f;
    nearestCell=base;
    for(int z=0;z<=1;++z)for(int y=0;y<=1;++y)for(int x=0;x<=1;++x){
        float3 neighbor=float3(float(x),float(y),float(z));
        float3 id=base+neighbor, rnd=vcfHash33(id);
        float3 feature=id+.5f+.28f*(rnd*2.f-1.f);
        float3 offset=feature-sample;
        float d=dot(offset,offset);
        if(d<first){third=second;second=first;first=d;nearestCell=id;}
        else if(d<second){third=second;second=d;}
        else if(d<third){third=d;}
    }
    float d1=sqrt(max(first,0.f)),d2=sqrt(max(second,first));
    float d3=sqrt(max(third,second));
    return float2((d2-d1)*.5f,(d3-d2)*.5f)/scale;
}
static float vcfField(float3 p,float t,thread float2& edges,thread float3& cell){
    edges=vcfEdges(p,t,cell);
    float membrane=edges.x-.0032f;
    float strut=max(edges.x,edges.y)-.0115f;
    float structure=min(membrane,strut),r=length(p);
    return max(max(structure,r-.79f),.16f-r);
}
static float vcfDistance(float3 p,float t){
    float2 edges;float3 cell;return vcfField(p,t,edges,cell);
}
static float3 vcfNormal(float3 p,float t){
    const float2 k=float2(1.f,-1.f);float e=.0018f;
    float3 gradient=
        k.xyy*vcfDistance(p+k.xyy*e,t)+
        k.yyx*vcfDistance(p+k.yyx*e,t)+
        k.yxy*vcfDistance(p+k.yxy*e,t)+
        k.xxx*vcfDistance(p+k.xxx*e,t);
    float gradient2=dot(gradient,gradient);
    return gradient2>1e-10f?gradient*rsqrt(gradient2):normalize(p);
}
static float3 vcfColor(float3 cell){
    return mix(float3(.08f,.28f,.42f),float3(.95f,.48f,.18f),
               vcfHash33(cell+19.17f));
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
    float3 ray=normalize(in.worldPos-camera),bg=float3(.002f,.006f,.010f);
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
    for(int step=0;step<64;++step){
        float3 p=ro+rd*travel,cell;float2 edges;
        float d=vcfField(p,time,edges,cell);
        closest=min(closest,abs(d));float eps=.0009f+travel*.00012f;
        if(d<eps){
            float3 n=vcfNormal(p,time),light=normalize(float3(-.38f,.86f,.34f));
            float dif=max(dot(n,light),0.f);
            float rim=pow(1.f-max(dot(n,-rd),0.f),2.6f);
            float spec=pow(max(dot(reflect(-light,n),-rd),0.f),44.f);
            float weight=1.f-smoothstep(.001f,.012f,max(edges.x,edges.y));
            float3 base=mix(vcfColor(cell),float3(.92f,.78f,.48f),weight*.65f);
            float3 color=base*(.20f+dif)+rim*float3(.16f,.48f,.78f)*.55f
                +spec*float3(1.f,.90f,.68f);
            return float4(color*exp(-travel*.11f),1);
        }
        travel+=clamp(max(d,eps)*.76f,.0009f,.052f);
        if(travel>exitDistance+.01f)break;
    }
    float glow=exp(-closest*95.f)*.035f;
    return float4(bg+glow*float3(.10f,.34f,.55f),1);
}
