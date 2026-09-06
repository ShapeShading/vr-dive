// Voxel tide — actual procedural voxels traversed with 3D DDA, not a line mesh.
// Each occupied grid cell contains a growing/shrinking solid box. Occupancy,
// pore shape and cell size evolve continuously with time, without whole-object spin.
// Traversal background: https://www.cs.yorku.ca/~amana/research/
// Runtime-only Metal: types and standard library come from DynamicBox's wrapper.
// The eye ray is not clipped to the Box entry/exit; only the raster aperture is.
static bool matterRay(DynamicBoxVertexOut in, constant DynamicBoxUniforms &u,
                      constant float4x4 *v2w, thread float3 &ro, thread float3 &rd) {
    uint vi=min(in.viewIndex,max(u.viewCount,1u)-1u);
    float3 eye=v2w[vi][3].xyz;
    ro=(eye-u.objectCenter.xyz)/max(u.boxScale,0.0001f);
    rd=normalize(in.worldPos-eye);
    if (!all(abs(ro)<DB_BOXDIMS-0.001f)) {
        float3 normal;
        if (db_boxHit(ro,rd,DB_BOXDIMS,normal,true)<0.0f) return false;
    }
    ro=(u.patternTransform*float4(ro,1)).xyz;
    rd=normalize((u.patternTransform*float4(rd,0)).xyz);
    return true;
}
static bool matterBound(float3 ro, float3 rd, float radius, thread float2 &span) {
    float b=dot(ro,rd), h=b*b-dot(ro,ro)+radius*radius;
    if (h<0.0f) return false;
    span=float2(max(0.0f,-b-sqrt(h)),-b+sqrt(h));
    return span.y>span.x;
}
static float3 matterShade(float3 normal, float3 rd, float3 base,
                          float occlusion, float translucency) {
    float3 n=dot(normal,rd)>0.0f ? -normal : normal;
    float3 light=normalize(float3(-0.55f,0.85f,0.65f));
    float diffuse=max(dot(n,light),0.0f);
    float back=pow(max(dot(-n,light),0.0f),2.0f)*translucency;
    float spec=pow(max(dot(n,normalize(light-rd)),0.0f),54.0f);
    float fresnel=pow(1.0f-max(dot(n,-rd),0.0f),4.0f);
    float3 color=base*(0.19f+0.95f*diffuse)*occlusion;
    color+=float3(1.0f,0.58f,0.30f)*back*0.32f;
    color+=float3(0.24f,0.52f,0.66f)*fresnel*0.25f;
    color+=float3(0.90f,0.92f,0.84f)*spec*0.5f;
    return color/(1.0f+0.18f*color);
}

static constant float VOXEL_SIZE=0.044f;
static float2 voxelField(float3 p, float time) {
    float envelope=1.0f-length(p/float3(0.93f,0.81f,0.86f))
                   +0.12f*sin(p.x*3.1f-time*0.18f)*sin(p.z*4.0f)*cos(p.y*3.0f);
    if (envelope < -0.04f) return float2(0);
    float3 q=p+0.065f*sin(p.yzx*4.0f+float3(0.25f,-0.19f,0.21f)*time);
    float frequency=8.3f+0.65f*sin(time*0.17f);
    q=q*frequency+float3(0.22f,-0.16f,0.13f)*time;
    float gyroid=dot(sin(q),cos(q.yzx));
    float thickness=0.57f+0.09f*sin(time*0.31f+p.y*4.0f);
    // An open, moving chamber reveals the interior voxel walls from a distance.
    float3 chamber=(p-float3(0.12f*sin(time*0.21f),0.03f,0.53f))/float3(1,0.9f,1);
    float cavity=length(chamber)-0.42f;
    float solid=min(min(thickness-abs(gyroid),envelope*5.0f),cavity*5.0f);
    return float2(smoothstep(-0.10f,0.23f,solid),gyroid);
}
static bool voxelBox(float3 ro,float3 rd,float3 center,float halfSize,
                     thread float &distance,thread float3 &normal) {
    bool3 parallel=abs(rd)<0.0000001f;
    if (any(parallel && (abs(ro-center)>halfSize))) return false;
    float3 inverse=1.0f/select(rd,float3(0.0000001f),abs(rd)<0.0000001f);
    float3 a=(center-halfSize-ro)*inverse, b=(center+halfSize-ro)*inverse;
    float3 nearV=min(a,b), farV=max(a,b);
    nearV=select(nearV,float3(-1.0e20f),parallel);
    farV=select(farV,float3(1.0e20f),parallel);
    float nearT=max(nearV.x,max(nearV.y,nearV.z)), farT=min(farV.x,min(farV.y,farV.z));
    if (nearT>farT || farT<0.0f) return false;
    bool inside=nearT<0.0f;
    distance=inside?farT:nearT;
    float3 face=inside?farV:nearV;
    // Deterministic tie-breaking at voxel edges gives a stable face normal.
    int axis=inside?(face.x<=face.y && face.x<=face.z?0:(face.y<=face.z?1:2)):
                    (face.x>=face.y && face.x>=face.z?0:(face.y>=face.z?1:2));
    normal=float3(0); normal[axis]=(inside?1.0f:-1.0f)*sign(rd[axis]);
    return true;
}

fragment float4 dynamicBoxFragment(
    DynamicBoxVertexOut in [[stage_in]],
    constant DynamicBoxUniforms &u [[buffer(0)]],
    constant float4x4 *v2w [[buffer(1)]],
    constant float4x4 *vp [[buffer(2)]]) {
    const float3 background=float3(0.003f,0.005f,0.010f);
    float3 ro,rd;
    if (!matterRay(in,u,v2w,ro,rd)) return float4(background,1);

    float2 span;
    if (!matterBound(ro,rd,1.17f,span)) return float4(background,1);
    float travel=span.x+0.00001f;
    int3 cell=int3(floor((ro+rd*travel)/VOXEL_SIZE));
    int3 direction=int3(sign(rd));
    float3 safeAbs=max(abs(rd),float3(0.0000001f));
    float3 delta=VOXEL_SIZE/safeAbs;
    float3 cellBoundary=(float3(cell)+select(float3(0),float3(1),rd>0.0f))*VOXEL_SIZE;
    float3 next=(cellBoundary-ro)/select(rd,float3(0.0000001f),abs(rd)<0.0000001f);
    next=select(next,float3(1.0e20f),abs(rd)<0.0000001f);
    // 104 covers diameter/cellSize times sqrt(3), plus boundary cells.
    for (int step=0; step<104 && travel<span.y; ++step) {
        float exitT=min(next.x,min(next.y,next.z));
        float3 center=(float3(cell)+0.5f)*VOXEL_SIZE;
        float2 sample=voxelField(center,u.time);
        if (sample.x>0.025f) {
            float halfSize=VOXEL_SIZE*0.485f*sqrt(sample.x);
            float hitT; float3 n;
            if (voxelBox(ro,rd,center,halfSize,hitT,n) && hitT>=travel-0.0001f && hitT<=exitT+0.0001f) {
                float3 p=ro+rd*hitT;
                float3 local=(p-center)/max(halfSize,0.0001f);
                // Mild bevel shading; occupancy and silhouette remain exact boxes.
                float3 shadeNormal=normalize(n+0.12f*local);
                float occ=0.0f;
                occ+=voxelField(center+n*VOXEL_SIZE,u.time).x;
                occ+=voxelField(center+n*VOXEL_SIZE+float3(0,VOXEL_SIZE,0),u.time).x*0.5f;
                occ=clamp(1.0f-occ*0.3f,0.5f,1.0f);
                float vein=0.5f+0.5f*sin(p.y*5.0f+p.z*3.0f+sample.y*1.5f);
                float3 base=mix(float3(0.025f,0.23f,0.29f),float3(0.18f,0.70f,0.54f),vein);
                float gold=smoothstep(0.10f,0.50f,sample.y);
                base=mix(base,float3(0.86f,0.50f,0.20f),gold);
                base*=0.70f+0.30f*sample.x;
                return float4(matterShade(shadeNormal,rd,base,occ,0.0f),1);
            }
        }
        // Advance every tied axis. Parallel rays never step their zero axis.
        bool3 crossed=next<=exitT+0.000001f;
        cell+=select(int3(0),direction,crossed);
        next+=select(float3(0),delta,crossed);
        travel=exitT;
    }
    return float4(background,1);
}
