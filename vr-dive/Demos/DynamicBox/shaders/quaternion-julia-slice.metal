// quaternion-julia-slice.metal - Animated 4D Quaternion Julia Slices
//
// This is not a second parameter-drifting Julia shader. It keeps a stable
// quaternion Julia set and moves a 3D observation slice through its fourth
// coordinate w, with a small XW/YW rotation before evaluation. The resulting
// growth, pinching, and splitting are different cross-sections of one 4D
// object. It reuses the proven low-risk escape-time DE cost of quaternion-julia
// rather than adding another high-iteration fractal family.
// Both shaders are intentionally retained: quaternion-julia animates the
// parameter c itself, while this one presents genuine changes of viewpoint
// through a stable four-dimensional object.
//
// ─── 设计方案 ────────────────────────────────────────────────────────────
// 思路: 对 ray-march 点 p 构造 q=(x,y,z,wSlice)，wSlice=0.32*sin(t*0.09)，
//       再在 XW/YW 两个 4D 平面做很小的旋转，最后执行固定 c 的四元数
//       Julia z→z²+c DE。与 quaternion-julia.metal 的 c 漂移不同，此处
//       改变的是观察 4D 形体的三维截面，因此局部会真实地出现/消失、合并
//       和分裂。迭代次数、march 上限沿用已验证设置，避免性能回退。
// 关键参数: wSlice 振幅 0.32、切片频率 0.09rad/s、11 次 DE 迭代、90 步 march。
// 性能特征: 与 quaternion-julia.metal 同为 11 次四元数 DE；仅额外 2 个
//           4D 平面旋转（少量 sin/cos），不增加 DE 循环或法线采样次数。
// 已知限制: 若性能日志表明该分形接近 100ms，应先将 DE 回退到 9-10 次，
//           不应提高切片频率或继续加迭代。

static float4 qmul(float4 a, float4 b) {
    return float4(a.x*b.x-a.y*b.y-a.z*b.z-a.w*b.w,
                  a.x*b.y+a.y*b.x+a.z*b.w-a.w*b.z,
                  a.x*b.z-a.y*b.w+a.z*b.x+a.w*b.y,
                  a.x*b.w+a.y*b.z-a.z*b.y+a.w*b.x);
}
static float4 qsquare(float4 q) {
    return float4(q.x*q.x-q.y*q.y-q.z*q.z-q.w*q.w,
                  2.0f*q.x*q.y, 2.0f*q.x*q.z, 2.0f*q.x*q.w);
}
static float4 rotateXW(float4 q, float angle) {
    float s = sin(angle), c = cos(angle);
    return float4(q.x*c-q.w*s, q.y, q.z, q.x*s+q.w*c);
}
static float4 rotateYW(float4 q, float angle) {
    float s = sin(angle), c = cos(angle);
    return float4(q.x, q.y*c-q.w*s, q.z, q.y*s+q.w*c);
}
static float juliaSliceDE(float3 p, float time) {
    float wSlice = 0.32f * sin(time * 0.09f);
    float4 z = float4(p, wSlice);
    z = rotateXW(z, 0.19f * sin(time * 0.07f));
    z = rotateYW(z, 0.14f * cos(time * 0.11f));
    float4 dz = float4(1.0f, 0.0f, 0.0f, 0.0f);
    const float4 c = float4(-0.2f, 0.55f, 0.2f, 0.15f);
    float m2 = dot(z, z);
    for (int i = 0; i < 11; i++) {
        if (m2 > 16.0f) break;
        dz = 2.0f * qmul(z, dz);
        z = qsquare(z) + c;
        m2 = dot(z, z);
    }
    float radius = sqrt(m2);
    return 0.5f * radius * log(max(radius, 1e-6f)) / max(length(dz), 1e-6f);
}
static float3 calcNormal(float3 p, float time) {
    float2 e = float2(0.0012f, 0.0f);
    return normalize(float3(juliaSliceDE(p+e.xyy,time)-juliaSliceDE(p-e.xyy,time),
                            juliaSliceDE(p+e.yxy,time)-juliaSliceDE(p-e.yxy,time),
                            juliaSliceDE(p+e.yyx,time)-juliaSliceDE(p-e.yyx,time)));
}
static float3 palette(float t) {
    return float3(0.35f,0.42f,0.55f)+float3(0.50f,0.46f,0.40f)*cos(6.28318f*(float3(0.8f,0.95f,0.65f)*t+float3(0.12f,0.31f,0.52f)));
}
fragment float4 dynamicBoxFragment(DynamicBoxVertexOut in [[stage_in]], constant DynamicBoxUniforms &uniforms [[buffer(0)]], constant float4x4 *v2wMats [[buffer(1)]], constant float4x4 *vpMatrices [[buffer(2)]]) {
    uint vi = min(in.viewIndex, uniforms.viewCount-1u);
    float3 camWorld = float3(v2wMats[vi][3].x,v2wMats[vi][3].y,v2wMats[vi][3].z);
    float3 boxEye = (camWorld-uniforms.objectCenter.xyz)/uniforms.boxScale;
    float3 boxRd = normalize(in.worldPos-camWorld);
    float3 bg = float3(0.005f,0.008f,0.025f), origin;
    if (!all(abs(boxEye)<(DB_BOXDIMS-1e-3f))) { float3 entryN; float entry=db_boxHit(boxEye,boxRd,DB_BOXDIMS,entryN,true); if(entry<0.0f)return float4(bg,1.0f); origin=boxEye+boxRd*(entry+1e-3f); } else origin=boxEye;
    float3 exitN; if(db_boxHit(origin,boxRd,DB_BOXDIMS,exitN,false)<=0.0f) discard_fragment();
    float3 ro=(uniforms.patternTransform*float4(origin,1.0f)).xyz;
    float3 rd=normalize(float3(uniforms.patternTransform*float4(boxRd,0.0f)));
    float time=uniforms.time, march=0.0f;
    for(int i=0;i<90;i++) { float3 p=ro+rd*march; float d=juliaSliceDE(p,time); if(d<0.0018f) { float3 n=calcNormal(p,time); float dif=max(dot(n,normalize(float3(0.35f,0.85f,0.45f))),0.0f); float rim=pow(1.0f-max(dot(-rd,n),0.0f),3.0f); float3 col=palette(length(p)*0.65f+time*0.025f+n.y*0.18f)*(0.28f+dif*1.05f)+float3(0.35f,0.7f,1.0f)*rim*0.65f; return float4(col*exp(-march*0.25f),1.0f); } march+=max(d*0.85f,0.0015f); if(march>25.0f)break; }
    return float4(bg,1.0f);
}
