// Column-major 4×4 float matrix — matches vek's Mat4<f32> layout.
typedef column_major float4x4 VekMat4;

// Sampler table constants and helper
// Layout: sampler_index = filter_index * OOB_COUNT + addr_index
static const uint SAMPLER_OOB_COUNT = 5u;

static const float PI = 3.14159265;

enum SamplerFilter
{
    FILTER_NEAREST = 0u,
    FILTER_LINEAR = 1u,
    FILTER_ANISOTROPIC = 2u,
};

enum SamplerAddress
{
    ADDR_REPEAT = 0u,
    ADDR_CLAMP_TO_EDGE = 1u,
    ADDR_MIRROR = 2u,
    ADDR_TRANSPARENT_BORDER = 3u,
    ADDR_CLAMP_TO_BORDER_WHITE = 4u,
};

uint getSamplerIndex(SamplerFilter f, SamplerAddress a)
{
    return uint(f) * SAMPLER_OOB_COUNT + uint(a);
}

// ── Set 0: per-frame ──────────────────────────────────────────────────────────

// TODO (PSIR): FrameUbo and PushConstants are temporary scaffolds.
// PSIR will absorb UBOs and push constants into a unified constant
// interface.
struct FrameUbo
{
    VekMat4 view_proj;
    float4 ambient;   // rgb = color, a = intensity
    uint light_count; // active entries in lights[]
    float3 _pad;
};

[[vk::binding(0, 0)]]
ConstantBuffer<FrameUbo> frame;

[[vk::binding(1, 0)]]
SamplerState samplers[];

// Light types stored in GpuLight::position_type.w.
static const float LIGHT_TYPE_POINT       = 0.0;
static const float LIGHT_TYPE_DIRECTIONAL = 1.0;
static const float LIGHT_TYPE_SPOT        = 2.0;

// GpuLight packs each float3+scalar pair into a float4 to avoid
// std430 vec3 alignment surprises.
struct GpuLight
{
    // xyz = world pos, w = type (0=point, 1=dir, 2=spot)
    float4 position_type;
    // xyz = color*intensity, w = falloff range
    float4 color_range;
    // xyz = normalized direction (dir/spot), w = inner cone angle
    float4 direction_inner;
    // x = outer cone angle, yz = oval angle_scale, w = unused
    float4 outer_scale_pad;
};

[[vk::binding(2, 0)]]
StructuredBuffer<GpuLight> lights;

// ── Set 1: material ───────────────────────────────────────────────────────────

[[vk::binding(0, 1)]]
Texture2D textures[];

static const uint MAT_HAS_EMISSIVE = 0x1u;

struct MaterialHot
{
    uint albedo_idx;
    uint normal_idx;
    uint orm_idx;
    uint flags;
    float4 base_color;
    float metallic_factor;
    float roughness_factor;
    float2 _pad;
};

[[vk::binding(1, 1)]]
StructuredBuffer<MaterialHot> materials_hot;

struct MaterialCold
{
    uint emissive_idx;
    uint3 _pad;
};

[[vk::binding(2, 1)]]
StructuredBuffer<MaterialCold> materials_cold;

// ── Push constants ────────────────────────────────────────────────────────────

struct PushConstants
{
    VekMat4 model;
    uint material_idx;
};

[[vk::push_constant]]
PushConstants push;

// ── Vertex input / output ─────────────────────────────────────────────────────

struct VertexInput
{
    [[vk::location(0)]]
    float3 position;
    [[vk::location(1)]]
    float2 tex_coord;
};

// PBR vertex: location order matches VertexPbr struct offsets.
struct VertexInputPbr
{
    [[vk::location(0)]]
    float3 position;
    [[vk::location(2)]]
    float3 norm;
    [[vk::location(3)]]
    float4 tan; // w = handedness ±1
    [[vk::location(1)]]
    float2 tex_coord;
};

struct VertexOutput
{
    [[vk::location(0)]]
    float2 tex_coord;
    float4 sv_position : SV_Position;
};

struct VertexOutputPbr
{
    float4 sv_pos    : SV_Position;
    float3 world_pos : TEXCOORD0;
    float3 world_n   : TEXCOORD1;
    float3 world_t   : TEXCOORD2;
    float3 world_bt  : TEXCOORD3;
    float2 uv        : TEXCOORD4;
};

// ── Non-PBR entry points (kept for reference) ─────────────────────────────────

[shader("vertex")]
VertexOutput vert_main(VertexInput input)
{
    VertexOutput output;
    output.sv_position = mul(
        frame.view_proj,
        mul(push.model, float4(input.position, 1.0)));
    output.tex_coord = input.tex_coord;
    return output;
}

[shader("fragment")]
float4 frag_main(VertexOutput in_vert)
    : SV_Target
{
    MaterialHot mat = materials_hot[push.material_idx];
    let samp = samplers[getSamplerIndex(
        SamplerFilter::FILTER_ANISOTROPIC,
        SamplerAddress::ADDR_REPEAT)];
    return textures[mat.albedo_idx].Sample(samp, in_vert.tex_coord);
}

// ── PBR math helpers ──────────────────────────────────────────────────────────

// GGX normal distribution function.
float ggx_ndf(float ndoth, float rough)
{
    float a  = rough * rough;
    float a2 = a * a;
    float d  = ndoth * ndoth * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Smith G1 (Schlick-GGX).
float smith_g1(float ndotv, float rough)
{
    float r = rough + 1.0;
    float k = (r * r) / 8.0;
    return ndotv / (ndotv * (1.0 - k) + k);
}

float smith_g(float ndotv, float ndotl, float rough)
{
    return smith_g1(ndotv, rough) * smith_g1(ndotl, rough);
}

// Fresnel-Schlick.
float3 fresnel_schlick(float cos_theta, float3 f0)
{
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

// ACES filmic tonemapping approximation.
float3 aces_tonemap(float3 x)
{
    float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// ── PBR surface ───────────────────────────────────────────────────────────────

struct PbrSurface
{
    float3 albedo;
    float3 N;        // world-space shading normal
    float  occlusion;
    float  roughness;
    float  metallic;
    float3 f0;       // Fresnel base reflectivity
    float3 emissive; // zero if MAT_HAS_EMISSIVE is unset
};

// Reconstruct a tangent-space normal from a BC5 RG sample and
// transform it to world space via TBN.
float3 reconstruct_world_normal(float2 rg, float3x3 tbn)
{
    float2 xy = rg * 2.0 - 1.0;
    float  z  = sqrt(max(0.0, 1.0 - dot(xy, xy)));
    return normalize(mul(transpose(tbn), float3(xy, z)));
}

// Sample all material textures for a surface hit. MaterialCold is
// only fetched when MAT_HAS_EMISSIVE is set.
PbrSurface sample_pbr_surface(
    MaterialHot  mat,
    SamplerState samp,
    float2       uv,
    float3x3     tbn,
    uint         material_idx)
{
    float3 albedo = textures[mat.albedo_idx].Sample(samp, uv).rgb
        * mat.base_color.rgb;

    float3 N = reconstruct_world_normal(
        textures[mat.normal_idx].Sample(samp, uv).rg, tbn);

    float3 orm       = textures[mat.orm_idx].Sample(samp, uv).rgb;
    float  occlusion = orm.r;
    float  roughness = saturate(orm.g * mat.roughness_factor);
    float  metallic  = saturate(orm.b * mat.metallic_factor);

    float3 f0 = lerp((float3)0.04, albedo, metallic);

    float3 emissive = (float3)0;
    if ((mat.flags & MAT_HAS_EMISSIVE) != 0u)
    {
        MaterialCold cold = materials_cold[material_idx];
        emissive = textures[cold.emissive_idx].Sample(samp, uv).rgb;
    }

    PbrSurface surf;
    surf.albedo    = albedo;
    surf.N         = N;
    surf.occlusion = occlusion;
    surf.roughness = roughness;
    surf.metallic  = metallic;
    surf.f0        = f0;
    surf.emissive  = emissive;
    return surf;
}

// ── Light incidence ───────────────────────────────────────────────────────────

struct LightIncidence
{
    float3 L;     // unit direction toward the light
    float  atten; // combined distance + cone attenuation
};

// Compute the incident direction and attenuation for a single GpuLight
// at the given world-space surface position and TBN axes.
LightIncidence eval_light_incidence(
    GpuLight light,
    float3   world_pos,
    float3   world_t,
    float3   world_bt)
{
    LightIncidence inc;

    if (light.position_type.w == LIGHT_TYPE_DIRECTIONAL)
    {
        inc.L     = normalize(light.direction_inner.xyz);
        inc.atten = 1.0;
        return inc;
    }

    float3 L_unnorm = light.position_type.xyz - world_pos;
    float  dist     = length(L_unnorm);
    inc.L = L_unnorm / dist;

    // Frostbite windowed inverse-square falloff.
    float range = light.color_range.w;
    float x     = dist / range;
    float wind  = saturate(1.0 - x * x * x * x);
    inc.atten   = (wind * wind) / (dist * dist + 1.0);

    if (light.position_type.w == LIGHT_TYPE_SPOT)
    {
        float  inner = light.direction_inner.w;
        float  outer = light.outer_scale_pad.x;
        float2 scale = light.outer_scale_pad.yz;

        // Project -L onto surface TBN axes and scale for oval cones.
        float2 spot_uv    = float2(dot(-inc.L, normalize(world_t)),
                                   dot(-inc.L, normalize(world_bt)));
        float  oval_angle = length(spot_uv / scale);
        inc.atten *= smoothstep(outer, inner, oval_angle);
    }

    return inc;
}

// ── Cook-Torrance BRDF ────────────────────────────────────────────────────────

// Evaluate the full Cook-Torrance BRDF (diffuse + specular) for one
// light direction. Returns the pre-multiplied radiance contribution;
// caller multiplies by light color and attenuation.
float3 eval_pbr_brdf(PbrSurface surf, float3 V, float3 L)
{
    float ndotl = saturate(dot(surf.N, L));
    if (ndotl <= 0.0)
        return (float3)0;

    float3 H     = normalize(V + L);
    float  ndotv = saturate(dot(surf.N, V));
    float  ndoth = saturate(dot(surf.N, H));
    float  vdoth = saturate(dot(V, H));

    float  D = ggx_ndf(ndoth, surf.roughness);
    float  G = smith_g(ndotv, ndotl, surf.roughness);
    float3 F = fresnel_schlick(vdoth, surf.f0);

    float3 spec = (D * G * F) / max(4.0 * ndotv * ndotl, 0.001);
    float3 kd   = (1.0 - F) * (1.0 - surf.metallic);
    float3 diff = kd * surf.albedo / PI;

    return (diff + spec) * ndotl;
}

// ── PBR vertex shader ─────────────────────────────────────────────────────────

[shader("vertex")]
VertexOutputPbr vert_pbr(VertexInputPbr v)
{
    float4   world = mul(push.model, float4(v.position, 1.0));
    float3x3 m3    = (float3x3)push.model;
    float3   n     = normalize(mul(m3, v.norm));
    float3   t     = normalize(mul(m3, v.tan.xyz));
    float3   bt    = cross(n, t) * v.tan.w;

    VertexOutputPbr o;
    o.sv_pos    = mul(frame.view_proj, world);
    o.world_pos = world.xyz;
    o.world_n   = n;
    o.world_t   = t;
    o.world_bt  = bt;
    o.uv        = v.tex_coord;
    return o;
}

// ── PBR fragment shader ───────────────────────────────────────────────────────

[shader("fragment")]
float4 frag_pbr(VertexOutputPbr frag) : SV_Target
{
    let samp = samplers[getSamplerIndex(
        SamplerFilter::FILTER_ANISOTROPIC,
        SamplerAddress::ADDR_REPEAT)];

    float3x3 tbn = float3x3(
        normalize(frag.world_t),
        normalize(frag.world_bt),
        normalize(frag.world_n));

    MaterialHot mat  = materials_hot[push.material_idx];
    PbrSurface  surf = sample_pbr_surface(
        mat, samp, frag.uv, tbn, push.material_idx);

    // TODO: pass camera world position in FrameUbo.
    float3 V = normalize(float3(0.0, 10.0, 5.0) - frag.world_pos);

    float3 Lo = (float3)0;
    for (uint i = 0u; i < frame.light_count; i++)
    {
        GpuLight       light = lights[i];
        LightIncidence inc   = eval_light_incidence(
            light, frag.world_pos, frag.world_t, frag.world_bt);

        Lo += eval_pbr_brdf(surf, V, inc.L)
            * light.color_range.xyz * inc.atten;
    }

    float3 ambient = frame.ambient.rgb * frame.ambient.a
        * surf.albedo * surf.occlusion;

    float3 color = ambient + Lo + surf.emissive;

    color = aces_tonemap(color);
    return float4(color, 1.0);
}
