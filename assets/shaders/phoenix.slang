// Column-major 4×4 float matrix — matches vek's Mat4<f32> layout.
typedef column_major float4x4 VekMat4;

// Sampler table constants and helper
// Layout: sampler_index = filter_index * OOB_COUNT + addr_index
static const uint SAMPLER_OOB_COUNT = 5u;

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

// Set 0 — camera: combined view-projection matrix, updated per frame, sampler
// state, updates when anisotropy settings are updated.
struct CameraUbo
{
    VekMat4 view_proj;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CameraUbo> camera;

// Set 0 — sampler table: runtime-provided immutable sampler objects.
[[vk::binding(1, 0)]]
SamplerState samplers[];

// Set 1 — material: bindless texture array (binding 0, unbounded,
// PARTIALLY_BOUND) and material SSBO (binding 1).
[[vk::binding(0, 1)]]
Texture2D textures[];

struct MaterialGpu
{
    uint albedo_idx;
    uint normal_idx;
    uint orm_idx;
    int emissive_idx; // negative = no emissive texture
};

[[vk::binding(1, 1)]]
StructuredBuffer<MaterialGpu> materials;

struct PushConstants
{
    VekMat4 model;
    uint material_idx;
};

[[vk::push_constant]]
PushConstants push;

struct VertexInput
{
    [[vk::location(0)]]
    float3 position;
    [[vk::location(1)]]
    float2 tex_coord;
};

struct VertexOutput
{
    [[vk::location(0)]]
    float2 tex_coord;
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexOutput vert_main(VertexInput input)
{
    VertexOutput output;
    // Transform: model-space -> world-space -> clip-space.
    // model is a push constant (cheap per-draw update);
    // view_proj comes from the UBO (updated each frame).
    output.sv_position = mul(
        camera.view_proj,
        mul(push.model, float4(input.position, 1.0)));
    output.tex_coord = input.tex_coord;
    return output;
}

[shader("fragment")]
float4 frag_main(VertexOutput in_vert)
    : SV_Target
{

    MaterialGpu mat = materials[push.material_idx];
    // Use the default sampler (Anisotropic + Repeat) from the sampler table.
    let sample = textures[mat.albedo_idx].Sample(
        samplers[getSamplerIndex(
            SamplerFilter::FILTER_ANISOTROPIC,
            SamplerAddress::ADDR_REPEAT)],
        in_vert.tex_coord, );
    return sample;
}
