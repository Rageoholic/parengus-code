// Column-major 4×4 float matrix — matches vek's Mat4<f32> layout.
typedef column_major float4x4 VekMat4;

// Set 0 — camera: combined view-projection matrix, updated per frame.
struct CameraUbo
{
    VekMat4 view_proj;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CameraUbo> camera;

// Set 1 — material: bindless texture array (binding 0, unbounded,
// PARTIALLY_BOUND) and material SSBO (binding 1).
[[vk::binding(0, 1)]]
Sampler2D textures[];

struct MaterialGpu
{
    uint albedo_idx;
    uint normal_idx;
    uint orm_idx;
    int  emissive_idx; // negative = no emissive texture
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
float4 frag_main(VertexOutput in_vert) : SV_Target
{
    // Look up this submesh's material and sample its albedo texture.
    // Normal, ORM, and emissive channels are wired up here but shading
    // is deferred to t023 (Blinn-Phong shader path).
    MaterialGpu mat = materials[push.material_idx];
    return textures[mat.albedo_idx].Sample(in_vert.tex_coord);
}
