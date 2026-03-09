// Column-major 4×4 float matrix — matches vek's Mat4<f32> layout.
typedef column_major float4x4 VekMat4;

// Set 0 — camera: combined view-projection matrix, updated per frame.
struct CameraUbo
{
    VekMat4 view_proj;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CameraUbo> camera;

// Set 1 — material: texture sampler, bound once at load time.
[[vk::binding(0, 1)]]
Sampler2D tex;

struct PushConstants
{
    VekMat4 model;
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
    return tex.Sample(in_vert.tex_coord);
}
