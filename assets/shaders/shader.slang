// Column-major 4×4 float matrix — matches vek's Mat4<f32> layout.
typedef column_major float4x4 VekMat4;

// Projection matrix only — changes on window resize.
// Uploaded once per frame via the per-frame UBO buffer.
struct Ubo
{
    VekMat4 projection;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Ubo> ubo;

[[vk::binding(1, 0)]]
Sampler2D tex;

struct PushConstants
{
    VekMat4 model_view;
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
    // Transform: model-space -> view-space -> clip-space.
    // model_view is a push constant (cheap per-draw update);
    // projection comes from the UBO (updated on resize).
    output.sv_position = mul(
        ubo.projection,
        mul(push.model_view, float4(input.position, 1.0)));
    output.tex_coord = input.tex_coord;
    return output;
}

[shader("fragment")]
float4 frag_main(VertexOutput in_vert) : SV_Target
{
    return tex.Sample(in_vert.tex_coord);
}
