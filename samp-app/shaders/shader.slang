struct Ubo
{
    column_major float4x4 model;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Ubo> ubo;

[[vk::binding(1, 0)]]
Sampler2D tex;

struct VertexInput
{
    [[vk::location(0)]]
    float2 position;
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
    output.sv_position = mul(
        ubo.model,
        float4(input.position, 0.0, 1.0)
    );
    output.tex_coord = input.tex_coord;
    return output;
}

[shader("fragment")]
float4 frag_main(VertexOutput in_vert) : SV_Target
{
    return tex.Sample(in_vert.tex_coord);
}
