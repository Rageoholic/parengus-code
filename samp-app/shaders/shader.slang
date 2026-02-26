struct VertexInput
{
    [[vk::location(0)]]
    float2 position;
    [[vk::location(1)]]
    float3 color;
};

struct VertexOutput
{
    [[vk::location(0)]]
    float3 color;
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexOutput vert_main(VertexInput input)
{
    VertexOutput output;
    output.sv_position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 frag_main(VertexOutput inVert) : SV_Target
{
    float3 color = inVert.color;
    return float4(color, 1.0);
}
