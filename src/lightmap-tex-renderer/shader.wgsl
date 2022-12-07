struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) fragPosition : vec3<f32>,
    @location(1) fragNormal : vec3<f32>,
    @location(2) @interpolate(flat) vertex_index: i32,
    @location(3) barycentric: vec3<f32>,
}

@group(0) @binding(0) var<storage> positions: array<f32>;
@group(0) @binding(1) var<storage> normals: array<f32>;
@group(0) @binding(2) var<storage> uvs: array<vec2<f32>>;

fn read_position(index: i32) -> vec3<f32> {
    var output: vec3<f32>;
    output.x = positions[index * 3];
    output.y = positions[index * 3 + 1];
    output.z = positions[index * 3 + 2];
    return output;
}

fn read_normal(index: i32) -> vec3<f32> {
    var output: vec3<f32>;
    output.x = normals[index * 3];
    output.y = normals[index * 3 + 1];
    output.z = normals[index * 3 + 2];
    return output;
}

@vertex
fn vertex(
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {

    var position = read_position(i32(vertex_index));

    var normal = read_normal(i32(vertex_index));

    var output : VertexOutput;
    output.fragPosition = position;
    output.fragNormal = normal;
    var uv: vec2<f32> = uvs[vertex_index] * 2.0 - 1.0;
    uv.y = -uv.y;
    output.Position = vec4(uv, 0.0, 1.0);
    output.vertex_index = i32(vertex_index);

    output.barycentric[i32(vertex_index) % 3] = 1.0;

    return output;
}

struct FragmentOutput {
    @location(0) positionOutput : vec4<f32>,
    @location(1) normalOutput : vec4<f32>,
}

// For `get_shadow_terminator_fix_shadow_origin`.
fn compute_vector_and_project_onto_tangent_plane(frag_pos: vec3<f32>, vertex_pos: vec3<f32>, vertex_normal: vec3<f32>) -> vec3<f32> {
    var vector_to_point = frag_pos - vertex_pos;

    var dot_product = min(0.0, dot(vector_to_point, vertex_normal));

    return vector_to_point - (dot_product * vertex_normal);
}

fn interpolate(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, barycentrics: vec3<f32>) -> vec3<f32> {
    return a * barycentrics.x + b * barycentrics.y + c * barycentrics.z;
}

@fragment
fn fragment(
    @location(0) fragPosition : vec3<f32>,
    @location(1) fragNormal : vec3<f32>,
    @location(2) @interpolate(flat) vertex_index: i32,
    @location(3) barycentric: vec3<f32>,
) -> FragmentOutput {
    var triangle_index = vertex_index / 3;
    var pos_a = read_position(triangle_index * 3);
    var pos_b = read_position(triangle_index * 3 + 1);
    var pos_c = read_position(triangle_index * 3 + 2);
    var normal_a = read_normal(triangle_index * 3);
    var normal_b = read_normal(triangle_index * 3 + 1);
    var normal_c = read_normal(triangle_index * 3 + 2);

    var offset_a = compute_vector_and_project_onto_tangent_plane(fragPosition, pos_a, normal_a);
    var offset_b = compute_vector_and_project_onto_tangent_plane(fragPosition, pos_b, normal_b);
    var offset_c = compute_vector_and_project_onto_tangent_plane(fragPosition, pos_c, normal_c);

    var interpolated_offset = interpolate(offset_a, offset_b, offset_c, barycentric);

    var output : FragmentOutput;
    var norm: vec3<f32> = normalize(fragNormal);
    output.positionOutput = vec4<f32>(fragPosition + interpolated_offset, 1.0);
    output.normalOutput = vec4<f32>(norm, 1.0);
    return output;
}
