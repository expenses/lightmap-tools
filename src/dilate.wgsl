
@group(0) @binding(0) var texture: texture_storage_2d<rgba32float,read_write>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba32float,read_write>;
@group(0) @binding(2) var<uniform> width: i32;

fn add_to_sample(
    sh0_accumulated: ptr<function, vec3<f32>>,
    sh1x_accumulated: ptr<function, vec3<f32>>,
    sh1y_accumulated: ptr<function, vec3<f32>>,
    sh1z_accumulated: ptr<function, vec3<f32>>,
    count: ptr<function, u32>,
    coord: vec2<i32>
) {
    var sh0 = textureLoad(texture, coord).xyz;
    var sh1x = textureLoad(texture, vec2(coord.x + width, coord.y)).xyz;
    var sh1y = textureLoad(texture, vec2(coord.x + width * 2, coord.y)).xyz;
    var sh1z = textureLoad(texture, vec2(coord.x + width * 3, coord.y)).xyz;

    if (any(sh0 >= vec3(0.0001))) {
        *sh0_accumulated += sh0;
        *sh1x_accumulated += sh1x;
        *sh1y_accumulated += sh1y;
        *sh1z_accumulated += sh1z;
        *count += 1u;
    }
}

@compute @workgroup_size(8, 8, 1)
fn compute(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
    var coord = vec2<i32>(global_invocation_id.xy);

    if coord.x > width {
        return;
    }

    var sh0 = textureLoad(texture, coord).xyz;
    var sh1x = textureLoad(texture, vec2(coord.x + width, coord.y)).xyz;
    var sh1y = textureLoad(texture, vec2(coord.x + width * 2, coord.y)).xyz;
    var sh1z = textureLoad(texture, vec2(coord.x + width * 3, coord.y)).xyz;


    if (all(sh0 < vec3(0.0001))) {
        var sh0_accumulated = vec3(0.0);
        var sh1x_accumulated = vec3(0.0);
        var sh1y_accumulated = vec3(0.0);
        var sh1z_accumulated = vec3(0.0);

        var count = 0u;

        for (var x = -1; x <= 1; x++) {
            for (var y = -1; y <= 1; y++) {
                if (x == 0 && y == 0) {
                    continue;
                }

                add_to_sample(&sh0_accumulated, &sh1x_accumulated, &sh1y_accumulated, &sh1z_accumulated, &count, coord + vec2(x, y));
            }
        }

        count = max(count, 1u);

        sh0 = sh0_accumulated / vec3(f32(count));
        sh1x = sh1x_accumulated / vec3(f32(count));
        sh1y = sh1y_accumulated / vec3(f32(count));
        sh1z = sh1z_accumulated / vec3(f32(count));
    }

    textureStore(output_texture, coord, vec4(sh0, 1.0));
    textureStore(output_texture, vec2(coord.x + width, coord.y), vec4(sh1x, 1.0));
    textureStore(output_texture, vec2(coord.x + width * 2, coord.y), vec4(sh1y, 1.0));
    textureStore(output_texture, vec2(coord.x + width * 3, coord.y), vec4(sh1z, 1.0));

    return;
}
