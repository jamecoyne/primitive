#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

layout(set = 0, binding = 0) uniform Uniforms {
    vec2 resolution;
    vec2 mouse;     // pixel coords, top-left origin (winit y-down)
    float time;
    float _pad0;
    float _pad1;
    float _pad2;
} u;

void main() {
    float aspect = u.resolution.x / u.resolution.y;

    // Screen-space NDC of this fragment ([-aspect, +aspect] × [-1, +1]).
    vec2 ndc = v_uv * 2.0 - 1.0;
    ndc.x *= aspect;

    // Mouse → NDC. winit y is top-down, NDC y is bottom-up, so flip.
    vec2 m_uv = u.mouse / u.resolution;
    vec2 m_ndc = vec2(m_uv.x * 2.0 - 1.0, 1.0 - m_uv.y * 2.0);
    m_ndc.x *= aspect;

    float zoom = 1.6 + 0.6 * sin(u.time * 0.15);
    vec2 fractal_origin = vec2(-0.745, 0.10);
    // Subtracting m_ndc translates the mandelbrot so its center sits under the cursor.
    vec2 c = fractal_origin + (ndc - m_ndc) * zoom;

    vec2 z = vec2(0.0);
    const int MAX_ITER = 384;
    int i = 0;
    for (; i < MAX_ITER; ++i) {
        if (dot(z, z) > 256.0) break;
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
    }

    if (i >= MAX_ITER) {
        o_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float log_zn = log(dot(z, z)) * 0.5;
    float nu = log(log_zn / log(2.0)) / log(2.0);
    float smooth_i = float(i) + 1.0 - nu;
    float t = smooth_i / float(MAX_ITER);

    vec3 col = 0.5 + 0.5 * cos(6.28318 * (t + vec3(0.0, 0.33, 0.67)) + u.time * 0.4);
    o_color = vec4(col, 1.0);
}
