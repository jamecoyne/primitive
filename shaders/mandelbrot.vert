#version 450

// Fullscreen triangle — a single 3-vertex triangle that covers the
// viewport. Avoids the need for a vertex buffer entirely.
layout(location = 0) out vec2 v_uv;

void main() {
    vec2 pos = vec2(
        float((gl_VertexIndex & 1) * 4) - 1.0,
        float((gl_VertexIndex & 2) * 2) - 1.0
    );
    v_uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
