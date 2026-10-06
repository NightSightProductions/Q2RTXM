/*
Copyright (C) 2024 Quake II RTX Metal port contributors

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

// Debug line rendering, the port of debug_line.vert / debug_line.frag. Lines
// arrive in view space; a negative z marks "no depth test". The fragment
// shader compares against the path tracer's view depth with a distance
// scaled bias so lines hugging geometry stay visible.

#include <metal_stdlib>
#include "../mtlpt_shared.h"

using namespace metal;

struct DebugLineVertex {
    packed_float3 view_pos;
    uint          color;
};

struct DebugLineInOut {
    float4 position [[position]];
    float4 color;
    float3 view_pos;
};

vertex DebugLineInOut debug_line_vertex(
    uint vid                                  [[vertex_id]],
    device const DebugLineVertex *verts       [[buffer(0)]],
    constant MTLDebugLineUniforms &u          [[buffer(1)]])
{
    DebugLineVertex v = verts[vid];
    float3 view_pos = float3(v.view_pos);
    bool depth_test = view_pos.z >= 0.0;
    view_pos.z = abs(view_pos.z);

    // Rectilinear projection: x right, y up, z forward in view space.
    float2 screen = float2(view_pos.x / (view_pos.z * u.tan_half_fov_x),
                           view_pos.y / (view_pos.z * u.tan_half_fov_y));

    DebugLineInOut out;
    if (depth_test) {
        out.position = float4(screen.x * view_pos.z, screen.y * view_pos.z, 0.5 * view_pos.z, view_pos.z);
        out.view_pos = view_pos;
    } else {
        out.position = float4(screen.x, screen.y, 0.0, 1.0);
        out.view_pos = float3(0.0);
    }

    uint c = v.color;
    out.color = float4((c & 0xffu), (c >> 8) & 0xffu, (c >> 16) & 0xffu, (c >> 24) & 0xffu) / 255.0;
    return out;
}

fragment float4 debug_line_fragment(
    DebugLineInOut in                              [[stage_in]],
    constant MTLDebugLineUniforms &u               [[buffer(1)]],
    texture2d<float, access::read> normal_depth    [[texture(0)]])
{
    // Fragment coordinates are in window pixels; the depth buffer is at
    // render resolution, in the path tracer's checkerboard field layout
    // (PT_VIEW_DEPTH: the two fields in the left and right halves).
    uint2 pixel = uint2(in.position.xy * float2(u.depth_scale_x, u.depth_scale_y));
    pixel = min(pixel, uint2(u.depth_width - 1, u.depth_height - 1));
    bool is_even_checkerboard = (pixel.x & 1u) == (pixel.y & 1u);
    pixel.x = pixel.x / 2u + (is_even_checkerboard ? 0u : u.depth_width / 2u);
    float view_depth = abs(normal_depth.read(pixel).x);
    if (view_depth <= 0.0)
        view_depth = 1.0e9;   // sky

    float dist = length(in.view_pos);
    float dist_log = log(dist) * 0.4342944819032518;
    if (dist > 0.0 && dist > view_depth + pow(10.0, floor(dist_log) - 2.0))
        discard_fragment();

    return float4(in.color.rgb * u.color_scale, in.color.a);
}
