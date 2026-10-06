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

// User interface rendering. The engine accumulates every 2D draw call into an
// array of MTLStretchPic records which this shader expands into quads, one
// instance per record.

#include <metal_stdlib>
#include "../mtlpt_shared.h"

using namespace metal;

struct StretchPicInOut {
    float4 position [[position]];
    float4 color;
    float2 texcoord;
    uint   tex_index [[flat]];
};

vertex StretchPicInOut stretch_pic_vertex(
    uint vertex_id                        [[vertex_id]],
    uint instance_id                      [[instance_id]],
    device const MTLStretchPic *pics      [[buffer(MTL_BUF_STRETCH_PICS)]])
{
    device const MTLStretchPic &pic = pics[instance_id];

    // Triangle strip corners: (0,0) (1,0) (0,1) (1,1)
    float2 corner = float2(float(vertex_id & 1u), float((vertex_id >> 1u) & 1u));

    float x = pic.x + corner.x * pic.w;
    float y = pic.y + corner.y * pic.h;

    StretchPicInOut out;
    // The engine works in Vulkan-style NDC where +Y is down, while Metal's NDC
    // has +Y up, so flip the vertical axis here.
    out.position = float4(x, -y, 0.0, 1.0);
    out.texcoord = float2(pic.s + corner.x * pic.w_s,
                          pic.t + corner.y * pic.h_t);

    uint c = pic.color;
    out.color = float4(float((c >>  0) & 0xffu),
                       float((c >>  8) & 0xffu),
                       float((c >> 16) & 0xffu),
                       float((c >> 24) & 0xffu)) * (1.0 / 255.0);
    out.tex_index = pic.tex_index;
    return out;
}

fragment float4 stretch_pic_fragment(
    StretchPicInOut in                        [[stage_in]],
    constant MTLDrawUniforms &ubo             [[buffer(MTL_BUF_DRAW_UNIFORMS)]],
    device const MTLTextureRef *textures      [[buffer(MTL_BUF_TEXTURE_TABLE)]],
    sampler tex_sampler                       [[sampler(0)]])
{
    texture2d<float> tex = textures[in.tex_index].tex;
    float4 color = tex.sample(tex_sampler, in.texcoord) * in.color;

    if (ubo.is_hdr != 0u) {
        // Lift the UI into the extended range so it stays legible against a
        // tone mapped HDR scene.
        color.rgb *= ubo.hdr_color_scale;
    }

    return color;
}
