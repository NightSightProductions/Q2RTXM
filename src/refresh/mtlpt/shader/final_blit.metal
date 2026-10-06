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

// Resolves the HDR path tracer output into the swapchain drawable, applying
// tone mapping for SDR output.

#include <metal_stdlib>
#include "../mtlpt_shared.h"

using namespace metal;

struct BlitInOut {
    float4 position [[position]];
    float2 texcoord;
};

vertex BlitInOut final_blit_vertex(uint vertex_id [[vertex_id]])
{
    // Oversized triangle covering the whole viewport.
    float2 uv = float2((vertex_id << 1u) & 2u, vertex_id & 2u);

    BlitInOut out;
    out.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    out.texcoord = float2(uv.x, 1.0 - uv.y);
    return out;
}

// Extended Reinhard on luminance only, the same curve the Vulkan backend uses
// in tone_mapping_apply.comp. Working on luminance instead of per channel keeps
// bright surfaces saturated instead of pulling them towards white, and the
// white point lets highlights run far above the exposure target before they
// clip -- which is what stops the sky from burning out.
static float3 tonemap_reinhard(float3 color, float white_point)
{
    float lum = max(dot(color, float3(0.2126, 0.7152, 0.0722)), 1e-4);
    float w2 = white_point * white_point;
    float mapped = (lum * (1.0 + lum / w2)) / (1.0 + lum);
    return saturate(color * (mapped / lum));
}

constant float M_PI_F_ = 3.14159265358979323846;

static inline float2 v_sel(float2 f, float val, float eq, float2 neq)
{
    return float2(f.x == val ? eq : neq.x, f.y == val ? eq : neq.y);
}

// Lanczos 3 upscale, from vkpt's final_blit.frag.
static float3 filter_lanczos(texture2d<float> img, sampler s, float2 uv)
{
    float2 size = float2(img.get_width(), img.get_height());

    float2 UV = uv * size;
    float2 tc = floor(UV - 0.5) + 0.5;
    float2 f = UV - tc + 2.0;

    float2 fpi = f * M_PI_F_, fpi3 = f * (M_PI_F_ / 3.0);
    float2 sinfpi = sin(fpi), sinfpi3 = sin(fpi3), cosfpi3 = cos(fpi3);
    const float r3 = sqrt(3.0);
    float2 w0 = v_sel(f, 0.0, M_PI_F_ * M_PI_F_ * 1.0 / 3.0, (sinfpi * sinfpi3) / (f * f));
    float2 w1 = v_sel(f, 1.0, M_PI_F_ * M_PI_F_ * 2.0 / 3.0, (-sinfpi * (sinfpi3 - r3 * cosfpi3)) / ((f - 1.0) * (f - 1.0)));
    float2 w2 = v_sel(f, 2.0, M_PI_F_ * M_PI_F_ * 2.0 / 3.0, (sinfpi * (-sinfpi3 - r3 * cosfpi3)) / ((f - 2.0) * (f - 2.0)));
    float2 w3 = v_sel(f, 3.0, M_PI_F_ * M_PI_F_ * 2.0 / 3.0, (-sinfpi * (-2.0 * sinfpi3)) / ((f - 3.0) * (f - 3.0)));
    float2 w4 = v_sel(f, 4.0, M_PI_F_ * M_PI_F_ * 2.0 / 3.0, (sinfpi * (-sinfpi3 + r3 * cosfpi3)) / ((f - 4.0) * (f - 4.0)));
    float2 w5 = v_sel(f, 5.0, M_PI_F_ * M_PI_F_ * 2.0 / 3.0, (-sinfpi * (sinfpi3 + r3 * cosfpi3)) / ((f - 5.0) * (f - 5.0)));

    float2 weight[5] = { w0, w1, w2 + w3, w4, w5 };
    float2 inv = 1.0 / size;
    float2 smp[5] = { inv * (tc - 2.0), inv * (tc - 1.0), inv * (tc + w3 / weight[2]), inv * (tc + 2.0), inv * (tc + 3.0) };

    float4 o = float4(0.0);
    #define TAP(i, j) o += float4(img.sample(s, float2(smp[i].x, smp[j].y), level(0)).rgb, 1.0) * weight[i].x * weight[j].y;
    TAP(0, 2) TAP(1, 1) TAP(1, 2) TAP(1, 3) TAP(2, 0) TAP(2, 1) TAP(2, 2)
    TAP(2, 3) TAP(2, 4) TAP(3, 1) TAP(3, 2) TAP(3, 3) TAP(4, 2)
    #undef TAP
    return o.rgb / o.w;
}

fragment float4 final_blit_fragment(
    BlitInOut in                    [[stage_in]],
    constant MTLDrawUniforms &ubo   [[buffer(MTL_BUF_DRAW_UNIFORMS)]],
    texture2d<float> source         [[texture(0)]])
{
    constexpr sampler linear_sampler(filter::linear, address::clamp_to_edge);

    float2 uv = in.texcoord;
    if (ubo.water_warp != 0u) {
        uv += float2(0.00666) * sin(uv.yx * float2(M_PI_F_ * 10.0) + ubo.time);
        // Warping pushes uv outside the rendered area near the borders.
        float2 input_dim_inv = 1.0 / float2(ubo.input_width, ubo.input_height);
        uv = clamp(uv, 0.5 * input_dim_inv, float2(1.0) - 1.5 * input_dim_inv);
    }

    // Only the lower left region of the target is rendered when dynamic
    // resolution scaling is active, so rescale the lookup accordingly.
    uv *= float2(ubo.uv_scale_x, ubo.uv_scale_y);

    float3 color = (ubo.filter_lanczos != 0u) ? filter_lanczos(source, linear_sampler, uv)
                                              : source.sample(linear_sampler, uv).rgb;

    if (ubo.tonemapped != 0u)
        return float4(color, 1.0);

    if (ubo.is_hdr != 0u) {
        // The drawable is extended-linear sRGB, so pass scene referred values
        // through and let the display handle the range.
        return float4(color * ubo.hdr_color_scale, 1.0);
    }

    // The drawable is an _sRGB format, so write linear values and let the
    // hardware apply the transfer function.
    return float4(tonemap_reinhard(color, max(ubo.tm_white_point, 1.0)), 1.0);
}
