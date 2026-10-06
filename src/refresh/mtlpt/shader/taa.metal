/*
Copyright (C) 2018 Christoph Schied
Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
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

// Temporal anti-aliasing and upscaling: a port of vkpt's asvgf_taau.comp.
// Runs in every mode: with AA off it passes the image through, and in the
// paused photo mode it accumulates the reference image.
//
// Metal: TAA_OUTPUT is written without vkpt's STORAGE_SCALE_HDR factor,
// since the Metal bloom and tone mapping passes expect scene radiance.

#include <metal_stdlib>
#include "vkpt_common.h"

using namespace metal;

#define GROUP_SIZE 16
#define FILTER_RADIUS 1
#define SHARED_SIZE (GROUP_SIZE + FILTER_RADIUS * 3)

constant float pq_m1 = 0.1593017578125;
constant float pq_m2 = 78.84375;
constant float pq_c1 = 0.8359375;
constant float pq_c2 = 18.8515625;
constant float pq_c3 = 18.6875;
constant float pq_C = 10000.0;

static float3 PQDecode(float3 image)
{
    float3 Np = pow(max(image, 0.0), float3(1.0 / pq_m2));
    float3 L = Np - pq_c1;
    L = L / (pq_c2 - pq_c3 * Np);
    L = pow(max(L, 0.0), float3(1.0 / pq_m1));
    return L * pq_C;
}

static float3 PQEncode(float3 image)
{
    float3 L = image / pq_C;
    float3 Lm = pow(max(L, 0.0), float3(pq_m1));
    float3 N = (pq_c1 + pq_c2 * Lm) / (1.0 + pq_c3 * Lm);
    image = pow(N, float3(pq_m2));
    return clamp(image, float3(0.0), float3(1.0));
}

static inline float2 hires_to_lores(thread const VkptCtx &ctx, int2 ipos)
{
    float2 input_size = float2(global_ubo.width, global_ubo.height);
    float2 output_size = float2(global_ubo.taa_output_width, global_ubo.taa_output_height);
    return (float2(ipos) + float2(0.5)) * (input_size / output_size) - float2(0.5) - float2(global_ubo.sub_pixel_jitter);
}

kernel void taa_main(
    uint2 gid                         [[thread_position_in_grid]],
    uint2 group_id                    [[threadgroup_position_in_grid]],
    uint local_index                  [[thread_index_in_threadgroup]],
    constant GlobalUbo &ubo           [[buffer(VKPT_BUF_UBO)]],
    device const VkptImages &img      [[buffer(VKPT_BUF_IMAGES)]])
{
    VkptCtx ctx = { &ubo, &img };

    threadgroup uint2 s_color_pq[SHARED_SIZE][SHARED_SIZE];
    threadgroup uint s_motion[SHARED_SIZE][SHARED_SIZE];

    int2 ipos = int2(gid);
    int2 group_base_hires = int2(group_id) * GROUP_SIZE;
    int2 group_base_lores = int2(hires_to_lores(ctx, group_base_hires));
    int2 group_bottomright_hires = int2(group_id) * GROUP_SIZE + int2(GROUP_SIZE - 1);
    int2 group_bottomright_lores = int2(hires_to_lores(ctx, group_bottomright_hires));

    // preload()
    {
        int2 group_size = group_bottomright_lores - group_base_lores + int2(1);
        int2 preload_size = min(group_size + int2(FILTER_RADIUS * 3), int2(SHARED_SIZE));
        for (uint linear_idx = local_index; linear_idx < uint(preload_size.x * preload_size.y); linear_idx += GROUP_SIZE * GROUP_SIZE) {
            float t = (float(linear_idx) + 0.5) / float(preload_size.x);
            int xx = int(floor(fract(t) * float(preload_size.x)));
            int yy = int(floor(t));

            int2 p = group_base_lores + int2(xx, yy) - int2(FILTER_RADIUS);
            p = clamp(p, int2(0), int2(global_ubo.width - 1, global_ubo.height - 1));
            float4 color = IMG_LOAD(FLAT_COLOR, p);
            float3 color_pq = PQEncode(color.rgb);
            float2 motion = IMG_LOAD(FLAT_MOTION, p).xy;

            s_color_pq[yy][xx] = packHalf4x16(float4(color_pq, color.a));
            s_motion[yy][xx] = packHalf2x16(motion);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (ipos.x >= global_ubo.taa_output_width || ipos.y >= global_ubo.taa_output_height) {
        IMG_STORE(TAA_OUTPUT, ipos, float4(0.0));
        return;
    }

    float2 nearest_render_pos = hires_to_lores(ctx, ipos);
    int2 int_render_pos = int2(round(nearest_render_pos.x), round(nearest_render_pos.y));
    int_render_pos = clamp(int_render_pos, int2(0), int2(global_ubo.width - 1, global_ubo.height - 1));

    int2 addr = int_render_pos - group_base_lores + int2(FILTER_RADIUS);
    float4 center_data = unpackHalf4x16(s_color_pq[addr.y][addr.x]);
    float3 color_center = center_data.rgb;
    int checkerboard_flags = int(center_data.a);

    float3 color_output = color_center;
    float3 linear_color_output;

    if (global_ubo.flt_taa != AA_MODE_OFF) {
        float3 mom1 = float3(0.0);
        float3 mom2 = float3(0.0);
        for (int yy = -FILTER_RADIUS; yy <= FILTER_RADIUS; yy++) {
            for (int xx = -FILTER_RADIUS; xx <= FILTER_RADIUS; xx++) {
                if (xx == 0 && yy == 0)
                    continue;
                int2 a = addr + int2(xx, yy);
                float3 c = unpackHalf4x16(s_color_pq[a.y][a.x]).rgb;
                mom1 += c;
                mom2 += c * c;
            }
        }
        const int num_pix = 9;

        if (global_ubo.flt_taa_anti_sparkle > 0.0) {
            float scale = pow(min(1.0, global_ubo.flt_taa_anti_sparkle), -0.25);
            color_center = min(color_center, scale * mom1 / float(num_pix - 1));
        }

        mom1 += color_center;
        mom2 += color_center * color_center;
        mom1 /= float(num_pix);
        mom2 /= float(num_pix);

        // Longest motion vector in a 3x3 window.
        float2 motion = float2(0.0);
        {
            float len = -1.0;
            for (int yy = -1; yy <= 1; yy++) {
                for (int xx = -1; xx <= 1; xx++) {
                    int2 a = addr + int2(xx, yy);
                    float2 m = unpackHalf2x16(s_motion[a.y][a.x]);
                    float l = dot(m, m);
                    if (l > len) {
                        len = l;
                        motion = m;
                    }
                }
            }
        }

        float2 pos_prev = ((float2(ipos) + float2(0.5)) / float2(global_ubo.taa_output_width, global_ubo.taa_output_height) + motion) *
                          float2(global_ubo.prev_taa_output_width, global_ubo.prev_taa_output_height);

        motion *= float2(global_ubo.taa_output_width, global_ubo.taa_output_height);

        if (all(int2(pos_prev) >= int2(1)) &&
            all(int2(pos_prev) < int2(global_ubo.taa_output_width, global_ubo.taa_output_height) - 1)) {
            float3 color_prev = sample_texture_catmull_rom(img.ASVGF_TAA_B_s, pos_prev).rgb;

            if (!any(isnan(color_prev))) {
                if (global_ubo.flt_taa_variance > 0.0) {
                    float variance_scale = global_ubo.flt_taa_variance;
                    if (checkerboard_flags == (CHECKERBOARD_FLAG_REFLECTION | CHECKERBOARD_FLAG_REFRACTION))
                        variance_scale *= 2.0;

                    float3 sigma = sqrt(max(float3(0.0), mom2 - mom1 * mom1));
                    float3 mi = mom1 - sigma * variance_scale;
                    float3 ma = mom1 + sigma * variance_scale;
                    color_prev = clamp(color_prev, mi, ma);
                }

                float motion_weight = smoothstep(0.0, 1.0, sqrt(dot(motion, motion)));
                float2 delta = nearest_render_pos - float2(int_render_pos);
                float sample_weight = clamp(1.0 - float(global_ubo.taa_output_width) * global_ubo.inv_width * dot(delta, delta), 0.0, 1.0);
                float pixel_weight = max(motion_weight, sample_weight) * 0.1;
                pixel_weight = clamp(pixel_weight, 0.0, 1.0);

                color_output = mix(color_prev, color_center, pixel_weight);
            }
        }

        linear_color_output = PQDecode(color_output);
    } else if (global_ubo.temporal_blend_factor > 0.0) {
        // Reference accumulation (paused photo mode).
        linear_color_output = PQDecode(color_output);
        if (global_ubo.temporal_blend_factor < 1.0) {
            float3 prev_color = IMG_LOAD(HQ_COLOR_INTERLEAVED, ipos).rgb;
            linear_color_output = mix(prev_color, linear_color_output, global_ubo.temporal_blend_factor);
        }
        IMG_STORE(HQ_COLOR_INTERLEAVED, ipos, float4(linear_color_output, 0.0));
        color_output = PQEncode(linear_color_output);
    } else {
        linear_color_output = PQDecode(color_output);
    }

    IMG_STORE(ASVGF_TAA_A, ipos, float4(color_output, 0.0));
    IMG_STORE(TAA_OUTPUT, ipos, float4(linear_color_output / STORAGE_SCALE_HDR, 1.0));
}
