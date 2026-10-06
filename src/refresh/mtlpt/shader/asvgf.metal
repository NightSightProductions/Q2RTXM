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

// The denoiser and compositing: ports of vkpt's asvgf_gradient_img.comp,
// asvgf_gradient_atrous.comp, asvgf_temporal.comp, asvgf_lf.comp,
// asvgf_atrous.comp, compositing.comp and checkerboard_interleave.comp.
// See vkpt/shader/asvgf.glsl for the overview. The screen images use vkpt's
// field layout: the two checkerboard fields de-interleaved into the left and
// right halves until checkerboard_interleave produces the flat image.
// (asvgf_gradient_reproject lives in path_tracer.metal: it needs the scene.)

#include <metal_stdlib>
#include "vkpt_common.h"

using namespace metal;

#define DENOISER_PARAMS \
    constant GlobalUbo &ubo           [[buffer(VKPT_BUF_UBO)]], \
    device const VkptImages &img      [[buffer(VKPT_BUF_IMAGES)]], \
    constant VkptPush &push           [[buffer(VKPT_BUF_PUSH)]], \
    texture2d_array<float> blue_noise [[texture(VKPT_TEX_BLUE_NOISE)]]

#define DENOISER_CTX VkptCtx ctx = { &ubo, &img };

static inline float3 demodulate_specular(thread const VkptCtx &ctx, float3 base_reflectivity, float3 specular)
{
    if (global_ubo.flt_enable == 0.0)
        return specular;
    return specular / max(float3(0.01), base_reflectivity);
}

static inline float3 modulate_specular(thread const VkptCtx &ctx, float3 base_reflectivity, float3 filtered_specular)
{
    if (global_ubo.flt_enable == 0.0)
        return filtered_specular;
    return filtered_specular * max(float3(0.01), base_reflectivity);
}

// brdf.glsl composite_color
static float3 composite_color(thread const VkptCtx &ctx, float3 surf_base_color, float surf_metallic, float3 throughput,
                              float3 projected_lf, float3 high_freq, float3 specular, float4 transparent)
{
    projected_lf *= global_ubo.flt_scale_lf;
    high_freq *= global_ubo.flt_scale_hf;
    specular *= global_ubo.flt_scale_spec;

    float3 albedo, base_reflectivity;
    get_reflectivity(surf_base_color, surf_metallic, albedo, base_reflectivity);

    specular = modulate_specular(ctx, base_reflectivity, specular);

    if (global_ubo.flt_fixed_albedo != 0.0)
        albedo = float3(global_ubo.flt_fixed_albedo);

    float3 final_color = (projected_lf + high_freq) * albedo + specular;
    final_color *= throughput;

    transparent *= global_ubo.flt_scale_overlay;
    final_color = final_color * (1.0 - transparent.a) + transparent.rgb;
    return final_color;
}

//
// asvgf_gradient_img.comp
//

static inline float get_gradient(float l_curr, float l_prev)
{
    float l_max = max(l_curr, l_prev);
    if (l_max == 0.0)
        return 0.0;
    float ret = abs(l_curr - l_prev) / l_max;
    return ret * ret;
}

static float2 get_lf_gradient(thread const VkptCtx &ctx, int2 ipos)
{
    float4 motion = IMG_LOAD(PT_MOTION, ipos);
    int2 pos_prev = int2(((float2(ipos) + float2(0.5)) * float2(global_ubo.inv_width * 2.0, global_ubo.inv_height) + motion.xy) *
                         float2(float(global_ubo.prev_width) * 0.5, float(global_ubo.prev_height)));

    int field_left = 0;
    int field_right = global_ubo.prev_width / 2;
    if (ipos.x >= global_ubo.width / 2) {
        field_left = field_right;
        field_right = global_ubo.prev_width;
    }

    if (pos_prev.x < field_left || pos_prev.x >= field_right || pos_prev.y < 0 || pos_prev.y >= global_ubo.height)
        return float2(0.0);

    float lum_curr = IMG_LOAD(PT_COLOR_LF_SH, ipos).w;
    float lum_prev = IMG_LOAD(ASVGF_HIST_COLOR_LF_SH_B, pos_prev).w;
    return float2(lum_curr, lum_prev);
}

kernel void asvgf_gradient_img(uint2 gid [[thread_position_in_grid]], DENOISER_PARAMS)
{
    DENOISER_CTX
    int2 ipos = int2(gid);
    if (any(ipos >= int2(global_ubo.current_gpu_slice_width, global_ubo.height) / GRAD_DWN))
        return;

    uint u = IMG_LOAD(ASVGF_GRAD_SMPL_POS_A, ipos).r;

    float2 grad_lf = float2(0.0);
    float grad_hf = 0.0;
    float grad_spec = 0.0;

    if (u != 0u) {
        int2 grad_strata_pos = int2(int(u >> (STRATUM_OFFSET_SHIFT * 0)), int(u >> (STRATUM_OFFSET_SHIFT * 1))) & STRATUM_OFFSET_MASK;
        int2 grad_sample_pos_curr = ipos * GRAD_DWN + grad_strata_pos;

        float2 prev_hf_spec_lum = IMG_LOAD(ASVGF_GRAD_HF_SPEC_PING, ipos).rg;

        float3 curr_hf = unpackRGBE(IMG_LOAD(PT_COLOR_HF, grad_sample_pos_curr).x);
        float3 curr_spec = unpackRGBE(IMG_LOAD(PT_COLOR_SPEC, grad_sample_pos_curr).x);

        grad_hf = get_gradient(luminance(curr_hf), prev_hf_spec_lum.x);
        grad_spec = get_gradient(luminance(curr_spec), prev_hf_spec_lum.y);

        // The weapon moves a lot; slow lighting beats noise on it.
        int checkerboard_flags = int(IMG_LOAD(PT_VIEW_DIRECTION, grad_sample_pos_curr).w);
        if ((checkerboard_flags & CHECKERBOARD_FLAG_WEAPON) != 0)
            grad_spec *= global_ubo.flt_grad_weapon;
    }

    for (int yy = 0; yy < GRAD_DWN; yy++)
        for (int xx = 0; xx < GRAD_DWN; xx++)
            grad_lf += get_lf_gradient(ctx, ipos * GRAD_DWN + int2(xx, yy));

    IMG_STORE(ASVGF_GRAD_LF_PING, ipos, float4(grad_lf, 0.0, 0.0));
    IMG_STORE(ASVGF_GRAD_HF_SPEC_PING, ipos, float4(grad_hf, grad_spec, 0.0, 0.0));
}

//
// asvgf_gradient_atrous.comp
//

static float2 gradient_filter_image(thread const VkptCtx &ctx, texture2d<float, access::read> img_in, int2 ipos, uint iteration)
{
    int2 grad_size = int2(global_ubo.current_gpu_slice_width, global_ubo.height) / GRAD_DWN;
    const int step_size = int(1u << iteration);

    float2 sum_color = float2(0.0);
    float sum_w = 0.0;
    for (int yy = -1; yy <= 1; yy++) {
        for (int xx = -1; xx <= 1; xx++) {
            int2 p = ipos + int2(xx, yy) * step_size;
            float2 c = img_load(img_in, p).xy;
            if (any(p >= grad_size))
                c = float2(0.0);
            float w = wavelet_kernel[abs(xx)][abs(yy)];
            sum_color += c * w;
            sum_w += w;
        }
    }
    return sum_color / sum_w;
}

kernel void asvgf_gradient_atrous(uint2 gid [[thread_position_in_grid]], DENOISER_PARAMS)
{
    DENOISER_CTX
    int2 ipos = int2(gid);
    int2 grad_size = int2(global_ubo.current_gpu_slice_width, global_ubo.height) / GRAD_DWN;
    if (any(ipos >= grad_size))
        return;

    uint iteration = push.iteration;
    float2 filtered_lf = float2(0.0);
    float2 filtered_hf_spec = float2(0.0);

    bool even = (iteration & 1u) == 0u;
    filtered_lf = gradient_filter_image(ctx, even ? img.ASVGF_GRAD_LF_PING_r : img.ASVGF_GRAD_LF_PONG_r, ipos, iteration);
    if (iteration < 3u)
        filtered_hf_spec = gradient_filter_image(ctx, even ? img.ASVGF_GRAD_HF_SPEC_PING_r : img.ASVGF_GRAD_HF_SPEC_PONG_r, ipos, iteration);

    if (iteration == 6u) {
        filtered_lf.x = get_gradient(filtered_lf.x, filtered_lf.y);
        filtered_lf.y = 0.0;
    }

    if (even) {
        IMG_STORE(ASVGF_GRAD_LF_PONG, ipos, float4(filtered_lf, 0.0, 0.0));
        if (iteration < 3u)
            IMG_STORE(ASVGF_GRAD_HF_SPEC_PONG, ipos, float4(filtered_hf_spec, 0.0, 0.0));
    } else {
        IMG_STORE(ASVGF_GRAD_LF_PING, ipos, float4(filtered_lf, 0.0, 0.0));
        if (iteration < 3u)
            IMG_STORE(ASVGF_GRAD_HF_SPEC_PING, ipos, float4(filtered_hf_spec, 0.0, 0.0));
    }
}

//
// asvgf_temporal.comp
//

#define TEMPORAL_GROUP_SIZE 15
#define TEMPORAL_FILTER_RADIUS 1
#define TEMPORAL_SHARED_SIZE (TEMPORAL_GROUP_SIZE + TEMPORAL_FILTER_RADIUS * 2)

kernel void asvgf_temporal(
    uint2 gid                       [[thread_position_in_grid]],
    uint2 group_id                  [[threadgroup_position_in_grid]],
    uint2 local_id                  [[thread_position_in_threadgroup]],
    uint local_index                [[thread_index_in_threadgroup]],
    DENOISER_PARAMS)
{
    DENOISER_CTX

    threadgroup uint2 s_normal_lum[TEMPORAL_SHARED_SIZE][TEMPORAL_SHARED_SIZE];
    threadgroup float s_depth[TEMPORAL_SHARED_SIZE][TEMPORAL_SHARED_SIZE];
    threadgroup float4 s_lf_shy[TEMPORAL_GROUP_SIZE][TEMPORAL_GROUP_SIZE];
    threadgroup float2 s_lf_cocg[TEMPORAL_GROUP_SIZE][TEMPORAL_GROUP_SIZE];
    threadgroup float s_depth_width[TEMPORAL_GROUP_SIZE / GRAD_DWN][TEMPORAL_GROUP_SIZE / GRAD_DWN];

    // preload()
    {
        int2 groupBase = int2(group_id) * TEMPORAL_GROUP_SIZE - TEMPORAL_FILTER_RADIUS;
        for (uint linear_idx = local_index; linear_idx < TEMPORAL_SHARED_SIZE * TEMPORAL_SHARED_SIZE;
             linear_idx += TEMPORAL_GROUP_SIZE * TEMPORAL_GROUP_SIZE) {
            float t = (float(linear_idx) + 0.5) / float(TEMPORAL_SHARED_SIZE);
            int xx = int(floor(fract(t) * float(TEMPORAL_SHARED_SIZE)));
            int yy = int(floor(t));

            int2 p = groupBase + int2(xx, yy);
            float depth = IMG_LOAD(PT_VIEW_DEPTH_A, p).x;
            float3 normal = decode_normal(IMG_LOAD(PT_NORMAL_A, p).x);
            float3 color_hf = unpackRGBE(IMG_LOAD(PT_COLOR_HF, p).x);

            s_normal_lum[yy][xx] = packHalf4x16(float4(normal, luminance(color_hf)));
            s_depth[yy][xx] = depth;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int2 ipos = int2(gid);
    float4 motion = IMG_LOAD(PT_MOTION, ipos);

    int checkerboard_flags = int(IMG_LOAD(PT_VIEW_DIRECTION, ipos).w);
    bool is_checkerboarded_surface = popcount_int(checkerboard_flags & CHECKERBOARD_FLAG_FIELD_MASK) > 1;

    // Regular surfaces can use both fields for a better history sample.
    bool sample_across_fields = !is_checkerboarded_surface && (global_ubo.current_gpu_slice_width == global_ubo.width);

    float2 pos_prev;
    if (sample_across_fields) {
        pos_prev = ((float2(checker_to_flat(ipos, global_ubo.width)) + float2(0.5)) * float2(global_ubo.inv_width, global_ubo.inv_height) + motion.xy) *
                   float2(global_ubo.prev_width, global_ubo.prev_height);
    } else {
        pos_prev = ((float2(ipos) + float2(0.5)) * float2(global_ubo.inv_width * 2.0, global_ubo.inv_height) + motion.xy) *
                   float2(float(global_ubo.prev_width / 2), float(global_ubo.prev_height));
    }

    float motion_length = length(motion.xy * float2(global_ubo.width, global_ubo.height));

    int2 shared_center = int2(local_id) + int2(TEMPORAL_FILTER_RADIUS);
    float depth_curr = s_depth[shared_center.y][shared_center.x];
    float4 normal_lum_curr = unpackHalf4x16(s_normal_lum[shared_center.y][shared_center.x]);
    float3 normal_curr = normal_lum_curr.xyz;
    float lum_curr_hf = normal_lum_curr.w;

    float2 metal_rough = IMG_LOAD(PT_METALLIC_A, ipos).xy;
    float shininess = clamp(2.0 / square(square(metal_rough.y)) - 2.0, 0.0, 32.0);

    float3 geo_normal_curr = decode_normal(IMG_LOAD(PT_GEO_NORMAL_A, ipos).x);

    bool temporal_sample_valid_diff = false;
    bool temporal_sample_valid_spec = false;
    SH temporal_color_lf = init_SH();
    float3 temporal_color_hf = float3(0.0);
    float4 temporal_color_histlen_spec = float4(0.0);
    float4 temporal_moments_histlen_hf = float4(0.0);
    {
        float temporal_sum_w_diff = 0.0;
        float temporal_sum_w_spec = 0.0;

        float2 pos_ld = floor(pos_prev - float2(0.5));
        float2 subpix = fract(pos_prev - float2(0.5) - pos_ld);

        int field_left = 0;
        int field_right = sample_across_fields ? global_ubo.prev_width : (global_ubo.prev_width / 2);
        if (!sample_across_fields && ipos.x >= global_ubo.width / 2) {
            field_left = field_right;
            field_right = global_ubo.prev_width;
        }

        const int2 off[4] = { int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1) };
        float w[4] = {
            (1.0 - subpix.x) * (1.0 - subpix.y),
            (subpix.x) * (1.0 - subpix.y),
            (1.0 - subpix.x) * (subpix.y),
            (subpix.x) * (subpix.y)
        };
        for (int i = 0; i < 4; i++) {
            int2 p = int2(pos_ld) + off[i];

            if (p.x < field_left || p.x >= field_right || p.y < 0 || p.y >= global_ubo.prev_height)
                continue;

            if (sample_across_fields)
                p = flat_to_checker(p, global_ubo.prev_width);

            float depth_prev = IMG_LOAD(PT_VIEW_DEPTH_B, p).x;
            float3 normal_prev = decode_normal(IMG_LOAD(PT_NORMAL_B, p).x);
            float3 geo_normal_prev = decode_normal(IMG_LOAD(PT_GEO_NORMAL_B, p).x);

            float dist_depth = abs(depth_curr - depth_prev + motion.z) / abs(depth_curr);
            float dot_normals = dot(normal_curr, normal_prev);
            float dot_geo_normals = dot(geo_normal_curr, geo_normal_prev);

            if (depth_curr < 0.0) {
                // Reflection/refraction motion vectors are often inaccurate.
                dist_depth *= 0.25;
            }

            if (dist_depth < 0.1 && dot_geo_normals > 0.5) {
                float w_diff = w[i] * max(dot_normals, 0.0);
                float w_spec = w[i] * pow(max(dot_normals, 0.0), shininess);

                SH hist_color_lf = LOAD_SH(ASVGF_HIST_COLOR_LF_SH_B, ASVGF_HIST_COLOR_LF_COCG_B, p);
                accumulate_SH(temporal_color_lf, hist_color_lf, w_diff);

                temporal_color_hf += unpackRGBE(IMG_LOAD(ASVGF_HIST_COLOR_HF, p).x) * w_diff;
                temporal_color_histlen_spec += IMG_LOAD(ASVGF_FILTERED_SPEC_B, p) * w_spec;
                temporal_moments_histlen_hf += IMG_LOAD(ASVGF_HIST_MOMENTS_HF_B, p) * w_diff;
                temporal_sum_w_diff += w_diff;
                temporal_sum_w_spec += w_spec;
            }
        }

        if (temporal_sum_w_diff > 1e-6) {
            float inv_w_diff = 1.0 / temporal_sum_w_diff;
            temporal_color_lf.shY *= inv_w_diff;
            temporal_color_lf.CoCg *= inv_w_diff;
            temporal_color_hf *= inv_w_diff;
            temporal_moments_histlen_hf *= inv_w_diff;
            temporal_sample_valid_diff = true;
        }

        if (temporal_sum_w_spec > 1e-6) {
            temporal_color_histlen_spec *= 1.0 / temporal_sum_w_spec;
            temporal_sample_valid_spec = true;
        }
    }

    // Spatial moments of the HF channel in a 3x3 window.
    float2 spatial_moments_hf = float2(lum_curr_hf, lum_curr_hf * lum_curr_hf);
    {
        float spatial_sum_w_hf = 1.0;
        for (int yy = -TEMPORAL_FILTER_RADIUS; yy <= TEMPORAL_FILTER_RADIUS; yy++) {
            for (int xx = -TEMPORAL_FILTER_RADIUS; xx <= TEMPORAL_FILTER_RADIUS; xx++) {
                if (xx == 0 && yy == 0)
                    continue;

                int2 sp = shared_center + int2(xx, yy);
                float depth = s_depth[sp.y][sp.x];
                float4 nl = unpackHalf4x16(s_normal_lum[sp.y][sp.x]);

                float dist_z = abs(depth_curr - depth) * motion.a;
                if (dist_z < 2.0) {
                    float w_hf = pow(max(0.0, dot(nl.xyz, normal_curr)), 128.0);
                    spatial_moments_hf += float2(nl.w * w_hf, nl.w * nl.w * w_hf);
                    spatial_sum_w_hf += w_hf;
                }
            }
        }
        spatial_moments_hf *= 1.0 / spatial_sum_w_hf;
    }

    SH color_curr_lf = LOAD_SH(PT_COLOR_LF_SH, PT_COLOR_LF_COCG, ipos);
    float3 color_curr_hf = unpackRGBE(IMG_LOAD(PT_COLOR_HF, ipos).x);
    float3 color_curr_spec = unpackRGBE(IMG_LOAD(PT_COLOR_SPEC, ipos).x);

    SH out_color_lf;
    float3 out_color_hf;
    float4 out_color_histlen_spec;
    float4 out_moments_histlen_hf;

    float grad_lf = IMG_LOAD(ASVGF_GRAD_LF_PONG, ipos / GRAD_DWN).r;
    float2 grad_hf_spec = IMG_LOAD(ASVGF_GRAD_HF_SPEC_PONG, ipos / GRAD_DWN).rg;
    grad_lf = clamp(grad_lf, 0.0, 1.0);
    grad_hf_spec = clamp(grad_hf_spec, float2(0.0), float2(1.0));

    if (temporal_sample_valid_diff) {
        float antilag_alpha_lf = clamp(mix(1.0, global_ubo.flt_antilag_lf * grad_lf, global_ubo.flt_temporal_lf), 0.0, 1.0);
        float antilag_alpha_hf = clamp(mix(1.0, global_ubo.flt_antilag_hf * grad_hf_spec.x, global_ubo.flt_temporal_hf), 0.0, 1.0);

        float hist_len_hf = min(temporal_moments_histlen_hf.b * pow(1.0 - antilag_alpha_hf, 10.0) + 1.0, 256.0);
        float hist_len_lf = min(temporal_moments_histlen_hf.a * pow(1.0 - antilag_alpha_lf, 10.0) + 1.0, 256.0);

        float alpha_color_lf = max(global_ubo.flt_min_alpha_color_lf, 1.0 / hist_len_lf);
        float alpha_color_hf = max(global_ubo.flt_min_alpha_color_hf, 1.0 / hist_len_hf);
        float alpha_moments_hf = max(global_ubo.flt_min_alpha_moments_hf, 1.0 / hist_len_hf);

        alpha_color_lf = mix(alpha_color_lf, 1.0, antilag_alpha_lf);
        alpha_color_hf = mix(alpha_color_hf, 1.0, antilag_alpha_hf);
        alpha_moments_hf = mix(alpha_moments_hf, 1.0, antilag_alpha_hf);

        out_color_lf = mix_SH(temporal_color_lf, color_curr_lf, alpha_color_lf);
        out_color_hf = mix(temporal_color_hf, color_curr_hf, alpha_color_hf);

        out_moments_histlen_hf.rg = mix(temporal_moments_histlen_hf.rg, spatial_moments_hf, alpha_moments_hf);
        out_moments_histlen_hf.b = hist_len_hf;
        out_moments_histlen_hf.a = hist_len_lf;
    } else {
        out_color_lf = color_curr_lf;
        out_color_hf = color_curr_hf;
        out_moments_histlen_hf = float4(spatial_moments_hf, 1.0, 1.0);
    }

    if (temporal_sample_valid_spec) {
        float antilag = grad_hf_spec.y * global_ubo.flt_antilag_spec + motion_length * global_ubo.flt_antilag_spec_motion;
        float antilag_alpha_spec = clamp(mix(1.0, antilag, global_ubo.flt_temporal_spec), 0.0, 1.0);
        float hist_len_spec = min(temporal_color_histlen_spec.a * pow(1.0 - antilag_alpha_spec, 10.0) + 1.0, 256.0);
        float alpha_color_spec = max(global_ubo.flt_min_alpha_color_spec, 1.0 / hist_len_spec);
        alpha_color_spec = mix(alpha_color_spec, 1.0, antilag_alpha_spec);
        out_color_histlen_spec.rgb = mix(temporal_color_histlen_spec.rgb, color_curr_spec, alpha_color_spec);
        out_color_histlen_spec.a = hist_len_spec;
    } else {
        out_color_histlen_spec = float4(color_curr_spec, 1.0);
    }

    IMG_STORE(ASVGF_HIST_MOMENTS_HF_A, ipos, out_moments_histlen_hf);
    STORE_SH(ASVGF_HIST_COLOR_LF_SH_A, ASVGF_HIST_COLOR_LF_COCG_A, ipos, out_color_lf);
    IMG_STORE(ASVGF_ATROUS_PING_HF, ipos, uint4(packRGBE(out_color_hf)));
    IMG_STORE(ASVGF_ATROUS_PING_SPEC, ipos, uint4(packRGBE(out_color_histlen_spec.rgb)));
    IMG_STORE(ASVGF_ATROUS_PING_MOMENTS, ipos, float4(out_moments_histlen_hf.xy, 0.0, 0.0));
    IMG_STORE(ASVGF_FILTERED_SPEC_A, ipos, out_color_histlen_spec);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    s_lf_shy[local_id.y][local_id.x] = out_color_lf.shY;
    s_lf_cocg[local_id.y][local_id.x] = out_color_lf.CoCg;
    if (local_id.x % GRAD_DWN == 1u && local_id.y % GRAD_DWN == 1u)
        s_depth_width[local_id.y / GRAD_DWN][local_id.x / GRAD_DWN] = motion.a;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 1/3 resolution LF, a bilateral filter anchored on the center pixel of
    // each 3x3 square.
    uint2 lowres_local_id;
    lowres_local_id.x = local_index % (TEMPORAL_GROUP_SIZE / GRAD_DWN);
    lowres_local_id.y = local_index / (TEMPORAL_GROUP_SIZE / GRAD_DWN);

    if (lowres_local_id.y >= uint(TEMPORAL_GROUP_SIZE / GRAD_DWN))
        return;

    uint2 center_shared_pos = lowres_local_id * GRAD_DWN + uint2(1);
    float3 center_normal = unpackHalf4x16(s_normal_lum[center_shared_pos.y + TEMPORAL_FILTER_RADIUS][center_shared_pos.x + TEMPORAL_FILTER_RADIUS]).xyz;
    float center_depth = s_depth[center_shared_pos.y + TEMPORAL_FILTER_RADIUS][center_shared_pos.x + TEMPORAL_FILTER_RADIUS];
    float depth_width = s_depth_width[lowres_local_id.y][lowres_local_id.x];

    SH center_lf;
    center_lf.shY = s_lf_shy[center_shared_pos.y][center_shared_pos.x];
    center_lf.CoCg = s_lf_cocg[center_shared_pos.y][center_shared_pos.x];

    float sum_w = 1.0;
    SH sum_lf = center_lf;

    for (int yy = -1; yy <= 1; yy++) {
        for (int xx = -1; xx <= 1; xx++) {
            if (yy == 0 && xx == 0)
                continue;

            float3 p_normal = unpackHalf4x16(s_normal_lum[int(center_shared_pos.y) + TEMPORAL_FILTER_RADIUS + yy][int(center_shared_pos.x) + TEMPORAL_FILTER_RADIUS + xx]).xyz;
            float p_depth = s_depth[int(center_shared_pos.y) + TEMPORAL_FILTER_RADIUS + yy][int(center_shared_pos.x) + TEMPORAL_FILTER_RADIUS + xx];

            float dist_depth = abs(p_depth - center_depth) * depth_width;
            if (dist_depth < 2.0) {
                float w = pow(max(dot(p_normal, center_normal), 0.0), 8.0);
                SH p_lf;
                p_lf.shY = s_lf_shy[int(center_shared_pos.y) + yy][int(center_shared_pos.x) + xx];
                p_lf.CoCg = s_lf_cocg[int(center_shared_pos.y) + yy][int(center_shared_pos.x) + xx];
                accumulate_SH(sum_lf, p_lf, w);
                sum_w += w;
            }
        }
    }

    float inv_w = 1.0 / sum_w;
    sum_lf.shY *= inv_w;
    sum_lf.CoCg *= inv_w;

    int2 ipos_lowres = int2(group_id) * (TEMPORAL_GROUP_SIZE / GRAD_DWN) + int2(lowres_local_id);
    STORE_SH(ASVGF_ATROUS_PING_LF_SH, ASVGF_ATROUS_PING_LF_COCG, ipos_lowres, sum_lf);
}

//
// asvgf_lf.comp
//

static SH lf_filter_image(thread const VkptCtx &ctx, texture2d<float, access::read> img_shY,
                          texture2d<float, access::read> img_CoCg, int2 ipos_lowres, uint iteration)
{
    int2 ipos_hires = ipos_lowres * GRAD_DWN + int2(1);

    SH color_center_lf = make_SH(img_load(img_shY, ipos_lowres), img_load(img_CoCg, ipos_lowres).xy);

    if (global_ubo.flt_atrous_lf <= float(iteration))
        return color_center_lf;

    float3 geo_normal_center = decode_normal(IMG_LOAD(PT_GEO_NORMAL_A, ipos_hires).x);
    float depth_center = IMG_LOAD(PT_VIEW_DEPTH_A, ipos_hires).x;
    float fwidth_depth = IMG_LOAD(PT_MOTION, ipos_hires).w;

    const int step_size = int(1u << (iteration - 1u));

    SH sum_color_lf = color_center_lf;
    float sum_w_lf = 1.0;

    int field_left = 0;
    int field_right = global_ubo.width / 2;
    if (ipos_hires.x >= field_right) {
        field_left = field_right;
        field_right = global_ubo.width;
    }

    for (int yy = -1; yy <= 1; yy++) {
        for (int xx = -1; xx <= 1; xx++) {
            int2 p_lowres = ipos_lowres + int2(xx, yy) * step_size;
            int2 p_hires = p_lowres * GRAD_DWN + int2(1);

            if (xx == 0 && yy == 0)
                continue;

            float w = float(all(p_hires >= int2(field_left, 0)) && all(p_hires < int2(field_right, global_ubo.height)));

            float3 geo_normal = decode_normal(IMG_LOAD(PT_GEO_NORMAL_A, p_hires).x);
            float depth = IMG_LOAD(PT_VIEW_DEPTH_A, p_hires).x;

            float dist_z = abs(depth_center - depth) * fwidth_depth * global_ubo.flt_atrous_depth;
            w *= exp(-dist_z / float(step_size * GRAD_DWN));
            w *= wavelet_kernel[abs(xx)][abs(yy)];

            float w_lf = w;
            if (global_ubo.flt_atrous_normal_lf > 0.0) {
                float GNdotGN = max(0.0, dot(geo_normal_center, geo_normal));
                w_lf *= pow(GNdotGN, global_ubo.flt_atrous_normal_lf);
            }

            SH c_lf = make_SH(img_load(img_shY, p_lowres), img_load(img_CoCg, p_lowres).xy);

            // Throw away too bright samples on the widest iteration (leaks).
            if (iteration == 3u)
                w_lf *= clamp(1.5 - c_lf.shY.w / color_center_lf.shY.w * 0.25, 0.0, 1.0);

            accumulate_SH(sum_color_lf, c_lf, w_lf);
            sum_w_lf += w_lf;
        }
    }

    SH filtered;
    filtered.shY = sum_color_lf.shY / sum_w_lf;
    filtered.CoCg = sum_color_lf.CoCg / sum_w_lf;
    return filtered;
}

static SH lf_deflicker_image(thread const VkptCtx &ctx, texture2d<float, access::read> img_shY,
                             texture2d<float, access::read> img_CoCg, int2 ipos_lowres)
{
    SH color_center_lf = make_SH(img_load(img_shY, ipos_lowres), img_load(img_CoCg, ipos_lowres).xy);

    SH sum_color_lf = init_SH();
    const float num_pixels = 8.0;
    for (int yy = -1; yy <= 1; yy++) {
        for (int xx = -1; xx <= 1; xx++) {
            int2 p_lowres = ipos_lowres + int2(xx, yy);
            if (xx == 0 && yy == 0)
                continue;
            accumulate_SH(sum_color_lf, make_SH(img_load(img_shY, p_lowres), img_load(img_CoCg, p_lowres).xy), 1.0);
        }
    }

    float max_lum = sum_color_lf.shY.w * global_ubo.flt_atrous_deflicker_lf / num_pixels;
    if (color_center_lf.shY.w > max_lum) {
        float ratio = max_lum / color_center_lf.shY.w;
        color_center_lf.shY *= ratio;
        color_center_lf.CoCg *= ratio;
    }
    return color_center_lf;
}

kernel void asvgf_lf(uint2 gid [[thread_position_in_grid]], DENOISER_PARAMS)
{
    DENOISER_CTX
    int2 ipos = int2(gid);
    if (any(ipos * GRAD_DWN >= int2(global_ubo.current_gpu_slice_width, global_ubo.height)))
        return;

    uint iteration = push.iteration;
    SH filtered_lf;
    switch (iteration) {
    case 0: filtered_lf = lf_deflicker_image(ctx, img.ASVGF_ATROUS_PING_LF_SH_r, img.ASVGF_ATROUS_PING_LF_COCG_r, ipos); break;
    case 1: filtered_lf = lf_filter_image(ctx, img.ASVGF_ATROUS_PONG_LF_SH_r, img.ASVGF_ATROUS_PONG_LF_COCG_r, ipos, iteration); break;
    case 2: filtered_lf = lf_filter_image(ctx, img.ASVGF_ATROUS_PING_LF_SH_r, img.ASVGF_ATROUS_PING_LF_COCG_r, ipos, iteration); break;
    default: filtered_lf = lf_filter_image(ctx, img.ASVGF_ATROUS_PONG_LF_SH_r, img.ASVGF_ATROUS_PONG_LF_COCG_r, ipos, iteration); break;
    }

    if ((iteration & 1u) == 0u)
        STORE_SH(ASVGF_ATROUS_PONG_LF_SH, ASVGF_ATROUS_PONG_LF_COCG, ipos, filtered_lf)
    else
        STORE_SH(ASVGF_ATROUS_PING_LF_SH, ASVGF_ATROUS_PING_LF_COCG, ipos, filtered_lf)
}

//
// asvgf_atrous.comp
//

static void atrous_filter_image(
    thread const VkptCtx &ctx, texture2d_array<float> blue_noise, int2 ipos, uint spec_iteration,
    texture2d<uint, access::read> img_hf, texture2d<uint, access::read> img_spec, texture2d<float, access::read> img_moments,
    thread float3 &filtered_hf, thread float3 &filtered_spec, thread float2 &filtered_moments)
{
    float3 color_center_hf = unpackRGBE(img_load(img_hf, ipos).x);
    float3 color_center_spec = unpackRGBE(img_load(img_spec, ipos).x);
    float2 moments_center = img_load(img_moments, ipos).xy;

    if (global_ubo.flt_atrous_hf <= float(spec_iteration) && global_ubo.flt_atrous_spec <= float(spec_iteration)) {
        filtered_hf = color_center_hf;
        filtered_spec = color_center_spec;
        filtered_moments = moments_center;
        return;
    }

    float3 normal_center = decode_normal(IMG_LOAD(PT_NORMAL_A, ipos).x);
    float depth_center = IMG_LOAD(PT_VIEW_DEPTH_A, ipos).x;
    float fwidth_depth = IMG_LOAD(PT_MOTION, ipos).w;
    float roughness_center = IMG_LOAD(PT_METALLIC_A, ipos).y;

    float lum_mean_hf = 0.0;
    float sigma_l_hf = 0.0;

    float hist_len_hf = IMG_LOAD(ASVGF_HIST_MOMENTS_HF_A, ipos).b;

    if (global_ubo.flt_atrous_lum_hf != 0.0 && hist_len_hf > 1.0) {
        lum_mean_hf = moments_center.x;
        float lum_variance_hf = max(1e-8, moments_center.y - moments_center.x * moments_center.x);
        sigma_l_hf = min(hist_len_hf, global_ubo.flt_atrous_lum_hf) / (2.0 * lum_variance_hf);
    } else {
        sigma_l_hf = 0.0;
    }

    float normal_weight_scale = clamp(hist_len_hf / 8.0, 0.0, 1.0);

    float normal_weight_hf = global_ubo.flt_atrous_normal_hf;
    normal_weight_hf *= normal_weight_scale;

    float normal_weight_spec = RoughnessSquareToSpecPower(square(roughness_center)) * global_ubo.flt_atrous_normal_spec;
    normal_weight_spec = clamp(normal_weight_spec, 8.0, 1024.0);
    normal_weight_spec *= normal_weight_scale;

    const int step_size = int(1u << spec_iteration);

    float3 sum_color_hf = color_center_hf;
    float3 sum_color_spec = color_center_spec;
    float2 sum_moments = moments_center;

    float sum_w_hf = 1.0;
    float sum_w_spec = 1.0;

    int field_left = 0;
    int field_right = global_ubo.width / 2;
    if (ipos.x >= field_right) {
        field_left = field_right;
        field_right = global_ubo.width;
    }

    // Jitter the taps with blue noise to hide the a-trous pattern.
    int2 jitter;
    {
        int texnum = global_ubo.current_frame_idx;
        uint2 texpos = uint2(ipos) & uint2(BLUE_NOISE_RES - 1);
        float jitter_x = blue_noise.read(texpos, uint((texnum + 0) & (NUM_BLUE_NOISE_TEX - 1))).r;
        float jitter_y = blue_noise.read(texpos, uint((texnum + 1) & (NUM_BLUE_NOISE_TEX - 1))).r;
        jitter = int2((float2(jitter_x, jitter_y) - 0.5) * float(step_size));
    }

    float spec_filter_width_scale = clamp(roughness_center * 30.0 - float(spec_iteration), 0.0, 1.0);

    for (int yy = -1; yy <= 1; yy++) {
        for (int xx = -1; xx <= 1; xx++) {
            int2 p = ipos + int2(xx, yy) * step_size + jitter;

            if (xx == 0 && yy == 0)
                continue;

            float w = float(all(p >= int2(field_left, 0)) && all(p < int2(field_right, global_ubo.height)));

            float3 normal = decode_normal(IMG_LOAD(PT_NORMAL_A, p).x);
            float depth = IMG_LOAD(PT_VIEW_DEPTH_A, p).x;
            float roughness = IMG_LOAD(PT_METALLIC_A, p).y;

            float dist_z = abs(depth_center - depth) * fwidth_depth * global_ubo.flt_atrous_depth;
            w *= exp(-dist_z / float(step_size));
            w *= wavelet_kernel[abs(xx)][abs(yy)];

            float w_hf = w;

            float3 c_hf = unpackRGBE(img_load(img_hf, p).x);
            float3 c_spec = unpackRGBE(img_load(img_spec, p).x);
            float2 c_mom = img_load(img_moments, p).xy;
            float l_hf = luminance(c_hf);
            float dist_l_hf = abs(lum_mean_hf - l_hf);

            w_hf *= exp(-dist_l_hf * dist_l_hf * sigma_l_hf);

            float w_spec = w_hf;
            w_spec *= max(0.0, 1.0 - 10.0 * abs(roughness - roughness_center));
            w_spec *= spec_filter_width_scale;

            float NdotN = max(0.0, dot(normal_center, normal));
            if (normal_weight_hf > 0.0)
                w_hf *= pow(NdotN, normal_weight_hf);
            if (normal_weight_spec > 0.0)
                w_spec *= pow(NdotN, normal_weight_spec);

            if (global_ubo.flt_atrous_hf <= float(spec_iteration))
                w_hf = 0.0;
            if (global_ubo.flt_atrous_spec <= float(spec_iteration))
                w_spec = 0.0;

            sum_color_hf += c_hf * w_hf;
            sum_color_spec += c_spec * w_spec;
            sum_moments += c_mom * w_hf;
            sum_w_hf += w_hf;
            sum_w_spec += w_spec;
        }
    }

    filtered_hf = sum_color_hf / sum_w_hf;
    filtered_spec = sum_color_spec / sum_w_spec;
    filtered_moments = sum_moments / sum_w_hf;
}

// The LF channel is denoised at 1/3 resolution; bilateral upsampling.
static SH interpolate_lf(thread const VkptCtx &ctx, int2 ipos)
{
    float depth_center = IMG_LOAD(PT_VIEW_DEPTH_A, ipos).x;
    float fwidth_depth = IMG_LOAD(PT_MOTION, ipos).w;
    float3 geo_normal_center = decode_normal(IMG_LOAD(PT_GEO_NORMAL_A, ipos).x);

    float2 pos_lowres = (float2(ipos) + float2(0.5)) / float(GRAD_DWN) - float2(0.5);
    float2 pos_ld = floor(pos_lowres);
    float2 subpix = fract(pos_lowres - pos_ld);

    SH sum_lf = init_SH();
    float sum_w = 0.0;

    const int2 off[4] = { int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1) };
    float w[4] = {
        (1.0 - subpix.x) * (1.0 - subpix.y),
        (subpix.x) * (1.0 - subpix.y),
        (1.0 - subpix.x) * (subpix.y),
        (subpix.x) * (subpix.y)
    };

    for (int i = 0; i < 4; i++) {
        int2 p_lowres = int2(pos_ld) + off[i];
        int2 p_hires = p_lowres * GRAD_DWN + int2(1);

        float p_depth = IMG_LOAD(PT_VIEW_DEPTH_A, p_hires).x;
        float3 p_geo_normal = decode_normal(IMG_LOAD(PT_GEO_NORMAL_A, p_hires).x);

        float p_w = w[i];
        float dist_depth = abs(p_depth - depth_center) * fwidth_depth;
        p_w *= exp(-dist_depth);
        p_w *= pow(max(0.0, dot(geo_normal_center, p_geo_normal)), 8.0);

        if (p_w > 0.0) {
            SH p_lf = LOAD_SH(ASVGF_ATROUS_PING_LF_SH, ASVGF_ATROUS_PING_LF_COCG, p_lowres);
            accumulate_SH(sum_lf, p_lf, p_w);
            sum_w += p_w;
        }
    }

    if (sum_w > 0.0) {
        float inv_w = 1.0 / sum_w;
        sum_lf.shY *= inv_w;
        sum_lf.CoCg *= inv_w;
    } else {
        sum_lf = LOAD_SH(ASVGF_HIST_COLOR_LF_SH_A, ASVGF_HIST_COLOR_LF_COCG_A, ipos);
    }
    return sum_lf;
}

kernel void asvgf_atrous(uint2 gid [[thread_position_in_grid]], DENOISER_PARAMS)
{
    DENOISER_CTX
    int2 ipos = int2(gid);
    if (any(ipos >= int2(global_ubo.current_gpu_slice_width, global_ubo.height)))
        return;

    uint spec_iteration = push.iteration;
    bool spec_enable_lf = push.bounce_index != 0;   // vkpt: num_bounce_rays >= 0.5

    float3 filtered_hf, filtered_spec;
    float2 filtered_moments;

    switch (spec_iteration) {
    case 0: atrous_filter_image(ctx, blue_noise, ipos, 0, img.ASVGF_ATROUS_PING_HF_r, img.ASVGF_ATROUS_PING_SPEC_r, img.ASVGF_ATROUS_PING_MOMENTS_r, filtered_hf, filtered_spec, filtered_moments); break;
    case 1: atrous_filter_image(ctx, blue_noise, ipos, 1, img.ASVGF_HIST_COLOR_HF_r, img.ASVGF_ATROUS_PONG_SPEC_r, img.ASVGF_ATROUS_PONG_MOMENTS_r, filtered_hf, filtered_spec, filtered_moments); break;
    case 2: atrous_filter_image(ctx, blue_noise, ipos, 2, img.ASVGF_ATROUS_PING_HF_r, img.ASVGF_ATROUS_PING_SPEC_r, img.ASVGF_ATROUS_PING_MOMENTS_r, filtered_hf, filtered_spec, filtered_moments); break;
    default: atrous_filter_image(ctx, blue_noise, ipos, 3, img.ASVGF_ATROUS_PONG_HF_r, img.ASVGF_ATROUS_PONG_SPEC_r, img.ASVGF_ATROUS_PONG_MOMENTS_r, filtered_hf, filtered_spec, filtered_moments); break;
    }

    switch (spec_iteration) {
    case 0:
        IMG_STORE(ASVGF_HIST_COLOR_HF, ipos, uint4(packRGBE(filtered_hf)));
        IMG_STORE(ASVGF_ATROUS_PONG_SPEC, ipos, uint4(packRGBE(filtered_spec)));
        IMG_STORE(ASVGF_ATROUS_PONG_MOMENTS, ipos, float4(filtered_moments, 0.0, 0.0));
        break;
    case 1:
        IMG_STORE(ASVGF_ATROUS_PING_HF, ipos, uint4(packRGBE(filtered_hf)));
        IMG_STORE(ASVGF_ATROUS_PING_SPEC, ipos, uint4(packRGBE(filtered_spec)));
        IMG_STORE(ASVGF_ATROUS_PING_MOMENTS, ipos, float4(filtered_moments, 0.0, 0.0));
        break;
    case 2:
        IMG_STORE(ASVGF_ATROUS_PONG_HF, ipos, uint4(packRGBE(filtered_hf)));
        IMG_STORE(ASVGF_ATROUS_PONG_SPEC, ipos, uint4(packRGBE(filtered_spec)));
        IMG_STORE(ASVGF_ATROUS_PONG_MOMENTS, ipos, float4(filtered_moments, 0.0, 0.0));
        break;
    default:
        break;
    }

    // Compositing on the last iteration.
    if (spec_iteration == 3u) {
        SH filtered_lf = interpolate_lf(ctx, ipos);
        filtered_lf.shY /= STORAGE_SCALE_LF;
        filtered_lf.CoCg /= STORAGE_SCALE_LF;
        filtered_hf /= STORAGE_SCALE_HF;
        filtered_spec /= STORAGE_SCALE_SPEC;

        float3 normal = decode_normal(IMG_LOAD(PT_NORMAL_A, ipos).x);
        float4 base_color = IMG_LOAD(PT_BASE_COLOR_A, ipos);
        float2 metallic_roughness = IMG_LOAD(PT_METALLIC_A, ipos).rg;
        float specular_factor = base_color.a;
        float checkerboard_flags = IMG_LOAD(PT_VIEW_DIRECTION, ipos).a;

        float metallic = metallic_roughness.x;
        float roughness = metallic_roughness.y;

        // Fake specular for rough materials from the LF spherical harmonics.
        if (spec_enable_lf && global_ubo.pt_fake_roughness_threshold < 1.0) {
            float fake_specular_weight = smoothstep(global_ubo.pt_fake_roughness_threshold,
                                                    global_ubo.pt_fake_roughness_threshold + 0.1, roughness);

            if (filtered_lf.shY.w > 0.0 && fake_specular_weight > 0.0) {
                float3 view_direction = IMG_LOAD(PT_VIEW_DIRECTION, ipos).xyz;

                float3 incoming_direction = filtered_lf.shY.xyz / filtered_lf.shY.w * (0.282095 / 0.488603);
                float incoming_len = length(incoming_direction);
                float directionality = incoming_len;

                float scale = 1.0;
                if (directionality >= 1.0) {
                    incoming_direction /= incoming_len;
                } else {
                    incoming_direction = mix(reflect(view_direction, normal), incoming_direction / (incoming_len + 1e-6), float3(directionality));
                    roughness = mix(1.0, roughness, pow(directionality, 3.0));
                    scale = pow(roughness + 1.0, 3.0);
                }

                float3 color = SH_to_irradiance(filtered_lf);
                float3 albedo, base_reflectivity;
                get_reflectivity(base_color.rgb, metallic, albedo, base_reflectivity);
                float3 F;
                float3 brdf = GGX_times_NdotL(view_direction, incoming_direction, normal, roughness, base_reflectivity, 0.0, specular_factor, F);
                float3 fake_specular = color * fake_specular_weight * brdf * scale;
                fake_specular = demodulate_specular(ctx, base_reflectivity, fake_specular);
                filtered_spec += fake_specular;
            }
        }

        float3 projected_lf = project_SH_irradiance(filtered_lf, normal);

        float4 transparent = IMG_LOAD(PT_TRANSPARENT, ipos);
        float3 throughput = IMG_LOAD(PT_THROUGHPUT, ipos).rgb;

        float3 final_color = composite_color(ctx, base_color.rgb, metallic, throughput, projected_lf, filtered_hf, filtered_spec, transparent);

        if (global_ubo.flt_show_gradients != 0.0) {
            float gradient_lf = IMG_LOAD(ASVGF_GRAD_LF_PONG, ipos / GRAD_DWN).r;
            float2 gradient_hf_spec = IMG_LOAD(ASVGF_GRAD_HF_SPEC_PONG, ipos / GRAD_DWN).rg;
            final_color.r += gradient_lf * global_ubo.flt_scale_lf;
            final_color.g += gradient_hf_spec.x * global_ubo.flt_scale_hf;
            final_color.b += gradient_hf_spec.y * global_ubo.flt_scale_spec;
        }

        final_color *= STORAGE_SCALE_HDR;
        IMG_STORE(ASVGF_COLOR, ipos, float4(final_color, checkerboard_flags));
    }
}

//
// compositing.comp (denoiser off)
//

kernel void compositing(uint2 gid [[thread_position_in_grid]], DENOISER_PARAMS)
{
    DENOISER_CTX
    int2 ipos = int2(gid);
    if (any(ipos >= int2(global_ubo.current_gpu_slice_width, global_ubo.height)))
        return;

    int checkerboard_flags = int(IMG_LOAD(PT_VIEW_DIRECTION, ipos).w);

    float3 low_freq = IMG_LOAD(PT_COLOR_LF_SH, ipos).rgb;
    float3 high_freq = unpackRGBE(IMG_LOAD(PT_COLOR_HF, ipos).x);
    float3 specular = unpackRGBE(IMG_LOAD(PT_COLOR_SPEC, ipos).x);
    float3 throughput = IMG_LOAD(PT_THROUGHPUT, ipos).rgb;

    low_freq /= STORAGE_SCALE_LF;
    high_freq /= STORAGE_SCALE_HF;
    specular /= STORAGE_SCALE_SPEC;

    float3 base_color = IMG_LOAD(PT_BASE_COLOR_A, ipos).rgb;
    float2 metal_rough = IMG_LOAD(PT_METALLIC_A, ipos).rg;
    float4 transparent = IMG_LOAD(PT_TRANSPARENT, ipos);

    float3 final_color = composite_color(ctx, base_color, metal_rough.r, throughput, low_freq, high_freq, specular, transparent);
    final_color *= STORAGE_SCALE_HDR;

    IMG_STORE(ASVGF_COLOR, ipos, float4(final_color, float(checkerboard_flags)));
}

//
// checkerboard_interleave.comp
//

kernel void checkerboard_interleave(uint2 gid [[thread_position_in_grid]], DENOISER_PARAMS)
{
    DENOISER_CTX

    int2 opos = int2(gid);
    if (opos.x >= global_ubo.width || opos.y >= global_ubo.height) {
        IMG_STORE(FLAT_COLOR, opos, float4(0.0));
        return;
    }

    // get_input_position()
    int2 ipos = int2(int(gid.x / 2), int(gid.y));
    int px = int(gid.x & 1u);
    int py = int(gid.y & 1u);
    int other_side_offset = global_ubo.width / 2;
    bool is_even_checkerboard = px == py;
    if (global_ubo.pt_swap_checkerboard != 0)
        is_even_checkerboard = !is_even_checkerboard;
    if (!is_even_checkerboard) {
        ipos.x += global_ubo.width / 2;
        other_side_offset = -other_side_offset;
    }

    float4 center = IMG_LOAD(ASVGF_COLOR, ipos);
    float4 color;
    bool use_center_mv = true;
    int checkerboard_flags = int(center.a);
    bool is_checkerboarded_surface = (global_ubo.flt_enable != 0.0) && (popcount_int(checkerboard_flags & CHECKERBOARD_FLAG_FIELD_MASK) > 1);

    if (is_checkerboarded_surface) {
        // safe_load: zero outside the image
        float4 a = IMG_LOAD(ASVGF_COLOR, ipos + int2(other_side_offset, 1));
        float4 b = IMG_LOAD(ASVGF_COLOR, ipos + int2(other_side_offset, -1));
        float4 c = IMG_LOAD(ASVGF_COLOR, ipos + int2(other_side_offset, 0));
        float4 d = IMG_LOAD(ASVGF_COLOR, ipos + int2(other_side_offset + (((opos.x & 1) != 0) ? 1 : -1), 0));

        if (gid.x == 0u || int(gid.x) == global_ubo.width - 1) {
            c = float4(0.0);
            d = float4(0.0);
        }

        float4 neighbors = (a + b + c + d) * 0.25;

        // Motion vectors of the brightest component, not a mix of both.
        if (luminance(neighbors.rgb) > luminance(center.rgb))
            use_center_mv = false;

        color.rgb = mix(center.rgb, neighbors.rgb, 0.5);
        color.a = center.a;
    } else {
        color = center;
    }

    IMG_STORE(FLAT_COLOR, opos, color);

    if (use_center_mv)
        IMG_STORE(FLAT_MOTION, opos, IMG_LOAD(PT_MOTION, ipos));
    else
        IMG_STORE(FLAT_MOTION, opos, IMG_LOAD(PT_MOTION, ipos + int2(other_side_offset, 0)));
}
