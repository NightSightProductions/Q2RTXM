/*
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

// Noise-aware histogram tone mapper (Eilertsen, Mantiuk, Unger), ported from
// tone_mapping_histogram.comp, tone_mapping_curve.comp and
// tone_mapping_apply.comp. See the extensive commentary in the Vulkan shaders
// for the theory; the structure here follows them one to one, with the
// subgroup reductions replaced by threadgroup memory reductions.

#include <metal_stdlib>
#include "../mtlpt_shared.h"

using namespace metal;

#define FIXED_POINT_FRAC_BITS 7
#define FIXED_POINT_FRAC_MULTIPLIER (1 << FIXED_POINT_FRAC_BITS)

constant float min_log_luminance = -24.0;
constant float max_log_luminance = 8.0;
constant float log_luminance_scale = 1.0 / (max_log_luminance - min_log_luminance);
constant float log_luminance_bias = -min_log_luminance * log_luminance_scale;

static inline float luminance(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

//
// Pass 1: histogram of log2 luminance, weighted towards the screen centre.
//

kernel void tone_mapping_histogram(
    uint2 tid                                        [[thread_position_in_grid]],
    uint  local_index                                [[thread_index_in_threadgroup]],
    texture2d<float, access::read> in_color          [[texture(0)]],
    device MTLToneMapBuffer &tm                      [[buffer(1)]],
    constant MTLToneMapUniforms &u                   [[buffer(0)]])
{
    threadgroup atomic_uint s_histogram[HISTOGRAM_BINS];

    int2 ipos = int2(tid);
    int2 screen_size = int2(u.width, u.height);
    bool valid = all(ipos < screen_size);

    float3 input_color = valid ? in_color.read(tid).rgb : float3(0.0);

    if (local_index < HISTOGRAM_BINS)
        atomic_store_explicit(&s_histogram[local_index], 0u, memory_order_relaxed);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid && luminance(input_color) > 0.0) {
        float lum = max(luminance(input_color), exp2(min_log_luminance));
        float biased_log_luminance = log2(lum) * log_luminance_scale + log_luminance_bias;
        float histogram_bin = clamp(biased_log_luminance * HISTOGRAM_BINS, 0.0, HISTOGRAM_BINS - 1.0);

        uint left_bin = uint(histogram_bin);
        uint right_bin = left_bin + 1;

        float weight = clamp(1.0 - length(float2(ipos) / float2(screen_size) - 0.5) * 1.5, 0.01, 1.0);

        float right_weight_f = fract(histogram_bin) * weight;
        float left_weight_f = weight - right_weight_f;

        atomic_fetch_add_explicit(&s_histogram[left_bin], uint(left_weight_f * FIXED_POINT_FRAC_MULTIPLIER),
                                  memory_order_relaxed);
        if (right_bin < HISTOGRAM_BINS)
            atomic_fetch_add_explicit(&s_histogram[right_bin], uint(right_weight_f * FIXED_POINT_FRAC_MULTIPLIER),
                                      memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_index < HISTOGRAM_BINS) {
        uint local_value = atomic_load_explicit(&s_histogram[local_index], memory_order_relaxed);
        if (local_value != 0u)
            atomic_fetch_add_explicit(&tm.accumulator[local_index], (int)local_value, memory_order_relaxed);
    }
}

//
// Pass 2: tone curve, one 128 thread group.
//

static float shared_sum(float val, uint idx, threadgroup float *s)
{
    s[idx] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 64; k >= 1; k /= 2) {
        if (idx < k)
            s[idx] += s[idx + k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    val = s[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return val;
}

static float shared_max(float val, uint idx, threadgroup float *s)
{
    s[idx] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 64; k >= 1; k /= 2) {
        if (idx < k)
            s[idx] = max(s[idx], s[idx + k]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    val = s[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return val;
}

// Inclusive prefix sum across the 128 threads. Leaves the result in shared
// memory too, which the noise floor blend below relies on.
static float prefix_sum(float val, uint idx, threadgroup float *s)
{
    s[idx] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 1; k < HISTOGRAM_BINS; k *= 2) {
        uint block_idx = idx / k;
        float add = 0.0;
        if ((block_idx % 2) == 1)
            add = s[k * block_idx - 1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s[idx] += add;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return s[idx];
}

kernel void tone_mapping_curve(
    uint idx                                         [[thread_index_in_threadgroup]],
    device MTLToneMapBuffer &tm                      [[buffer(1)]],
    constant MTLToneMapUniforms &u                   [[buffer(0)]])
{
    threadgroup float s_shared[HISTOGRAM_BINS];

    float original_hist = 1.0 + float(atomic_load_explicit(&tm.accumulator[idx], memory_order_relaxed)) /
                                FIXED_POINT_FRAC_MULTIPLIER;
    float hist_sum = shared_sum(original_hist, idx, s_shared);
    float hist_max = shared_max(original_hist, idx, s_shared);

    tm.normalized[idx] = original_hist / hist_max;

    original_hist /= hist_sum;

    float bin_log_luminance = (float(idx) / float(HISTOGRAM_BINS)) * (max_log_luminance - min_log_luminance) + min_log_luminance;

    float histogram_cdf = prefix_sum(original_hist, idx, s_shared);
    float histogram_cdf_prev = histogram_cdf - original_hist;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float lower_limit = u.tm_low_percentile * 0.01;
    float upper_limit = u.tm_high_percentile * 0.01;

    float weight_sum = 0.0;
    float bin_sum = 0.0;
    if (lower_limit <= histogram_cdf && histogram_cdf_prev <= upper_limit) {
        weight_sum = bin_log_luminance * original_hist;
        bin_sum = original_hist;
    }

    weight_sum = shared_sum(weight_sum, idx, s_shared);
    bin_sum = shared_sum(bin_sum, idx, s_shared);

    float log_target_lum = weight_sum / max(0.0001, bin_sum);
    log_target_lum = clamp(log_target_lum, log2(u.tm_min_luminance), log2(u.tm_max_luminance));

    if (u.reset_curve == 0.0) {
        float log_old_lum = tm.adapted_luminance;
        if (log_old_lum > 0.0)
            log_old_lum = log2(log_old_lum);
        float speed = (log_old_lum < log_target_lum) ? u.tm_exposure_speed_up : u.tm_exposure_speed_down;
        log_target_lum = mix(log_target_lum, log_old_lum, exp(-u.frame_time * speed));
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (idx == 0)
        tm.adapted_luminance = exp2(log_target_lum);

    if (bin_log_luminance < u.tm_noise_stops)
        original_hist = 0.0;

    float r = u.tm_dyn_range_stops;
    float delta = (max_log_luminance - min_log_luminance) / HISTOGRAM_BINS;
    float r_over_delta = r / delta;

    float rcp_hist = (original_hist > 0.0) ? 1.0 / original_hist : 0.0;
    float thresh = 1e-16;
    float sum_recip = 0.0;
    float len_omega = 0.0;
    float thresh_passed = 0.0;
    for (uint i = 0; i < 16; i++) {
        thresh_passed = step(thresh, original_hist);
        len_omega = shared_sum(thresh_passed, idx, s_shared);
        sum_recip = shared_sum(rcp_hist * thresh_passed, idx, s_shared);
        thresh = (len_omega - r_over_delta) / sum_recip;
    }

    thresh_passed = step(thresh, original_hist);
    len_omega = shared_sum(thresh_passed, idx, s_shared);
    sum_recip = shared_sum(rcp_hist * thresh_passed, idx, s_shared);
    float my_slope = (1.0 + rcp_hist * (r_over_delta - len_omega) / sum_recip) * thresh_passed;

    // Blur the slopes with the symmetric kernel supplied by the host.
    s_shared[idx] = my_slope;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float blurred = my_slope * u.weights[0];
    for (int dx = -13; dx <= 13; dx++) {
        if (dx != 0)
            blurred += u.weights[abs(dx)] * s_shared[clamp(int(idx) + dx, 0, HISTOGRAM_BINS - 1)];
    }
    my_slope = blurred;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float my_tonecurve = prefix_sum(my_slope, idx, s_shared);
    my_tonecurve = (my_tonecurve - my_slope) * delta - r;

    float noise_stop_bin = clamp((u.tm_noise_stops * log_luminance_scale + log_luminance_bias) * HISTOGRAM_BINS,
                                 0.0, HISTOGRAM_BINS - 1.0);
    if (float(idx) < noise_stop_bin) {
        float my_tonecurve_at_ns = s_shared[max(int(noise_stop_bin) - 1, 0)] * delta - r;
        float bin_log_luminance_at_ns = ((noise_stop_bin - 1.0) / float(HISTOGRAM_BINS)) *
                                        (max_log_luminance - min_log_luminance) + min_log_luminance;
        float fudge = -(my_tonecurve_at_ns - bin_log_luminance_at_ns) / log_target_lum;

        float tone_curve_ae = bin_log_luminance - log_target_lum * fudge;
        my_tonecurve = mix(tone_curve_ae, my_tonecurve,
                           mix(smoothstep(0.5 * noise_stop_bin, noise_stop_bin, float(idx)), 1.0, u.tm_noise_blend));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (u.reset_curve == 0.0) {
        float my_old_tonecurve = tm.curve[idx];
        float blend_speed = (my_old_tonecurve < my_tonecurve) ? u.tm_exposure_speed_up : u.tm_exposure_speed_down;
        my_tonecurve = mix(my_tonecurve, my_old_tonecurve, exp(-u.frame_time * blend_speed));
    }

    tm.curve[idx] = my_tonecurve;
    // The apply pass reads curve[left_bin + 1] for the last bin.
    if (idx == HISTOGRAM_BINS - 1)
        tm.curve[HISTOGRAM_BINS] = my_tonecurve;

    atomic_store_explicit(&tm.accumulator[idx], 0, memory_order_relaxed);
}

//
// Pass 3: apply the curve, the autoexposure Reinhard, and the knee.
//

static inline float linear_to_srgb(float x)
{
    return (x <= 0.0031308) ? x * 12.92 : 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

static inline float srgb_to_linear(float x)
{
    return (x <= 0.04045) ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
}

static inline float hash_noise(uint2 p, uint frame)
{
    uint v = p.x * 1973u + p.y * 9277u + frame * 26699u;
    v = (v ^ 61u) ^ (v >> 16u);
    v *= 9u;
    v ^= v >> 4u;
    v *= 0x27d4eb2du;
    v ^= v >> 15u;
    return float(v & 0xffffffu) / 16777216.0;
}

// The Vulkan backend dithers with blue noise; an interleaved hash serves the
// same purpose of breaking up banding in dark gradients.
static float3 srgb_dither(float3 color, uint2 ipos, uint frame)
{
    float3 srgb = float3(linear_to_srgb(color.r), linear_to_srgb(color.g), linear_to_srgb(color.b));
    float3 bumped = srgb + 1.0 / 256.0;
    float3 diff = float3(srgb_to_linear(bumped.r), srgb_to_linear(bumped.g), srgb_to_linear(bumped.b)) - color;
    float noise = hash_noise(ipos, frame);
    return max(float3(0.0), color + diff * (noise - 0.5) * 2.0);
}

static float3 apply_saturation_scale(float3 color, float saturation_scale)
{
    float lum = luminance(color);
    return max(float3(lum) + (color - lum) * saturation_scale, float3(0.0));
}

static float4 show_bar_chart(int2 pos, bool is_curve, constant MTLToneMapUniforms &u, device MTLToneMapBuffer &tm)
{
    float scale = float(u.height) / float(u.unscaled_height);
    float bin_width = 2.0 * scale;
    int2 size = int2(HISTOGRAM_BINS * bin_width, 100.0 * scale);

    int2 TL, BR;
    TL.x = int(10.0 * scale);
    BR.y = int(u.height) - int(22.0 * scale);
    TL.y = BR.y - size.y - 1;
    BR.x = TL.x + size.x + 1;

    if ((pos.x == TL.x || pos.x == BR.x) && (pos.y >= TL.y && pos.y <= BR.y))
        return float4(0.5);
    if ((pos.y == TL.y || pos.y == BR.y) && (pos.x >= TL.x && pos.x <= BR.x))
        return float4(0.5);
    if (pos.x < TL.x || pos.y < TL.y || pos.x > BR.x || pos.y > BR.y)
        return float4(0.0);

    int bin = clamp(int(float(pos.x - TL.x - 1) / bin_width), 0, HISTOGRAM_BINS - 1);
    float pix_value = float(BR.y - pos.y + 1) / float(size.y);
    float bin_value = is_curve ? 0.5 * tm.curve[bin] / u.tm_dyn_range_stops + 1.0 : tm.normalized[bin];

    return (pix_value <= bin_value) ? float4(1.0) : float4(0.0, 0.0, 0.0, 1.0);
}

kernel void tone_mapping_apply(
    uint2 tid                                        [[thread_position_in_grid]],
    texture2d<float, access::read_write> color       [[texture(0)]],
    device MTLToneMapBuffer &tm                      [[buffer(1)]],
    constant MTLToneMapUniforms &u                   [[buffer(0)]])
{
    if (tid.x >= u.width || tid.y >= u.height)
        return;

    int2 ipos = int2(tid);
    int2 screen_size = int2(u.width, u.height);

    float3 input_color = color.read(tid).rgb;

    // Colorization (IR goggles): fs_colorize.rgb is the colour, .a the strength.
    float3 colorized = float3(u.fs_colorize.x, u.fs_colorize.y, u.fs_colorize.z);
    float colorize_lum = luminance(colorized);
    if (colorize_lum > 0.0)
        colorized *= 1.0 / colorize_lum;
    colorized *= luminance(input_color);
    input_color = mix(input_color, colorized, u.fs_colorize.w);

    // Full screen blend (damage flashes, pickups), scaled to the scene exposure.
    float4 blend_color = float4(u.fs_blend_color.x, u.fs_blend_color.y, u.fs_blend_color.z, u.fs_blend_color.w);
    blend_color.rgb *= tm.adapted_luminance / max(luminance(blend_color.rgb), exp2(min_log_luminance));
    blend_color.a = min(blend_color.a, u.tm_blend_max_alpha);

    float2 norm_pos = (float2(ipos) / float2(screen_size) - 0.5) * (2.0 / max(u.tm_blend_distance_factor, 0.1));
    float blend_factor = pow(min(length(norm_pos), 1.0), u.tm_blend_scale_fade_exp);
    float blend_color_scale = mix(u.tm_blend_scale_center, u.tm_blend_scale_border, smoothstep(0.0, 1.0, blend_factor));

    input_color = mix(input_color, blend_color.rgb, blend_color.a * saturate(blend_color_scale));

    float lum = max(luminance(input_color), exp2(min_log_luminance));

    float biased_log_luminance = log2(lum) * log_luminance_scale + log_luminance_bias;
    float histogram_bin = clamp(biased_log_luminance * HISTOGRAM_BINS, 0.0, float(HISTOGRAM_BINS));
    uint left_bin = uint(histogram_bin);
    uint right_bin = min(left_bin + 1, uint(HISTOGRAM_BINS));
    float right_weight_f = fract(histogram_bin);
    float left_weight_f = 1.0 - right_weight_f;

    float out_log_luminance = left_weight_f * tm.curve[left_bin] + right_weight_f * tm.curve[right_bin];
    float out_luminance = exp2(out_log_luminance + u.tm_exposure_bias);

    float3 mapped_color = input_color * out_luminance / lum;

    if (u.is_hdr == 0u) {
        float3 step_value = step(u.tm_knee_start, mapped_color);
        mapped_color = mix(mapped_color,
                           (u.knee_w * mapped_color + u.knee_a) / max(float3(1e-6), mapped_color + u.knee_b),
                           step_value);
    }

    // Autoexposure Reinhard on luminance only.
    float adapted_luminance = tm.adapted_luminance;
    float scaled_luminance = exp2(u.tm_exposure_bias - 2.0) * lum / adapted_luminance;
    float white_point = (u.is_hdr != 0u) ? max(u.tm_white_point, u.tm_hdr_peak_nits / 80.0) : u.tm_white_point;
    float white_point_squared = white_point * white_point;
    float mapped_luminance = (scaled_luminance * (1.0 + scaled_luminance / white_point_squared)) / (1.0 + scaled_luminance);
    float3 ae_mapped_color = input_color * mapped_luminance / lum;

    mapped_color = mix(mapped_color, ae_mapped_color, u.tm_reinhard);

    if (u.is_hdr == 0u) {
        mapped_color = saturate(mapped_color);
        mapped_color = srgb_dither(mapped_color, tid, u.frame_index);
    } else {
        float nits_factor = u.tm_hdr_peak_nits / 80.0;
        float color_scale = mix(nits_factor, min(nits_factor, u.ui_color_scale), u.hdr_clamp_strength);
        mapped_color *= color_scale;
        mapped_color = apply_saturation_scale(mapped_color, u.tm_hdr_saturation_scale * 0.01);
    }

    if (u.tm_debug != 0u) {
        float4 overlay = show_bar_chart(ipos, u.tm_debug == 2u, u, tm);
        if (u.is_hdr != 0u)
            overlay.rgb *= u.ui_color_scale;
        mapped_color = mix(mapped_color, overlay.rgb, overlay.a);
    }

    color.write(float4(mapped_color, 1.0), tid);
}
