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

// Bloom: port of bloom_downscale.comp, bloom_blur.comp and
// bloom_composite.comp. The image is box-downsampled 4x, blurred with a
// separable Gaussian, then blended back over the full resolution image.

#include <metal_stdlib>
#include "../mtlpt_shared.h"

using namespace metal;

kernel void bloom_downscale(
    uint2 tid                                       [[thread_position_in_grid]],
    texture2d<float, access::read>  in_color        [[texture(0)]],
    texture2d<float, access::write> out_bloom       [[texture(1)]],
    constant MTLBloomUniforms &u                    [[buffer(0)]])
{
    if (tid.x >= out_bloom.get_width() || tid.y >= out_bloom.get_height())
        return;

    float4 result = float4(0.0);
    int count = 0;

    for (uint yy = 0; yy < 4; yy++) {
        for (uint xx = 0; xx < 4; xx++) {
            uint2 input_pos = tid * 4 + uint2(xx, yy);
            if (input_pos.x < u.output_width && input_pos.y < u.output_height) {
                result += in_color.read(input_pos);
                count++;
            }
        }
    }

    if (count > 0)
        result /= float(count);

    out_bloom.write(result, tid);
}

kernel void bloom_blur(
    uint2 tid                                       [[thread_position_in_grid]],
    texture2d<float>                in_bloom        [[texture(0)]],
    texture2d<float, access::write> out_bloom       [[texture(1)]],
    constant MTLBloomUniforms &u                    [[buffer(0)]])
{
    if (tid.x >= out_bloom.get_width() || tid.y >= out_bloom.get_height())
        return;

    constexpr sampler linear_sampler(filter::linear, address::clamp_to_edge, coord::normalized);

    int2 ipos = int2(tid);
    int2 bloom_extent = int2(u.output_width / 4, u.output_height / 4);

    if (any(ipos >= bloom_extent)) {
        out_bloom.write(float4(0.0), tid);
        return;
    }

    float2 bloom_sample_extent = float2(bloom_extent) - 0.5;
    float2 tex_size = float2(in_bloom.get_width(), in_bloom.get_height());
    float2 pixstep = float2(u.pixstep_x, u.pixstep_y);

    float4 bloom_input = in_bloom.read(tid);
    if (any(isnan(bloom_input)) || any(isinf(bloom_input))) {
        out_bloom.write(float4(0.0, 0.0, 0.0, 1.0), tid);
        return;
    }

    float3 result = bloom_input.rgb;

    for (float x = 1.0; x < float(u.num_samples); x += 2.0) {
        float w1 = exp(x * x * u.argument_scale);
        float w2 = exp((x + 1.0) * (x + 1.0) * u.argument_scale);

        float w12 = w1 + w2;
        float p = w2 / w12;
        float2 offset = pixstep * (x + p);

        float2 pos1 = clamp(float2(ipos) + 0.5 + offset, float2(0.0), bloom_sample_extent);
        float2 pos2 = clamp(float2(ipos) + 0.5 - offset, float2(0.0), bloom_sample_extent);

        float3 pix = in_bloom.sample(linear_sampler, pos1 / tex_size, level(0)).rgb;
        if (any(isnan(pix)))
            pix = float3(0.0);
        result += pix * w12;

        pix = in_bloom.sample(linear_sampler, pos2 / tex_size, level(0)).rgb;
        if (any(isnan(pix)))
            pix = float3(0.0);
        result += pix * w12;
    }

    result *= u.normalization_scale;

    out_bloom.write(float4(result, bloom_input.a), tid);
}

kernel void bloom_composite(
    uint2 tid                                            [[thread_position_in_grid]],
    texture2d<float>                     in_bloom        [[texture(0)]],
    texture2d<float, access::read_write> color           [[texture(1)]],
    constant MTLBloomUniforms &u                         [[buffer(0)]])
{
    if (tid.x >= u.output_width || tid.y >= u.output_height)
        return;

    constexpr sampler linear_sampler(filter::linear, address::clamp_to_edge, coord::normalized);

    // Both textures are allocated at the full window size; only the
    // output_width x output_height corner is live, and the bloom texture is
    // a quarter of that, so the same normalized uv addresses both.
    float2 uv = saturate((float2(tid) + 0.5) / float2(color.get_width(), color.get_height()));

    float4 src = in_bloom.sample(linear_sampler, uv, level(0));
    float4 dst = color.read(tid);

    if (any(isnan(dst)) || any(isinf(dst)))
        return;

    src.a = 0.0;
    dst.a = 1.0;

    float4 result = mix(dst, src, u.intensity);
    color.write(result, tid);
}
