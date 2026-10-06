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

// Volumetric sun lighting: ports of vkpt's god_rays.comp and
// god_rays_filter.comp, marching through a shadow map rasterized from the
// sun (shadow_map.c). Runs in the checkerboard field layout after the primary
// rays (pass 0) and the first reflection pass (pass 1), and blends the result
// into PT_TRANSPARENT before compositing.
//
// Metal: the shadow map only has the opaque layer, so the inscatter under
// water is not attenuated by the distance to the water surface.

#include <metal_stdlib>
#include "vkpt_common.h"

using namespace metal;

static inline int2 GetRotatedGridOffset(int2 pixelPos)
{
    return int2(pixelPos.y & 1, 1 - (pixelPos.x & 1));
}

static inline float3 shadow_ndc(constant MTLGodRaysUniforms &g, float3 p)
{
    return float3(dot(float3(g.shadow_rows[0].xyz), p) + g.shadow_rows[0].w,
                  dot(float3(g.shadow_rows[1].xyz), p) + g.shadow_rows[1].w,
                  dot(float3(g.shadow_rows[2].xyz), p) + g.shadow_rows[2].w);
}

static float GetShadow(depth2d<float> shadow_map, constant MTLGodRaysUniforms &g, float3 worldPos)
{
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    float3 ndc = shadow_ndc(g, worldPos);
    float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - 0.5 * ndc.y);
    if (any(uv < 0.0) || any(uv > 1.0))
        return 1.0;
    // Metal: the nearest occluder of the 2x2 texel footprint. Rasterization
    // leaves single texel pinholes at the BSP's T-junctions (where cliffs
    // meet the ground), which otherwise show as chains of lit dots.
    float4 d = shadow_map.gather(s, uv);
    return min(min(d.x, d.y), min(d.z, d.w)) > ndc.z ? 1.0 : 0.0;
}

// Depth only pass of the shadow map; vertex_id is the index buffer value.
struct ShadowMapOut {
    float4 position [[position]];
};

vertex ShadowMapOut shadow_map_vertex(
    uint vertex_id                          [[vertex_id]],
    device const MTLTriVertex *vertices     [[buffer(0)]],
    constant MTLGodRaysUniforms &g          [[buffer(1)]])
{
    ShadowMapOut out;
    out.position = float4(shadow_ndc(g, float3(vertices[vertex_id].position)), 1.0);
    return out;
}

static float ScatterPhase_HenyeyGreenstein(float cosa, float g)
{
    // "normalized" Henyey-Greenstein
    float g_sqr = g * g;
    float num = (1.0 - abs(g));
    float denom = sqrt(max(1.0 - 2.0 * g * cosa + g_sqr, 0.0));
    float frac = num / denom;
    float scale = g_sqr + (1.0 - g_sqr) / (4.0 * M_PI);
    return scale * (frac * frac * frac);
}

static bool IntersectRayBox(float3 origin, float3 direction, float3 mins, float3 maxs, thread float &tIn, thread float &tOut)
{
    float3 t1 = (mins - origin) / direction;
    float3 t2 = (maxs - origin) / direction;
    tIn = max3(min(t1, t2).x, min(t1, t2).y, min(t1, t2).z);
    tOut = min3(max(t1, t2).x, max(t1, t2).y, max(t1, t2).z);
    return tIn < tOut && tOut > 0.0;
}

static inline float getDensity(thread const VkptCtx &ctx, float3 p)
{
    float3 bounds = clamp(3.0 - 2.0 * abs((p - global_ubo.world_center.xyz) * global_ubo.world_half_size_inv.xyz), float3(0.0), float3(1.0));
    return bounds.x * bounds.y * bounds.z;
}

static inline float getStep(float t, float density, float min_step)
{
    return max(min_step, mix(20.0, 5.0, density));
}

static inline float3 get_extinction_factors(thread const VkptCtx &ctx, int medium)
{
    float3 factors = float3(0.0);
    if (medium == MEDIUM_WATER)
        factors = float3(0.035, 0.013, 0.012);
    else if (medium == MEDIUM_SLIME)
        factors = float3(0.200, 0.010, 0.050);
    else if (medium == MEDIUM_LAVA)
        factors = float3(0.001, 0.100, 0.300);
    return factors * global_ubo.pt_water_density;
}

kernel void god_rays_trace(
    uint2 gid                                         [[thread_position_in_grid]],
    constant GlobalUbo &ubo                           [[buffer(VKPT_BUF_UBO)]],
    device const VkptImages &img                      [[buffer(VKPT_BUF_IMAGES)]],
    constant VkptPush &push                           [[buffer(VKPT_BUF_PUSH)]],
    constant MTLGodRaysUniforms &g                    [[buffer(VKPT_BUF_PUSH + 1)]],
    texture2d_array<float> blue_noise                 [[texture(VKPT_TEX_BLUE_NOISE)]],
    depth2d<float> shadow_map                         [[texture(3)]],
    texture2d<float, access::read_write> intermediate [[texture(4)]])
{
    VkptCtx ctx = { &ubo, &img };

    float3 sun_color = float3(global_ubo.sun_color);
    if (luminance(sun_color) == 0.0) {
        intermediate.write(float4(0.0), gid);
        return;
    }

    int2 pixelPos = int2(gid);
    int2 sourcePixelPos = pixelPos * 2 + GetRotatedGridOffset(pixelPos);

    if (sourcePixelPos.x >= global_ubo.width || sourcePixelPos.y >= global_ubo.height)
        return;

    if (push.iteration != 0u) {
        float view_depth = IMG_LOAD(PT_VIEW_DEPTH_A, sourcePixelPos).x;
        if (view_depth >= 0.0)
            return;
    }

    float4 position_material = IMG_LOAD(PT_SHADING_POSITION, sourcePixelPos);
    float3 surface_pos = position_material.xyz;
    uint material_id = as_type<uint>(position_material.w);
    int medium = int((material_id & MATERIAL_LIGHT_STYLE_MASK) >> MATERIAL_LIGHT_STYLE_SHIFT);
    float3 direction = IMG_LOAD(PT_VIEW_DIRECTION, sourcePixelPos).xyz;
    float4 throughput_distance = IMG_LOAD(PT_GODRAYS_THROUGHPUT_DIST, sourcePixelPos);
    float distance = throughput_distance.w;

    float3 original_pos = surface_pos - direction * distance;

    float cosa = dot(direction, float3(global_ubo.sun_direction));
    float eccentricity = global_ubo.god_rays_eccentricity;
    if (medium != MEDIUM_NONE)
        eccentricity = 0.5;
    float phase = ScatterPhase_HenyeyGreenstein(cosa, eccentricity);

    float offset = 0.0;
    float tIn, tOut;
    if (IntersectRayBox(original_pos, direction,
                        global_ubo.world_center.xyz - global_ubo.world_size.xyz * 0.75,
                        global_ubo.world_center.xyz + global_ubo.world_size.xyz * 0.75, tIn, tOut)) {
        if (tIn > distance) {
            intermediate.write(float4(0.0), gid);
            return;
        }
        offset = max(tIn, offset);
        distance = min(tOut, distance);
    } else {
        intermediate.write(float4(0.0), gid);
        return;
    }

    // Metal: bound the march on very long rays (mtl_gr_max_steps).
    float min_step = max(1.0, (distance - offset) / float(max(g.max_steps, 1u)));

    float3 currentPos = original_pos + direction * offset;
    float density = getDensity(ctx, currentPos);

    float3 extinction_factors = (medium == MEDIUM_NONE) ? float3(0.0001) : get_extinction_factors(ctx, medium);

    float3 throughput = throughput_distance.xyz;
    float3 inscatter = float3(0.0);

    {
        uint texnum = uint(global_ubo.current_frame_idx) & (NUM_BLUE_NOISE_TEX - 1);
        uint2 texpos = uint2(pixelPos) & uint2(BLUE_NOISE_RES - 1);
        float noise = blue_noise.read(texpos, texnum).r;
        offset += getStep(offset, density, min_step) * (noise - 1.0);
    }

    for (uint i = 0; i < 4096u; i++) {
        float d = getDensity(ctx, currentPos);
        float stepLength = getStep(offset, d, min_step);

        offset += stepLength;
        if (offset >= distance)
            break;

        currentPos = original_pos + direction * offset;

        const float3 shadowBias = -float3(global_ubo.sun_direction) * 20.0;
        float shadow = GetShadow(shadow_map, g, currentPos + shadowBias);

        float3 differentialInscatter = (shadow * phase * stepLength * d) * throughput;
        inscatter += differentialInscatter;
        throughput *= exp(-(stepLength * d) * extinction_factors);
    }

    float3 inscatterColor = inscatter * sun_color * global_ubo.god_rays_intensity * 0.0001;
    if (medium != MEDIUM_NONE)
        inscatterColor *= global_ubo.pt_water_density * 30.0;

    float4 prevColor = (push.iteration != 0u) ? intermediate.read(gid) : float4(0.0);
    intermediate.write(float4(inscatterColor + prevColor.rgb, 1.0), gid);
}

kernel void god_rays_filter(
    uint2 gid                                         [[thread_position_in_grid]],
    constant GlobalUbo &ubo                           [[buffer(VKPT_BUF_UBO)]],
    device const VkptImages &img                      [[buffer(VKPT_BUF_IMAGES)]],
    texture2d<float, access::read_write> intermediate [[texture(4)]],
    texture2d<float, access::read_write> transparent  [[texture(5)]])
{
    VkptCtx ctx = { &ubo, &img };

    if (any(int2(gid) >= int2(global_ubo.current_gpu_slice_width, global_ubo.height)))
        return;

    int2 i_position = int2(gid);

    float4 result = float4(0.0);
    float4 fallbackResult = float4(0.0);
    float weightSum = 0.0;
    float referenceViewDepth = IMG_LOAD(PT_VIEW_DEPTH_A, i_position).r;
    int2 lowResOrigin = i_position >> 1;

    int field_left = 0;
    int field_right = global_ubo.width / 2;
    if (i_position.x >= field_right) {
        field_left = field_right;
        field_right = global_ubo.width;
    }

    int2 low_size = int2(intermediate.get_width(), intermediate.get_height());
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int2 lowResPos = lowResOrigin + int2(dx, dy);
            int2 highResPos = lowResPos * 2 + GetRotatedGridOffset(lowResPos);

            if (highResPos.x < field_left || highResPos.x >= field_right || highResPos.y < 0 || highResPos.y >= global_ubo.height)
                continue;
            if (any(lowResPos < 0) || any(lowResPos >= low_size))
                continue;

            float4 color = intermediate.read(uint2(lowResPos));
            float viewDepth = IMG_LOAD(PT_VIEW_DEPTH_A, highResPos).r;

            float weight = clamp(1.0 - 10.0 * abs(viewDepth - referenceViewDepth) / abs(referenceViewDepth), 0.0, 1.0);
            weight *= clamp(5.0 - length(float2(highResPos - i_position)), 0.0, 1.0);

            result += color * weight;
            fallbackResult += color;
            weightSum += weight;
        }
    }

    result = (weightSum > 0.0) ? result / weightSum : fallbackResult / 16.0;

    // Additive, like vkpt: alpha blending would darken areas the sun never
    // reaches, since only the extinction of the sun is modelled.
    float4 originalColor = transparent.read(gid);
    transparent.write(float4(originalColor.rgb + result.rgb, originalColor.a), gid);
}
