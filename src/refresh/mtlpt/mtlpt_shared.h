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

// Structures shared between the Metal shaders and the host code.
// Included from both .metal and .mm translation units, so it must only use
// constructs both compilers understand.

#ifndef MTLPT_SHARED_H
#define MTLPT_SHARED_H

// Material kind and flag bits, shared with the Vulkan backend and its shaders.
#include "../vkpt/shader/constants.h"

#ifdef __METAL_VERSION__
typedef packed_float2 mtl_float2;
typedef packed_float3 mtl_float3;
typedef packed_float4 mtl_float4;
typedef uint          mtl_uint;
typedef int           mtl_int;

// Entry of the bindless texture table. Metal does not allow a bare pointer to
// a texture type, so each slot is a struct holding one texture. The struct is
// the size of an MTLResourceID, which is what the host writes into the buffer.
struct MTLTextureRef {
    metal::texture2d<float> tex;
};
#else
#include <stdint.h>
typedef struct { float x, y; }       mtl_float2;
typedef struct { float x, y, z; }    mtl_float3;
typedef struct { float x, y, z, w; } mtl_float4;
typedef uint32_t                     mtl_uint;
typedef int32_t                      mtl_int;
#endif

// Buffer binding slots. Kept in sync with the [[buffer(n)]] attributes in the
// shaders so host and device agree without string lookups.
enum {
    MTL_BUF_STRETCH_PICS   = 0,
    MTL_BUF_DRAW_UNIFORMS  = 1,
    MTL_BUF_TEXTURE_TABLE  = 2,
};

//
// Physical sky, shared with physical_sky.metal.
//

#define PHYSICAL_SKY_FLAG_NONE           0
#define PHYSICAL_SKY_FLAG_USE_SKYBOX     (1 << 0)
#define PHYSICAL_SKY_FLAG_DRAW_MOUNTAINS (1 << 1)
#define PHYSICAL_SKY_FLAG_DRAW_CLOUDS    (1 << 2)

// Fixed point scale for the sky colour accumulator, which has to be integer
// because it is written with atomics from every face of the cube map.
#define MTL_SKY_ACCUM_SCALE 1024.0f

// Bruneton atmosphere description, uploaded once per preset.
typedef struct {
    mtl_float3 star_irradiance;
    float      star_angular_radius;

    mtl_float3 rayleigh_scattering;
    float      planet_surface_radius;

    mtl_float3 mie_scattering;
    float      planet_atmosphere_radius;

    float      mie_henyey_greenstein_g;
    float      sq_distance_to_horizontal_boundary;
    float      atmosphere_height;
    float      reserved;
} MTLAtmosphereParams;

typedef struct {
    mtl_float3 sun_direction;
    float      sun_cos_half_angle;
    mtl_float3 sun_color;
    float      sun_solid_angle;
    mtl_float3 ground_radiance;
    mtl_uint   face_size;
    mtl_uint   flags;
    mtl_uint   pad0;
    mtl_uint   pad1;
    mtl_uint   pad2;
} MTLPhysicalSkyUniforms;

typedef struct {
    int color[3];
    int count;
} MTLSkyAccumulator;

// Material slots reserved after the world texinfo materials for entity skins.
#define MTL_MAX_ENTITY_MATERIALS 256

// Upper bounds for the per-frame entity geometry buffers.
#define MTL_MAX_ENTITY_VERTICES 262144
#define MTL_MAX_ENTITY_INDICES  (MTL_MAX_ENTITY_VERTICES * 3)

// Transparent effects (the port of transparency.c plus the explosion any-hit
// shader): particles, beams and sprites are camera facing quads, explosion
// and muzzle flash models are their alias triangles. All live in a separate
// acceleration structure with one MTLEffectPrim per triangle.
#define MTL_MAX_EFFECT_VERTICES  131072
#define MTL_MAX_EFFECT_TRIANGLES 131072

#define MTL_EFFECT_PARTICLE  0
#define MTL_EFFECT_BEAM      1
#define MTL_EFFECT_SPRITE    2
#define MTL_EFFECT_EXPLOSION 3   // MATERIAL_KIND_EXPLOSION colour ramp
#define MTL_EFFECT_ADDITIVE  4   // other MCLASS_FLASH style additive models

typedef struct {
    mtl_float2 texcoord;
    mtl_float3 normal;
    mtl_uint   pad;
} MTLEffectVertex;

typedef struct {
    mtl_float4 color;           // rgb already scaled by the hdr factor, a = alpha
    mtl_uint   type;            // MTL_EFFECT_*
    mtl_uint   texture;         // sprites/models: bindless texture index
    float      radius;          // beams
    float      length;          // beams
    // Beams: world -> beam space (beam starts at the origin, points +Z), as
    // three rows so p' = (row0.p, row1.p, row2.p) with p.w = 1.
    mtl_float4 world_to_beam[3];
} MTLEffectPrim;

// Fog volume, same layout as vkpt's ShaderFogVolume (fog.c fills it, and
// indexes the vectors as float arrays, hence the C-side array types).
typedef struct {
#ifdef __METAL_VERSION__
    mtl_float3 mins;
    mtl_uint   is_active;
    mtl_float3 maxs;
    float      pad2;
    mtl_float3 color;
    float      pad3;
    mtl_float4 density;         // xyz: gradient, w: constant
#else
    float      mins[3];
    uint32_t   is_active;
    float      maxs[3];
    float      pad2;
    float      color[3];
    float      pad3;
    float      density[4];
#endif
} MTLFogVolume;

// One queued 2D UI quad. Positions are already in normalized device coords.
typedef struct {
    float    x, y, w, h;
    float    s, t, w_s, h_t;
    mtl_uint color;
    mtl_uint tex_index;
    mtl_uint pad0;
    mtl_uint pad1;
} MTLStretchPic;

typedef struct {
    float    hdr_color_scale;
    float    hdr_saturation_scale;
    mtl_uint is_hdr;
    // Set when the source is already display referred (tone mapper ran).
    mtl_uint tonemapped;
    // Fraction of the source texture actually rendered, for dynamic resolution.
    float    uv_scale_x;
    float    uv_scale_y;
    // Luminance, relative to the exposure target, that maps to pure white.
    float    tm_white_point;
    // Final blit: underwater screen warp (pt_waterwarp), Lanczos upscale when
    // the rendered image is smaller than the window, and the time for the warp.
    mtl_uint water_warp;
    mtl_uint filter_lanczos;
    float    time;
    // Rendered size of the source in pixels, for clamping the warped lookup.
    float    input_width;
    float    input_height;
    float    pad4;
    float    pad5;
} MTLDrawUniforms;

// A single world/model triangle vertex in the ray tracing geometry buffer.
typedef struct {
    mtl_float3 position;
    // Where this vertex was last frame (vkpt's transform_prev / prev pose);
    // equals `position` for static world geometry.
    mtl_float3 prev_position;
    mtl_float3 normal;
    mtl_float3 tangent;
    mtl_float2 texcoord;
    mtl_uint   material;       // low bits: material index; top bits: security camera id
    mtl_uint   alpha_bits;     // float bits: 1.0 opaque, 0.33/0.66 for SURF_TRANS, entity alpha
    mtl_int    cluster;        // BSP cluster the triangle (or its entity) sits in, -1 if none
    mtl_uint   flags;          // MTL_VERTEX_FLAG_*
} MTLTriVertex;

#define MTL_VERTEX_CAMERA_SHIFT   28
#define MTL_VERTEX_MATERIAL_MASK  ((1u << MTL_VERTEX_CAMERA_SHIFT) - 1u)
#define MTL_VERTEX_FLAG_LIGHT     1u   // triangle is in the light poly list (MATERIAL_FLAG_LIGHT per prim)
// Entity triangles carry their entity id above the flag bits (path tracer
// visibility buffer and the gradient reprojection's entity tables).
#define MTL_VERTEX_ENTITY_SHIFT   8u

typedef struct {
    mtl_float3 emissive;
    float      roughness;
    mtl_float3 base_color;
    float      metalness;
    mtl_uint   base_texture;
    mtl_uint   flags;
    mtl_uint   normal_texture;
    mtl_uint   emissive_texture;
    float      bump_scale;
    float      specular_factor;
    mtl_uint   kind_flags;     // MATERIAL_KIND_* | MATERIAL_FLAG_* from the .mat
    float      emissive_factor; // scale for the emissive texture (radiance * material factor)
    mtl_uint   shell;           // SHELL_* bits for entity power-up shells
    mtl_uint   pad0;
    mtl_uint   pad1;
    mtl_uint   pad2;
} MTLMaterial;

// An emissive world triangle, sampled directly so that map lighting does not
// depend on a random bounce happening to find a light surface.
typedef struct {
    mtl_float3 v0;
    float      area;
    mtl_float3 v1;
    float      pad0;
    mtl_float3 v2;
    float      pad1;
    mtl_float3 radiance;
    float      pad2;
} MTLLightPoly;

// Bloom, matching bloom.c's push constants plus the image extents.
typedef struct {
    mtl_uint   output_width;    // live region of the colour image
    mtl_uint   output_height;
    float      pixstep_x;
    float      pixstep_y;
    float      argument_scale;
    float      normalization_scale;
    mtl_uint   num_samples;
    float      intensity;
} MTLBloomUniforms;

// Volumetric sun lighting.
typedef struct {
    mtl_float3 world_center;
    float      intensity;
    mtl_float3 world_size;          // full extent
    float      eccentricity;
    mtl_float3 world_half_size_inv;
    mtl_uint   max_steps;
    // Sun space orthographic projection of the shadow map: NDC component i is
    // dot(rows[i].xyz, world_pos) + rows[i].w, with z in [0, 1].
    mtl_float4 shadow_rows[3];
} MTLGodRaysUniforms;

// FidelityFX Super Resolution 1.0 constants (FsrEasuCon / FsrRcasCon).
typedef struct {
    mtl_float4 easu_con0;
    mtl_float4 easu_con1;
    mtl_float4 easu_con2;
    mtl_float4 easu_con3;
    float      rcas_sharpness;      // exp2(-flt_fsr_sharpness)
    float      container_inv_width; // 1 / full texture size
    float      container_inv_height;
    mtl_uint   is_hdr;
    mtl_uint   input_width;         // rendered region of the input
    mtl_uint   input_height;
    mtl_uint   output_width;        // display size
    mtl_uint   output_height;
    mtl_uint   easu_to_display;     // no RCAS after EASU
    mtl_uint   rcas_after_easu;     // RCAS reads the EASU output, else the TAA output
    mtl_uint   pad0;
    mtl_uint   pad1;
} MTLFsrUniforms;

typedef struct {
    float    tan_half_fov_x;
    float    tan_half_fov_y;
    float    depth_scale_x;     // window pixel -> render pixel
    float    depth_scale_y;
    mtl_uint depth_width;
    mtl_uint depth_height;
    float    color_scale;       // ui_color_scale in HDR
    float    pad;
} MTLDebugLineUniforms;

// Tone mapper working set, the port of vkpt's tonemap_buffer.
typedef struct {
#ifdef __METAL_VERSION__
    metal::atomic_int accumulator[HISTOGRAM_BINS];
#else
    int32_t    accumulator[HISTOGRAM_BINS];
#endif
    float      curve[HISTOGRAM_BINS + 1];   // +1: apply reads bin+1 for the last bin
    float      normalized[HISTOGRAM_BINS];
    float      adapted_luminance;
    float      pad[2];
} MTLToneMapBuffer;

typedef struct {
    mtl_uint   width;               // live region of the colour image
    mtl_uint   height;
    mtl_uint   unscaled_height;     // window height, for the debug chart
    mtl_uint   frame_index;

    mtl_float4 fs_blend_color;
    mtl_float4 fs_colorize;

    float      reset_curve;
    float      frame_time;
    float      knee_w;
    float      knee_a;
    float      knee_b;
    mtl_uint   is_hdr;
    mtl_uint   tm_debug;
    float      hdr_clamp_strength;
    float      ui_color_scale;

    float      tm_exposure_bias;
    float      tm_dyn_range_stops;
    float      tm_exposure_speed_up;
    float      tm_exposure_speed_down;
    float      tm_low_percentile;
    float      tm_high_percentile;
    float      tm_min_luminance;
    float      tm_max_luminance;
    float      tm_noise_stops;
    float      tm_noise_blend;
    float      tm_reinhard;
    float      tm_white_point;
    float      tm_knee_start;
    float      tm_hdr_peak_nits;
    float      tm_hdr_saturation_scale;
    float      tm_blend_scale_border;
    float      tm_blend_scale_center;
    float      tm_blend_scale_fade_exp;
    float      tm_blend_distance_factor;
    float      tm_blend_max_alpha;

    float      weights[14];         // half of the symmetric slope blur kernel
    float      pad[2];
} MTLToneMapUniforms;



#endif // MTLPT_SHARED_H
