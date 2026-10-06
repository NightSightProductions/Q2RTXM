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

// The Vulkan backend's global uniform buffer (global_ubo.h) and screen image
// table (global_textures.h), shared by the host and the Metal shaders so the
// path tracer and denoiser shaders can be ported from vkpt line by line.
//
// Layout rules: vec3/vec2 are packed (4 byte aligned), vec4/mat4/uvec4 are 16
// byte aligned, on both sides. vkpt orders the members for std140, which keeps
// every vec4 and mat4 on a 16 byte boundary, so both compilers produce the
// same offsets.

#ifndef MTLPT_VKPT_H
#define MTLPT_VKPT_H

#include "mtlpt_shared.h"

#ifdef __METAL_VERSION__
typedef metal::packed_float2 ubo_vec2;
typedef metal::packed_float3 ubo_vec3;
typedef metal::float4        ubo_vec4;
typedef metal::uint4         ubo_uvec4;
typedef metal::float4x4      ubo_mat4;
typedef int           ubo_int;
typedef uint          ubo_uint;
#else
typedef struct { float x, y; }    ubo_vec2;
typedef struct { float x, y, z; } ubo_vec3;
typedef struct __attribute__((aligned(16))) { float x, y, z, w; } ubo_vec4;
typedef struct __attribute__((aligned(16))) { uint32_t x, y, z, w; } ubo_uvec4;
typedef struct __attribute__((aligned(16))) { float m[16]; } ubo_mat4;   // column major
typedef int32_t  ubo_int;
typedef uint32_t ubo_uint;
#endif

// global_ubo.h DynLightData.
typedef struct {
    ubo_vec3 center;
    float    radius;
    ubo_vec3 color;
    ubo_uint type;              // DYNLIGHT_* | (spot emission profile << 16)
    ubo_vec3 spot_direction;
    ubo_uint spot_data;
} VkptDynLight;

// The UBO_CVAR_LIST of global_ubo.h: registered as console variables with
// these defaults and copied into the UBO every frame.
#define VKPT_UBO_CVAR_LIST \
    UBO_CVAR_DO(flt_antilag_hf, 1) \
    UBO_CVAR_DO(flt_antilag_lf, 0.2) \
    UBO_CVAR_DO(flt_antilag_spec, 2) \
    UBO_CVAR_DO(flt_antilag_spec_motion, 0.004) \
    UBO_CVAR_DO(flt_atrous_depth, 0.5) \
    UBO_CVAR_DO(flt_atrous_deflicker_lf, 2) \
    UBO_CVAR_DO(flt_atrous_hf, 4) \
    UBO_CVAR_DO(flt_atrous_lf, 4) \
    UBO_CVAR_DO(flt_atrous_spec, 3) \
    UBO_CVAR_DO(flt_atrous_lum_hf, 16) \
    UBO_CVAR_DO(flt_atrous_normal_hf, 64) \
    UBO_CVAR_DO(flt_atrous_normal_lf, 8) \
    UBO_CVAR_DO(flt_atrous_normal_spec, 1) \
    UBO_CVAR_DO(flt_enable, 1) \
    UBO_CVAR_DO(flt_fixed_albedo, 0) \
    UBO_CVAR_DO(flt_grad_weapon, 0.25) \
    UBO_CVAR_DO(flt_min_alpha_color_hf, 0.02) \
    UBO_CVAR_DO(flt_min_alpha_color_lf, 0.01) \
    UBO_CVAR_DO(flt_min_alpha_color_spec, 0.01) \
    UBO_CVAR_DO(flt_min_alpha_moments_hf, 0.01) \
    UBO_CVAR_DO(flt_scale_hf, 1) \
    UBO_CVAR_DO(flt_scale_lf, 1) \
    UBO_CVAR_DO(flt_scale_overlay, 1.0) \
    UBO_CVAR_DO(flt_scale_spec, 1) \
    UBO_CVAR_DO(flt_show_gradients, 0) \
    UBO_CVAR_DO(flt_taa, 2) \
    UBO_CVAR_DO(flt_taa_anti_sparkle, 0.25) \
    UBO_CVAR_DO(flt_taa_variance, 1.0) \
    UBO_CVAR_DO(flt_taa_history_weight, 0.95) \
    UBO_CVAR_DO(flt_temporal_hf, 1) \
    UBO_CVAR_DO(flt_temporal_lf, 1) \
    UBO_CVAR_DO(flt_temporal_spec, 1) \
    UBO_CVAR_DO(pt_aperture, 2.0) \
    UBO_CVAR_DO(pt_aperture_angle, 0) \
    UBO_CVAR_DO(pt_aperture_type, 0) \
    UBO_CVAR_DO(pt_beam_softness, 1.0) \
    UBO_CVAR_DO(pt_bump_scale, 1.0) \
    UBO_CVAR_DO(pt_cameras, 1) \
    UBO_CVAR_DO(pt_direct_polygon_lights, 1) \
    UBO_CVAR_DO(pt_direct_roughness_threshold, 0.18) \
    UBO_CVAR_DO(pt_direct_dyn_lights, 1) \
    UBO_CVAR_DO(pt_direct_sun_light, 1) \
    UBO_CVAR_DO(pt_envmap_brightness, 0.5) \
    UBO_CVAR_DO(pt_envmap_desaturate, 0.1) \
    UBO_CVAR_DO(pt_explosion_brightness, 4.0) \
    UBO_CVAR_DO(pt_fake_roughness_threshold, 0.20) \
    UBO_CVAR_DO(pt_focus, 200) \
    UBO_CVAR_DO(pt_fog_brightness, 0.01) \
    UBO_CVAR_DO(pt_indirect_polygon_lights, 1) \
    UBO_CVAR_DO(pt_indirect_dyn_lights, 1) \
    UBO_CVAR_DO(pt_light_stats, 1) \
    UBO_CVAR_DO(pt_max_log_sky_luminance, -3) \
    UBO_CVAR_DO(pt_min_log_sky_luminance, -10) \
    UBO_CVAR_DO(pt_metallic_override, -1) \
    UBO_CVAR_DO(pt_ndf_trim, 0.9) \
    UBO_CVAR_DO(pt_num_bounce_rays, 1) \
    UBO_CVAR_DO(pt_particle_softness, 0.7) \
    UBO_CVAR_DO(pt_particle_brightness, 100) \
    UBO_CVAR_DO(pt_reflect_refract, 2) \
    UBO_CVAR_DO(pt_roughness_override, -1) \
    UBO_CVAR_DO(pt_specular_anti_flicker, 2) \
    UBO_CVAR_DO(pt_specular_mis, 1) \
    UBO_CVAR_DO(pt_show_sky, 0) \
    UBO_CVAR_DO(pt_sun_bounce_range, 2000) \
    UBO_CVAR_DO(pt_sun_specular, 1.0) \
    UBO_CVAR_DO(pt_texture_lod_bias, 0) \
    UBO_CVAR_DO(pt_toksvig, 1) \
    UBO_CVAR_DO(pt_thick_glass, 0) \
    UBO_CVAR_DO(pt_water_density, 0.5)

typedef struct {
    ubo_int    current_frame_idx;
    ubo_int    width;
    ubo_int    height;
    ubo_int    current_gpu_slice_width;

    ubo_int    medium;
    float      time;
    ubo_int    first_person_model;
    ubo_int    environment_type;

    ubo_vec3   sun_direction;
    float      bloom_intensity;
    ubo_vec3   sun_tangent;
    float      sun_tan_half_angle;
    ubo_vec3   sun_bitangent;
    float      sun_bounce_scale;
    ubo_vec3   sun_color;
    float      sun_cos_half_angle;
    ubo_vec3   sun_direction_envmap;
    ubo_int    sun_visible;

    float      sky_transmittance;
    float      sky_phase_g;
    float      sky_amb_phase_g;
    float      sun_solid_angle;

    ubo_vec3   physical_sky_ground_radiance;
    ubo_int    physical_sky_flags;

    float      sky_scattering;
    float      temporal_blend_factor;
    ubo_int    planet_albedo_map;
    ubo_int    planet_normal_map;

    ubo_int    num_dyn_lights;
    ubo_int    num_static_lights;
    ubo_uint   num_static_primitives;
    ubo_int    cluster_debug_index;

    ubo_int    water_normal_texture;
    float      pt_env_scale;
    float      cylindrical_hfov;
    float      cylindrical_hfov_prev;

    ubo_int    pt_swap_checkerboard;
    float      shadow_map_depth_scale;
    float      god_rays_intensity;
    float      god_rays_eccentricity;

    ubo_int    num_cameras;
    ubo_int    screen_image_width;
    ubo_int    screen_image_height;
    ubo_int    prev_gpu_slice_width;

    ubo_int    prev_width;
    ubo_int    prev_height;
    float      inv_width;
    float      inv_height;

    ubo_int    unscaled_width;
    ubo_int    unscaled_height;
    ubo_int    taa_image_width;
    ubo_int    taa_image_height;

    ubo_int    taa_output_width;
    ubo_int    taa_output_height;
    ubo_int    prev_taa_output_width;
    ubo_int    prev_taa_output_height;

    ubo_vec2   projection_fov_scale;
    ubo_vec2   projection_fov_scale_prev;

    ubo_vec3   padding;
    ubo_int    pt_projection;

    ubo_vec2   sub_pixel_jitter;
    float      prev_adapted_luminance;
    float      tonemap_hdr_clamp_strength;

    ubo_vec4   world_center;
    ubo_vec4   world_size;
    ubo_vec4   world_half_size_inv;

    VkptDynLight dyn_light_data[MAX_LIGHT_SOURCES];
    ubo_vec4   cam_pos;
    ubo_mat4   V;
    ubo_mat4   invV;
    ubo_mat4   V_prev;
    ubo_mat4   P;
    ubo_mat4   invP;
    ubo_mat4   P_prev;
    ubo_mat4   invP_prev;
    ubo_mat4   environment_rotation_matrix;
    ubo_mat4   security_camera_data[MAX_CAMERAS];
    MTLFogVolume fog_volumes[MAX_FOG_VOLUMES];

    ubo_int    weapon_left_handed;
    float      ui_color_scale;

    // Metal additions: the scene layer differs from vkpt's (see path_tracer.metal).
    ubo_uint   num_effect_triangles;
    ubo_uint   world_transparent_first_prim;   // world index buffer offsets of TLAS instances
    ubo_uint   world_sky_first_prim;
    ubo_uint   world_masked_first_prim;
    ubo_uint   entity_viewer_first_prim;       // entity index buffer: regular | viewer | weapon
    ubo_uint   entity_weapon_first_prim;
    ubo_uint   entity_transparent_first_prim;  // regular entities: opaque | transparent
    ubo_uint   num_clusters;                   // 0: no vis data
    ubo_uint   light_stats_size;               // uints per light stats buffer (one of three)
    ubo_uint   has_skybox;
    ubo_uint   has_physical_sky;
    float      sky_rotate;
    ubo_vec3   sky_axis;
    float      sky_luminance;                  // stands in for sun_color_ubo.sky_luminance
    ubo_vec2   viewport_scale;                 // player setup view (vkpt viewport_proj)
    ubo_vec2   viewport_offset;
    float      lava_emissive;
    ubo_uint   has_world;
    ubo_uint   has_masked_world;               // the alpha tested world instance exists
    ubo_uint   material_filter;                // pt_nearest: 0 anisotropic, 1 mixed, 2 nearest

#define UBO_CVAR_DO(name, default_value) float name;
    VKPT_UBO_CVAR_LIST
#undef UBO_CVAR_DO
} GlobalUbo;

//
// Screen images, global_textures.h LIST_IMAGES / LIST_IMAGES_A_B.
// IMG_DO(name, pixel format, shader element type, size class)
// Size classes: FULL = render target size, GRAD = 1/3 of it rounded up.
//

#define VKPT_LIST_IMAGES \
    IMG_DO(PT_MOTION,                  RGBA16Float, float, FULL) \
    IMG_DO(PT_TRANSPARENT,             RGBA16Float, float, FULL) \
    IMG_DO(ASVGF_HIST_COLOR_HF,        R32Uint,     uint,  FULL) \
    IMG_DO(ASVGF_ATROUS_PING_LF_SH,    RGBA16Float, float, GRAD) \
    IMG_DO(ASVGF_ATROUS_PONG_LF_SH,    RGBA16Float, float, GRAD) \
    IMG_DO(ASVGF_ATROUS_PING_LF_COCG,  RG16Float,   float, GRAD) \
    IMG_DO(ASVGF_ATROUS_PONG_LF_COCG,  RG16Float,   float, GRAD) \
    IMG_DO(ASVGF_ATROUS_PING_HF,       R32Uint,     uint,  FULL) \
    IMG_DO(ASVGF_ATROUS_PONG_HF,       R32Uint,     uint,  FULL) \
    IMG_DO(ASVGF_ATROUS_PING_SPEC,     R32Uint,     uint,  FULL) \
    IMG_DO(ASVGF_ATROUS_PONG_SPEC,     R32Uint,     uint,  FULL) \
    IMG_DO(ASVGF_ATROUS_PING_MOMENTS,  RG16Float,   float, FULL) \
    IMG_DO(ASVGF_ATROUS_PONG_MOMENTS,  RG16Float,   float, FULL) \
    IMG_DO(ASVGF_COLOR,                RGBA16Float, float, FULL) \
    IMG_DO(ASVGF_GRAD_LF_PING,         RG16Float,   float, GRAD) \
    IMG_DO(ASVGF_GRAD_LF_PONG,         RG16Float,   float, GRAD) \
    IMG_DO(ASVGF_GRAD_HF_SPEC_PING,    RG16Float,   float, GRAD) \
    IMG_DO(ASVGF_GRAD_HF_SPEC_PONG,    RG16Float,   float, GRAD) \
    IMG_DO(PT_SHADING_POSITION,        RGBA32Float, float, FULL) \
    IMG_DO(FLAT_COLOR,                 RGBA16Float, float, FULL) \
    IMG_DO(FLAT_MOTION,                RGBA16Float, float, FULL) \
    IMG_DO(PT_GODRAYS_THROUGHPUT_DIST, RGBA16Float, float, FULL) \
    IMG_DO(TAA_OUTPUT,                 RGBA16Float, float, FULL) \
    IMG_DO(PT_VIEW_DIRECTION,          RGBA16Float, float, FULL) \
    IMG_DO(PT_VIEW_DIRECTION2,         RGBA16Float, float, FULL) \
    IMG_DO(PT_THROUGHPUT,              RGBA16Float, float, FULL) \
    IMG_DO(PT_BOUNCE_THROUGHPUT,       RGBA16Float, float, FULL) \
    IMG_DO(HQ_COLOR_INTERLEAVED,       RGBA32Float, float, FULL) \
    IMG_DO(PT_COLOR_LF_SH,             RGBA16Float, float, FULL) \
    IMG_DO(PT_COLOR_LF_COCG,           RG16Float,   float, FULL) \
    IMG_DO(PT_COLOR_HF,                R32Uint,     uint,  FULL) \
    IMG_DO(PT_COLOR_SPEC,              R32Uint,     uint,  FULL) \
    IMG_DO(PT_GEO_NORMAL2,             R32Uint,     uint,  FULL)

// PT_VISBUF_BARY is RG32F where vkpt uses RG16F: the gradient reprojection
// rebuilds last frame's hit position from it, and on large world triangles
// half precision moved it far enough to change light selection and shadow
// rays, so many gradient samples reported false lighting changes, history was
// discarded, and the denoiser output turned blotchy.
//
// History pairs: _A is written this frame, _B holds the previous frame. The
// host swaps the textures behind the names every frame (vkpt's two
// descriptor sets).
#define VKPT_LIST_IMAGES_A_B \
    IMG_AB(PT_VISBUF_PRIM,             RG32Uint,    uint,  FULL) \
    IMG_AB(PT_VISBUF_BARY,             RG32Float,   float, FULL) \
    IMG_AB(PT_CLUSTER,                 R16Uint,     uint,  FULL) \
    IMG_AB(PT_BASE_COLOR,              RGBA16Float, float, FULL) \
    IMG_AB(PT_METALLIC,                RG8Unorm,    float, FULL) \
    IMG_AB(PT_VIEW_DEPTH,              R16Float,    float, FULL) \
    IMG_AB(PT_NORMAL,                  R32Uint,     uint,  FULL) \
    IMG_AB(PT_GEO_NORMAL,              R32Uint,     uint,  FULL) \
    IMG_AB(ASVGF_FILTERED_SPEC,        RGBA16Float, float, FULL) \
    IMG_AB(ASVGF_HIST_MOMENTS_HF,      RGBA16Float, float, FULL) \
    IMG_AB(ASVGF_TAA,                  RGBA16Float, float, FULL) \
    IMG_AB(ASVGF_RNG_SEED,             R32Uint,     uint,  FULL) \
    IMG_AB(ASVGF_HIST_COLOR_LF_SH,     RGBA16Float, float, FULL) \
    IMG_AB(ASVGF_HIST_COLOR_LF_COCG,   RG16Float,   float, FULL) \
    IMG_AB(ASVGF_GRAD_SMPL_POS,        R32Uint,     uint,  GRAD)

enum {
#define IMG_DO(name, fmt, type, size) VKPT_IMG_##name,
#define IMG_AB(name, fmt, type, size) VKPT_IMG_##name##_A, VKPT_IMG_##name##_B,
    VKPT_LIST_IMAGES
    VKPT_LIST_IMAGES_A_B
#undef IMG_DO
#undef IMG_AB
    VKPT_NUM_IMAGES
};

// The argument buffer: a read and a write handle for every image, plus a
// sampling handle for the float ones. Images whose formats Metal cannot open
// read_write (RG16Float, RG8Unorm, RG32Uint) are still usable this way.
#ifdef __METAL_VERSION__
#define VKPT_SAMPLE_FIELD_float(name) metal::texture2d<float, metal::access::sample> name##_s;
#define VKPT_SAMPLE_FIELD_uint(name)
#define VKPT_IMAGE_FIELDS(name, type) \
    metal::texture2d<type, metal::access::read>  name##_r; \
    metal::texture2d<type, metal::access::write> name##_w; \
    VKPT_SAMPLE_FIELD_##type(name)
#else
#define VKPT_SAMPLE_FIELD_float(name) MTLResourceID name##_s;
#define VKPT_SAMPLE_FIELD_uint(name)
#define VKPT_IMAGE_FIELDS(name, type) \
    MTLResourceID name##_r; \
    MTLResourceID name##_w; \
    VKPT_SAMPLE_FIELD_##type(name)
#endif

#if defined(__METAL_VERSION__) || defined(__OBJC__)
typedef struct {
#define IMG_DO(name, fmt, type, size) VKPT_IMAGE_FIELDS(name, type)
#define IMG_AB(name, fmt, type, size) VKPT_IMAGE_FIELDS(name##_A, type) VKPT_IMAGE_FIELDS(name##_B, type)
    VKPT_LIST_IMAGES
    VKPT_LIST_IMAGES_A_B
#undef IMG_DO
#undef IMG_AB
} VkptImages;
#endif

// Path tracer and denoiser buffer bindings.
enum {
    VKPT_BUF_UBO              = 0,
    VKPT_BUF_IMAGES           = 1,
    VKPT_BUF_TEXTURES         = 2,
    VKPT_BUF_WORLD_VERTICES   = 3,
    VKPT_BUF_WORLD_INDICES    = 4,
    VKPT_BUF_ENT_VERTICES     = 5,
    VKPT_BUF_ENT_INDICES      = 6,
    VKPT_BUF_MATERIALS        = 7,
    VKPT_BUF_LIGHT_POLYS      = 8,
    VKPT_BUF_LIGHT_OFFSETS    = 9,
    VKPT_BUF_LIGHT_LIGHTS     = 10,
    VKPT_BUF_SKY_VISIBILITY   = 11,
    VKPT_BUF_LIGHT_COUNTS     = 12,   // LIGHT_COUNT_HISTORY slices of MAX_LIGHT_LISTS
    VKPT_BUF_LIGHT_STATS      = 13,   // NUM_LIGHT_STATS_BUFFERS slices of light_stats_size
    VKPT_BUF_EFFECT_PRIMS     = 14,
    VKPT_BUF_EFFECT_VERTS     = 15,
    VKPT_BUF_EFFECT_INDICES   = 16,
    VKPT_BUF_ACCEL            = 17,
    VKPT_BUF_EFFECTS_ACCEL    = 18,
    VKPT_BUF_ENTITY_TABLE     = 19,   // current frame VkptEntitySlot table
    VKPT_BUF_ENTITY_TABLE_PREV= 20,   // previous frame
    VKPT_BUF_PUSH             = 21,   // push constants (pass / bounce index)
};

// Metal only: instance mask bit of the alpha tested world (vkpt keeps it
// under AS_FLAG_OPAQUE; see trace_geometry_ray in path_tracer.metal).
#define AS_FLAG_MASKED (1 << 6)

// Texture bindings outside the argument buffer.
enum {
    VKPT_TEX_SKY          = 0,
    VKPT_TEX_PHYSICAL_SKY = 1,
    VKPT_TEX_BLUE_NOISE   = 2,
};

// vkpt keeps 3 with two frames in flight; with MTL_FRAMES_IN_FLIGHT = 3 the
// slice written for frame N could still be read by frame N - 2's gradient
// samples, so one more. A power of two keeps (frame & 0x7fff) % N, which is
// what the 15 bit frame number in the RNG seed yields, consistent.
#define LIGHT_COUNT_HISTORY      4
#define NUM_LIGHT_STATS_BUFFERS  3
#define MAX_LIGHT_LISTS          (1 << 14)

// Per entity bookkeeping that gives dynamic geometry a stable identity across
// frames (vkpt's model_current_to_prev): hashed by entity id.
#define VKPT_ENTITY_TABLE_SIZE 1024
typedef struct {
    ubo_uint id;          // entity id, 0 = empty
    ubo_uint first_prim;  // first triangle in the entity index buffer this frame
    ubo_uint prim_count;
    ubo_uint pad;
} VkptEntitySlot;

typedef struct {
    ubo_int  bounce_index;
    ubo_uint iteration;
    ubo_uint pad0, pad1;
} VkptPush;

#endif // MTLPT_VKPT_H
