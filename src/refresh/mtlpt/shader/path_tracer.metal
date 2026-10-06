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

// The path tracer: a port of vkpt's path_tracer_rgen.h, light_lists.h,
// water.glsl, path_tracer_hit_shaders.h and the four tracing stages
// (primary_rays.rgen, reflect_refract.rgen, direct_lighting.rgen,
// indirect_lighting.rgen) as compute kernels with Metal ray queries.
// See vkpt/shader/path_tracer.h for the overview of the stages.
//
// Differences from vkpt are confined to the scene layer: geometry, materials
// and lights come from the Metal backend's buffers (MTLTriVertex,
// MTLMaterial, MTLLightPoly) through the adapters below, and the transparent
// effects are walked front to back instead of through any-hit shaders.

#include <metal_stdlib>
#include <metal_raytracing>
#include "vkpt_common.h"

using namespace metal;
using namespace raytracing;

// TLAS instance ids (path_tracer.m refresh_tlas); masks are AS_FLAG_*.
constant uint INSTANCE_WORLD = 0u;
constant uint INSTANCE_ENTITIES = 1u;
constant uint INSTANCE_WORLD_TRANSPARENT = 2u;
constant uint INSTANCE_VIEWER_MODELS = 3u;
constant uint INSTANCE_VIEWER_WEAPON = 4u;
constant uint INSTANCE_WORLD_SKY = 5u;
constant uint INSTANCE_WORLD_MASKED = 6u;     // alpha tested (vkpt geom_masked), non-opaque
constant uint INSTANCE_ENTITIES_TRANSPARENT = 7u;   // vkpt blas_transparent_models

#define RNG_SEED_SHIFT_X        0u
#define RNG_SEED_SHIFT_Y        8u
#define RNG_SEED_SHIFT_ISODD    16u
#define RNG_SEED_SHIFT_FRAME    17u

#define RNG_PRIMARY_OFF_X   0
#define RNG_PRIMARY_OFF_Y   1
#define RNG_PRIMARY_APERTURE_X   2
#define RNG_PRIMARY_APERTURE_Y   3

#define RNG_NEE_LIGHT_SELECTION(bounce)   (4 + 0 + 9 * (bounce))
#define RNG_NEE_TRI_X(bounce)             (4 + 1 + 9 * (bounce))
#define RNG_NEE_TRI_Y(bounce)             (4 + 2 + 9 * (bounce))
#define RNG_NEE_LIGHT_TYPE(bounce)        (4 + 3 + 9 * (bounce))
#define RNG_BRDF_X(bounce)                (4 + 4 + 9 * (bounce))
#define RNG_BRDF_Y(bounce)                (4 + 5 + 9 * (bounce))
#define RNG_BRDF_FRESNEL(bounce)          (4 + 6 + 9 * (bounce))
#define RNG_SUNLIGHT_X(bounce)            (4 + 7 + 9 * (bounce))
#define RNG_SUNLIGHT_Y(bounce)            (4 + 8 + 9 * (bounce))

#define PRIMARY_RAY_CULL_MASK        (AS_FLAG_OPAQUE | AS_FLAG_TRANSPARENT | AS_FLAG_VIEWER_WEAPON | AS_FLAG_SKY)
#define REFLECTION_RAY_CULL_MASK     (AS_FLAG_OPAQUE | AS_FLAG_SKY)
#define BOUNCE_RAY_CULL_MASK         (AS_FLAG_OPAQUE | AS_FLAG_SKY | AS_FLAG_CUSTOM_SKY)
#define SHADOW_RAY_CULL_MASK         (AS_FLAG_OPAQUE)

#define MAX_OUTPUT_VALUE 1000.0
#define MAX_BRUTEFORCE_SAMPLING 8

//
// Context: everything vkpt reads through globals and descriptor sets.
//

struct PtCtx {
    constant GlobalUbo *ubo;
    device const VkptImages *img;
    device const MTLTextureRef *textures;
    device const MTLTriVertex *world_vertices;
    device const uint *world_indices;
    device const MTLTriVertex *entity_vertices;
    device const uint *entity_indices;
    device const MTLMaterial *materials;
    device const MTLLightPoly *light_polys;
    device const uint *light_list_offsets;
    device const uint *light_list_lights;
    device const uint *sky_visibility;
    device const uint *light_counts;
    device atomic_uint *light_stats;
    device const MTLEffectPrim *effect_prims;
    device const MTLEffectVertex *effect_verts;
    device const uint *effect_indices;
    device const VkptEntitySlot *entity_table;
    device const VkptEntitySlot *entity_table_prev;
    instance_acceleration_structure accel;
    instance_acceleration_structure effects_accel;
    texture2d_array<float> sky;
    texture2d_array<float> physical_sky;
    texture2d_array<float> blue_noise;
    uint rng_seed;
    uint3 launch_id;
    int bounce_index;
};

#define PT_KERNEL_PARAMS \
    uint3 launch_id                                   [[thread_position_in_grid]], \
    constant GlobalUbo &ubo                           [[buffer(VKPT_BUF_UBO)]], \
    device const VkptImages &img                      [[buffer(VKPT_BUF_IMAGES)]], \
    device const MTLTextureRef *textures              [[buffer(VKPT_BUF_TEXTURES)]], \
    device const MTLTriVertex *world_vertices         [[buffer(VKPT_BUF_WORLD_VERTICES)]], \
    device const uint *world_indices                  [[buffer(VKPT_BUF_WORLD_INDICES)]], \
    device const MTLTriVertex *entity_vertices        [[buffer(VKPT_BUF_ENT_VERTICES)]], \
    device const uint *entity_indices                 [[buffer(VKPT_BUF_ENT_INDICES)]], \
    device const MTLMaterial *materials               [[buffer(VKPT_BUF_MATERIALS)]], \
    device const MTLLightPoly *light_polys            [[buffer(VKPT_BUF_LIGHT_POLYS)]], \
    device const uint *light_list_offsets             [[buffer(VKPT_BUF_LIGHT_OFFSETS)]], \
    device const uint *light_list_lights              [[buffer(VKPT_BUF_LIGHT_LIGHTS)]], \
    device const uint *sky_visibility                 [[buffer(VKPT_BUF_SKY_VISIBILITY)]], \
    device const uint *light_counts                   [[buffer(VKPT_BUF_LIGHT_COUNTS)]], \
    device atomic_uint *light_stats                   [[buffer(VKPT_BUF_LIGHT_STATS)]], \
    device const MTLEffectPrim *effect_prims          [[buffer(VKPT_BUF_EFFECT_PRIMS)]], \
    device const MTLEffectVertex *effect_verts        [[buffer(VKPT_BUF_EFFECT_VERTS)]], \
    device const uint *effect_indices                 [[buffer(VKPT_BUF_EFFECT_INDICES)]], \
    instance_acceleration_structure accel             [[buffer(VKPT_BUF_ACCEL)]], \
    instance_acceleration_structure effects_accel     [[buffer(VKPT_BUF_EFFECTS_ACCEL)]], \
    device const VkptEntitySlot *entity_table         [[buffer(VKPT_BUF_ENTITY_TABLE)]], \
    device const VkptEntitySlot *entity_table_prev    [[buffer(VKPT_BUF_ENTITY_TABLE_PREV)]], \
    constant VkptPush &push                           [[buffer(VKPT_BUF_PUSH)]], \
    texture2d_array<float> sky                        [[texture(VKPT_TEX_SKY)]], \
    texture2d_array<float> physical_sky               [[texture(VKPT_TEX_PHYSICAL_SKY)]], \
    texture2d_array<float> blue_noise                 [[texture(VKPT_TEX_BLUE_NOISE)]]

#define PT_CTX_INIT \
    PtCtx ctx; \
    ctx.ubo = &ubo; ctx.img = &img; ctx.textures = textures; \
    ctx.world_vertices = world_vertices; ctx.world_indices = world_indices; \
    ctx.entity_vertices = entity_vertices; ctx.entity_indices = entity_indices; \
    ctx.materials = materials; ctx.light_polys = light_polys; \
    ctx.light_list_offsets = light_list_offsets; ctx.light_list_lights = light_list_lights; \
    ctx.sky_visibility = sky_visibility; ctx.light_counts = light_counts; ctx.light_stats = light_stats; \
    ctx.effect_prims = effect_prims; ctx.effect_verts = effect_verts; ctx.effect_indices = effect_indices; \
    ctx.entity_table = entity_table; ctx.entity_table_prev = entity_table_prev; \
    ctx.accel = accel; ctx.effects_accel = effects_accel; \
    ctx.sky = sky; ctx.physical_sky = physical_sky; ctx.blue_noise = blue_noise; \
    ctx.rng_seed = 0u; ctx.launch_id = launch_id; ctx.bounce_index = push.bounce_index;

struct Ray {
    float3 origin, direction;
    float t_min, t_max;
};

//
// Scene adapters: vkpt's Triangle and MaterialInfo built from the Metal
// backend's vertex, index and material buffers.
//

struct Triangle {
    float3x3 positions;
    float3x3 positions_prev;
    float3x3 normals;
    float3x2 tex_coords;
    float3x3 tangents;
    uint     material_id;
    uint     material_index;    // MTLMaterial slot (vkpt keeps it in MATERIAL_INDEX_MASK)
    uint     shell;
    int      cluster;
    uint     instance_index;    // entity id, ~0u for the static world
    uint     instance_prim;
    float    emissive_factor;
    float    alpha;
};

struct MaterialInfo {
    uint   base_texture;
    uint   normals_texture;
    uint   emissive_texture;
    uint   mask_texture;
    float  bump_scale;
    float  roughness_override;
    float  metalness_factor;
    float  emissive_factor;
    float  specular_factor;
    float3 base_factor;
    float3 emissive_constant;   // Metal: emission of lights without an emissive map
};

static inline MaterialInfo get_material_info(thread const PtCtx &ctx, uint material_index)
{
    device const MTLMaterial &m = ctx.materials[material_index];
    MaterialInfo minfo;
    minfo.base_texture = m.base_texture;
    minfo.normals_texture = m.normal_texture;
    minfo.emissive_texture = m.emissive_texture;
    minfo.mask_texture = m.pad0;
    minfo.bump_scale = m.bump_scale;
    minfo.roughness_override = m.roughness;
    minfo.metalness_factor = m.metalness;
    minfo.emissive_factor = m.emissive_factor;
    minfo.specular_factor = m.specular_factor;
    minfo.base_factor = float3(m.base_color);
    minfo.emissive_constant = float3(m.emissive);
    return minfo;
}

static inline bool is_entity_instance(uint instance)
{
    return instance == INSTANCE_ENTITIES || instance == INSTANCE_ENTITIES_TRANSPARENT ||
           instance == INSTANCE_VIEWER_MODELS || instance == INSTANCE_VIEWER_WEAPON;
}

// Primitive index in the world or entity index buffer for a TLAS hit.
static inline uint global_prim_index(thread const PtCtx &ctx, uint instance, uint primitive_id)
{
    if (instance == INSTANCE_WORLD_TRANSPARENT)
        return primitive_id + global_ubo.world_transparent_first_prim;
    if (instance == INSTANCE_WORLD_SKY)
        return primitive_id + global_ubo.world_sky_first_prim;
    if (instance == INSTANCE_WORLD_MASKED)
        return primitive_id + global_ubo.world_masked_first_prim;
    if (instance == INSTANCE_ENTITIES_TRANSPARENT)
        return primitive_id + global_ubo.entity_transparent_first_prim;
    if (instance == INSTANCE_VIEWER_MODELS)
        return primitive_id + global_ubo.entity_viewer_first_prim;
    if (instance == INSTANCE_VIEWER_WEAPON)
        return primitive_id + global_ubo.entity_weapon_first_prim;
    return primitive_id;
}

static Triangle load_triangle(thread const PtCtx &ctx, bool is_entity, bool is_weapon, uint prim)
{
    device const MTLTriVertex *vertices = is_entity ? ctx.entity_vertices : ctx.world_vertices;
    device const uint *indices = is_entity ? ctx.entity_indices : ctx.world_indices;

    device const MTLTriVertex &v0 = vertices[indices[prim * 3 + 0]];
    device const MTLTriVertex &v1 = vertices[indices[prim * 3 + 1]];
    device const MTLTriVertex &v2 = vertices[indices[prim * 3 + 2]];

    Triangle t;
    t.positions = float3x3(float3(v0.position), float3(v1.position), float3(v2.position));
    t.positions_prev = float3x3(float3(v0.prev_position), float3(v1.prev_position), float3(v2.prev_position));
    t.normals = float3x3(float3(v0.normal), float3(v1.normal), float3(v2.normal));
    t.tex_coords = float3x2(float2(v0.texcoord), float2(v1.texcoord), float2(v2.texcoord));
    t.tangents = float3x3(float3(v0.tangent), float3(v1.tangent), float3(v2.tangent));

    t.material_index = v0.material & MTL_VERTEX_MATERIAL_MASK;
    device const MTLMaterial &m = ctx.materials[t.material_index];

    // vkpt material_id: kind | flags | light style bits. The index bits stay
    // zero; the slot travels in material_index instead.
    uint material_id = m.kind_flags & ~(uint(MATERIAL_INDEX_MASK) | uint(MATERIAL_LIGHT_STYLE_MASK) | uint(MATERIAL_FLAG_LIGHT));
    if ((m.kind_flags & MATERIAL_FLAG_LIGHT) != 0u || (v0.flags & MTL_VERTEX_FLAG_LIGHT) != 0u)
        material_id |= MATERIAL_FLAG_LIGHT;
    if (is_weapon)
        material_id |= MATERIAL_FLAG_WEAPON;
    if ((material_id & MATERIAL_KIND_MASK) == MATERIAL_KIND_CAMERA) {
        // Camera id in the light style bits, as bsp_mesh.c assigns it.
        uint camera_id = v0.material >> MTL_VERTEX_CAMERA_SHIFT;
        material_id |= ((camera_id << 2) << MATERIAL_LIGHT_STYLE_SHIFT) & MATERIAL_LIGHT_STYLE_MASK;
    }
    t.material_id = material_id;

    t.shell = m.shell;
    t.cluster = v0.cluster;
    t.instance_index = is_entity ? (v0.flags >> MTL_VERTEX_ENTITY_SHIFT) : ~0u;
    t.instance_prim = prim;
    t.emissive_factor = 1.0;
    t.alpha = as_type<float>(v0.alpha_bits);
    return t;
}

static inline Triangle load_hit_triangle(thread const PtCtx &ctx, uint instance, uint primitive_id)
{
    return load_triangle(ctx, is_entity_instance(instance), instance == INSTANCE_VIEWER_WEAPON,
                         global_prim_index(ctx, instance, primitive_id));
}

//
// Texture lookups (global_textureLod / global_textureGrad).
//

static inline float4 global_textureLod(thread const PtCtx &ctx, uint tex, float2 uv, float lod)
{
    return ctx.textures[tex].tex.sample(vkpt_material_sampler, uv, level(lod));
}

static inline float4 global_textureGrad(thread const PtCtx &ctx, uint tex, float2 uv, float2 dx, float2 dy)
{
    // vkpt samples materials with textureGrad and a 16x anisotropic sampler.
    // The isotropic explicit LOD used here before blurred surfaces seen at an
    // angle, and the blurrier normal maps raised their Toksvig roughness.
    constexpr sampler aniso(filter::linear, mip_filter::linear, address::repeat, max_anisotropy(16));
    return ctx.textures[tex].tex.sample(aniso, uv, gradient2d(dx, dy));
}

// Wall and skin textures, with the sampler vkpt picks for them by pt_nearest
// (vkpt_textures_update_descriptor_set): anisotropic, nearest magnification
// with anisotropic minification ("mixed"), or nearest everything.
constexpr sampler material_sampler_mixed(mag_filter::nearest, min_filter::linear, mip_filter::linear,
                                         address::repeat, max_anisotropy(16));
constexpr sampler material_sampler_nearest(filter::nearest, mip_filter::nearest, address::repeat);

static inline float4 material_textureGrad(thread const PtCtx &ctx, uint tex, float2 uv, float2 dx, float2 dy)
{
    uint mode = global_ubo.material_filter;
    if (mode == 1u)
        return ctx.textures[tex].tex.sample(material_sampler_mixed, uv, gradient2d(dx, dy));
    if (mode >= 2u)
        return ctx.textures[tex].tex.sample(material_sampler_nearest, uv, gradient2d(dx, dy));
    return global_textureGrad(ctx, tex, uv, dx, dy);
}

static inline float4 material_textureLod(thread const PtCtx &ctx, uint tex, float2 uv, float lod)
{
    uint mode = global_ubo.material_filter;
    if (mode == 1u)
        return ctx.textures[tex].tex.sample(material_sampler_mixed, uv, level(lod));
    if (mode >= 2u)
        return ctx.textures[tex].tex.sample(material_sampler_nearest, uv, level(lod));
    return global_textureLod(ctx, tex, uv, lod);
}

//
// Random numbers: blue noise, indexed by the per-pixel seed (path_tracer_rgen.h).
//

static inline float get_rng(thread const PtCtx &ctx, uint idx)
{
    uint seed = ctx.rng_seed;
    uint3 p = uint3(seed >> RNG_SEED_SHIFT_X, seed >> RNG_SEED_SHIFT_Y, seed >> RNG_SEED_SHIFT_ISODD);
    p.z = (p.z >> 1) + (p.z & 1u);
    p.z = (p.z + idx);
    p &= uint3(BLUE_NOISE_RES - 1, BLUE_NOISE_RES - 1, NUM_BLUE_NOISE_TEX - 1);
    return min(ctx.blue_noise.read(uint2(p.xy), p.z).r, 0.9999999999999);
}

//
// Material kinds
//

static inline bool is_water(uint m)       { return (m & MATERIAL_KIND_MASK) == MATERIAL_KIND_WATER; }
static inline bool is_slime(uint m)       { return (m & MATERIAL_KIND_MASK) == MATERIAL_KIND_SLIME; }
static inline bool is_lava(uint m)        { return (m & MATERIAL_KIND_MASK) == MATERIAL_KIND_LAVA; }
static inline bool is_glass(uint m)       { return (m & MATERIAL_KIND_MASK) == MATERIAL_KIND_GLASS; }
static inline bool is_sky(uint m)         { return (m & MATERIAL_KIND_MASK) == MATERIAL_KIND_SKY; }
static inline bool is_screen(uint m)      { return (m & MATERIAL_KIND_MASK) == MATERIAL_KIND_SCREEN; }
static inline bool is_camera(uint m)      { return (m & MATERIAL_KIND_MASK) == MATERIAL_KIND_CAMERA; }
static inline bool is_transparent(uint m)
{
    uint kind = m & MATERIAL_KIND_MASK;
    return kind == MATERIAL_KIND_TRANSPARENT || kind == MATERIAL_KIND_TRANSP_MODEL;
}
static inline bool is_chrome(uint m)
{
    uint kind = m & MATERIAL_KIND_MASK;
    return kind == MATERIAL_KIND_CHROME || kind == MATERIAL_KIND_CHROME_MODEL;
}

static inline float3 correct_emissive(float3 emissive)
{
    return max(float3(0.0), emissive + float3(EMISSIVE_TRANSFORM_BIAS));
}

static inline uint set_kind(uint material_id, uint kind)
{
    return (material_id & ~uint(MATERIAL_KIND_MASK)) | kind;
}

//
// Environment (env_map). The Metal backend stores the sky box and the
// physical sky as six face arrays in the classic renderer's face layout.
//

static float3 sample_skybox(texture2d_array<float> sky, float3 dir)
{
    constexpr sampler sky_sampler(filter::linear, address::clamp_to_edge);

    float3 a = abs(dir);
    uint axis;
    float dv, s, t;

    if (a.x > a.y && a.x > a.z) {
        if (dir.x > 0.0) { axis = 0u; dv = dir.x;  s = -dir.y / dv; }
        else             { axis = 1u; dv = -dir.x; s =  dir.y / dv; }
        t = dir.z / dv;
    } else if (a.y > a.z) {
        if (dir.y > 0.0) { axis = 2u; dv = dir.y;  s =  dir.x / dv; }
        else             { axis = 3u; dv = -dir.y; s = -dir.x / dv; }
        t = dir.z / dv;
    } else {
        if (dir.z > 0.0) { axis = 4u; dv = dir.z;  s = -dir.y / dv; t = -dir.x / dv; }
        else             { axis = 5u; dv = -dir.z; s = -dir.y / dv; t =  dir.x / dv; }
    }

    float2 uv = float2(s, t) * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    uv = clamp(uv, 1.0 / 512.0, 511.0 / 512.0);
    return sky.sample(sky_sampler, uv, axis).rgb;
}

static float3 env_map(thread const PtCtx &ctx, float3 direction, bool remove_sun)
{
    float3 dir = direction;
    if (global_ubo.sky_rotate != 0.0) {
        float3 axis = normalize(float3(global_ubo.sky_axis));
        float c = cos(global_ubo.sky_rotate), s = sin(global_ubo.sky_rotate);
        dir = dir * c + cross(axis, dir) * s + axis * dot(axis, dir) * (1.0 - c);
    }

    float3 envmap = float3(0.0);
    if (global_ubo.environment_type == ENVIRONMENT_DYNAMIC) {
        envmap = sample_skybox(ctx.physical_sky, dir);
        if (remove_sun) {
            // roughly remove the sun from the env map
            envmap = min(envmap, float3((1.0 - dot(dir, float3(global_ubo.sun_direction_envmap))) * 200.0));
        }
    } else if (global_ubo.environment_type == ENVIRONMENT_STATIC) {
        envmap = sample_skybox(ctx.sky, dir);
        float avg = (envmap.x + envmap.y + envmap.z) / 3.0;
        envmap = mix(envmap, float3(avg), global_ubo.pt_envmap_desaturate) * global_ubo.pt_envmap_brightness;
    }
    return envmap;
}

//
// water.glsl
//

static float3 get_water_normal(thread const PtCtx &ctx, uint material_id, float3 geo_normal, float3 tangent,
                               float3 position, bool local_space)
{
    constexpr sampler water_sampler(filter::linear, mip_filter::none, address::repeat);

    if ((material_id & MATERIAL_FLAG_FLOWING) != 0u)
        position -= tangent * global_ubo.time * 32.0;

    float3 unsigned_geo_normal = abs(geo_normal);
    float3x3 basis = construct_ONB_frisvad(unsigned_geo_normal);
    float2 p = float2(dot(position, basis[0]), dot(position, basis[2]));

    texture2d<float> tex = ctx.textures[global_ubo.water_normal_texture].tex;
    const float speed = 2.5;

    float2 uv1 = p * 0.006 + global_ubo.time * float2(0.01, 0.02) * speed;
    float3 a = tex.sample(water_sampler, uv1, level(0.0)).xyz;
    a.xy = a.xy * 2.0 - 1.0;
    a.xy *= 0.3;

    float2 uv2 = p * 0.003 + global_ubo.time * float2(0.013, 0.014) * speed;
    float3 b = tex.sample(water_sampler, uv2, level(0.0)).xyz;
    b.xy = b.xy * 2.0 - 1.0;
    b.xy *= 0.5;

    float2 uv3 = p * 0.0061 + global_ubo.time * float2(-0.01, -0.02) * speed;
    float3 c = tex.sample(water_sampler, uv3, level(0.0)).xyz;
    c.xy = c.xy * 2.0 - 1.0;
    c.xy *= 0.3;

    float3 n = normalize(a + b + c).xzy;
    if (local_space)
        return n;

    n = basis * n;
    if (geo_normal.x < 0.0) n.x = -n.x;
    if (geo_normal.y < 0.0) n.y = -n.y;
    if (geo_normal.z < 0.0) n.z = -n.z;
    return n;
}

static inline float3 get_extinction_factors(thread const PtCtx &ctx, int medium)
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

static inline float3 extinction(thread const PtCtx &ctx, int medium, float distance)
{
    return exp(-get_extinction_factors(ctx, medium) * distance);
}

//
// Ray tracing (trace_geometry_ray / trace_shadow_ray / trace_caustic_ray).
// Everything but the alpha tested world is opaque and goes through one
// hardware traversal (intersector), which is much cheaper on Apple GPUs than
// stepping an intersection_query. The alpha tested world has its own
// instance mask bit (AS_FLAG_MASKED) and, when the map has any, is resolved
// by a query limited to that instance and to the opaque hit distance; the
// candidates are alpha tested there (path_tracer_masked.rahit).
//

struct HitInfo {
    bool   found;
    uint   instance;
    uint   primitive;
    float2 barycentric;
    float  hit_distance;
};

static bool pt_logic_masked(thread const PtCtx &ctx, uint instance, uint primitive_id, float2 bary)
{
    Triangle triangle = load_hit_triangle(ctx, instance, primitive_id);
    MaterialInfo minfo = get_material_info(ctx, triangle.material_index);
    if (minfo.mask_texture == 0u)
        return true;

    float2 tex_coord = triangle.tex_coords * float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    perturb_tex_coord(triangle.material_id, global_ubo.time, tex_coord);
    float4 mask_value = material_textureLod(ctx, minfo.mask_texture, tex_coord, 0.0);
    return mask_value.x >= 0.5;
}

static inline bool trace_masked_world(thread const PtCtx &ctx, int instance_mask)
{
    return (uint(instance_mask) & AS_FLAG_OPAQUE) != 0u && global_ubo.has_masked_world != 0u;
}

static HitInfo trace_geometry_ray(thread const PtCtx &ctx, Ray ray, int instance_mask)
{
    HitInfo hit;
    hit.found = false;
    hit.instance = 0u;
    hit.primitive = ~0u;
    hit.barycentric = float2(0.0);
    hit.hit_distance = 0.0;

    intersector<instancing, triangle_data> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.set_triangle_cull_mode(triangle_cull_mode::none);
    isect.force_opacity(forced_opacity::opaque);
    auto r = isect.intersect(raytracing::ray(ray.origin, ray.direction, ray.t_min, ray.t_max), ctx.accel,
                             uint(instance_mask) & ~uint(AS_FLAG_MASKED));

    if (r.type == intersection_type::triangle) {
        hit.found = true;
        hit.instance = r.instance_id;
        hit.primitive = r.primitive_id;
        hit.barycentric = r.triangle_barycentric_coord;
        hit.hit_distance = r.distance;
    }

    if (trace_masked_world(ctx, instance_mask)) {
        intersection_params params;
        params.assume_geometry_type(geometry_type::triangle);
        params.set_triangle_cull_mode(triangle_cull_mode::none);

        float t_max = hit.found ? hit.hit_distance : ray.t_max;
        intersection_query<instancing, triangle_data> q;
        q.reset(raytracing::ray(ray.origin, ray.direction, ray.t_min, t_max), ctx.accel, uint(AS_FLAG_MASKED), params);
        while (q.next()) {
            if (q.get_candidate_intersection_type() == intersection_type::triangle &&
                pt_logic_masked(ctx, q.get_candidate_instance_id(), q.get_candidate_primitive_id(),
                                q.get_candidate_triangle_barycentric_coord()))
                q.commit_triangle_intersection();
        }

        if (q.get_committed_intersection_type() == intersection_type::triangle) {
            hit.found = true;
            hit.instance = q.get_committed_instance_id();
            hit.primitive = q.get_committed_primitive_id();
            hit.barycentric = q.get_committed_triangle_barycentric_coord();
            hit.hit_distance = q.get_committed_distance();
        }
    }
    return hit;
}

static inline float3 get_hit_barycentric(thread const HitInfo &hit)
{
    return float3(1.0 - hit.barycentric.x - hit.barycentric.y, hit.barycentric.x, hit.barycentric.y);
}

static Ray get_shadow_ray(float3 p1, float3 p2, float tmin)
{
    float3 l = p2 - p1;
    float dist = length(l);
    l /= dist;

    Ray ray;
    ray.origin = p1 + l * tmin;
    ray.t_min = 0.0;
    ray.t_max = dist - tmin - 0.01;
    ray.direction = l;
    return ray;
}

static float trace_shadow_ray(thread const PtCtx &ctx, Ray ray, int cull_mask)
{
    if (cull_mask == 0 || !(ray.t_max > ray.t_min))
        return 1.0;

    intersector<instancing, triangle_data> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.set_triangle_cull_mode(triangle_cull_mode::none);
    isect.force_opacity(forced_opacity::opaque);
    isect.accept_any_intersection(true);
    auto r = isect.intersect(raytracing::ray(ray.origin, ray.direction, ray.t_min, ray.t_max), ctx.accel,
                             uint(cull_mask) & ~uint(AS_FLAG_MASKED));
    if (r.type != intersection_type::none)
        return 0.0;

    if (trace_masked_world(ctx, cull_mask)) {
        intersection_params params;
        params.assume_geometry_type(geometry_type::triangle);
        params.set_triangle_cull_mode(triangle_cull_mode::none);
        params.accept_any_intersection(true);

        intersection_query<instancing, triangle_data> q;
        q.reset(raytracing::ray(ray.origin, ray.direction, ray.t_min, ray.t_max), ctx.accel, uint(AS_FLAG_MASKED), params);
        while (q.next()) {
            if (q.get_candidate_intersection_type() == intersection_type::triangle &&
                pt_logic_masked(ctx, q.get_candidate_instance_id(), q.get_candidate_primitive_id(),
                                q.get_candidate_triangle_barycentric_coord()))
                q.commit_triangle_intersection();
        }
        if (q.get_committed_intersection_type() != intersection_type::none)
            return 0.0;
    }
    return 1.0;
}

static float3 trace_caustic_ray(thread const PtCtx &ctx, Ray ray, int surface_medium)
{
    intersector<instancing, triangle_data> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.set_triangle_cull_mode(triangle_cull_mode::none);
    isect.force_opacity(forced_opacity::opaque);
    auto r = isect.intersect(raytracing::ray(ray.origin, ray.direction, ray.t_min, ray.t_max), ctx.accel, uint(AS_FLAG_TRANSPARENT));

    float extinction_distance = ray.t_max - ray.t_min;
    float3 throughput = float3(1.0);

    if (r.type == intersection_type::triangle) {
        float hit_distance = r.distance;
        Triangle triangle = load_hit_triangle(ctx, r.instance_id, r.primitive_id);

        float3 geo_normal = triangle.normals[0];
        bool is_vertical = abs(geo_normal.z) < 0.1;

        if ((is_water(triangle.material_id) || is_slime(triangle.material_id)) && !is_vertical) {
            float3 position = ray.origin + ray.direction * hit_distance;
            float3 w = get_water_normal(ctx, triangle.material_id, geo_normal, triangle.tangents[0], position, true);

            float caustic = clamp((1.0 - pow(clamp(1.0 - length(w.xz), 0.0, 1.0), 2.0)) * 100.0, 0.0, 8.0);
            caustic = mix(1.0, caustic, clamp(hit_distance * 0.02, 0.0, 1.0));
            throughput = float3(caustic);

            if (surface_medium != MEDIUM_NONE) {
                extinction_distance = hit_distance;
            } else {
                surface_medium = is_water(triangle.material_id) ? MEDIUM_WATER : MEDIUM_SLIME;
                extinction_distance = max(0.0, ray.t_max - hit_distance);
            }
        } else if (is_glass(triangle.material_id) || (is_water(triangle.material_id) && is_vertical)) {
            float2 bary2 = r.triangle_barycentric_coord;
            float3 bary = float3(1.0 - bary2.x - bary2.y, bary2.x, bary2.y);
            float2 tex_coord = triangle.tex_coords * bary;

            MaterialInfo minfo = get_material_info(ctx, triangle.material_index);
            float3 base_color = minfo.base_factor;
            if (minfo.base_texture > 0u)
                base_color *= material_textureLod(ctx, minfo.base_texture, tex_coord, 2.0).rgb;
            throughput = clamp(base_color, float3(0.0), float3(1.0));
        } else {
            throughput = float3(clamp(1.0 - triangle.alpha, 0.0, 1.0));
        }
    }

    return extinction(ctx, surface_medium, extinction_distance) * throughput;
}

//
// Transparent effects: particles, beams, sprites and explosions
// (path_tracer_hit_shaders.h, path_tracer_transparency.glsl). Metal's ray
// query reports opaque hits in no particular order, so the effect quads are
// walked front to back by re-casting past each one, which also makes the
// blending exact rather than vkpt's two-distance approximation.
//

static bool solve_quadratic(float a, float b, float c, thread float2 &t)
{
    float discrim = b * b - 4.0 * a * c;
    if (discrim < 0.0)
        return false;
    float q = (b < 0.0) ? -0.5 * (b - sqrt(discrim)) : -0.5 * (b + sqrt(discrim));
    if (a == 0.0 || q == 0.0)
        return false;
    float t0 = q / a;
    float t1 = c / q;
    t = float2(min(t0, t1), max(t0, t1));
    return true;
}

static bool hit_cylinder(float3 o, float3 d, float radius, thread float2 &t)
{
    return solve_quadratic(dot(d.xy, d.xy), 2.0 * dot(d.xy, o.xy), dot(o.xy, o.xy) - radius * radius, t);
}

static bool hit_sphere(float3 o, float3 d, float radius, thread float2 &t)
{
    return solve_quadratic(dot(d, d), 2.0 * dot(d, o), dot(o, o) - radius * radius, t);
}

// pt_logic_beam_intersection
static bool beam_intersection(device const MTLEffectPrim &beam, float3 world_origin, float3 world_dir,
                              float t_min, float t_max, thread float2 &fade_and_thickness, thread float &t_hit)
{
    fade_and_thickness = float2(0.0);
    t_hit = 0.0;

    float4 r0 = float4(beam.world_to_beam[0]);
    float4 r1 = float4(beam.world_to_beam[1]);
    float4 r2 = float4(beam.world_to_beam[2]);
    float4 wo = float4(world_origin, 1.0);
    float4 wd = float4(world_dir, 0.0);
    float3 o = float3(dot(r0, wo), dot(r1, wo), dot(r2, wo));
    float3 d = float3(dot(r0, wd), dot(r1, wd), dot(r2, wd));

    float radius = beam.radius;
    float beam_length = beam.length;

    float2 t;
    if (!hit_cylinder(o, d, radius, t))
        return false;

    float2 hit_z = float2(o.z) + float2(d.z) * t;
    bool2 hit_below_0 = hit_z < float2(0.0);
    if (any(hit_below_0)) {
        float2 t_sphere;
        if (!hit_sphere(o, d, radius, t_sphere))
            return false;
        if (hit_below_0.x) t.x = max(t.x, t_sphere.x);
        if (hit_below_0.y) t.y = min(t.y, t_sphere.y);
    }
    bool2 hit_above_end = hit_z > float2(beam_length);
    if (any(hit_above_end)) {
        float2 t_sphere;
        if (!hit_sphere(o - float3(0.0, 0.0, beam_length), d, radius, t_sphere))
            return false;
        if (hit_above_end.x) t.x = max(t.x, t_sphere.x);
        if (hit_above_end.y) t.y = min(t.y, t_sphere.y);
    }

    if (t.x >= t_max || t.y < t_min)
        return false;

    t_hit = t.x;
    if (t_hit < t_min) {
        t_hit = t.y;
        if (t_hit >= t_max)
            return false;
    }

    float3 perp_norm = normalize(float3(d.y, -d.x, 0.0));
    float3 n2 = float3(-perp_norm.y, perp_norm.x, 0.0);
    float t1 = dot(-o, n2) / dot(d, n2);
    float3 n1 = cross(d, perp_norm);
    float t2 = dot(o, n1) / n1.z;

    float3 c_ray = o + t1 * d;
    float3 c_beam = float3(0.0, 0.0, t2);

    float dist_side = distance(c_ray, float3(0.0, 0.0, clamp(t2, 0.0, beam_length)));
    float dist_head = distance(c_ray, c_beam);
    float dist = mix(dist_side, dist_head, abs(d.z));

    float fade = 1.0 - dist / radius;
    float thickness = t.y - t.x;
    fade *= clamp(thickness / (2.0 * radius), 0.0, 1.0);

    fade_and_thickness = float2(fade, thickness);
    return true;
}

// Fog volumes reduced to a 1D density along the ray (find_fog_volumes).
struct RayFog {
    float3 color;
    float2 bounds;      // t_in, t_out; y == 0 means unused
    float2 density;     // per t, constant
};

static float4 evaluate_fog(RayFog fog, float t1, float t2)
{
    t1 = max(t1, fog.bounds.x);
    t2 = min(t2, fog.bounds.y);
    if (t1 >= t2)
        return float4(0.0);
    float alpha = 1.0 - exp((t1 * t1 - t2 * t2) * fog.density.x + (t1 - t2) * fog.density.y);
    return float4(fog.color * alpha, alpha);
}

static void blend_fogs_behind(RayFog fog1, RayFog fog2, float t1, float t2, thread float4 &accumulated)
{
    float4 seg = float4(0.0);
    if (fog2.bounds.y != 0.0)
        seg = alpha_blend_premultiplied(evaluate_fog(fog2, t1, t2), seg);
    if (fog1.bounds.y != 0.0)
        seg = alpha_blend_premultiplied(evaluate_fog(fog1, t1, t2), seg);
    accumulated = alpha_blend_premultiplied(accumulated, seg);
}

static void find_fog_volumes(thread const PtCtx &ctx, Ray ray, thread RayFog &fog1, thread RayFog &fog2)
{
    fog1.bounds = fog2.bounds = float2(0.0);
    fog1.color = fog2.color = float3(0.0);
    fog1.density = fog2.density = float2(0.0);

    float3 inv_dir = 1.0 / ray.direction;
    for (int i = 0; i < MAX_FOG_VOLUMES; i++) {
        constant MTLFogVolume &volume = global_ubo.fog_volumes[i];
        if (volume.is_active == 0u)
            return;

        float3 t1 = (float3(volume.mins) - ray.origin) * inv_dir;
        float3 t2 = (float3(volume.maxs) - ray.origin) * inv_dir;
        float3 tmin = min(t1, t2), tmax = max(t1, t2);
        float t_in = max(max3(tmin.x, tmin.y, tmin.z), ray.t_min);
        float t_out = min(min3(tmax.x, tmax.y, tmax.z), ray.t_max);
        if (t_out <= t_in)
            continue;

        bool replaces_first = t_in < fog1.bounds.x || fog1.bounds.y == 0.0;
        bool replaces_second = t_in < fog2.bounds.x || fog2.bounds.y == 0.0;
        if (!replaces_first && !replaces_second)
            continue;

        RayFog f;
        f.color = float3(volume.color) * global_ubo.pt_fog_brightness;
        f.bounds = float2(t_in, t_out);
        f.density = float2(dot(float3(volume.density.xyz), ray.direction) * 0.5,
                           dot(float3(volume.density.xyz), ray.origin) + volume.density.w);
        if (replaces_first) {
            fog2 = fog1;
            fog1 = f;
        } else {
            fog2 = f;
        }
    }
}

// trace_effects_ray: skip_procedural drops beams and fog (specular bounces).
static float4 trace_effects_ray(thread const PtCtx &ctx, Ray ray, bool skip_procedural)
{
    float4 accumulated = float4(0.0);

    RayFog fog1, fog2;
    fog1.bounds = fog2.bounds = float2(0.0);
    if (!skip_procedural)
        find_fog_volumes(ctx, ray, fog1, fog2);
    bool have_fog = fog1.bounds.y != 0.0;
    float fog_t = ray.t_min;

    if (global_ubo.num_effect_triangles != 0u) {
        // Blue noise for the beam flicker (pt_logic_beam).
        uint texnum = uint(global_ubo.current_frame_idx) & (NUM_BLUE_NOISE_TEX - 1);
        uint2 texpos = ctx.launch_id.xy & uint2(BLUE_NOISE_RES - 1);
        float noise = ctx.blue_noise.read(texpos, texnum).r;

        float t_start = ray.t_min;
        for (uint iteration = 0; iteration < 16u; iteration++) {
            intersector<instancing, triangle_data> isect;
            isect.assume_geometry_type(geometry_type::triangle);
            isect.set_triangle_cull_mode(triangle_cull_mode::none);
            isect.force_opacity(forced_opacity::opaque);
            auto r = isect.intersect(raytracing::ray(ray.origin, ray.direction, t_start, ray.t_max), ctx.effects_accel, 0xffu);
            if (r.type != intersection_type::triangle)
                break;

            float hit_t = r.distance;
            uint prim = r.primitive_id;
            device const MTLEffectPrim &info = ctx.effect_prims[prim];

            float2 bary = r.triangle_barycentric_coord;
            float3 b = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
            device const MTLEffectVertex &v0 = ctx.effect_verts[ctx.effect_indices[prim * 3 + 0]];
            device const MTLEffectVertex &v1 = ctx.effect_verts[ctx.effect_indices[prim * 3 + 1]];
            device const MTLEffectVertex &v2 = ctx.effect_verts[ctx.effect_indices[prim * 3 + 2]];
            float2 uv = float2(v0.texcoord) * b.x + float2(v1.texcoord) * b.y + float2(v2.texcoord) * b.z;

            float4 color = float4(0.0);

            if (info.type == MTL_EFFECT_PARTICLE) {
                float factor = pow(clamp(1.0 - length(float2(0.5) - uv) * 2.0, 0.0, 1.0), global_ubo.pt_particle_softness);
                if (factor > 0.0) {
                    color = float4(info.color);
                    color.a *= factor;
                    color.rgb *= color.a;
                    color.rgb *= global_ubo.prev_adapted_luminance * global_ubo.pt_particle_brightness;
                }
            } else if (info.type == MTL_EFFECT_BEAM) {
                float2 fade_and_thickness;
                float t_shape;
                if (!skip_procedural &&
                    beam_intersection(info, ray.origin, ray.direction, ray.t_min, ray.t_max, fade_and_thickness, t_shape)) {
                    float factor = pow(clamp(fade_and_thickness.x, 0.0, 1.0), global_ubo.pt_beam_softness);
                    if (factor > 0.0) {
                        color = float4(info.color);
                        color.a *= factor;
                        color.rgb *= color.a;
                        color.rgb *= global_ubo.prev_adapted_luminance * 20.0;
                        color.rgb *= noise * noise + 0.1;
                        float thickness = fade_and_thickness.y;
                        if (thickness > 0.0)
                            color *= clamp((ray.t_max - t_shape) / thickness, 0.0, 1.0);
                    }
                }
            } else if (info.type == MTL_EFFECT_SPRITE) {
                color = global_textureLod(ctx, info.texture, uv, 0.0);
                float alpha = info.color.w;
                color.a *= alpha;
                float lum = luminance(color.rgb);
                if (lum > 0.0) {
                    float lum2 = pow(lum, 2.2);
                    color.rgb = color.rgb * (lum2 / lum) * color.a * alpha;
                    color.rgb *= global_ubo.prev_adapted_luminance * 2000.0;
                }
            } else {
                // pt_logic_explosion
                float4 emission = global_textureLod(ctx, info.texture, uv, 0.0);
                float alpha = info.color.w;
                if (info.type == MTL_EFFECT_EXPLOSION) {
                    float3 normal = float3(v0.normal) * b.x + float3(v1.normal) * b.y + float3(v2.normal) * b.z;
                    emission.rgb = mix(emission.rgb, get_explosion_color(normal, ray.direction), alpha);
                    emission.rgb *= global_ubo.pt_explosion_brightness;
                }
                emission.a *= alpha;
                emission.rgb *= emission.a;
                emission.rgb *= global_ubo.prev_adapted_luminance * 500.0;
                color = emission;
            }

            if (color.a > 0.0) {
                if (have_fog) {
                    blend_fogs_behind(fog1, fog2, fog_t, hit_t, accumulated);
                    fog_t = hit_t;
                }
                accumulated = alpha_blend_premultiplied(accumulated, color);
            }

            t_start = hit_t + max(hit_t * 1e-4, 0.01);
            if (t_start >= ray.t_max)
                break;
        }
    }

    if (have_fog)
        blend_fogs_behind(fog1, fog2, fog_t, ray.t_max, accumulated);
    return accumulated;
}

//
// light_lists.h
//

struct LightPolygon {
    float3x3 positions;
    float3   color;               // negative for sky lights
    float    light_style_scale;
    float    prev_style_scale;
};

static inline LightPolygon get_light_polygon(thread const PtCtx &ctx, uint index)
{
    device const MTLLightPoly &p = ctx.light_polys[index];
    LightPolygon light;
    light.positions = float3x3(float3(p.v0), float3(p.v1), float3(p.v2));
    light.color = (p.pad0 != 0.0) ? -max(float3(p.radiance), float3(1e-6)) : float3(p.radiance);
    light.light_style_scale = 1.0;
    light.prev_style_scale = 1.0;
    return light;
}

static float spherical_tri_area(float3x3 positions, float3 p, float3 n, float3 V, float phong_exp, float phong_scale, float phong_weight)
{
    positions[0] = positions[0] - p;
    positions[1] = positions[1] - p;
    positions[2] = positions[2] - p;

    float3 g = cross(positions[1] - positions[0], positions[2] - positions[0]);
    if (dot(n, positions[0]) <= 0.0 && dot(n, positions[1]) <= 0.0 && dot(n, positions[2]) <= 0.0)
        return 0.0;
    if (dot(g, positions[0]) >= 0.0 && dot(g, positions[1]) >= 0.0 && dot(g, positions[2]) >= 0.0)
        return 0.0;

    float3 L = normalize(positions * float3(1.0 / 3.0));
    float specular = phong(n, L, V, phong_exp) * phong_scale;
    float brdf = mix(1.0, specular, phong_weight);

    float3 A = normalize(positions[0]);
    float3 B = normalize(positions[1]);
    float3 C = normalize(positions[2]);

    float area = 2.0 * atan2(abs(dot(A, cross(B, C))), 1.0 + dot(A, B) + dot(B, C) + dot(A, C));
    float pa = max(area - 1e-5, 0.0);
    return pa * brdf;
}

static float get_spherical_triangle_pdfw(float3x3 positions)
{
    float3 A = normalize(positions[0]);
    float3 B = normalize(positions[1]);
    float3 C = normalize(positions[2]);
    float area = 2.0 * atan2(abs(dot(A, cross(B, C))), 1.0 + dot(A, B) + dot(B, C) + dot(A, C));
    return 1.0 / area;
}

static float3x3 project_triangle(float3x3 positions, float3 p)
{
    positions[0] = normalize(positions[0] - p);
    positions[1] = normalize(positions[1] - p);
    positions[2] = normalize(positions[2] - p);
    return positions;
}

// Arvo, "Stratified sampling of spherical triangles".
static float3 sample_projected_triangle(float3 pt, float3x3 positions, float2 rnd, thread float3 &light_normal, thread float &pdfw)
{
    light_normal = normalize(cross(positions[1] - positions[0], positions[2] - positions[0]));

    positions[0] = positions[0] - pt;
    positions[1] = positions[1] - pt;
    positions[2] = positions[2] - pt;

    float o = dot(light_normal, positions[0]);

    float3 A = normalize(positions[0]);
    float3 B = normalize(positions[1]);
    float3 C = normalize(positions[2]);
    float3 cross_BC = cross(B, C);
    float3 norm_AB = normalize(cross(A, B));
    float3 norm_CA = normalize(cross(C, A));
    float cos_c = dot(A, B);
    float cos_alpha = dot(norm_AB, -norm_CA);

    float area = 2.0 * atan2(abs(dot(A, cross_BC)), 1.0 + cos_c + dot(B, C) + dot(A, C));
    float new_area = rnd.x * area;

    float sin_alpha = sqrt(1.0 - cos_alpha * cos_alpha);
    float sin_new_area = sin(new_area);
    float cos_new_area = cos(new_area);
    float p = sin_new_area * cos_alpha - cos_new_area * sin_alpha;
    float q = cos_new_area * cos_alpha + sin_new_area * sin_alpha;

    float u = q - cos_alpha;
    float v = p + sin_alpha * cos_c;

    float cos_b = clamp(((v * q - u * p) * cos_alpha - v) / ((v * p + u * q) * sin_alpha), -1.0, 1.0);
    float3 new_C = cos_b * A + sqrt(1.0 - cos_b * cos_b) * normalize(C - dot(C, A) * A);

    float z = 1.0 - rnd.y * (1.0 - dot(new_C, B));
    float3 direction = z * B + sqrt(1.0 - z * z) * normalize(new_C - dot(new_C, B) * B);

    float3 lo = direction * (o / dot(light_normal, direction));
    pdfw = 1.0 / area;
    return pt + lo;
}

static inline uint get_light_stats_addr(thread const PtCtx &ctx, uint cluster, uint light, uint side)
{
    uint addr = cluster;
    addr = addr * uint(global_ubo.num_static_lights) + light;
    addr = addr * 6u + side;
    addr = addr * 2u;
    return addr;
}

static void sample_polygonal_lights(
    thread const PtCtx &ctx,
    uint list_idx,
    float3 p, float3 n, float3 gn, float3 V,
    float phong_exp, float phong_scale, float phong_weight,
    bool is_gradient,
    thread float3 &position_light, thread float3 &light_color, thread int &light_index,
    thread float &pdfw, thread bool &is_sky_light,
    float3 rng)
{
    position_light = float3(0.0);
    light_index = -1;
    light_color = float3(0.0);
    pdfw = 0.0;
    is_sky_light = false;

    uint list_start, list_end, light_count;
    bool use_lists = global_ubo.num_clusters != 0u;
    if (use_lists) {
        if (list_idx == ~0u || list_idx >= global_ubo.num_clusters)
            return;
        list_start = ctx.light_list_offsets[list_idx];
        list_end = ctx.light_list_offsets[list_idx + 1];
        // The light count the selection is based on may differ from the current
        // one so that gradient samples pick the same light as last frame.
        uint history_index = (ctx.rng_seed >> RNG_SEED_SHIFT_FRAME) % LIGHT_COUNT_HISTORY;
        light_count = ctx.light_counts[history_index * MAX_LIGHT_LISTS + list_idx];
    } else {
        // Metal: maps without vis data sample every light.
        list_start = 0u;
        list_end = uint(global_ubo.num_static_lights);
        light_count = list_end;
    }

    float partitions = ceil(float(light_count) / float(MAX_BRUTEFORCE_SAMPLING));
    rng.x *= partitions;
    float fpart = min(floor(rng.x), partitions - 1.0);
    rng.x -= fpart;
    list_start += uint(fpart);
    int stride = int(partitions);

    float mass = 0.0;
    float light_masses[MAX_BRUTEFORCE_SAMPLING];

    for (uint i = 0, n_idx = list_start; i < MAX_BRUTEFORCE_SAMPLING; i++, n_idx += uint(stride)) {
        if (n_idx >= list_start + light_count)
            break;
        if (n_idx >= list_end) {
            light_masses[i] = 0.0;
            continue;
        }

        uint current_idx = use_lists ? ctx.light_list_lights[n_idx] : n_idx;
        LightPolygon light = get_light_polygon(ctx, current_idx);

        float m = spherical_tri_area(light.positions, p, n, V, phong_exp, phong_scale, phong_weight);

        float light_lum = luminance(light.color);
        light_lum *= is_gradient ? light.prev_style_scale : light.light_style_scale;

        if (light_lum < 0.0 && global_ubo.environment_type == ENVIRONMENT_DYNAMIC) {
            // Limit the sky luminance used for light selection.
            m *= clamp(global_ubo.sky_luminance, global_ubo.pt_min_log_sky_luminance, global_ubo.pt_max_log_sky_luminance);
        } else {
            m *= abs(light_lum);
        }

        // CDF adjustment from the shadowing statistics of a previous frame.
        if (global_ubo.pt_light_stats != 0.0 && m > 0.0 && use_lists &&
            current_idx < uint(global_ubo.num_static_lights)) {
            uint buffer_idx = uint(global_ubo.current_frame_idx);
            buffer_idx += is_gradient ? (NUM_LIGHT_STATS_BUFFERS - 2) : (NUM_LIGHT_STATS_BUFFERS - 1);
            buffer_idx = buffer_idx % NUM_LIGHT_STATS_BUFFERS;

            uint addr = get_light_stats_addr(ctx, list_idx, current_idx, get_primary_direction(n));
            device atomic_uint *stats = ctx.light_stats + buffer_idx * global_ubo.light_stats_size;
            uint num_hits = atomic_load_explicit(&stats[addr], memory_order_relaxed);
            uint num_misses = atomic_load_explicit(&stats[addr + 1], memory_order_relaxed);
            uint num_total = num_hits + num_misses;
            if (num_total > 0u)
                m *= max(float(num_hits) / float(num_total), 0.1);
        }

        mass += m;
        light_masses[i] = m;
    }

    if (mass <= 0.0)
        return;

    rng.x *= mass;
    int current_idx = -1;
    mass *= partitions;
    float pdf = 0.0;

    for (uint i = 0, n_idx = list_start; i < MAX_BRUTEFORCE_SAMPLING; i++, n_idx += uint(stride)) {
        if (n_idx >= list_start + light_count)
            break;
        pdf = light_masses[i];
        current_idx = int(n_idx);
        rng.x -= pdf;
        if (rng.x <= 0.0)
            break;
    }

    if (rng.x > 0.0)
        return;

    pdf /= mass;

    if (current_idx >= 0) {
        current_idx = use_lists ? int(ctx.light_list_lights[current_idx]) : current_idx;
        LightPolygon light = get_light_polygon(ctx, uint(current_idx));

        float3 light_normal;
        position_light = sample_projected_triangle(p, light.positions, rng.yz, light_normal, pdfw);

        float3 L = normalize(position_light - p);
        if (dot(L, gn) <= 0.0)
            pdfw = 0.0;

        if (pdfw > 0.0) {
            float LdotNL = max(0.0, -dot(light_normal, L));
            float spotlight = sqrt(LdotNL);
            float inv_pdfw = 1.0 / pdfw;

            if (light.color.r >= 0.0) {
                light_color = light.color * (inv_pdfw * spotlight * light.light_style_scale);
            } else {
                light_color = env_map(ctx, L, true) * inv_pdfw * global_ubo.pt_env_scale;
                is_sky_light = true;
            }
        }

        light_index = current_idx;
    }

    light_color /= pdf;
}

static float compute_dynlight_sphere(thread const PtCtx &ctx, uint light_idx, float3 light_center, float3 p,
                                     thread float3 &position_light, float3 rng)
{
    float3 c = light_center - p;
    float dist = length(c);
    float rdist = 1.0 / dist;
    float3 L = c * rdist;

    float sphere_radius = global_ubo.dyn_light_data[light_idx].radius;
    float irradiance = 2.0 * (1.0 - sqrt(max(0.0, 1.0 - square(sphere_radius * rdist))));

    float3x3 onb = construct_ONB_frisvad(L);
    float3 diskpt;
    diskpt.xy = sample_disk(rng.yz);
    diskpt.z = sqrt(max(0.0, 1.0 - diskpt.x * diskpt.x - diskpt.y * diskpt.y));

    position_light = light_center + (onb[0] * diskpt.x + onb[2] * diskpt.y - L * diskpt.z) * sphere_radius;
    return irradiance;
}

static float compute_dynlight_spot(thread const PtCtx &ctx, uint light_idx, uint spot_style, float3 light_center, float3 p,
                                   thread float3 &position_light, float3 rng)
{
    float3x3 onb = construct_ONB_frisvad(float3(global_ubo.dyn_light_data[light_idx].spot_direction));
    float emitter_radius = global_ubo.dyn_light_data[light_idx].radius;
    float2 diskpt = sample_disk(rng.yz);
    position_light = light_center + (onb[0] * diskpt.x + onb[2] * diskpt.y) * emitter_radius;

    float3 c = position_light - p;
    float dist = length(c);
    float rdist = 1.0 / dist;
    float3 L = c * rdist;

    float3 L_l = -L * onb;
    float cosTheta = L_l.y;
    float falloff = 0.0;

    if (spot_style == DYNLIGHT_SPOT_EMISSION_PROFILE_FALLOFF) {
        float2 spot_falloff = unpackHalf2x16(global_ubo.dyn_light_data[light_idx].spot_data);
        float cosTotalWidth = spot_falloff.x;
        float cosFalloffStart = spot_falloff.y;
        if (cosTheta < cosTotalWidth)
            falloff = 0.0;
        else if (cosTheta > cosFalloffStart)
            falloff = 1.0;
        else {
            float delta = (cosTheta - cosTotalWidth) / (cosFalloffStart - cosTotalWidth);
            falloff = (delta * delta) * (delta * delta);
        }
    } else if (spot_style == DYNLIGHT_SPOT_EMISSION_PROFILE_AXIS_ANGLE_TEXTURE) {
        uint spot_data = global_ubo.dyn_light_data[light_idx].spot_data;
        float theta = acos(cosTheta);
        float totalWidth = unpackHalf2x16(spot_data).x;
        uint texture_num = spot_data >> 16;
        if (cosTheta >= 0.0) {
            float tc = clamp(theta / totalWidth, 0.0, 1.0);
            falloff = global_textureLod(ctx, texture_num, float2(tc, 0.0), 0.0).r;
        }
    }

    return 2.0 * falloff * square(rdist);
}

static void sample_dynamic_lights(thread const PtCtx &ctx, float3 p, float3 n, float3 gn, float max_solid_angle,
                                  thread float3 &position_light, thread float3 &light_color, float3 rng)
{
    position_light = float3(0.0);
    light_color = float3(0.0);

    if (global_ubo.num_dyn_lights == 0)
        return;

    float random_light = rng.x * float(global_ubo.num_dyn_lights);
    uint light_idx = min(uint(global_ubo.num_dyn_lights - 1), uint(random_light));

    float3 light_center = float3(global_ubo.dyn_light_data[light_idx].center);
    light_color = float3(global_ubo.dyn_light_data[light_idx].color);

    uint light_type = global_ubo.dyn_light_data[light_idx].type & 0xffffu;
    uint light_style = global_ubo.dyn_light_data[light_idx].type >> 16;

    float irradiance;
    if (light_type == DYNLIGHT_SPHERE)
        irradiance = compute_dynlight_sphere(ctx, light_idx, light_center, p, position_light, rng);
    else
        irradiance = compute_dynlight_spot(ctx, light_idx, light_style, light_center, p, position_light, rng);
    irradiance = min(irradiance, max_solid_angle);
    irradiance *= float(global_ubo.num_dyn_lights);

    light_color *= irradiance;

    if (dot(position_light - p, gn) <= 0.0)
        light_color = float3(0.0);
}

//
// path_tracer_rgen.h: lighting and materials
//

static inline float3 rgbToNormal(float3 rgb, thread float &len)
{
    float3 n = float3(rgb.xy * 2.0 - 1.0, rgb.z);
    len = length(n);
    return len > 0.0 ? n / len : float3(0.0);
}

static inline float AdjustRoughnessToksvig(thread const PtCtx &ctx, float roughness, float normalMapLen, float mip_level)
{
    float effect = global_ubo.pt_toksvig * clamp(mip_level, 0.0, 1.0);
    float shininess = RoughnessSquareToSpecPower(roughness) * effect;
    float ft = normalMapLen / mix(shininess, 1.0, normalMapLen);
    ft = max(ft, 0.01);
    return SpecPowerToRoughnessSquare(ft * shininess / effect);
}

static inline float get_specular_sampled_lighting_weight(float roughness, float3 N, float3 V, float3 L, float pdfw)
{
    float ggxVndfPdf = ImportanceSampleGGX_VNDF_PDF(max(roughness, 0.01), N, V, L);
    return clamp(pdfw / (pdfw + ggxVndfPdf), 0.0, 1.0);
}

static float3 ImportanceSampleGGX_VNDF(thread const PtCtx &ctx, float2 u, float roughness, float3 V, float3x3 basis)
{
    float alpha = square(roughness);

    float3 Ve = -float3(dot(V, basis[0]), dot(V, basis[2]), dot(V, basis[1]));

    float3 Vh = normalize(float3(alpha * Ve.x, alpha * Ve.y, Ve.z));

    float lensq = square(Vh.x) + square(Vh.y);
    float3 T1 = lensq > 0.0 ? float3(-Vh.y, Vh.x, 0.0) * rsqrt(lensq) : float3(1.0, 0.0, 0.0);
    float3 T2 = cross(Vh, T1);

    float r = sqrt(u.x * global_ubo.pt_ndf_trim);
    float phi = 2.0 * M_PI * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - square(t1)) + s * t2;

    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - square(t1) - square(t2))) * Vh;

    float3 Ne = float3(alpha * Nh.x, max(0.0, Nh.z), alpha * Nh.y);
    return normalize(basis * Ne);
}

static inline float3 demodulate_specular(thread const PtCtx &ctx, float3 base_reflectivity, float3 specular)
{
    if (global_ubo.flt_enable == 0.0)
        return specular;
    return specular / max(float3(0.01), base_reflectivity);
}

static void get_direct_illumination(
    thread const PtCtx &ctx,
    float3 position, float3 normal, float3 geo_normal,
    uint cluster_idx, uint material_id, int shadow_cull_mask,
    float3 view_direction, float3 albedo, float3 base_reflectivity,
    float specular_factor, float roughness, int surface_medium, bool enable_caustics,
    float direct_specular_weight, bool enable_polygonal, bool enable_dynamic,
    bool is_gradient, int bounce,
    thread float3 &diffuse, thread float3 &specular)
{
    diffuse = float3(0.0);
    specular = float3(0.0);

    float3 pos_on_light_polygonal = float3(0.0);
    float3 pos_on_light_dynamic = float3(0.0);
    float3 contrib_polygonal = float3(0.0);
    float3 contrib_dynamic = float3(0.0);

    float alpha = square(roughness);
    float phong_exp = RoughnessSquareToSpecPower(alpha);
    float phong_scale = min(100.0, 1.0 / (M_PI * square(alpha)));
    float phong_weight = clamp(specular_factor * luminance(base_reflectivity) / (luminance(base_reflectivity) + luminance(albedo)), 0.0, 0.9);

    int polygonal_light_index = -1;
    float polygonal_light_pdfw = 0.0;
    bool polygonal_light_is_sky = false;

    float3 rng = float3(
        get_rng(ctx, RNG_NEE_LIGHT_SELECTION(bounce)),
        get_rng(ctx, RNG_NEE_TRI_X(bounce)),
        get_rng(ctx, RNG_NEE_TRI_Y(bounce)));

    if (enable_polygonal) {
        sample_polygonal_lights(ctx, cluster_idx, position, normal, geo_normal, view_direction,
                                phong_exp, phong_scale, phong_weight, is_gradient,
                                pos_on_light_polygonal, contrib_polygonal, polygonal_light_index,
                                polygonal_light_pdfw, polygonal_light_is_sky, rng);
    }

    bool is_polygonal = true;
    float vis = 1.0;

    if (enable_dynamic) {
        float max_solid_angle = (bounce == 0) ? 2.0 * M_PI : 0.02;
        sample_dynamic_lights(ctx, position, normal, geo_normal, max_solid_angle, pos_on_light_dynamic, contrib_dynamic, rng);
    }

    float spec_polygonal = phong(normal, normalize(pos_on_light_polygonal - position), view_direction, phong_exp) * phong_scale;
    float spec_dynamic = phong(normal, normalize(pos_on_light_dynamic - position), view_direction, phong_exp) * phong_scale;

    float l_polygonal = luminance(abs(contrib_polygonal)) * mix(1.0, spec_polygonal, phong_weight);
    float l_dynamic = luminance(abs(contrib_dynamic)) * mix(1.0, spec_dynamic, phong_weight);
    float l_sum = l_polygonal + l_dynamic;

    bool null_light = (l_sum == 0.0);

    float w = null_light ? 0.5 : l_polygonal / (l_polygonal + l_dynamic);

    float rng2 = get_rng(ctx, RNG_NEE_LIGHT_TYPE(bounce));
    is_polygonal = (rng2 < w);
    vis = is_polygonal ? (1.0 / w) : (1.0 / (1.0 - w));
    float3 pos_on_light = null_light ? position : (is_polygonal ? pos_on_light_polygonal : pos_on_light_dynamic);
    float3 contrib = is_polygonal ? contrib_polygonal : contrib_dynamic;

    Ray shadow_ray = get_shadow_ray(position - view_direction * 0.01, pos_on_light, 0.0);

    vis *= trace_shadow_ray(ctx, shadow_ray, null_light ? 0 : shadow_cull_mask);
    if (enable_caustics)
        contrib *= trace_caustic_ray(ctx, shadow_ray, surface_medium);

    // Light shadowing statistics for the next frame's light selection.
    if (global_ubo.pt_light_stats != 0.0 && is_polygonal && !null_light && global_ubo.num_clusters != 0u &&
        cluster_idx != ~0u && polygonal_light_index >= 0 && polygonal_light_index < global_ubo.num_static_lights) {
        uint addr = get_light_stats_addr(ctx, cluster_idx, uint(polygonal_light_index), get_primary_direction(normal));
        if (vis == 0.0)
            addr += 1u;
        uint buffer_idx = uint(global_ubo.current_frame_idx) % NUM_LIGHT_STATS_BUFFERS;
        atomic_fetch_add_explicit(ctx.light_stats + buffer_idx * global_ubo.light_stats_size + addr, 1u, memory_order_relaxed);
    }

    if (null_light)
        return;

    float3 radiance = vis * contrib;

    float3 L = normalize(pos_on_light - position);

    if (is_polygonal && direct_specular_weight > 0.0 && polygonal_light_is_sky && global_ubo.pt_specular_mis != 0.0) {
        direct_specular_weight *= get_specular_sampled_lighting_weight(roughness, normal, -view_direction, L, polygonal_light_pdfw);
    }

    float3 F = float3(0.0);

    if (vis > 0.0 && direct_specular_weight > 0.0) {
        float3 specular_brdf = GGX_times_NdotL(view_direction, normalize(pos_on_light - position),
                                               normal, roughness, base_reflectivity, 0.0, specular_factor, F);
        specular = radiance * specular_brdf * direct_specular_weight;
    }

    float NdotL = max(0.0, dot(normal, L));
    float diffuse_brdf = NdotL / M_PI;
    diffuse = radiance * diffuse_brdf * (float3(1.0) - F);
}

static void get_sunlight(
    thread const PtCtx &ctx,
    uint cluster_idx, uint material_id,
    float3 position, float3 normal, float3 geo_normal, float3 view_direction,
    float3 base_reflectivity, float specular_factor, float roughness,
    int surface_medium, bool enable_caustics, bool sun_shape,
    thread float3 &diffuse, thread float3 &specular, int shadow_cull_mask)
{
    diffuse = float3(0.0);
    specular = float3(0.0);

    if (global_ubo.sun_visible == 0)
        return;

    bool visible = (cluster_idx == ~0u) || global_ubo.num_clusters == 0u ||
                   (ctx.sky_visibility[cluster_idx >> 5] & (1u << (cluster_idx & 31u))) != 0u;
    if (!visible)
        return;

    float2 rng3 = float2(get_rng(ctx, RNG_SUNLIGHT_X(0)), get_rng(ctx, RNG_SUNLIGHT_Y(0)));
    float2 disk = sample_disk(rng3);
    disk.xy *= global_ubo.sun_tan_half_angle;

    float3 direction = normalize(float3(global_ubo.sun_direction) + float3(global_ubo.sun_tangent) * disk.x +
                                 float3(global_ubo.sun_bitangent) * disk.y);

    float NdotL = dot(direction, normal);
    float GNdotL = dot(direction, geo_normal);
    if (NdotL <= 0.0 || GNdotL <= 0.0)
        return;

    Ray shadow_ray = get_shadow_ray(position - view_direction * 0.01, position + direction * 10000.0, 0.0);

    float vis = trace_shadow_ray(ctx, shadow_ray, shadow_cull_mask);
    if (vis == 0.0)
        return;

    float3 radiance;
    if (sun_shape) {
        // ENABLE_SUN_SHAPE: the sun as seen in the environment, so partially
        // occluded suns cast properly shaped shadows.
        radiance = (global_ubo.sun_solid_angle * global_ubo.pt_env_scale) * env_map(ctx, direction, false);
    } else {
        radiance = float3(global_ubo.sun_color);
    }

    if (enable_caustics)
        radiance *= trace_caustic_ray(ctx, shadow_ray, surface_medium);

    float3 F = float3(0.0);
    if (global_ubo.pt_sun_specular > 0.0) {
        float NoH_offset = 0.5 * square(global_ubo.sun_tan_half_angle);
        float3 specular_brdf = GGX_times_NdotL(view_direction, float3(global_ubo.sun_direction),
                                               normal, roughness, base_reflectivity, NoH_offset, specular_factor, F);
        specular = radiance * specular_brdf;
    }

    float diffuse_brdf = NdotL / M_PI;
    diffuse = radiance * diffuse_brdf * (float3(1.0) - F);
}

static inline float3 clamp_output(float3 c)
{
    if (any(isnan(c)) || any(isinf(c)))
        return float3(0.0);
    return clamp(c, float3(0.0), float3(MAX_OUTPUT_VALUE));
}

static float3 sample_emissive_texture(thread const PtCtx &ctx, MaterialInfo minfo, float2 tex_coord,
                                      float2 tex_coord_x, float2 tex_coord_y, float mip_level)
{
    if (minfo.emissive_texture != 0u) {
        float4 image3;
        if (mip_level >= 0.0)
            image3 = material_textureLod(ctx, minfo.emissive_texture, tex_coord, mip_level);
        else
            image3 = material_textureGrad(ctx, minfo.emissive_texture, tex_coord, tex_coord_x, tex_coord_y);
        return correct_emissive(image3.rgb) * minfo.emissive_factor;
    }
    // Metal: synthesized lights without an emissive map carry a constant.
    return minfo.emissive_constant;
}

static float3 get_emissive_shell(thread const PtCtx &ctx, uint material_id, uint shell)
{
    float3 c = float3(0.0);
    if ((shell & SHELL_MASK) != 0u) {
        if ((shell & SHELL_HALF_DAM) != 0u) c = float3(0.56, 0.59, 0.45);
        if ((shell & SHELL_DOUBLE) != 0u) { c.r = 0.9; c.g = 0.7; }
        if ((shell & SHELL_LITE_GREEN) != 0u) c = float3(0.7, 1.0, 0.7);
        if ((shell & SHELL_RED) != 0u) c.r += 1.0;
        if ((shell & SHELL_GREEN) != 0u) c.g += 1.0;
        if ((shell & SHELL_BLUE) != 0u) c.b += 1.0;
        if ((material_id & MATERIAL_FLAG_WEAPON) != 0u) c *= 0.2;
    }
    // tonemap_buffer.adapted_luminance
    if (global_ubo.prev_adapted_luminance > 0.0)
        c *= global_ubo.prev_adapted_luminance * 100.0;
    return c;
}

static bool get_is_gradient(thread const PtCtx &ctx, int2 ipos)
{
    if (global_ubo.flt_enable != 0.0) {
        uint u = IMG_LOAD(ASVGF_GRAD_SMPL_POS_A, ipos / GRAD_DWN).r;
        int2 grad_strata_pos = int2(int(u >> (STRATUM_OFFSET_SHIFT * 0)), int(u >> (STRATUM_OFFSET_SHIFT * 1))) & STRATUM_OFFSET_MASK;
        return (u > 0u && all(grad_strata_pos == ipos % GRAD_DWN));
    }
    return false;
}

static void get_material(
    thread const PtCtx &ctx,
    Triangle triangle, float3 bary, float2 tex_coord, float2 tex_coord_x, float2 tex_coord_y,
    float mip_level, float3 geo_normal,
    thread float3 &base_color, thread float3 &normal, thread float &metallic, thread float &roughness,
    thread float3 &emissive, thread float &specular_factor)
{
    MaterialInfo minfo = get_material_info(ctx, triangle.material_index);

    perturb_tex_coord(triangle.material_id, global_ubo.time, tex_coord);

    float4 image1 = float4(1.0);
    if (minfo.base_texture != 0u) {
        if (mip_level >= 0.0)
            image1 = material_textureLod(ctx, minfo.base_texture, tex_coord, mip_level);
        else
            image1 = material_textureGrad(ctx, minfo.base_texture, tex_coord, tex_coord_x, tex_coord_y);
    }

    base_color = image1.rgb * minfo.base_factor;
    base_color = clamp(base_color, float3(0.0), float3(1.0));

    normal = geo_normal;
    metallic = 0.0;
    roughness = 1.0;

    if (minfo.normals_texture != 0u) {
        float4 image2;
        if (mip_level >= 0.0)
            image2 = material_textureLod(ctx, minfo.normals_texture, tex_coord, mip_level);
        else
            image2 = material_textureGrad(ctx, minfo.normals_texture, tex_coord, tex_coord_x, tex_coord_y);

        float normalMapLen;
        float3 local_normal = rgbToNormal(image2.rgb, normalMapLen);

        if (dot(triangle.tangents[0], triangle.tangents[0]) > 0.0) {
            float3 tangent = normalize(triangle.tangents * bary);
            float3 bitangent = cross(geo_normal, tangent);

            if ((triangle.material_id & MATERIAL_FLAG_HANDEDNESS) != 0u)
                bitangent = -bitangent;

            normal = tangent * local_normal.x + bitangent * local_normal.y + geo_normal * local_normal.z;

            float bump_scale = global_ubo.pt_bump_scale * minfo.bump_scale;
            if (is_glass(triangle.material_id))
                bump_scale *= 0.2;

            normal = normalize(mix(geo_normal, normal, bump_scale));
        }

        metallic = clamp(image2.a * minfo.metalness_factor, 0.0, 1.0);

        if (minfo.roughness_override >= 0.0)
            roughness = max(image1.a, minfo.roughness_override);
        else
            roughness = image1.a;

        roughness = clamp(roughness, 0.0, 1.0);

        float effective_mip = mip_level;
        if (effective_mip < 0.0) {
            texture2d<float> nt = ctx.textures[minfo.normals_texture].tex;
            float2 texSize = float2(nt.get_width(), nt.get_height());
            float2 tx = tex_coord_x * texSize;
            float2 ty = tex_coord_y * texSize;
            float d = max(dot(tx, tx), dot(ty, ty));
            effective_mip = 0.5 * log2(d);
        }

        bool is_mirror = (roughness < MAX_MIRROR_ROUGHNESS) && (is_chrome(triangle.material_id) || is_screen(triangle.material_id));

        if (normalMapLen > 0.0 && global_ubo.pt_toksvig > 0.0 && effective_mip > 0.0 && !is_mirror)
            roughness = AdjustRoughnessToksvig(ctx, roughness, normalMapLen, effective_mip);
    }

    if (global_ubo.pt_roughness_override >= 0.0) roughness = global_ubo.pt_roughness_override;
    if (global_ubo.pt_metallic_override >= 0.0) metallic = global_ubo.pt_metallic_override;

    specular_factor = mix(minfo.specular_factor, 1.0, metallic);

    if (triangle.emissive_factor > 0.0) {
        emissive = sample_emissive_texture(ctx, minfo, tex_coord, tex_coord_x, tex_coord_y, mip_level);
        emissive *= triangle.emissive_factor;
    } else {
        emissive = float3(0.0);
    }

    // Metal: lava is emissive through its albedo (mtl_pt_lava_emissive).
    if ((triangle.material_id & MATERIAL_KIND_MASK) == MATERIAL_KIND_LAVA)
        emissive += base_color * global_ubo.lava_emissive;

    emissive += get_emissive_shell(ctx, triangle.material_id, triangle.shell) * base_color * (1.0 - metallic * 0.9);
}

static bool get_camera_uv(float2 tex_coord, thread float2 &cameraUV)
{
    const float2 minUV = float2(11.0 / 256.0, 14.0 / 256.0);
    const float2 maxUV = float2(245.0 / 256.0, 148.0 / 256.0);

    tex_coord = fract(tex_coord);
    cameraUV = (tex_coord - minUV) / (maxUV - minUV);
    return all(cameraUV > float2(0.0)) && all(cameraUV < float2(1.0));
}

// "Improved Shader and Texture Level of Detail Using Ray Cones", section 5.
static void compute_anisotropic_texture_gradients(
    float3 intersection, float3 normal, float3 ray_direction, float cone_radius,
    float3x3 positions, float3x2 tex_coords, float2 tex_coords_at_intersection,
    thread float2 &texGradient1, thread float2 &texGradient2, thread float &fwidth_depth)
{
    float3 a1 = ray_direction - dot(normal, ray_direction) * normal;
    float3 p1 = a1 - dot(ray_direction, a1) * ray_direction;
    a1 *= cone_radius / max(0.0001, length(p1));

    float3 a2 = cross(normal, a1);
    float3 p2 = a2 - dot(ray_direction, a2) * ray_direction;
    a2 *= cone_radius / max(0.0001, length(p2));

    float3 eP, delta = intersection - positions[0];
    float3 e1 = positions[1] - positions[0];
    float3 e2 = positions[2] - positions[0];
    float inv_tri_area = 1.0 / dot(normal, cross(e1, e2));

    eP = delta + a1;
    float u1 = dot(normal, cross(eP, e2)) * inv_tri_area;
    float v1 = dot(normal, cross(e1, eP)) * inv_tri_area;
    texGradient1 = (1.0 - u1 - v1) * tex_coords[0] + u1 * tex_coords[1] + v1 * tex_coords[2] - tex_coords_at_intersection;

    eP = delta + a2;
    float u2 = dot(normal, cross(eP, e2)) * inv_tri_area;
    float v2 = dot(normal, cross(e1, eP)) * inv_tri_area;
    texGradient2 = (1.0 - u2 - v2) * tex_coords[0] + u2 * tex_coords[1] + v2 * tex_coords[2] - tex_coords_at_intersection;

    fwidth_depth = 1.0 / max(0.1, abs(dot(a1, ray_direction)) + abs(dot(a2, ray_direction)));
}

// Launch layout: x over half the width, z selects the checkerboard field.
static inline int2 launch_ipos(thread const PtCtx &ctx)
{
    int2 ipos = int2(ctx.launch_id.xy);
    if (ctx.launch_id.z != 0u)
        ipos.x += global_ubo.width / 2;
    return ipos;
}

static inline bool launch_is_odd(thread const PtCtx &ctx) { return ctx.launch_id.z != 0u; }

static inline bool launch_in_range(thread const PtCtx &ctx)
{
    return int(ctx.launch_id.x) < global_ubo.width / 2 && int(ctx.launch_id.y) < global_ubo.height;
}

//
// primary_rays.rgen
//

static Ray get_primary_ray(thread const PtCtx &ctx, float2 screen_pos)
{
    float3 view_dir = projection_screen_to_view(ctx, screen_pos, 1.0, false);
    view_dir = normalize((global_ubo.invV * float4(view_dir, 0.0)).xyz);

    Ray ray;
    ray.origin = float3(0.0);
    ray.direction = view_dir;
    ray.t_min = 0.0;
    ray.t_max = PRIMARY_RAY_T_MAX;

    if (global_ubo.pt_aperture > 0.0) {
        float3 right = global_ubo.invV[0].xyz;
        float3 up = global_ubo.invV[1].xyz;
        float3 forward = global_ubo.invV[2].xyz;

        float distance = global_ubo.pt_focus / dot(view_dir, forward);
        float3 focal_point = view_dir * distance;

        float2 uv = float2(get_rng(ctx, RNG_PRIMARY_APERTURE_X), get_rng(ctx, RNG_PRIMARY_APERTURE_Y));
        float2 planar_offset;

        if (global_ubo.pt_aperture_type < 3.0) {
            planar_offset = sample_disk(uv);
        } else {
            float triangle = uv.x * global_ubo.pt_aperture_type;
            uv.x = fract(triangle);
            triangle = floor(triangle);

            float3 bary = sample_triangle(uv);
            float section_angle = 2.0 * M_PI / global_ubo.pt_aperture_type;
            float a1 = section_angle * (triangle + global_ubo.pt_aperture_angle);
            float a2 = section_angle * (triangle + global_ubo.pt_aperture_angle + 1.0);
            planar_offset = bary.x * float2(cos(a1), sin(a1)) + bary.y * float2(cos(a2), sin(a2));
        }

        planar_offset *= global_ubo.pt_aperture;
        float3 offset = planar_offset.x * right + planar_offset.y * up;

        ray.origin = offset;
        ray.direction = normalize(focal_point - ray.origin);
    }

    ray.origin += global_ubo.cam_pos.xyz;
    return ray;
}

static void generate_rng_seed(thread PtCtx &ctx, int2 ipos, bool is_odd_checkerboard)
{
    int frame_num = global_ubo.current_frame_idx;
    uint frame_offset = uint(frame_num) / NUM_BLUE_NOISE_TEX;

    uint rng_seed = 0u;
    rng_seed |= (uint(ipos.x + int(frame_offset)) % BLUE_NOISE_RES) << RNG_SEED_SHIFT_X;
    rng_seed |= (uint(ipos.y + int(frame_offset << 4)) % BLUE_NOISE_RES) << RNG_SEED_SHIFT_Y;
    rng_seed |= uint(is_odd_checkerboard) << RNG_SEED_SHIFT_ISODD;
    rng_seed |= uint(frame_num) << RNG_SEED_SHIFT_FRAME;
    ctx.rng_seed = rng_seed;

    IMG_STORE(ASVGF_RNG_SEED_A, ipos, uint4(rng_seed));
}

static int2 get_image_position(thread const PtCtx &ctx)
{
    int2 pos;
    bool is_even_checkerboard = ctx.launch_id.z == 0u;
    if (global_ubo.pt_swap_checkerboard != 0)
        is_even_checkerboard = !is_even_checkerboard;

    if (is_even_checkerboard)
        pos.x = int(ctx.launch_id.x * 2) + int(ctx.launch_id.y & 1u);
    else
        pos.x = int(ctx.launch_id.x * 2 + 1) - int(ctx.launch_id.y & 1u);
    pos.y = int(ctx.launch_id.y);
    return pos;
}

kernel void pt_primary_rays(PT_KERNEL_PARAMS)
{
    PT_CTX_INIT
    if (!launch_in_range(ctx))
        return;

    int2 ipos = launch_ipos(ctx);
    bool is_odd_checkerboard = launch_is_odd(ctx);

    generate_rng_seed(ctx, ipos, is_odd_checkerboard);

    float2 pixel_offset;
    if (global_ubo.flt_taa == AA_MODE_TAA || global_ubo.temporal_blend_factor > 0.0) {
        pixel_offset = float2(get_rng(ctx, RNG_PRIMARY_OFF_X), get_rng(ctx, RNG_PRIMARY_OFF_Y));
        pixel_offset -= float2(0.5);
    } else {
        pixel_offset = float2(global_ubo.sub_pixel_jitter);
    }

    const int2 image_position = get_image_position(ctx);
    const float2 pixel_center = float2(image_position) + float2(0.5);
    const float2 inUV = (pixel_center + pixel_offset) / float2(global_ubo.width, global_ubo.height);

    Ray ray = get_primary_ray(ctx, inUV);

    HitInfo hit = trace_geometry_ray(ctx, ray, PRIMARY_RAY_CULL_MASK);

    Triangle triangle;
    if (hit.found) {
        ray.t_max = hit.hit_distance;
        triangle = load_hit_triangle(ctx, hit.instance, hit.primitive);
    }

    float4 effects = trace_effects_ray(ctx, ray, false);

    float3 direction = ray.direction;

    if (!hit.found || (is_sky(triangle.material_id) && global_ubo.pt_show_sky == 0.0)) {
        float3 env = env_map(ctx, ray.direction, false);
        env *= global_ubo.pt_env_scale;

        float4 transparent = alpha_blend(effects, float4(env, 1.0));

        // Motion vector for the sky, so TAA does not blur it.
        float3 prev_view_dir = (global_ubo.V_prev * float4(direction, 0.0)).xyz;
        float2 prev_screen_pos;
        float prev_distance;
        projection_view_to_screen(ctx, prev_view_dir, prev_screen_pos, prev_distance, true);
        float2 motion = prev_screen_pos - inUV;

        IMG_STORE(PT_NORMAL_A, ipos, uint4(0u));
        IMG_STORE(PT_GEO_NORMAL_A, ipos, uint4(0u));
        IMG_STORE(PT_VIEW_DEPTH_A, ipos, float4(PRIMARY_RAY_T_MAX));
        IMG_STORE(PT_GODRAYS_THROUGHPUT_DIST, ipos, float4(1.0, 1.0, 1.0, PRIMARY_RAY_T_MAX));
        IMG_STORE(PT_VIEW_DIRECTION, ipos, float4(direction, 0.0));
        IMG_STORE(PT_SHADING_POSITION, ipos, float4(global_ubo.cam_pos.xyz + direction * PRIMARY_RAY_T_MAX, 0.0));
        IMG_STORE(PT_MOTION, ipos, float4(motion, 0.0, 0.0));
        IMG_STORE(PT_VISBUF_PRIM_A, ipos, uint4(0u));
        IMG_STORE(PT_VISBUF_BARY_A, ipos, float4(0.0));
        IMG_STORE(PT_BASE_COLOR_A, ipos, float4(0.0));
        IMG_STORE(PT_TRANSPARENT, ipos, transparent);
        // Metal: the remaining G-buffer channels are reset too, since these
        // images are not cleared between frames.
        IMG_STORE(PT_CLUSTER_A, ipos, uint4(0xffffu));
        IMG_STORE(PT_METALLIC_A, ipos, float4(0.0));
        IMG_STORE(PT_THROUGHPUT, ipos, float4(1.0, 1.0, 1.0, PRIMARY_RAY_T_MAX));
        return;
    }

    float3 bary = get_hit_barycentric(hit);

    IMG_STORE(PT_VISBUF_PRIM_A, ipos, uint4(triangle.instance_index, triangle.instance_prim, 0u, 0u));
    IMG_STORE(PT_VISBUF_BARY_A, ipos, float4(bary.yz, 0.0, 0.0));

    float3 position = triangle.positions * bary;
    float2 tex_coord = triangle.tex_coords * bary;
    float3 geo_normal = normalize(triangle.normals * bary);

    float3 flat_normal = normalize(cross(triangle.positions[1] - triangle.positions[0],
                                         triangle.positions[2] - triangle.positions[1]));

    // Metal: the flat normal is oriented along the vertex normals, which
    // also covers mirrored (left handed) weapons without vkpt's sign flag.
    if (dot(flat_normal, geo_normal) < 0.0)
        flat_normal = -flat_normal;

    if (dot(flat_normal, direction) > 0.0)
        geo_normal = -geo_normal;

    // View-space derivatives of depth, and the ray cone.
    Ray ray_0 = get_primary_ray(ctx, inUV);
    Ray ray_x = get_primary_ray(ctx, inUV + float2(1.0 / float(global_ubo.width), 0.0));
    Ray ray_y = get_primary_ray(ctx, inUV + float2(0.0, 1.0 / float(global_ubo.height)));

    float half_cone_angle = sqrt(1.0 - square(min(dot(ray_0.direction, ray_x.direction), dot(ray_0.direction, ray_y.direction))));

    float2 tex_coord_x, tex_coord_y;
    float fwidth_depth;
    compute_anisotropic_texture_gradients(position, flat_normal, ray.direction, hit.hit_distance * half_cone_angle,
                                          triangle.positions, triangle.tex_coords, tex_coord,
                                          tex_coord_x, tex_coord_y, fwidth_depth);

    if (global_ubo.pt_texture_lod_bias != 0.0) {
        tex_coord_x *= pow(2.0, global_ubo.pt_texture_lod_bias);
        tex_coord_y *= pow(2.0, global_ubo.pt_texture_lod_bias);
    }

    float3 pos_ws_curr = position;
    float3 pos_ws_prev = triangle.positions_prev * bary;

    float2 screen_pos_curr, screen_pos_prev;
    float distance_curr, distance_prev;
    projection_view_to_screen(ctx, (global_ubo.V * float4(pos_ws_curr, 1.0)).xyz, screen_pos_curr, distance_curr, false);
    projection_view_to_screen(ctx, (global_ubo.V_prev * float4(pos_ws_prev, 1.0)).xyz, screen_pos_prev, distance_prev, true);

    float3 motion;
    motion.xy = screen_pos_prev - screen_pos_curr;
    motion.z = distance_prev - distance_curr;

    IMG_STORE(PT_VIEW_DEPTH_A, ipos, float4(distance_curr));
    IMG_STORE(PT_MOTION, ipos, float4(motion, fwidth_depth));

    float3 primary_base_color = float3(1.0);
    float primary_metallic = 0.0;
    float primary_roughness = 1.0;
    float3 primary_emissive = float3(0.0);
    float primary_specular_factor = 1.0;
    float3 throughput = float3(1.0);
    float3 normal;

    get_material(ctx, triangle, bary, tex_coord, tex_coord_x, tex_coord_y, -1.0, geo_normal,
                 primary_base_color, normal, primary_metallic, primary_roughness, primary_emissive,
                 primary_specular_factor);

    if (global_ubo.medium != MEDIUM_NONE)
        throughput *= extinction(ctx, global_ubo.medium, length(position - global_ubo.cam_pos.xyz));

    uint material_id = triangle.material_id;
    uint transparency_kind = (material_id & MATERIAL_KIND_MASK) == MATERIAL_KIND_TRANSPARENT ? MATERIAL_KIND_TRANSPARENT : MATERIAL_KIND_TRANSP_MODEL;

    if (((is_chrome(material_id) || is_screen(material_id) || is_camera(material_id)) && primary_roughness >= MAX_MIRROR_ROUGHNESS) ||
        is_transparent(material_id))
        material_id = set_kind(material_id, MATERIAL_KIND_REGULAR);

    if (is_camera(material_id) && (global_ubo.pt_cameras == 0.0 || global_ubo.pt_reflect_refract == 0.0))
        material_id = set_kind(material_id, MATERIAL_KIND_SCREEN);

    int checkerboard_flags = CHECKERBOARD_FLAG_PRIMARY;

    float2 cameraUV = float2(0.0);
    if (is_camera(material_id)) {
        if (get_camera_uv(tex_coord, cameraUV)) {
            throughput *= 2.0;
            checkerboard_flags = CHECKERBOARD_FLAG_REFRACTION | CHECKERBOARD_FLAG_REFLECTION;
            primary_emissive = float3(0.0);

            if (is_odd_checkerboard) {
                material_id = set_kind(material_id, MATERIAL_KIND_SCREEN);
            } else {
                uint packed = (uint(cameraUV.x * 65535.0) & 0xffffu) | ((uint(cameraUV.y * 65535.0) & 0xffffu) << 16);
                IMG_STORE(PT_NORMAL_A, ipos, uint4(packed));
            }
        } else {
            material_id = set_kind(material_id, MATERIAL_KIND_REGULAR);
        }
    }

    if (is_screen(material_id) && luminance(primary_emissive) > 0.0) {
        // Odd field: reflection. Even field: the emissive surface on black.
        throughput *= 2.0;
        checkerboard_flags = CHECKERBOARD_FLAG_PRIMARY | CHECKERBOARD_FLAG_REFLECTION;

        if (!is_odd_checkerboard) {
            primary_roughness = 1.0;
            primary_metallic = 1.0;
            primary_base_color = float3(0.0);
            material_id = set_kind(material_id, MATERIAL_KIND_REGULAR);
        } else {
            primary_emissive = float3(0.0);
        }
    }

    if (triangle.alpha < 1.0) {
        // Translucent objects: split the path.
        throughput *= 2.0;
        checkerboard_flags = CHECKERBOARD_FLAG_PRIMARY | CHECKERBOARD_FLAG_REFRACTION;

        if (!is_odd_checkerboard) {
            throughput *= triangle.alpha;
        } else {
            material_id = set_kind(material_id, transparency_kind);
            throughput *= 1.0 - triangle.alpha;
        }
    }

    if (is_water(material_id) || is_slime(material_id)) {
        normal = get_water_normal(ctx, material_id, geo_normal, triangle.tangents[0], position, false);
        if (abs(geo_normal.z) < 0.1)  // vertical "water" is a force field
            material_id = set_kind(material_id, MATERIAL_KIND_GLASS);
    }

    if (is_camera(material_id)) {
        uint camera_id = (material_id & MATERIAL_LIGHT_STYLE_MASK) >> (MATERIAL_LIGHT_STYLE_SHIFT + 2);
        uint packed = (uint(cameraUV.x * float(0x3fff)) & 0x3fffu) | ((uint(cameraUV.y * float(0x3fff)) & 0x3fffu) << 14) | (camera_id << 28);
        IMG_STORE(PT_NORMAL_A, ipos, uint4(packed));
    } else {
        IMG_STORE(PT_NORMAL_A, ipos, uint4(encode_normal(normal)));
    }

    // Replace the material light style with the medium.
    material_id = (material_id & ~uint(MATERIAL_LIGHT_STYLE_MASK)) | ((uint(global_ubo.medium) << MATERIAL_LIGHT_STYLE_SHIFT) & MATERIAL_LIGHT_STYLE_MASK);

    if ((material_id & MATERIAL_FLAG_WEAPON) != 0u)
        checkerboard_flags |= CHECKERBOARD_FLAG_WEAPON;

    IMG_STORE(PT_GEO_NORMAL_A, ipos, uint4(encode_normal(geo_normal)));
    IMG_STORE(PT_SHADING_POSITION, ipos, float4(position, as_type<float>(material_id)));
    IMG_STORE(PT_VIEW_DIRECTION, ipos, float4(direction, float(checkerboard_flags)));
    IMG_STORE(PT_THROUGHPUT, ipos, float4(throughput, distance_curr));
    IMG_STORE(PT_BOUNCE_THROUGHPUT, ipos, float4(1.0, 1.0, 1.0, half_cone_angle));
    IMG_STORE(PT_CLUSTER_A, ipos, uint4(uint(triangle.cluster) & 0xffffu));
    IMG_STORE(PT_BASE_COLOR_A, ipos, float4(primary_base_color, primary_specular_factor));
    IMG_STORE(PT_METALLIC_A, ipos, float4(primary_metallic, primary_roughness, 0.0, 0.0));
    IMG_STORE(PT_GODRAYS_THROUGHPUT_DIST, ipos, float4(1.0, 1.0, 1.0, distance_curr));

    // Transparency starts from the primary surface emission, with zero alpha.
    float4 transparent = float4(primary_emissive * throughput, 0.0);

    if (global_ubo.pt_show_sky != 0.0 && is_sky(triangle.material_id)) {
        if (any(bary < float3(0.02)))
            transparent = alpha_blend(float4(1.0, 0.0, 0.0, 0.1), transparent);
        if ((triangle.material_id & MATERIAL_FLAG_LIGHT) != 0u)
            transparent = alpha_blend(float4(0.0, 0.0, 1.0, 0.1), transparent);
    }

    transparent = alpha_blend_premultiplied(effects, transparent);
    IMG_STORE(PT_TRANSPARENT, ipos, transparent);
}

//
// reflect_refract.rgen
//

static inline float3 reflect_point_vs_plane(float3 plane_pt, float3 plane_normal, float3 point)
{
    return point - 2.0 * plane_normal * dot(plane_normal, point - plane_pt);
}

static inline uint cluster_from_image(uint c) { return (c == 0xffffu) ? ~0u : c; }

kernel void pt_reflect_refract(PT_KERNEL_PARAMS)
{
    PT_CTX_INIT
    if (!launch_in_range(ctx))
        return;

    const int spec_bounce_index = ctx.bounce_index;
    int2 ipos = launch_ipos(ctx);

    float4 position_material = IMG_LOAD(PT_SHADING_POSITION, ipos);
    float3 position = position_material.xyz;
    uint material_id = as_type<uint>(position_material.w);

    bool primary_is_water = is_water(material_id);
    bool primary_is_slime = is_slime(material_id);
    bool primary_is_glass = is_glass(material_id);
    bool primary_is_chrome = is_chrome(material_id);
    bool primary_is_screen = is_screen(material_id);
    bool primary_is_camera = is_camera(material_id);
    bool primary_is_transparent = is_transparent(material_id);

    if (!(primary_is_water || primary_is_slime || primary_is_glass || primary_is_chrome ||
          primary_is_screen || primary_is_camera || primary_is_transparent))
        return;

    bool is_odd_checkerboard = launch_is_odd(ctx);

    ctx.rng_seed = IMG_LOAD(ASVGF_RNG_SEED_A, ipos).r;

    float4 view_direction = IMG_LOAD(PT_VIEW_DIRECTION, ipos);
    float3 direction = view_direction.xyz;
    int checkerboard_flags = int(view_direction.w);
    int checkerboard_weapon_flag = checkerboard_flags & CHECKERBOARD_FLAG_WEAPON;
    checkerboard_flags &= CHECKERBOARD_FLAG_FIELD_MASK;

    float4 throughput_distance = IMG_LOAD(PT_THROUGHPUT, ipos);
    float3 throughput = throughput_distance.rgb;
    float optical_path_length = throughput_distance.a;
    float4 transparent = IMG_LOAD(PT_TRANSPARENT, ipos);
    float3 primary_base_color = IMG_LOAD(PT_BASE_COLOR_A, ipos).rgb;
    float3 geo_normal = decode_normal(IMG_LOAD(PT_GEO_NORMAL_A, ipos).x);
    float3 normal = decode_normal(IMG_LOAD(PT_NORMAL_A, ipos).x);
    uint cluster_idx = cluster_from_image(IMG_LOAD(PT_CLUSTER_A, ipos).r);

    int primary_medium = int((material_id & MATERIAL_LIGHT_STYLE_MASK) >> MATERIAL_LIGHT_STYLE_SHIFT);
    bool primary_is_weapon = (material_id & MATERIAL_FLAG_WEAPON) != 0u;

    int correct_motion_vector = 0; // 1 -> flat reflection, 2 -> flat refraction
    bool include_player_model = !primary_is_weapon;

    if (primary_is_water || primary_is_slime) {
        float3 reflected_direction = reflect(direction, normal);
        float n_dot_v = abs(dot(direction, normal));
        const float index_of_refraction = 1.34;

        if (primary_medium != MEDIUM_NONE) {
            // Looking up from under water
            float3 refracted_direction = refract(direction, normal, index_of_refraction);
            n_dot_v = 1.0 - (1.0 - n_dot_v) * 3.0;

            if (n_dot_v <= 0.0 || dot(refracted_direction, refracted_direction) == 0.0) {
                direction = reflected_direction;
                if (spec_bounce_index == 0)
                    checkerboard_flags = CHECKERBOARD_FLAG_REFLECTION;
                correct_motion_vector = 1;
            } else {
                float F = pow(1.0 - n_dot_v, 5.0);
                float correction_factor;
                bool do_split, do_refraction;
                if (spec_bounce_index == 0) {
                    do_split = true;
                    do_refraction = is_odd_checkerboard;
                } else {
                    do_split = popcount_int(checkerboard_flags) == 1;
                    do_refraction = do_split ? is_odd_checkerboard : (F < 0.1);
                }

                if (do_refraction) {
                    direction = refracted_direction;
                    correction_factor = (1.0 - F);
                    primary_medium = MEDIUM_NONE;
                    correct_motion_vector = 2;
                } else {
                    direction = reflected_direction;
                    correction_factor = F;
                    correct_motion_vector = 1;
                }

                if (do_split) {
                    correction_factor *= 2.0;
                    checkerboard_flags = CHECKERBOARD_FLAG_REFLECTION | CHECKERBOARD_FLAG_REFRACTION;
                }
                throughput *= correction_factor;
            }
            include_player_model = false;
        } else {
            // Looking down on the water surface
            float3 refracted_direction = refract(direction, normal, 1.0 / index_of_refraction);
            float F = 0.1 + 0.9 * pow(1.0 - n_dot_v, 5.0);
            float correction_factor;
            bool do_split, do_refraction;
            if (spec_bounce_index == 0) {
                do_split = true;
                do_refraction = is_odd_checkerboard;
            } else {
                do_split = popcount_int(checkerboard_flags) == 1;
                do_refraction = do_split ? is_odd_checkerboard : (F < 0.1);
            }

            if (do_refraction) {
                primary_medium = primary_is_water ? MEDIUM_WATER : primary_is_slime ? MEDIUM_SLIME : MEDIUM_NONE;
                direction = refracted_direction;
                correction_factor = (1.0 - F);
                include_player_model = false;
                correct_motion_vector = 2;
            } else {
                direction = reflect(direction, normal);
                correction_factor = F;
                correct_motion_vector = 1;
            }

            if (do_split) {
                correction_factor *= 2.0;
                checkerboard_flags = CHECKERBOARD_FLAG_REFLECTION | CHECKERBOARD_FLAG_REFRACTION;
            }
            throughput *= correction_factor;
        }
    } else if (primary_is_screen) {
        float n_dot_v = abs(dot(direction, normal));
        float F = 0.05 + 0.95 * pow(1.0 - n_dot_v, 5.0);
        throughput *= F;
        direction = reflect(direction, normal);
        if (checkerboard_flags == CHECKERBOARD_FLAG_PRIMARY)
            checkerboard_flags = CHECKERBOARD_FLAG_REFLECTION;
        correct_motion_vector = 1;
    } else if (primary_is_camera) {
        normal = geo_normal;
        uint packedUV = IMG_LOAD(PT_NORMAL_A, ipos).x;

        float2 uv;
        uv.x = float(packedUV & 0x3fffu) / float(0x3fff);
        uv.y = float((packedUV >> 14) & 0x3fffu) / float(0x3fff);

        uint camera_id = (packedUV >> 28) & (MAX_CAMERAS - 1);

        float4x4 camera_data = global_ubo.security_camera_data[camera_id];
        position = camera_data[0].xyz;
        direction = camera_data[1].xyz + camera_data[2].xyz * uv.x + camera_data[3].xyz * uv.y;
        direction = normalize(direction);
    } else if (primary_is_chrome) {
        throughput *= primary_base_color;
        direction = reflect(direction, normal);
        if (spec_bounce_index == 0) {
            if (checkerboard_flags == CHECKERBOARD_FLAG_PRIMARY)
                checkerboard_flags = CHECKERBOARD_FLAG_REFLECTION;
            if ((material_id & MATERIAL_KIND_MASK) != MATERIAL_KIND_CHROME_MODEL)
                correct_motion_vector = 1;
        }
    } else if (primary_is_transparent) {
        correct_motion_vector = 2;
        if ((material_id & MATERIAL_KIND_MASK) == MATERIAL_KIND_TRANSPARENT) {
            float index_of_refraction = 1.005;
            float3 refracted1 = refract(direction, normal, 1.0 / index_of_refraction);
            float3 refracted2 = refract(refracted1, geo_normal, index_of_refraction);
            if (length(refracted2) > 0.0) {
                direction = refracted2;
                include_player_model = false;
            }
        }
    } else {
        // Glass
        float index_of_refraction = 1.52;

        float gn_dot_v = dot(direction, geo_normal);
        if (gn_dot_v > 0.0) {
            geo_normal = -geo_normal;
            normal = -normal;
            gn_dot_v = -gn_dot_v;
        }

        float n_dot_v = dot(direction, -normal);
        float3 reflected_direction = reflect(direction, normal);

        if (global_ubo.pt_thick_glass == 2.0 || (global_ubo.pt_thick_glass == 1.0 && global_ubo.flt_enable == 0.0)) {
            float rand = get_rng(ctx, RNG_BRDF_FRESNEL(ctx.bounce_index));

            if (primary_medium == MEDIUM_GLASS)
                n_dot_v = 1.0 - (1.0 - n_dot_v) * 4.0;
            else
                index_of_refraction = 1.0 / index_of_refraction;

            float3 refracted_direction = refract(direction, normal, index_of_refraction);
            float correction_factor = 1.0;

            if (n_dot_v <= 0.0 || dot(refracted_direction, refracted_direction) == 0.0) {
                direction = reflected_direction;
                correct_motion_vector = 1;
            } else {
                float F = 0.05 + 0.95 * pow(1.0 - n_dot_v, 5.0);
                bool do_refraction;
                if (popcount_int(checkerboard_flags) == 1) {
                    do_refraction = is_odd_checkerboard && F < 1.0;
                    correction_factor = do_refraction ? 1.0 - F : F;
                    correction_factor *= 2.0;
                    checkerboard_flags = CHECKERBOARD_FLAG_REFLECTION | CHECKERBOARD_FLAG_REFRACTION;
                } else if (global_ubo.flt_enable != 0.0) {
                    do_refraction = F < 0.75;
                    correction_factor = do_refraction ? 1.0 - F : F;
                } else {
                    float corrected_F = clamp(F, 0.25, 0.75);
                    do_refraction = corrected_F < rand;
                    correction_factor = do_refraction ? (1.0 - F) / (1.0 - corrected_F) : F / corrected_F;
                }

                if (do_refraction) {
                    direction = refracted_direction;
                    if (primary_medium == MEDIUM_NONE) {
                        throughput *= primary_base_color;
                        primary_medium = MEDIUM_GLASS;
                    } else {
                        primary_medium = MEDIUM_NONE;
                    }
                    correct_motion_vector = 2;
                } else {
                    direction = reflected_direction;
                    correct_motion_vector = 1;
                }
            }
            throughput *= correction_factor;
        } else {
            float F = 0.05 + 0.95 * pow(1.0 - abs(n_dot_v), 5.0);
            if (dot(reflected_direction, geo_normal) < 0.01)
                F = 0.0;

            float correction_factor = 1.0;
            bool do_split, do_refraction;
            if (spec_bounce_index == 0) {
                do_split = F > 0.0;
                do_refraction = is_odd_checkerboard;
            } else {
                do_split = popcount_int(checkerboard_flags) == 1 && F > 0.0;
                do_refraction = do_split ? is_odd_checkerboard : (F < 0.1);
            }

            if (do_refraction) {
                float3 refracted1 = refract(direction, normal, 1.0 / index_of_refraction);
                float3 refracted2 = refract(refracted1, geo_normal, index_of_refraction);
                if (length(refracted2) > 0.0) {
                    direction = refracted2;
                    include_player_model = false;
                }
                correction_factor = (1.0 - F);
                throughput *= primary_base_color;
                correct_motion_vector = 2;
            } else {
                direction = reflected_direction;
                correction_factor = F;
                correct_motion_vector = 1;
            }

            if (abs(dot(normal, geo_normal)) < 0.99999)
                correct_motion_vector = 0;

            if (do_split) {
                correction_factor *= 2.0;
                checkerboard_flags = CHECKERBOARD_FLAG_REFLECTION | CHECKERBOARD_FLAG_REFRACTION;
            }
            throughput *= correction_factor;
        }
    }

    int reflection_cull_mask = REFLECTION_RAY_CULL_MASK;
    if (global_ubo.first_person_model != 0 && include_player_model)
        reflection_cull_mask |= AS_FLAG_VIEWER_MODELS;
    else if (!(primary_is_weapon && primary_is_transparent))
        reflection_cull_mask |= AS_FLAG_VIEWER_WEAPON;

    if (ctx.bounce_index < int(global_ubo.pt_reflect_refract - 1.0))
        reflection_cull_mask |= AS_FLAG_TRANSPARENT;

    Ray reflection_ray;
    reflection_ray.origin = position;
    reflection_ray.direction = direction;
    reflection_ray.t_min = 0.0;
    reflection_ray.t_max = PRIMARY_RAY_T_MAX;

    if (dot(direction, normal) >= 0.0)
        reflection_ray.origin -= view_direction.xyz * 0.01;
    else
        reflection_ray.origin -= normal.xyz * 0.001;

    HitInfo hit = trace_geometry_ray(ctx, reflection_ray, reflection_cull_mask);

    Triangle triangle;
    if (hit.found) {
        reflection_ray.t_max = hit.hit_distance;
        triangle = load_hit_triangle(ctx, hit.instance, hit.primitive);
    }

    float4 effects = trace_effects_ray(ctx, reflection_ray, false);
    transparent = alpha_blend_premultiplied(transparent, effects * float4(throughput, 1.0));

    if (!hit.found || is_sky(triangle.material_id)) {
        float3 env = env_map(ctx, direction, false);
        env *= global_ubo.pt_env_scale;

        transparent = alpha_blend(transparent, float4(env * throughput, 1.0));

        material_id = (uint(primary_medium) << MATERIAL_LIGHT_STYLE_SHIFT) & MATERIAL_LIGHT_STYLE_MASK;

        IMG_STORE(PT_NORMAL_A, ipos, uint4(0u));
        IMG_STORE(PT_GEO_NORMAL_A, ipos, uint4(0u));
        IMG_STORE(PT_BASE_COLOR_A, ipos, float4(0.0));
        IMG_STORE(PT_METALLIC_A, ipos, float4(0.0));
        IMG_STORE(PT_TRANSPARENT, ipos, transparent);
        IMG_STORE(PT_SHADING_POSITION, ipos, float4(position + direction * PRIMARY_RAY_T_MAX, as_type<float>(material_id)));
        IMG_STORE(PT_VIEW_DIRECTION, ipos, float4(direction, float(checkerboard_flags | checkerboard_weapon_flag)));
        IMG_STORE(PT_VIEW_DEPTH_A, ipos, float4(-PRIMARY_RAY_T_MAX));
        IMG_STORE(PT_GODRAYS_THROUGHPUT_DIST, ipos, float4(throughput, PRIMARY_RAY_T_MAX));
        return;
    }

    IMG_STORE(PT_GODRAYS_THROUGHPUT_DIST, ipos, float4(throughput, hit.hit_distance));

    if (primary_medium != MEDIUM_NONE)
        throughput *= extinction(ctx, primary_medium, hit.hit_distance);

    float3 bary = get_hit_barycentric(hit);
    float2 tex_coord = triangle.tex_coords * bary;
    float3 new_position = triangle.positions * bary;
    float3 new_pos_prev = triangle.positions_prev * bary;
    float3 new_geo_normal = normalize(triangle.normals * bary);
    material_id = triangle.material_id;
    cluster_idx = uint(triangle.cluster);

    float3 new_flat_normal = normalize(cross(triangle.positions[1] - triangle.positions[0],
                                             triangle.positions[2] - triangle.positions[1]));
    if (dot(new_flat_normal, new_geo_normal) < 0.0)
        new_flat_normal = -new_flat_normal;
    if (dot(new_flat_normal, direction) > 0.0)
        new_geo_normal = -new_geo_normal;

    optical_path_length += hit.hit_distance;

    float half_cone_angle = IMG_LOAD(PT_BOUNCE_THROUGHPUT, ipos).a;

    float2 tex_coord_x, tex_coord_y;
    float fwidth_depth;
    compute_anisotropic_texture_gradients(new_position, new_flat_normal, direction,
                                          optical_path_length * half_cone_angle, triangle.positions,
                                          triangle.tex_coords, tex_coord, tex_coord_x, tex_coord_y, fwidth_depth);

    if (global_ubo.pt_texture_lod_bias != 0.0) {
        tex_coord_x *= pow(2.0, global_ubo.pt_texture_lod_bias);
        tex_coord_y *= pow(2.0, global_ubo.pt_texture_lod_bias);
    }

    if ((correct_motion_vector == 1 && spec_bounce_index == 0) || correct_motion_vector == 2) {
        float3 ref_pos_curr, ref_pos_prev;
        if (correct_motion_vector == 1) {
            ref_pos_curr = reflect_point_vs_plane(position, geo_normal, new_position);
            ref_pos_prev = reflect_point_vs_plane(position, geo_normal, new_pos_prev);
        } else {
            ref_pos_curr = new_position;
            ref_pos_prev = new_pos_prev;
        }

        float2 screen_pos_curr, screen_pos_prev;
        float distance_curr, distance_prev;
        projection_view_to_screen(ctx, (global_ubo.V * float4(ref_pos_curr, 1.0)).xyz, screen_pos_curr, distance_curr, false);
        projection_view_to_screen(ctx, (global_ubo.V_prev * float4(ref_pos_prev, 1.0)).xyz, screen_pos_prev, distance_prev, true);

        float3 motion;
        motion.xy = screen_pos_prev - screen_pos_curr;
        motion.z = distance_prev - distance_curr;

        // Negative depth for reflections and refractions keeps the filters
        // from filtering across reflection boundaries.
        motion.z = -motion.z;
        distance_curr = -distance_curr;

        IMG_STORE(PT_VIEW_DEPTH_A, ipos, float4(distance_curr));
        IMG_STORE(PT_MOTION, ipos, float4(motion, fwidth_depth));
    } else if (spec_bounce_index == 0) {
        IMG_STORE(PT_VIEW_DEPTH_A, ipos, -IMG_LOAD(PT_VIEW_DEPTH_A, ipos));
        float4 motion = IMG_LOAD(PT_MOTION, ipos);
        motion.z = -motion.z;
        IMG_STORE(PT_MOTION, ipos, motion);
    }

    position = new_position;
    geo_normal = new_geo_normal;
    if (dot(direction, geo_normal) > 0.0)
        geo_normal = -geo_normal;

    float primary_metallic = 0.0;
    float primary_roughness = 1.0;
    float3 primary_emissive = float3(0.0);
    float primary_specular_factor = 1.0;

    get_material(ctx, triangle, bary, tex_coord, tex_coord_x, tex_coord_y, -1.0, geo_normal,
                 primary_base_color, normal, primary_metallic, primary_roughness, primary_emissive,
                 primary_specular_factor);

    uint transparency_kind = (material_id & MATERIAL_KIND_MASK) == MATERIAL_KIND_TRANSPARENT ? MATERIAL_KIND_TRANSPARENT : MATERIAL_KIND_TRANSP_MODEL;

    if (((is_chrome(material_id) || is_screen(material_id) || is_camera(material_id)) && primary_roughness >= MAX_MIRROR_ROUGHNESS) ||
        is_transparent(material_id))
        material_id = set_kind(material_id, MATERIAL_KIND_REGULAR);

    if (is_water(material_id) || is_slime(material_id)) {
        normal = get_water_normal(ctx, material_id, geo_normal, triangle.tangents[0], position, false);
        if (abs(geo_normal.z) < 0.1)
            material_id = set_kind(material_id, MATERIAL_KIND_GLASS);
    }

    if (is_camera(material_id) && (global_ubo.pt_cameras == 0.0 || float(ctx.bounce_index) >= global_ubo.pt_reflect_refract - 1.0))
        material_id = set_kind(material_id, MATERIAL_KIND_SCREEN);

    float2 cameraUV = float2(0.0);
    if (is_camera(material_id)) {
        if (get_camera_uv(tex_coord, cameraUV)) {
            bool do_split = popcount_int(checkerboard_flags) == 1;
            if (do_split) {
                throughput *= 2.0;
                checkerboard_flags = CHECKERBOARD_FLAG_REFRACTION | CHECKERBOARD_FLAG_REFLECTION;
                if (is_odd_checkerboard)
                    material_id = set_kind(material_id, MATERIAL_KIND_SCREEN);
            }
            primary_emissive = float3(0.0);
        } else {
            material_id = set_kind(material_id, MATERIAL_KIND_REGULAR);
        }
    }

    if (triangle.alpha < 1.0) {
        if (popcount_int(checkerboard_flags & CHECKERBOARD_FLAG_FIELD_MASK) == 1) {
            throughput *= 2.0;
            checkerboard_flags = CHECKERBOARD_FLAG_PRIMARY | CHECKERBOARD_FLAG_REFRACTION;
            if (!is_odd_checkerboard) {
                throughput *= triangle.alpha;
            } else {
                material_id = set_kind(material_id, transparency_kind);
                throughput *= 1.0 - triangle.alpha;
            }
        } else {
            // Split on the same model before: keep going through it rather
            // than showing its internal geometry.
            uint primary_instance_index = IMG_LOAD(PT_VISBUF_PRIM_A, ipos).x;
            if (primary_instance_index == triangle.instance_index)
                material_id = set_kind(material_id, transparency_kind);
        }
    }

    if (luminance(primary_emissive) > 0.0)
        transparent = alpha_blend_premultiplied(transparent, float4(primary_emissive * throughput, 0.0));

    IMG_STORE(PT_VISBUF_PRIM_A, ipos, uint4(triangle.instance_index, triangle.instance_prim, 0u, 0u));
    IMG_STORE(PT_VISBUF_BARY_A, ipos, float4(bary.yz, 0.0, 0.0));

    if (is_camera(material_id)) {
        uint camera_id = (material_id & MATERIAL_LIGHT_STYLE_MASK) >> (MATERIAL_LIGHT_STYLE_SHIFT + 2);
        uint packed = (uint(cameraUV.x * float(0x3fff)) & 0x3fffu) | ((uint(cameraUV.y * float(0x3fff)) & 0x3fffu) << 14) | (camera_id << 28);
        IMG_STORE(PT_NORMAL_A, ipos, uint4(packed));
    } else {
        IMG_STORE(PT_NORMAL_A, ipos, uint4(encode_normal(normal)));
    }

    material_id = (material_id & ~uint(MATERIAL_LIGHT_STYLE_MASK)) | ((uint(primary_medium) << MATERIAL_LIGHT_STYLE_SHIFT) & MATERIAL_LIGHT_STYLE_MASK);

    IMG_STORE(PT_GEO_NORMAL_A, ipos, uint4(encode_normal(geo_normal)));
    IMG_STORE(PT_SHADING_POSITION, ipos, float4(position, as_type<float>(material_id)));
    IMG_STORE(PT_VIEW_DIRECTION, ipos, float4(direction, float(checkerboard_flags | checkerboard_weapon_flag)));
    IMG_STORE(PT_THROUGHPUT, ipos, float4(throughput, optical_path_length));
    IMG_STORE(PT_TRANSPARENT, ipos, transparent);
    IMG_STORE(PT_CLUSTER_A, ipos, uint4(cluster_idx & 0xffffu));
    IMG_STORE(PT_BASE_COLOR_A, ipos, float4(primary_base_color, primary_specular_factor));
    IMG_STORE(PT_METALLIC_A, ipos, float4(primary_metallic, primary_roughness, 0.0, 0.0));
}

//
// direct_lighting.rgen
//

static void direct_lighting(thread PtCtx &ctx, int2 ipos, bool enable_caustics, thread float3 &high_freq, thread float3 &o_specular)
{
    high_freq = float3(0.0);
    o_specular = float3(0.0);

    ctx.rng_seed = IMG_LOAD(ASVGF_RNG_SEED_A, ipos).r;

    float4 position_material = IMG_LOAD(PT_SHADING_POSITION, ipos);
    float3 position = position_material.xyz;
    uint material_id = as_type<uint>(position_material.w);

    if (material_id == 0u)
        return;

    float4 view_direction = IMG_LOAD(PT_VIEW_DIRECTION, ipos);
    float3 normal = decode_normal(IMG_LOAD(PT_NORMAL_A, ipos).x);
    float3 geo_normal = decode_normal(IMG_LOAD(PT_GEO_NORMAL_A, ipos).x);
    float4 primary_base_color = IMG_LOAD(PT_BASE_COLOR_A, ipos);
    float primary_specular_factor = primary_base_color.a;
    float2 metal_rough = IMG_LOAD(PT_METALLIC_A, ipos).xy;
    float primary_metallic = metal_rough.x;
    float primary_roughness = metal_rough.y;
    uint cluster_idx = cluster_from_image(IMG_LOAD(PT_CLUSTER_A, ipos).x);

    bool primary_is_weapon = (material_id & MATERIAL_FLAG_WEAPON) != 0u;
    int primary_medium = int((material_id & MATERIAL_LIGHT_STYLE_MASK) >> MATERIAL_LIGHT_STYLE_SHIFT);

    int shadow_cull_mask = SHADOW_RAY_CULL_MASK;
    if (global_ubo.first_person_model != 0 && !primary_is_weapon)
        shadow_cull_mask |= AS_FLAG_VIEWER_MODELS;
    else
        shadow_cull_mask |= AS_FLAG_VIEWER_WEAPON;

    float direct_specular_weight = smoothstep(global_ubo.pt_direct_roughness_threshold - 0.02,
                                              global_ubo.pt_direct_roughness_threshold + 0.02, primary_roughness);

    bool is_gradient = get_is_gradient(ctx, ipos);

    float3 primary_albedo, primary_base_reflectivity;
    get_reflectivity(primary_base_color.rgb, primary_metallic, primary_albedo, primary_base_reflectivity);

    float3 direct_diffuse, direct_specular;
    get_direct_illumination(ctx, position, normal, geo_normal, cluster_idx, material_id, shadow_cull_mask,
                            view_direction.xyz, primary_albedo, primary_base_reflectivity,
                            primary_specular_factor, primary_roughness, primary_medium, enable_caustics,
                            direct_specular_weight,
                            global_ubo.pt_direct_polygon_lights > 0.0, global_ubo.pt_direct_dyn_lights > 0.0,
                            is_gradient, 0, direct_diffuse, direct_specular);

    high_freq += direct_diffuse;
    o_specular += direct_specular;

    if (global_ubo.pt_direct_sun_light != 0.0) {
        float3 direct_sun_diffuse, direct_sun_specular;
        get_sunlight(ctx, cluster_idx, material_id, position, normal, geo_normal, view_direction.xyz,
                     primary_base_reflectivity, primary_specular_factor, primary_roughness,
                     primary_medium, enable_caustics, false, direct_sun_diffuse, direct_sun_specular, shadow_cull_mask);
        high_freq += direct_sun_diffuse;
        o_specular += direct_sun_specular;
    }

    o_specular = demodulate_specular(ctx, primary_base_reflectivity, o_specular);

    high_freq = clamp_output(high_freq);
    o_specular = clamp_output(o_specular);
}

kernel void pt_direct_lighting(PT_KERNEL_PARAMS)
{
    PT_CTX_INIT
    if (!launch_in_range(ctx))
        return;

    int2 ipos = launch_ipos(ctx);

    float3 high_freq, specular;
    direct_lighting(ctx, ipos, push.iteration != 0u, high_freq, specular);

    high_freq *= STORAGE_SCALE_HF;
    specular *= STORAGE_SCALE_SPEC;

    IMG_STORE(PT_COLOR_LF_SH, ipos, float4(0.0));
    IMG_STORE(PT_COLOR_LF_COCG, ipos, float4(0.0));
    IMG_STORE(PT_COLOR_HF, ipos, uint4(packRGBE(high_freq)));
    IMG_STORE(PT_COLOR_SPEC, ipos, uint4(packRGBE(specular)));
}

//
// indirect_lighting.rgen
//

static void indirect_lighting(thread PtCtx &ctx, int2 ipos, bool half_res, int spec_bounce_index,
                              thread float3 &bounce_direction, thread float3 &bounce_contrib, thread bool &is_specular_ray)
{
    ctx.rng_seed = IMG_LOAD(ASVGF_RNG_SEED_A, ipos).r;

    bounce_direction = float3(0.0);
    bounce_contrib = float3(0.0);
    is_specular_ray = false;

    float4 position_material = IMG_LOAD(PT_SHADING_POSITION, ipos);
    float3 position = position_material.xyz;
    uint material_id = as_type<uint>(position_material.w);

    // No lighting for invalid surfaces, nor for lava which lights itself.
    if (material_id == 0u || is_lava(material_id))
        return;

    float4 throughput = IMG_LOAD(PT_BOUNCE_THROUGHPUT, ipos);
    float4 view_direction;
    float3 normal;
    float3 geo_normal;
    float4 primary_base_color = float4(0.0);
    float primary_metallic = 0.0;
    float primary_roughness = 1.0;
    float primary_specular_factor = 0.0;

    if (half_res)
        throughput.rgb *= 2.0;

    if (spec_bounce_index == 0) {
        view_direction = IMG_LOAD(PT_VIEW_DIRECTION, ipos);
        normal = decode_normal(IMG_LOAD(PT_NORMAL_A, ipos).x);
        geo_normal = decode_normal(IMG_LOAD(PT_GEO_NORMAL_A, ipos).x);
        primary_base_color = IMG_LOAD(PT_BASE_COLOR_A, ipos);
        primary_specular_factor = primary_base_color.a;
        if (!half_res) {
            float2 metal_rough = IMG_LOAD(PT_METALLIC_A, ipos).xy;
            primary_metallic = metal_rough.x;
            primary_roughness = metal_rough.y;
        }
    } else {
        view_direction = IMG_LOAD(PT_VIEW_DIRECTION2, ipos);
        geo_normal = decode_normal(IMG_LOAD(PT_GEO_NORMAL2, ipos).x);
        normal = geo_normal;
    }

    float3 primary_albedo, primary_base_reflectivity;
    get_reflectivity(primary_base_color.rgb, primary_metallic, primary_albedo, primary_base_reflectivity);

    bool primary_is_weapon = (material_id & MATERIAL_FLAG_WEAPON) != 0u;

    float direct_specular_weight = smoothstep(global_ubo.pt_direct_roughness_threshold - 0.02,
                                              global_ubo.pt_direct_roughness_threshold + 0.02, primary_roughness);
    float fake_specular_weight = smoothstep(global_ubo.pt_fake_roughness_threshold,
                                            global_ubo.pt_fake_roughness_threshold + 0.1, primary_roughness);

    float NoV = max(0.0, -dot(normal, view_direction.xyz));

    bool first_bounce_is_specular = throughput.a != 0.0;

    float3 bounce_throughput = throughput.rgb;

    {
        float2 rng3 = float2(get_rng(ctx, RNG_BRDF_X(spec_bounce_index + 1)), get_rng(ctx, RNG_BRDF_Y(spec_bounce_index + 1)));
        float rng_frensel = get_rng(ctx, RNG_BRDF_FRESNEL(spec_bounce_index + 1));

        float specular_pdf = 0.0;

        if (spec_bounce_index == 0) {
            specular_pdf = (primary_metallic == 1.0 && fake_specular_weight == 0.0) ? 1.0 : 0.5;

            if (rng_frensel < specular_pdf) {
                float3x3 basis = construct_ONB_frisvad(normal);

                float3 N = normal;
                float3 V = view_direction.xyz;
                float3 H = ImportanceSampleGGX_VNDF(ctx, rng3, primary_roughness, V, basis);
                float3 L = reflect(V, H);

                float NoL = max(0.0, dot(N, L));
                float VoH = max(0.0, -dot(V, H));

                if (NoL > 0.0 && NoV > 0.0) {
                    float G1_NoL = G1_Smith(primary_roughness, NoL);
                    float3 F = schlick_fresnel(primary_base_reflectivity, VoH, primary_specular_factor);

                    bounce_throughput *= G1_NoL * F;
                    bounce_throughput *= 1.0 / specular_pdf;
                    is_specular_ray = true;
                    bounce_direction = normalize(L);
                }
            }
        }

        if (!is_specular_ray) {
            float3 basis_normal, dir_sphere;
            if (spec_bounce_index == 0 && global_ubo.flt_enable != 0.0) {
                dir_sphere = sample_cos_hemisphere_multi(0.0, 1.0, rng3, HEMISPHERE_UNIFORMISH);
                basis_normal = geo_normal;
            } else {
                dir_sphere = sample_cos_hemisphere(rng3);
                basis_normal = normal;
            }

            float3x3 basis = construct_ONB_frisvad(basis_normal);
            bounce_direction = normalize(basis * dir_sphere);
            bounce_throughput *= 1.0 / (1.0 - specular_pdf);

            float3 L = bounce_direction;
            float3 V = -view_direction.xyz;
            float3 H = normalize(V + L);
            float VoH = max(0.0, dot(V, H));

            float3 F = schlick_fresnel(primary_base_reflectivity, VoH, primary_specular_factor);
            bounce_throughput *= float3(1.0) - F;
        }
    }

    Ray bounce_ray;
    bounce_ray.origin = position;
    bounce_ray.direction = bounce_direction;
    bounce_ray.t_min = 0.0;
    bounce_ray.t_max = 10000.0;
    bounce_ray.origin -= view_direction.xyz * 0.01;

    int bounce_cull_mask = BOUNCE_RAY_CULL_MASK;
    int shadow_cull_mask = SHADOW_RAY_CULL_MASK;
    if (global_ubo.first_person_model != 0 && !primary_is_weapon) {
        bounce_cull_mask |= AS_FLAG_VIEWER_MODELS;
        shadow_cull_mask |= AS_FLAG_VIEWER_MODELS;
    } else {
        bounce_cull_mask |= AS_FLAG_VIEWER_WEAPON;
        shadow_cull_mask |= AS_FLAG_VIEWER_WEAPON;
    }

    HitInfo hit = trace_geometry_ray(ctx, bounce_ray, bounce_cull_mask);

    bounce_contrib = float3(0.0);

    if (is_specular_ray) {
        if (hit.found)
            bounce_ray.t_max = hit.hit_distance;
        float4 transparency = trace_effects_ray(ctx, bounce_ray, true);
        bounce_contrib += transparency.rgb * transparency.a * bounce_throughput * (1.0 - direct_specular_weight);
    }

    if (!hit.found) {
        if (float(spec_bounce_index) < global_ubo.pt_num_bounce_rays - 1.0)
            IMG_STORE(PT_SHADING_POSITION, ipos, float4(0.0));
        return;
    }

    Triangle triangle = load_hit_triangle(ctx, hit.instance, hit.primitive);

    float indirect_specular_weight = 1.0;
    if (is_specular_ray) {
        float cone_size = hit.hit_distance * primary_roughness;
        indirect_specular_weight = 1.0 - fake_specular_weight;
        indirect_specular_weight /= max(1.0, cone_size * global_ubo.pt_specular_anti_flicker * 0.01);
    }

    if (is_sky(triangle.material_id)) {
        bool is_analytic_light = (triangle.material_id & MATERIAL_FLAG_LIGHT) != 0u &&
            ((spec_bounce_index == 0 && global_ubo.pt_direct_polygon_lights >= 0.0) ||
             (spec_bounce_index == 1 && global_ubo.pt_indirect_polygon_lights >= 0.0));

        if (!is_analytic_light || is_specular_ray) {
            float3 env = env_map(ctx, bounce_direction, true);
            env *= global_ubo.pt_env_scale;

            if (!is_analytic_light) {
                bounce_contrib = bounce_throughput * env * indirect_specular_weight;
            } else if (is_specular_ray) {
                if (global_ubo.pt_specular_mis != 0.0) {
                    float3x3 projected_positions = project_triangle(triangle.positions, position);
                    float pdfw = get_spherical_triangle_pdfw(projected_positions);
                    direct_specular_weight *= get_specular_sampled_lighting_weight(primary_roughness,
                        normal, -view_direction.xyz, bounce_direction, pdfw);
                }
                bounce_contrib = bounce_throughput * env * indirect_specular_weight * (1.0 - direct_specular_weight);
            }
        }

        if (float(spec_bounce_index) < global_ubo.pt_num_bounce_rays - 1.0)
            IMG_STORE(PT_SHADING_POSITION, ipos, float4(0.0));
    } else {
        float3 bary = get_hit_barycentric(hit);
        float2 tex_coord = triangle.tex_coords * bary;
        uint bounce_material_id = triangle.material_id;

        if ((bounce_material_id & MATERIAL_FLAG_WARP) != 0u)
            tex_coord = lava_uv_warp(tex_coord, global_ubo.time);

        MaterialInfo bounce_minfo = get_material_info(ctx, triangle.material_index);

        float3 bounce_position = triangle.positions * bary;
        float3 bounce_geo_normal = normalize(triangle.normals * bary);

        float3 bounce_flat_normal = normalize(cross(triangle.positions[1] - triangle.positions[0],
                                                    triangle.positions[2] - triangle.positions[1]));
        if (dot(bounce_flat_normal, bounce_geo_normal) < 0.0)
            bounce_flat_normal = -bounce_flat_normal;
        if (dot(bounce_flat_normal, bounce_direction) > 0.0)
            bounce_geo_normal = -bounce_geo_normal;

        float3 bounce_normal = bounce_geo_normal;
        float3 bounce_base_color = bounce_minfo.base_factor;
        if (bounce_minfo.base_texture != 0u)
            bounce_base_color *= material_textureLod(ctx, bounce_minfo.base_texture, tex_coord, 2.0).rgb;
        bounce_base_color = clamp(bounce_base_color, float3(0.0), float3(1.0));
        uint bounce_cluster_idx = uint(triangle.cluster);

        float3 emissive = sample_emissive_texture(ctx, bounce_minfo, tex_coord, float2(0.0), float2(0.0), is_specular_ray ? 2.0 : 3.0);
        emissive *= triangle.emissive_factor;
        if ((bounce_material_id & MATERIAL_KIND_MASK) == MATERIAL_KIND_LAVA)
            emissive += bounce_base_color * global_ubo.lava_emissive;

        emissive += get_emissive_shell(ctx, triangle.material_id, triangle.shell) * bounce_base_color;

        if (luminance(emissive) > 0.0) {
            emissive *= bounce_throughput;

            float spotlight = sqrt(max(0.0, -dot(bounce_direction, bounce_normal)));
            emissive *= spotlight;

            bool is_analytic_light = (bounce_material_id & MATERIAL_FLAG_LIGHT) != 0u &&
                ((spec_bounce_index == 0 && global_ubo.pt_direct_polygon_lights >= 0.0) ||
                 (spec_bounce_index == 1 && global_ubo.pt_indirect_polygon_lights >= 0.0));

            if (is_specular_ray) {
                if (is_analytic_light)
                    bounce_contrib += emissive * (1.0 - direct_specular_weight);
                else
                    bounce_contrib += emissive * indirect_specular_weight;
            } else {
                if (!is_analytic_light)
                    bounce_contrib += emissive;
            }
        }

        if (dot(bounce_direction, bounce_normal) > 0.0)
            bounce_normal = -bounce_normal;

        bounce_throughput *= bounce_base_color;
        bounce_throughput *= indirect_specular_weight;

        float sun_bounce_range = global_ubo.pt_sun_bounce_range;
        if (is_specular_ray)
            sun_bounce_range *= (1.0 - sqrt(primary_roughness));

        float sun_attenuation = square(clamp(1.0 - square(square(hit.hit_distance / sun_bounce_range)), 0.0, 1.0));

        if (spec_bounce_index == 0) {
            bool is_gradient = get_is_gradient(ctx, ipos);

            float3 bounce_diffuse, bounce_specular;
            get_direct_illumination(ctx, bounce_position, bounce_geo_normal, bounce_geo_normal, bounce_cluster_idx,
                                    bounce_material_id, shadow_cull_mask, bounce_direction, bounce_base_color,
                                    float3(0.0), 0.0, 1.0, MEDIUM_NONE, false, 0.0,
                                    global_ubo.pt_indirect_polygon_lights > 0.0, global_ubo.pt_indirect_dyn_lights > 0.0,
                                    is_gradient, 1, bounce_diffuse, bounce_specular);
            bounce_contrib += bounce_throughput * bounce_diffuse;
        }

        if (sun_attenuation > 0.0) {
            float3 bounce_sun_diffuse, bounce_sun_specular;
            get_sunlight(ctx, bounce_cluster_idx, bounce_material_id, bounce_position, bounce_normal, bounce_geo_normal,
                         bounce_direction, float3(0.0), 0.0, 1.0, MEDIUM_NONE, false, false,
                         bounce_sun_diffuse, bounce_sun_specular, shadow_cull_mask);
            bounce_contrib += bounce_throughput * bounce_sun_diffuse * global_ubo.sun_bounce_scale * sun_attenuation;
        }

        if (float(spec_bounce_index) < global_ubo.pt_num_bounce_rays - 1.0) {
            IMG_STORE(PT_GEO_NORMAL2, ipos, uint4(encode_normal(bounce_geo_normal)));
            IMG_STORE(PT_SHADING_POSITION, ipos, float4(bounce_position, as_type<float>(triangle.material_id)));
            IMG_STORE(PT_VIEW_DIRECTION2, ipos, float4(bounce_direction, 0.0));
            IMG_STORE(PT_BOUNCE_THROUGHPUT, ipos, float4(bounce_throughput, is_specular_ray ? 1.0 : 0.0));
        }
    }

    if (is_specular_ray)
        bounce_contrib = demodulate_specular(ctx, primary_base_reflectivity, bounce_contrib);

    if (spec_bounce_index > 0 && first_bounce_is_specular)
        is_specular_ray = true;

    if (spec_bounce_index > 0)
        bounce_direction = view_direction.xyz;
}

kernel void pt_indirect_lighting(PT_KERNEL_PARAMS)
{
    PT_CTX_INIT

    int2 ipos = launch_ipos(ctx);

    // Half resolution tracing for the "low" GI setting: the launch grid is
    // half the height and alternates rows between frames.
    bool half_res = global_ubo.pt_num_bounce_rays == 0.5;
    if (half_res)
        ipos.y = ipos.y * 2 + (global_ubo.current_frame_idx & 1);

    if (int(launch_id.x) >= global_ubo.width / 2 || ipos.y >= global_ubo.height)
        return;

    float3 bounce_direction;
    float3 bounce_contrib;
    bool is_specular_ray;

    indirect_lighting(ctx, ipos, half_res, ctx.bounce_index, bounce_direction, bounce_contrib, is_specular_ray);

    if (any(isinf(bounce_contrib)) || any(isnan(bounce_contrib)) || any(isinf(bounce_direction)) || any(isnan(bounce_direction)))
        return;

    bounce_contrib = clamp_output(bounce_contrib);

    if (all(bounce_contrib == float3(0.0)))
        return;

    if (is_specular_ray) {
        bounce_contrib *= STORAGE_SCALE_SPEC;
        float3 specular = unpackRGBE(IMG_LOAD(PT_COLOR_SPEC, ipos).x);
        specular += bounce_contrib;
        IMG_STORE(PT_COLOR_SPEC, ipos, uint4(packRGBE(specular)));
    } else {
        bounce_contrib *= STORAGE_SCALE_LF;
        SH low_freq = LOAD_SH(PT_COLOR_LF_SH, PT_COLOR_LF_COCG, ipos);
        if (global_ubo.flt_enable == 0.0)
            low_freq.shY.xyz += bounce_contrib;
        else
            accumulate_SH(low_freq, irradiance_to_SH(bounce_contrib, bounce_direction), 1.0);
        STORE_SH(PT_COLOR_LF_SH, PT_COLOR_LF_COCG, ipos, low_freq);
    }
}

//
// asvgf_gradient_reproject.comp: picks one gradient sample per 3x3 stratum,
// the brightest pixel whose surface also existed last frame, and patches the
// G-buffer so the lighting passes re-shade that surface with last frame's
// random sequence. Needs the scene adapter, hence it lives here.
//

#define GROUP_SIZE_GRAD 8
#define GROUP_SIZE_PIXELS (GROUP_SIZE_GRAD * GRAD_DWN)

// Maps last frame's visibility buffer entry to this frame's triangle
// (vkpt's model_prev_to_current for dynamic geometry).
static bool map_prev_visbuf(thread const PtCtx &ctx, uint2 vis_buf, thread bool &is_entity, thread uint &prim)
{
    if (vis_buf.x == ~0u) {
        is_entity = false;
        prim = vis_buf.y;
        return true;
    }

    uint id = vis_buf.x;
    device const VkptEntitySlot &prev = ctx.entity_table_prev[id % VKPT_ENTITY_TABLE_SIZE];
    device const VkptEntitySlot &curr = ctx.entity_table[id % VKPT_ENTITY_TABLE_SIZE];
    if (prev.id != id || curr.id != id || vis_buf.y < prev.first_prim)
        return false;
    uint local = vis_buf.y - prev.first_prim;
    if (local >= prev.prim_count || local >= curr.prim_count)
        return false;

    is_entity = true;
    prim = curr.first_prim + local;
    return true;
}

static void patch_position(thread const PtCtx &ctx, int2 ipos, int2 found_pos_prev)
{
    uint2 vis_buf = IMG_LOAD(PT_VISBUF_PRIM_B, found_pos_prev).xy;

    bool is_entity;
    uint prim;
    if (!map_prev_visbuf(ctx, vis_buf, is_entity, prim))
        return;

    Triangle triangle = load_triangle(ctx, is_entity, false, prim);

    float3 bary;
    bary.yz = IMG_LOAD(PT_VISBUF_BARY_B, found_pos_prev).xy;
    bary.x = clamp(1.0 - bary.y - bary.z, 0.0, 1.0);

    float3 position = triangle.positions * bary;

    float materialId = IMG_LOAD(PT_SHADING_POSITION, ipos).w;
    IMG_STORE(PT_SHADING_POSITION, ipos, float4(position, materialId));

    // For primary surfaces, reconstruct the exact view direction.
    uint checkerboard_flags = uint(int(IMG_LOAD(PT_VIEW_DIRECTION, ipos).w));
    if ((checkerboard_flags & CHECKERBOARD_FLAG_FIELD_MASK) == CHECKERBOARD_FLAG_PRIMARY) {
        float3 view_direction = normalize(position - global_ubo.cam_pos.xyz);
        IMG_STORE(PT_VIEW_DIRECTION, ipos, float4(view_direction, float(checkerboard_flags)));
    }
}

kernel void asvgf_gradient_reproject(
    PT_KERNEL_PARAMS,
    uint3 group_id                                    [[threadgroup_position_in_grid]],
    uint3 local_id                                    [[thread_position_in_threadgroup]],
    uint local_index                                  [[thread_index_in_threadgroup]])
{
    PT_CTX_INIT
    threadgroup float4 s_reprojected_pixels[GROUP_SIZE_PIXELS][GROUP_SIZE_PIXELS];

    // First pass: every thread matches its pixel with the previous frame.
    int2 ipos = int2(launch_id.xy);
    {
        int field_left = 0;
        int field_right = global_ubo.prev_width / 2;
        if (ipos.x >= global_ubo.width / 2) {
            field_left = field_right;
            field_right = global_ubo.prev_width;
        }

        int2 lp = int2(local_id.xy);
        s_reprojected_pixels[lp.y][lp.x] = float4(0.0);

        int2 p = ipos;
        float4 motion = IMG_LOAD(PT_MOTION, p);
        float2 pos_prev = ((float2(p) + float2(0.5)) * float2(global_ubo.inv_width * 2.0, global_ubo.inv_height) + motion.xy) *
                          float2(float(global_ubo.prev_width / 2), float(global_ubo.prev_height));
        int2 pp = int2(floor(pos_prev));

        bool valid = !(pp.x < field_left || pp.x >= field_right || pp.y < 0 || pp.y >= global_ubo.prev_height) &&
                     p.x < global_ubo.width && p.y < global_ubo.height;

        if (valid) {
            int2 pos_grad_prev = pp / GRAD_DWN;
            uint prev_grad_sample_pos = IMG_LOAD(ASVGF_GRAD_SMPL_POS_B, pos_grad_prev).x;
            int2 stratum_prev = int2(int(prev_grad_sample_pos >> (STRATUM_OFFSET_SHIFT * 0)),
                                     int(prev_grad_sample_pos >> (STRATUM_OFFSET_SHIFT * 1))) & STRATUM_OFFSET_MASK;

            // A gradient pixel last frame must not be one again.
            if (all(pos_grad_prev * GRAD_DWN + stratum_prev == pp))
                valid = false;
        }

        if (valid) {
            uint cluster_curr = IMG_LOAD(PT_CLUSTER_A, p).x;
            uint cluster_prev = IMG_LOAD(PT_CLUSTER_B, pp).x;
            float depth_curr = IMG_LOAD(PT_VIEW_DEPTH_A, p).x;
            float depth_prev = IMG_LOAD(PT_VIEW_DEPTH_B, pp).x;
            float3 geo_normal_curr = decode_normal(IMG_LOAD(PT_GEO_NORMAL_A, p).x);
            float3 geo_normal_prev = decode_normal(IMG_LOAD(PT_GEO_NORMAL_B, pp).x);

            float dist_depth = abs(depth_curr - depth_prev + motion.z) / abs(depth_curr);
            float dot_geo_normals = dot(geo_normal_curr, geo_normal_prev);

            if (cluster_curr == cluster_prev && dist_depth < 0.1 && dot_geo_normals > 0.9) {
                float3 prev_hf = unpackRGBE(IMG_LOAD(PT_COLOR_HF, pp).x);
                float3 prev_spec = unpackRGBE(IMG_LOAD(PT_COLOR_SPEC, pp).x);
                float2 prev_lum = float2(luminance(prev_hf), luminance(prev_spec));
                s_reprojected_pixels[lp.y][lp.x] = float4(float2(pp), prev_lum);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Second pass: the first GROUP_SIZE_GRAD^2 threads pick the brightest
    // matching pixel in each 3x3 square.
    int2 local_pos;
    local_pos.x = int(local_index) % GROUP_SIZE_GRAD;
    local_pos.y = int(local_index) / GROUP_SIZE_GRAD;
    if (local_pos.y >= GROUP_SIZE_GRAD)
        return;

    int2 pos_grad = int2(group_id.xy) * GROUP_SIZE_GRAD + local_pos;
    ipos = pos_grad * GRAD_DWN;
    if (ipos.x >= global_ubo.width || ipos.y >= global_ubo.height)
        return;

    bool found = false;
    int2 found_offset = int2(0);
    int2 found_pos_prev = int2(0);
    float2 found_prev_lum = float2(0.0);

    for (int offy = 0; offy < GRAD_DWN; offy++) {
        for (int offx = 0; offx < GRAD_DWN; offx++) {
            int2 p = local_pos * GRAD_DWN + int2(offx, offy);
            float4 reprojected_pixel = s_reprojected_pixels[p.y][p.x];
            float2 prev_lum = reprojected_pixel.zw;
            if (prev_lum.x + prev_lum.y > found_prev_lum.x + found_prev_lum.y) {
                found_prev_lum = prev_lum;
                found_offset = int2(offx, offy);
                found_pos_prev = int2(reprojected_pixel.xy);
                found = true;
            }
        }
    }

    if (!found) {
        IMG_STORE(ASVGF_GRAD_SMPL_POS_A, pos_grad, uint4(0u));
        return;
    }

    ipos += found_offset;

    uint gradient_idx = (1u << 31) | (uint(found_offset.x) << (STRATUM_OFFSET_SHIFT * 0)) |
                        (uint(found_offset.y) << (STRATUM_OFFSET_SHIFT * 1));

    IMG_STORE(ASVGF_GRAD_SMPL_POS_A, pos_grad, uint4(gradient_idx));
    IMG_STORE(ASVGF_GRAD_HF_SPEC_PING, pos_grad, float4(found_prev_lum, 0.0, 0.0));
    IMG_STORE(ASVGF_RNG_SEED_A, ipos, IMG_LOAD(ASVGF_RNG_SEED_B, found_pos_prev));
    IMG_STORE(PT_NORMAL_A, ipos, IMG_LOAD(PT_NORMAL_B, found_pos_prev));
    IMG_STORE(PT_BASE_COLOR_A, ipos, IMG_LOAD(PT_BASE_COLOR_B, found_pos_prev));
    IMG_STORE(PT_METALLIC_A, ipos, IMG_LOAD(PT_METALLIC_B, found_pos_prev));

    patch_position(ctx, ipos, found_pos_prev);
}
