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

// The path tracer host side: scene geometry, acceleration structures and
// lights, and the frame, which follows vkpt's R_RenderFrame_RTX pass for
// pass (primary rays, god rays, reflections and refractions, gradient
// reprojection, direct and indirect lighting, A-SVGF, checkerboard
// interleave, TAAU, bloom, tone mapping, FSR) with the same global UBO and
// screen images (mtlpt_vkpt.h). Metal has no ray tracing pipeline, so the
// vkpt ray generation shaders are compute kernels with inline ray queries.

#include "mtlpt_metal.h"
#include "mtlpt_profiler.h"
#include "mtlpt_vkpt.h"
#include "physical_sky.h"
#include "material_compat.h"
#include "../vkpt/material.h"
#include "../vkpt/fog.h"
#include "../../client/client.h"
#include "../../client/ui/ui.h"
#include "common/math.h"
#include "../stb/stb_image.h"
#include <stdatomic.h>

extern uiStatic_t uis;

#define MTL_MATERIAL_FLAG_SKY    BIT(0)
#define MTL_MATERIAL_FLAG_LIGHT  BIT(1)
#define MTL_MATERIAL_FLAG_WARP   BIT(2)
#define MTL_MATERIAL_FLAG_TRANS  BIT(3)

// The tracing and denoising passes, one pipeline per vkpt shader
// (path_tracer.c, asvgf.c, god_rays.c).
static id<MTLComputePipelineState>  pipeline_primary_rays;
static id<MTLComputePipelineState>  pipeline_reflect_refract;
static id<MTLComputePipelineState>  pipeline_direct_lighting;
static id<MTLComputePipelineState>  pipeline_indirect_lighting;
static id<MTLComputePipelineState>  pipeline_gradient_reproject;
static id<MTLComputePipelineState>  pipeline_gradient_img;
static id<MTLComputePipelineState>  pipeline_gradient_atrous;
static id<MTLComputePipelineState>  pipeline_temporal;
static id<MTLComputePipelineState>  pipeline_atrous_lf;
static id<MTLComputePipelineState>  pipeline_atrous;
static id<MTLComputePipelineState>  pipeline_compositing;
static id<MTLComputePipelineState>  pipeline_interleave;
static id<MTLComputePipelineState>  pipeline_taa;
static id<MTLComputePipelineState>  pipeline_god_rays;
static id<MTLComputePipelineState>  pipeline_god_rays_filter;

// FidelityFX Super Resolution 1.0.
static id<MTLComputePipelineState>  pipeline_fsr_easu;
static id<MTLComputePipelineState>  pipeline_fsr_rcas;
static id<MTLTexture>               fsr_easu_output;
static id<MTLTexture>               fsr_rcas_output;
static bool                         fsr_active;          // this frame
cvar_t *cvar_flt_fsr_enable;   // read by the profiler overlay
static cvar_t *cvar_flt_fsr_easu;
static cvar_t *cvar_flt_fsr_rcas;
static cvar_t *cvar_flt_fsr_sharpness;

// Half resolution god rays accumulation (god_rays.c intermediate image).
static id<MTLTexture>               tex_god_rays;
// Sun space depth of the opaque scene for the god rays march (shadow_map.c).
// SHADOWMAP_SIZE in vkpt; at 2048^2 the texels along wall/floor seams
// alias into a chain of lit dots in the god rays.
#define MTL_SHADOW_MAP_SIZE 4096
static id<MTLRenderPipelineState>   pipeline_shadow_map;
static id<MTLDepthStencilState>     shadow_map_depth_state;
static id<MTLTexture>               tex_shadow_map;
static vec3_t                       world_aabb_min, world_aabb_max;

static cvar_t *cvar_gr_enable;
static cvar_t *cvar_gr_intensity;
static cvar_t *cvar_gr_eccentricity;
static cvar_t *cvar_gr_max_steps;
static cvar_t *cvar_pt_nearest;
static cvar_t *cvar_physical_sky_space;

// The screen images of global_textures.h (see mtlpt_vkpt.h). History pairs
// are fixed textures; the image table of each frame parity swaps which one
// the shaders see as _A, like vkpt's two descriptor sets.
static id<MTLTexture>               vkpt_images[VKPT_NUM_IMAGES];
static id<MTLBuffer>                image_tables[2];
static int                          screen_image_width, screen_image_height;   // FULL size class
static int                          unscaled_width, unscaled_height;           // 3D base resolution (see base_resolution)
static int                          display_width, display_height;             // the drawable
static bool                         resize_pending;
static cvar_t                      *cvar_mtl_hidpi;

// Blue noise (textures.c load_blue_noise), a stand-in sky box, and a zero
// buffer for optional bindings that have no data this frame.
static id<MTLTexture>               blue_noise_texture;
static id<MTLTexture>               dummy_sky_texture;
static id<MTLBuffer>                dummy_buffer;

// vkpt's global UBO. The host copy persists so the previous frame's
// matrices are at hand; each frame slot gets its own GPU copy.
static GlobalUbo                    ubo_host;
static id<MTLBuffer>                uniform_buffers[MTL_FRAMES_IN_FLIGHT];

#define UBO_CVAR_DO(name, default_value) static cvar_t *ubo_cvar_##name;
VKPT_UBO_CVAR_LIST
#undef UBO_CVAR_DO

// Frame state (main.c): frames traced so far, which drives the history
// parity, the blue noise sequence and the light statistics ring.
static uint32_t                     pt_frame_counter;
static uint32_t                     prev_traced_slot;
static bool                         temporal_frame_valid;
static bool                         prev_render_world = true;
static bool                         tlas_valid;           // tlas[current slot] was rebuilt this frame
static int                          effective_aa_mode;    // AA_MODE_*
static int                          taa_output_width, taa_output_height;
static int                          prev_render_width, prev_render_height;

// Reference (photo) mode, the port of vkpt's reference_mode_t.
static cvar_t *cvar_pt_accumulation_rendering;
static cvar_t *cvar_pt_accumulation_rendering_framenum;
static int     num_accumulated_frames;
static bool    accumulation_active;
static float   temporal_blend_factor;
static bool    reset_accumulation_requested;

#define NUM_TAA_SAMPLES 128
static vec2_t                       taa_samples[NUM_TAA_SAMPLES];

// Light statistics (adaptive shadow ray budgeting): NUM_LIGHT_STATS_BUFFERS
// slices, one cleared per frame. Light counts: the per-cluster list lengths
// of the last LIGHT_COUNT_HISTORY frames, so gradient samples pick the same
// light as the frame they reproduce.
static id<MTLBuffer>                light_stats_buffer;
static uint32_t                     light_stats_size;    // uints per slice
static id<MTLBuffer>                light_counts_buffer;

// Per entity primitive ranges of each frame (vkpt model_current_to_prev),
// for mapping last frame's visibility buffer onto this frame's geometry.
static id<MTLBuffer>                entity_table_buffers[MTL_FRAMES_IN_FLIGHT];

// Sky box faces, stored as a 6 slice array in the axis order the shader uses.
static id<MTLTexture>               sky_texture;
static float                        sky_rotate;
static bool                         sky_autorotate;
static vec3_t                       sky_axis;
static vec3_t                       sky_average_color;

// Dynamic resolution scaling. The screen images are allocated at the window
// size; only the used sub-rectangle changes.
static int      drs_current_scale;
static int      drs_effective_scale;
static bool     drs_last_frame_world;   // the previous traced frame was the world view
static int      render_width, render_height;

static cvar_t *cvar_drs_enable;
static cvar_t *cvar_drs_target;
static cvar_t *cvar_drs_minscale;
static cvar_t *cvar_drs_maxscale;
static cvar_t *cvar_drs_gain;
static cvar_t *cvar_drs_adjust_up;
static cvar_t *cvar_drs_adjust_down;
static cvar_t *cvar_drs_last_scale;
static cvar_t *cvar_scr_viewsize;

static id<MTLBuffer>                vertex_buffer;
static id<MTLBuffer>                index_buffer;
static id<MTLBuffer>                material_buffers[MTL_FRAMES_IN_FLIGHT];
static id<MTLBuffer>                light_poly_buffer;
// Per frame: the static world list followed by this frame's dynamic lights
// (emissive inline models, beams). This is what the tracer samples.
#define MTL_MAX_DYNAMIC_LIGHT_POLYS 4096
static id<MTLBuffer>                frame_light_poly_buffers[MTL_FRAMES_IN_FLIGHT];
static uint32_t                     frame_light_poly_capacity;
static uint32_t                     num_frame_light_polys;

// Emissive triangles of each inline model in model space, instanced per frame.
typedef struct {
    uint32_t first;
    uint32_t count;
} mtl_submodel_lights_t;
static MTLLightPoly          *submodel_light_polys;
static uint32_t              *submodel_light_poly_material;
static uint32_t               num_submodel_light_polys;
static mtl_submodel_lights_t *submodel_lights;

static id<MTLAccelerationStructure> world_blas;
static id<MTLAccelerationStructure> world_transparent_blas;
static id<MTLBuffer>                transparent_index_buffer;
static id<MTLBuffer>                opaque_index_buffer;
static id<MTLBuffer>                sky_index_buffer;
static id<MTLAccelerationStructure> world_sky_blas;
static id<MTLAccelerationStructure> world_masked_blas;
static id<MTLBuffer>                masked_index_buffer;
// World index buffer: [0, masked) opaque, [masked, opaque) alpha tested,
// [opaque, sky) transparent, [sky, num) sky.
static uint32_t num_masked_first_index;
static uint32_t num_opaque_indices;
static uint32_t num_sky_first_index;
static image_t *water_normal_image;
static id<MTLAccelerationStructure> tlas[MTL_FRAMES_IN_FLIGHT];
static id<MTLBuffer>                instance_buffers[MTL_FRAMES_IN_FLIGHT];

// Entity geometry is rebuilt every frame, because Quake's alias models are
// vertex animated and there is no skinning pass yet. Everything written per
// frame is ring buffered, since up to MTL_FRAMES_IN_FLIGHT frames may still be
// reading the previous contents on the GPU.
static id<MTLBuffer>                entity_vertex_buffers[MTL_FRAMES_IN_FLIGHT];
static id<MTLBuffer>                entity_index_buffers[MTL_FRAMES_IN_FLIGHT];
static id<MTLAccelerationStructure> entity_blas[MTL_FRAMES_IN_FLIGHT];
static id<MTLAccelerationStructure> entity_transparent_blas[MTL_FRAMES_IN_FLIGHT];
static id<MTLBuffer>                entity_transparent_scratch[MTL_FRAMES_IN_FLIGHT];
static id<MTLAccelerationStructure> viewer_blas[MTL_FRAMES_IN_FLIGHT];
static id<MTLAccelerationStructure> weapon_blas[MTL_FRAMES_IN_FLIGHT];
static id<MTLBuffer>                viewer_scratch[MTL_FRAMES_IN_FLIGHT];
static id<MTLBuffer>                weapon_scratch[MTL_FRAMES_IN_FLIGHT];
// Index buffer ranges of the entity groups: opaque | transparent | viewer | weapon.
static uint32_t                     entity_transparent_first_index;
static uint32_t                     entity_viewer_first_index;
static uint32_t                     entity_weapon_first_index;
static id<MTLBuffer>                entity_scratch[MTL_FRAMES_IN_FLIGHT];
static id<MTLBuffer>                tlas_scratch[MTL_FRAMES_IN_FLIGHT];
static uint32_t                     num_entity_vertices;
static uint32_t                     num_entity_indices;
static uint32_t                     num_entity_materials;
static int                          world_material_count;

// TLAS instances, see INSTANCE_* in path_tracer.metal.
#define MTL_TLAS_INSTANCES 8

// Index of the ring slot currently being recorded.
static uint32_t                     current_frame_slot;

// Last frame's pose of every entity, keyed by entity_t::id (stable per
// object, unlike the fd->entities slot). This is what gives moving entities
// correct motion vectors; vkpt keeps the equivalent in transform_prev.
#define MTL_ENTITY_POSE_CACHE_SIZE 1024

typedef struct {
    vec3_t origin;      // interpolated (entity_lerped_origin), like transform_prev
    vec3_t angles;
    int    frame;
    int    oldframe;
    float  backlerp;
    float  scale;
} mtl_pose_t;

typedef struct {
    int        id;          // 0 = empty
    int        frame_seen;  // mtl.frame_counter `curr` was recorded in
    mtl_pose_t curr;        // the pose in frame_seen
    mtl_pose_t prev;        // the pose the frame before frame_seen
} mtl_entity_pose_t;

static mtl_entity_pose_t entity_pose_cache[MTL_ENTITY_POSE_CACHE_SIZE];

// Direct mapped; a collision only costs the evicted entity one frame of zero
// motion.
static mtl_entity_pose_t *entity_pose_slot(int id)
{
    return &entity_pose_cache[(uint32_t)id % MTL_ENTITY_POSE_CACHE_SIZE];
}

static void entity_lerped_origin(const entity_t *ent, vec3_t out);

// The interpolated origin, not the raw one: re-lerping last frame's origin
// with this frame's oldorigin and backlerp gives a wrong previous position
// (the view weapon's oldorigin is its current origin).
static void entity_current_pose(const entity_t *ent, mtl_pose_t *pose)
{
    entity_lerped_origin(ent, pose->origin);
    VectorCopy(ent->angles, pose->angles);
    pose->frame = ent->frame;
    pose->oldframe = ent->oldframe;
    pose->backlerp = ent->backlerp;
    pose->scale = ent->scale;
}

// Fetches last frame's pose for `ent` (falling back to the current one) and
// records the current pose for next frame. Called once per mesh, so every
// mesh of a multi-mesh model (the MD3 view weapons keep the hands in a mesh
// of their own) must get the same answer within a frame.
static void entity_prev_pose(const entity_t *ent, vec3_t prev_origin, vec3_t prev_angles,
                             int *prev_frame, int *prev_oldframe, float *prev_backlerp, float *prev_scale)
{
    mtl_entity_pose_t *slot = entity_pose_slot(ent->id);
    int now = (int)mtl.frame_counter;

    if (!(ent->id != 0 && slot->id == ent->id && slot->frame_seen == now)) {
        mtl_pose_t curr;
        entity_current_pose(ent, &curr);
        // Only a pose from exactly the previous frame is usable history.
        bool valid = ent->id != 0 && slot->id == ent->id && now - slot->frame_seen == 1;
        slot->prev = valid ? slot->curr : curr;
        slot->curr = curr;
        slot->id = ent->id;
        slot->frame_seen = now;
    }

    const mtl_pose_t *prev = &slot->prev;
    VectorCopy(prev->origin, prev_origin);
    VectorCopy(prev->angles, prev_angles);
    *prev_frame = prev->frame;
    *prev_oldframe = prev->oldframe;
    *prev_backlerp = prev->backlerp;
    *prev_scale = prev->scale;
}

// Transparent effects: one camera facing quad per particle, beam or sprite,
// in their own acceleration structure so opaque rays never see them.
typedef struct { float x, y, z; } mtl_packed_float3_t;
static id<MTLBuffer>                effect_vertex_buffers[MTL_FRAMES_IN_FLIGHT];
static id<MTLBuffer>                effect_attr_buffers[MTL_FRAMES_IN_FLIGHT];
static id<MTLBuffer>                effect_index_buffers[MTL_FRAMES_IN_FLIGHT];
static id<MTLBuffer>                effect_prim_buffers[MTL_FRAMES_IN_FLIGHT];
static id<MTLAccelerationStructure> effect_blas[MTL_FRAMES_IN_FLIGHT];
static id<MTLAccelerationStructure> effect_tlas[MTL_FRAMES_IN_FLIGHT];
static id<MTLBuffer>                effect_scratch[MTL_FRAMES_IN_FLIGHT];
static id<MTLBuffer>                effect_tlas_scratch[MTL_FRAMES_IN_FLIGHT];
static id<MTLBuffer>                effect_instance_buffers[MTL_FRAMES_IN_FLIGHT];
static uint32_t                     num_effect_vertices;
static uint32_t                     num_effect_triangles;

static cvar_t *cvar_pt_particle_size;
static cvar_t *cvar_pt_beam_width;
static cvar_t *cvar_pt_enable_particles;
static cvar_t *cvar_pt_enable_beams;
static cvar_t *cvar_pt_enable_sprites;
static cvar_t *cvar_pt_particle_emissive;

static uint32_t num_indices;
static uint32_t num_vertices;
static uint32_t num_light_polys;
static bool     have_sky_lights;
static bool     world_ready;
static bsp_t   *world_bsp;
static float    prev_frame_time;
static bool     last_rdflags_underwater;
static int      current_rdflags;

static int  light_poly_cluster_of(const bsp_t *bsp, const vec3_t p0, const vec3_t p1, const vec3_t p2);
static int  triangle_cluster(const bsp_t *bsp, const vec3_t p0, const vec3_t p1, const vec3_t p2);
static int  point_cluster(const vec3_t p);
static void allocate_light_list_buffers(void);
static void collect_cluster_lights(const bsp_t *bsp);
static void free_light_lists(void);
static void build_frame_light_lists(uint32_t slot, uint32_t num_dynamic);
static int *light_poly_cluster;            // per static light poly
static uint32_t num_unclustered_faces;

// Renderer-private entity flag: the mesh has mirrored texture handedness.
#define RF_MTL_HANDEDNESS BIT(31)
static cvar_t  *cvar_pt_waterwarp;

// Index ranges of the inline BSP models inside the world geometry buffers.
typedef struct {
    uint32_t first_index;
    uint32_t index_count;
} mtl_submodel_t;

static mtl_submodel_t *submodels;
static uint32_t        num_submodels;

// maps/sky/<map>.txt: BSP clusters whose sky faces are converted to area
// lights, plus the "!all_lava" flag (bsp_mesh.c load_sky_and_lava_clusters).
#define MTL_MAX_SKY_CLUSTERS 1024
static int      sky_clusters[MTL_MAX_SKY_CLUSTERS];
static int      num_sky_clusters;
static bool     have_sky_cluster_list;
static bool     all_lava_emissive;
static char     world_map_name[MAX_QPATH];
static cvar_t  *cvar_pt_bsp_sky_lights;

// Animated world materials (blinking signs etc.), the port of
// animate_materials.comp. Each texinfo tracks which frame of its chain is
// current; the world material slots are re-copied from the pristine set.
static MTLMaterial *world_materials_base;
static int         *texinfo_anim_current;
static uint32_t    *light_poly_material;   // texinfo per light poly, ~0u for sky
static int          world_anim_frame = -1;
static int          material_slots_dirty;

// Inverse texture dimensions per texinfo, used to normalize lightmap-style
// texture axes into [0,1) coordinates.
static vec2_t  *texinfo_inv_size;

// Emissive radiance per texinfo, zero for non-light surfaces.
static vec3_t  *texinfo_radiance;

static cvar_t *mtl_pt_light_scale;
static cvar_t *cvar_pt_caustics;
static cvar_t *cvar_pt_enable_nodraw;
static cvar_t *cvar_pt_projection;
static cvar_t *cvar_pt_enable_surface_lights;
static cvar_t *cvar_pt_enable_surface_lights_warp;
static cvar_t *cvar_mtl_view_override;

// Security cameras (cameras.c): maps/cameras/<map>.txt lists "(pos) (dir)".
typedef struct { vec3_t pos, dir; } mtl_camera_t;
static mtl_camera_t cameras[MAX_CAMERAS];
static uint32_t     num_cameras;

static void load_cameras(const char *map_name)
{
    num_cameras = 0;

    char filename[MAX_QPATH];
    Q_snprintf(filename, sizeof(filename), "maps/cameras/%s.txt", map_name);

    char *filebuf = NULL;
    FS_LoadFile(filename, (void **)&filebuf);
    if (!filebuf) {
        Com_DPrintf("Couldn't read %s\n", filename);
        return;
    }

    const char *ptr = filebuf;
    char linebuf[1024];
    while (sgets(linebuf, sizeof(linebuf), &ptr)) {
        { char *t = strchr(linebuf, '#'); if (t) *t = 0; }
        { char *t = strchr(linebuf, '\n'); if (t) *t = 0; }

        vec3_t pos, dir;
        if (sscanf(linebuf, "(%f, %f, %f) (%f, %f, %f)", &pos[0], &pos[1], &pos[2], &dir[0], &dir[1], &dir[2]) != 6)
            continue;
        if (num_cameras >= MAX_CAMERAS) {
            Com_WPrintf("Map has too many cameras (max: %i)\n", MAX_CAMERAS);
            break;
        }
        VectorCopy(pos, cameras[num_cameras].pos);
        VectorCopy(dir, cameras[num_cameras].dir);
        num_cameras++;
    }

    Z_Free(filebuf);
}

// Port of vkpt's prepare_camera(): origin, top-left corner direction and the
// two screen extents, for a 1.75 aspect, 90 degree view.
static void prepare_camera(const vec3_t position, const vec3_t direction, mtl_float4 data[4])
{
    vec3_t forward, right, up;
    VectorCopy(direction, forward);
    VectorNormalize(forward);

    if (fabsf(forward[2]) < 0.99f)
        VectorSet(up, 0.0f, 0.0f, 1.0f);
    else
        VectorSet(up, 0.0f, 1.0f, 0.0f);

    CrossProduct(forward, up, right);
    CrossProduct(right, forward, up);
    VectorNormalize(up);
    VectorNormalize(right);

    float aspect = 1.75f;
    float tan_half_fov_x = 1.0f;
    float tan_half_fov_y = tan_half_fov_x / aspect;

    vec3_t corner, ext_x, ext_y;
    VectorCopy(forward, corner);
    VectorMA(corner, -tan_half_fov_x, right, corner);
    VectorMA(corner, tan_half_fov_y, up, corner);
    VectorScale(right, 2.0f * tan_half_fov_x, ext_x);
    VectorScale(up, -2.0f * tan_half_fov_y, ext_y);

    data[0] = (mtl_float4){ position[0], position[1], position[2], 0.0f };
    data[1] = (mtl_float4){ corner[0], corner[1], corner[2], 0.0f };
    data[2] = (mtl_float4){ ext_x[0], ext_x[1], ext_x[2], 0.0f };
    data[3] = (mtl_float4){ ext_y[0], ext_y[1], ext_y[2], 0.0f };
}

static const refdef_t *camera_cmd_refdef;

// `camera` prints the current view in the cameras file format.
static void Camera_Cmd_f(void)
{
    if (!camera_cmd_refdef)
        return;
    vec3_t forward;
    AngleVectors(camera_cmd_refdef->viewangles, forward, NULL, NULL);
    Com_Printf("(%f, %f, %f) (%f, %f, %f)\n",
               camera_cmd_refdef->vieworg[0], camera_cmd_refdef->vieworg[1], camera_cmd_refdef->vieworg[2],
               forward[0], forward[1], forward[2]);
}
static cvar_t *cvar_tm_white_point;

// Sun and sky, using the same cvar names as the Vulkan backend.
static cvar_t *cvar_sun_azimuth;
static cvar_t *cvar_sun_elevation;
static cvar_t *cvar_sun_color[3];
static cvar_t *cvar_sun_brightness;
static cvar_t *cvar_sky_brightness;
static cvar_t *cvar_lava_emissive;
static cvar_t *cvar_physical_sky_brightness;

#if !REF_VKPT
// The client consults this to decide whether to add explicit rail trail
// lights. The Vulkan backend owns it when both backends are compiled in.
cvar_t *cvar_pt_beam_lights = NULL;
#else
extern cvar_t *cvar_pt_beam_lights;
#endif

static void effect_color(int color_index, const color_t *rgba, float hdr_factor, MTLEffectPrim *prim);
static void entity_lerped_origin(const entity_t *ent, vec3_t out);
static void release_render_targets(void);
static void apply_resize(void);
static void mtl_hidpi_changed(cvar_t *self);

static float halton(int base, int index)
{
    float f = 1.0f, r = 0.0f;
    while (index > 0) {
        f /= (float)base;
        r += f * (float)(index % base);
        index /= base;
    }
    return r;
}

static id<MTLComputePipelineState> create_compute_pipeline(const char *name)
{
    NSError *err = nil;
    id<MTLFunction> fn = mtl_new_function([NSString stringWithUTF8String:name]);
    if (!fn) {
        Com_EPrintf("Metal: kernel '%s' missing from the library\n", name);
        return nil;
    }

    id<MTLComputePipelineState> pipeline = [mtl.device newComputePipelineStateWithFunction:fn error:&err];
    [fn release];

    if (!pipeline)
        mtl_log_error(name, err);

    return pipeline;
}

static bool create_shadow_map_resources(void)
{
    id<MTLFunction> vs = mtl_new_function(@"shadow_map_vertex");
    if (!vs) {
        Com_EPrintf("Metal: shadow_map_vertex missing from the shader library\n");
        return false;
    }

    // Depth only: no fragment function and no colour attachment.
    MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.label = @"shadow map";
    desc.vertexFunction = vs;
    desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    NSError *err = nil;
    pipeline_shadow_map = [mtl.device newRenderPipelineStateWithDescriptor:desc error:&err];
    [desc release];
    [vs release];
    if (!pipeline_shadow_map)
        return mtl_log_error("newRenderPipelineStateWithDescriptor(shadow map)", err);

    MTLDepthStencilDescriptor *ds = [[MTLDepthStencilDescriptor alloc] init];
    ds.depthCompareFunction = MTLCompareFunctionLess;
    ds.depthWriteEnabled = YES;
    shadow_map_depth_state = [mtl.device newDepthStencilStateWithDescriptor:ds];
    [ds release];

    MTLTextureDescriptor *td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                           width:MTL_SHADOW_MAP_SIZE
                                                          height:MTL_SHADOW_MAP_SIZE
                                                       mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    tex_shadow_map = [mtl.device newTextureWithDescriptor:td];
    tex_shadow_map.label = @"sun shadow map";

    return shadow_map_depth_state && tex_shadow_map;
}

// vkpt_shadow_map_setup(): an orthographic view along the sun direction that
// covers the world bounds, stored as three NDC rows for the shaders.
static void shadow_map_setup(const vec3_t sun_direction, MTLGodRaysUniforms *g)
{
    vec3_t up_dir = { 0.0f, 0.0f, 1.0f };
    if (sun_direction[2] >= 0.99f)
        VectorSet(up_dir, 1.0f, 0.0f, 0.0f);

    vec3_t look_dir, left_dir;
    VectorScale(sun_direction, -1.0f, look_dir);
    VectorNormalize(look_dir);
    CrossProduct(up_dir, look_dir, left_dir);
    VectorNormalize(left_dir);
    CrossProduct(look_dir, left_dir, up_dir);
    VectorNormalize(up_dir);

    const float *axes[3] = { left_dir, up_dir, look_dir };
    vec3_t vmin = { FLT_MAX, FLT_MAX, FLT_MAX }, vmax = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    for (int i = 0; i < 8; i++) {
        vec3_t corner = {
            (i & 1) ? world_aabb_max[0] : world_aabb_min[0],
            (i & 2) ? world_aabb_max[1] : world_aabb_min[1],
            (i & 4) ? world_aabb_max[2] : world_aabb_min[2],
        };
        for (int a = 0; a < 3; a++) {
            float v = DotProduct(corner, axes[a]);
            vmin[a] = min(vmin[a], v);
            vmax[a] = max(vmax[a], v);
        }
    }

    // Square texels.
    float max_xy = max(vmax[0] - vmin[0], vmax[1] - vmin[1]);
    for (int a = 0; a < 2; a++) {
        float pad = (max_xy - (vmax[a] - vmin[a])) * 0.5f;
        vmin[a] -= pad;
        vmax[a] += pad;
    }

    // x, y to [-1, 1], z to [0, 1].
    for (int a = 0; a < 3; a++) {
        float extent = max(vmax[a] - vmin[a], 1.0f);
        float scale = (a < 2) ? 2.0f / extent : 1.0f / extent;
        float offset = (a < 2) ? -(vmax[a] + vmin[a]) / extent : -vmin[a] / extent;
        g->shadow_rows[a].x = axes[a][0] * scale;
        g->shadow_rows[a].y = axes[a][1] * scale;
        g->shadow_rows[a].z = axes[a][2] * scale;
        g->shadow_rows[a].w = offset;
    }
}

// shadow_map.c: the opaque world and the regular entities, depth only.
static void render_shadow_map(id<MTLCommandBuffer> cmd, uint32_t slot, const MTLGodRaysUniforms *g)
{
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.depthAttachment.texture = tex_shadow_map;
    pass.depthAttachment.loadAction = MTLLoadActionClear;
    pass.depthAttachment.storeAction = MTLStoreActionStore;
    pass.depthAttachment.clearDepth = 1.0;

    id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:pass];
    enc.label = @"sun shadow map";
    [enc setRenderPipelineState:pipeline_shadow_map];
    [enc setDepthStencilState:shadow_map_depth_state];
    [enc setVertexBytes:g length:sizeof(*g) atIndex:1];

    // Every face: a Quake 2 wall only has faces on the playable side, so
    // culling either side drops real occluders and lets god rays through.
    [enc setCullMode:MTLCullModeNone];

    if (vertex_buffer && index_buffer && num_opaque_indices) {
        [enc setVertexBuffer:vertex_buffer offset:0 atIndex:0];
        [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                        indexCount:num_opaque_indices
                         indexType:MTLIndexTypeUInt32
                       indexBuffer:index_buffer
                 indexBufferOffset:0];
    }
    // Regular entities (doors, monsters, items); not the viewer's own models.
    // Transparent entities do not cast sun shadows, as in vkpt.
    if (entity_transparent_first_index) {
        [enc setVertexBuffer:entity_vertex_buffers[slot] offset:0 atIndex:0];
        [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                        indexCount:entity_transparent_first_index
                         indexType:MTLIndexTypeUInt32
                       indexBuffer:entity_index_buffers[slot]
                 indexBufferOffset:0];
    }
    [enc endEncoding];
}

// A compute pipeline whose threadgroups must hold `threads` threads.
static id<MTLComputePipelineState> create_compute_pipeline_threads(const char *name, NSUInteger threads)
{
    id<MTLFunction> fn = mtl_new_function([NSString stringWithUTF8String:name]);
    if (!fn) {
        Com_EPrintf("Metal: kernel '%s' missing from the library\n", name);
        return nil;
    }

    MTLComputePipelineDescriptor *desc = [[MTLComputePipelineDescriptor alloc] init];
    desc.computeFunction = fn;
    desc.label = [NSString stringWithUTF8String:name];
    // No maxTotalThreadsPerThreadgroup hint: telling the compiler a group
    // only needs `threads` lets it hand each thread more registers, which
    // cost the ray tracing kernels resident threads (about 6% of the frame).

    NSError *err = nil;
    id<MTLComputePipelineState> pipeline =
        [mtl.device newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:nil error:&err];
    [desc release];

    if (pipeline && getenv("Q2RTX_MTL_TIMING")) {
        id<MTLComputePipelineState> probe = [mtl.device newComputePipelineStateWithFunction:fn error:nil];
        fprintf(stderr, "pipeline %-28s max threads %4u (unconstrained %4u)\n", name,
                (unsigned)pipeline.maxTotalThreadsPerThreadgroup, (unsigned)probe.maxTotalThreadsPerThreadgroup);
        [probe release];
    }
    [fn release];

    if (!pipeline) {
        mtl_log_error(name, err);
        return nil;
    }
    if (pipeline.maxTotalThreadsPerThreadgroup < threads) {
        Com_EPrintf("Metal: kernel '%s' fits only %u threads per group, needs %u\n", name,
                    (unsigned)pipeline.maxTotalThreadsPerThreadgroup, (unsigned)threads);
        [pipeline release];
        return nil;
    }
    return pipeline;
}

// `mtl_find_texture <name>`: prints where the world uses a texture (face
// centre and normal), for placing the camera with mtl_view_override.
static void FindTexture_Cmd_f(void)
{
    if (!world_bsp || Cmd_Argc() < 2)
        return;
    const char *name = Cmd_Argv(1);
    int found = 0;
    for (int i = 0; i < world_bsp->numfaces && found < 40; i++) {
        const mface_t *surf = &world_bsp->faces[i];
        if (!surf->texinfo || Q_stricmp(surf->texinfo->name, name) || surf->numsurfedges < 3)
            continue;
        vec3_t c = { 0, 0, 0 };
        for (int e = 0; e < surf->numsurfedges; e++)
            VectorAdd(c, surf->firstsurfedge[e].edge->v[surf->firstsurfedge[e].vert]->point, c);
        VectorScale(c, 1.0f / surf->numsurfedges, c);
        vec3_t n;
        VectorCopy(surf->plane->normal, n);
        if (surf->drawflags & DSURF_PLANEBACK)
            VectorNegate(n, n);
        Com_Printf("%s: center (%.0f %.0f %.0f) normal (%.2f %.2f %.2f)\n", name, c[0], c[1], c[2], n[0], n[1], n[2]);
        found++;
    }
    if (!found)
        Com_Printf("%s: not in this map\n", name);
}

static void accumulation_cvar_changed(cvar_t *self)
{
    reset_accumulation_requested = true;
}

static void temporal_cvar_changed(cvar_t *self)
{
    temporal_frame_valid = false;
}

// textures.c load_blue_noise(): 128 RGBA16 images, each channel one layer.
static bool load_blue_noise(void)
{
    const int res = BLUE_NOISE_RES;

    MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
    desc.textureType = MTLTextureType2DArray;
    desc.pixelFormat = MTLPixelFormatR16Unorm;
    desc.width = res;
    desc.height = res;
    desc.arrayLength = NUM_BLUE_NOISE_TEX;
    desc.storageMode = MTLStorageModeShared;
    desc.usage = MTLTextureUsageShaderRead;
    blue_noise_texture = [mtl.device newTextureWithDescriptor:desc];
    [desc release];
    if (!blue_noise_texture)
        return false;
    blue_noise_texture.label = @"blue noise";

    uint16_t *layer = Z_Malloc(sizeof(uint16_t) * res * res);
    bool ok = true;

    for (int i = 0; i < NUM_BLUE_NOISE_TEX / 4 && ok; i++) {
        char path[MAX_QPATH];
        Q_snprintf(path, sizeof(path), "blue_noise/%d_%d/HDR_RGBA_%04d.png", res, res, i);

        byte *filedata = NULL;
        int filelen = FS_LoadFile(path, (void **)&filedata);
        uint16_t *data = NULL;
        int w = 0, h = 0, n = 0;
        if (filedata) {
            data = stbi_load_16_from_memory(filedata, filelen, &w, &h, &n, 4);
            FS_FreeFile(filedata);
        }
        if (!data || w != res || h != res) {
            Com_EPrintf("Metal: error loading blue noise texture %s\n", path);
            if (data)
                stbi_image_free(data);
            ok = false;
            break;
        }

        for (int k = 0; k < 4; k++) {
            for (int j = 0; j < res * res; j++)
                layer[j] = data[j * 4 + k];
            [blue_noise_texture replaceRegion:MTLRegionMake2D(0, 0, res, res)
                                  mipmapLevel:0
                                        slice:i * 4 + k
                                    withBytes:layer
                                  bytesPerRow:sizeof(uint16_t) * res
                                bytesPerImage:0];
        }
        stbi_image_free(data);
    }

    Z_Free(layer);
    return ok;
}

bool mtl_pt_init(void)
{
    // Scales the BSP surface light value only; same name, meaning and default
    // as the Vulkan backend.
    mtl_pt_light_scale = Cvar_Get("pt_bsp_radiance_scale", "0.001", CVAR_ARCHIVE);

    // Registered before the UBO list so the archive flags stick, as vkpt's
    // menu and config files expect.
    Cvar_Get("flt_enable", "1", CVAR_ARCHIVE);
    Cvar_Get("flt_taa", "2", CVAR_ARCHIVE);
    Cvar_Get("pt_reflect_refract", "2", CVAR_ARCHIVE);
    Cvar_Get("pt_cameras", "1", CVAR_ARCHIVE);
    mtl_freecam_init();   // pt_dof, pt_aperture, pt_focus

#define UBO_CVAR_DO(name, default_value) ubo_cvar_##name = Cvar_Get(#name, #default_value, 0);
    VKPT_UBO_CVAR_LIST
#undef UBO_CVAR_DO

    ubo_cvar_flt_temporal_hf->changed = temporal_cvar_changed;
    ubo_cvar_flt_temporal_lf->changed = temporal_cvar_changed;
    ubo_cvar_flt_temporal_spec->changed = temporal_cvar_changed;
    ubo_cvar_flt_enable->changed = temporal_cvar_changed;
    ubo_cvar_pt_aperture_type->changed = accumulation_cvar_changed;
    ubo_cvar_pt_aperture_angle->changed = accumulation_cvar_changed;
    ubo_cvar_pt_num_bounce_rays->flags |= CVAR_ARCHIVE;

    cvar_pt_caustics = Cvar_Get("pt_caustics", "1", CVAR_ARCHIVE);
    cvar_pt_enable_nodraw = Cvar_Get("pt_enable_nodraw", "0", 0);
    // Wall and skin texture filtering (the "texture filtering" menu item).
    cvar_pt_nearest = Cvar_Get("pt_nearest", "0", CVAR_ARCHIVE);
    cvar_pt_projection = Cvar_Get("pt_projection", "0", CVAR_ARCHIVE);
    cvar_pt_projection->changed = accumulation_cvar_changed;
    cvar_pt_enable_surface_lights = Cvar_Get("pt_enable_surface_lights", "1", CVAR_FILES);
    cvar_pt_enable_surface_lights_warp = Cvar_Get("pt_enable_surface_lights_warp", "0", CVAR_FILES);
    Cvar_Get("sun_bounce", "1", CVAR_ARCHIVE);
    Cmd_AddCommand("camera", Camera_Cmd_f);
    Cmd_AddCommand("mtl_find_texture", FindTexture_Cmd_f);
    cvar_mtl_view_override = Cvar_Get("mtl_view_override", "", 0);
    cvar_tm_white_point = Cvar_Get("tm_white_point", "10.0", CVAR_ARCHIVE);

    for (int i = 0; i < NUM_TAA_SAMPLES; i++) {
        taa_samples[i][0] = halton(2, i + 1) - 0.5f;
        taa_samples[i][1] = halton(3, i + 1) - 0.5f;
    }

    // Same defaults as vkpt's physical_sky.c; whichever registers first wins.
    cvar_sun_azimuth = Cvar_Get("sun_azimuth", "345", 0);
    cvar_sun_elevation = Cvar_Get("sun_elevation", "45", 0);
    cvar_sun_color[0] = Cvar_Get("sun_color_r", "1.0", 0);
    cvar_sun_color[1] = Cvar_Get("sun_color_g", "1.0", 0);
    cvar_sun_color[2] = Cvar_Get("sun_color_b", "1.0", 0);
    cvar_sun_brightness = Cvar_Get("sun_brightness", "10", 0);
    cvar_sky_brightness = Cvar_Get("sky_brightness", "1.0", CVAR_ARCHIVE);
    cvar_lava_emissive = Cvar_Get("mtl_pt_lava_emissive", "2.0", CVAR_ARCHIVE);
    cvar_physical_sky_brightness = Cvar_Get("physical_sky_brightness", "0", 0);

    // These drive the resolution options in the Q2RTX menu.
    cvar_drs_enable = Cvar_Get("drs_enable", "0", CVAR_ARCHIVE);
    cvar_drs_target = Cvar_Get("drs_target", "60", CVAR_ARCHIVE);
    cvar_drs_minscale = Cvar_Get("drs_minscale", "50", 0);
    cvar_drs_maxscale = Cvar_Get("drs_maxscale", "100", 0);
    cvar_drs_gain = Cvar_Get("drs_gain", "20", 0);
    cvar_drs_adjust_up = Cvar_Get("drs_adjust_up", "0.92", 0);
    cvar_drs_adjust_down = Cvar_Get("drs_adjust_down", "0.98", 0);
    cvar_drs_last_scale = Cvar_Get("drs_last_scale", "0", CVAR_ARCHIVE);
    cvar_scr_viewsize = Cvar_Get("scr_viewsize", "100", CVAR_ARCHIVE);

#if !REF_VKPT
    cvar_pt_beam_lights = Cvar_Get("pt_beam_lights", "1.0", 0);
#else
    if (!cvar_pt_beam_lights)
        cvar_pt_beam_lights = Cvar_Get("pt_beam_lights", "1.0", 0);
#endif

    if (!mtl_device_supports_raytracing()) {
        Com_EPrintf("Metal: this GPU does not support ray tracing\n");
        return false;
    }

    // Threadgroup sizes are vkpt's local sizes.
    pipeline_primary_rays = create_compute_pipeline_threads("pt_primary_rays", 64);
    pipeline_reflect_refract = create_compute_pipeline_threads("pt_reflect_refract", 64);
    pipeline_direct_lighting = create_compute_pipeline_threads("pt_direct_lighting", 64);
    pipeline_indirect_lighting = create_compute_pipeline_threads("pt_indirect_lighting", 64);
    pipeline_gradient_reproject = create_compute_pipeline_threads("asvgf_gradient_reproject", 24 * 24);
    pipeline_gradient_img = create_compute_pipeline_threads("asvgf_gradient_img", 256);
    pipeline_gradient_atrous = create_compute_pipeline_threads("asvgf_gradient_atrous", 256);
    pipeline_temporal = create_compute_pipeline_threads("asvgf_temporal", 15 * 15);
    pipeline_atrous_lf = create_compute_pipeline_threads("asvgf_lf", 256);
    pipeline_atrous = create_compute_pipeline_threads("asvgf_atrous", 256);
    pipeline_compositing = create_compute_pipeline_threads("compositing", 256);
    pipeline_interleave = create_compute_pipeline_threads("checkerboard_interleave", 256);
    pipeline_taa = create_compute_pipeline_threads("taa_main", 256);
    pipeline_god_rays = create_compute_pipeline_threads("god_rays_trace", 64);
    pipeline_god_rays_filter = create_compute_pipeline_threads("god_rays_filter", 64);
    pipeline_fsr_easu = create_compute_pipeline("fsr_easu");
    pipeline_fsr_rcas = create_compute_pipeline("fsr_rcas");

    if (!pipeline_primary_rays || !pipeline_reflect_refract || !pipeline_direct_lighting ||
        !pipeline_indirect_lighting || !pipeline_gradient_reproject || !pipeline_gradient_img ||
        !pipeline_gradient_atrous || !pipeline_temporal || !pipeline_atrous_lf || !pipeline_atrous ||
        !pipeline_compositing || !pipeline_interleave || !pipeline_taa || !pipeline_god_rays ||
        !pipeline_god_rays_filter || !pipeline_fsr_easu || !pipeline_fsr_rcas)
        return false;

    if (!create_shadow_map_resources())
        return false;

    if (!load_blue_noise()) {
        Com_EPrintf("Metal: could not load the blue noise textures\n");
        return false;
    }

    cvar_flt_fsr_enable = Cvar_Get("flt_fsr_enable", "0", CVAR_ARCHIVE);
    cvar_flt_fsr_easu = Cvar_Get("flt_fsr_easu", "1", CVAR_ARCHIVE);
    cvar_flt_fsr_rcas = Cvar_Get("flt_fsr_rcas", "1", CVAR_ARCHIVE);
    cvar_flt_fsr_sharpness = Cvar_Get("flt_fsr_sharpness", "0.2", CVAR_ARCHIVE);
    cvar_pt_waterwarp = Cvar_Get("pt_waterwarp", "0", CVAR_ARCHIVE);
    cvar_pt_accumulation_rendering = Cvar_Get("pt_accumulation_rendering", "1", CVAR_ARCHIVE);
    cvar_pt_accumulation_rendering_framenum = Cvar_Get("pt_accumulation_rendering_framenum", "500", 0);
    cvar_pt_bsp_sky_lights = Cvar_Get("pt_bsp_sky_lights", "1", 0);
    vkpt_fog_init();

    cvar_gr_enable = Cvar_Get("gr_enable", "1", 0);
    cvar_gr_intensity = Cvar_Get("gr_intensity", "2.0", 0);
    cvar_gr_eccentricity = Cvar_Get("gr_eccentricity", "0.75", 0);
    // Metal only: caps the shadow map taps per half-res pixel on long view
    // rays; vkpt marches uncapped in 5..20 unit steps.
    cvar_gr_max_steps = Cvar_Get("mtl_gr_max_steps", "512", 0);
    cvar_physical_sky_space = Cvar_Get("physical_sky_space", "0", 0);

    if (!mtl_bloom_init())
        return false;
    if (!mtl_tone_mapping_init())
        return false;

    for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++) {
        uniform_buffers[i] = [mtl.device newBufferWithLength:sizeof(GlobalUbo)
                                                     options:MTLResourceStorageModeShared];
        entity_table_buffers[i] = [mtl.device newBufferWithLength:sizeof(VkptEntitySlot) * VKPT_ENTITY_TABLE_SIZE
                                                          options:MTLResourceStorageModeShared];
        if (!uniform_buffers[i] || !entity_table_buffers[i]) {
            Com_EPrintf("Metal: could not allocate path tracer buffers\n");
            return false;
        }
        uniform_buffers[i].label = @"global ubo";
        entity_table_buffers[i].label = @"entity table";
        memset(entity_table_buffers[i].contents, 0, entity_table_buffers[i].length);
    }

    for (int i = 0; i < 2; i++) {
        image_tables[i] = [mtl.device newBufferWithLength:sizeof(VkptImages) options:MTLResourceStorageModeShared];
        image_tables[i].label = @"screen image table";
    }
    light_counts_buffer = [mtl.device newBufferWithLength:sizeof(uint32_t) * MAX_LIGHT_LISTS * LIGHT_COUNT_HISTORY
                                                  options:MTLResourceStorageModeShared];
    light_counts_buffer.label = @"light counts history";
    memset(light_counts_buffer.contents, 0, light_counts_buffer.length);
    dummy_buffer = [mtl.device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    dummy_buffer.label = @"placeholder";
    memset(dummy_buffer.contents, 0, dummy_buffer.length);

    {
        MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
        desc.textureType = MTLTextureType2DArray;
        desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
        desc.width = desc.height = 1;
        desc.arrayLength = 6;
        desc.storageMode = MTLStorageModeShared;
        desc.usage = MTLTextureUsageShaderRead;
        dummy_sky_texture = [mtl.device newTextureWithDescriptor:desc];
        [desc release];
        const uint32_t black = 0;
        for (int i = 0; i < 6; i++)
            [dummy_sky_texture replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 slice:i
                                   withBytes:&black bytesPerRow:4 bytesPerImage:0];
    }

    for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++) {
        entity_vertex_buffers[i] = [mtl.device newBufferWithLength:sizeof(MTLTriVertex) * MTL_MAX_ENTITY_VERTICES
                                                           options:MTLResourceStorageModeShared];
        entity_index_buffers[i] = [mtl.device newBufferWithLength:sizeof(uint32_t) * MTL_MAX_ENTITY_INDICES
                                                          options:MTLResourceStorageModeShared];
        instance_buffers[i] = [mtl.device newBufferWithLength:sizeof(MTLAccelerationStructureInstanceDescriptor) * MTL_TLAS_INSTANCES
                                                     options:MTLResourceStorageModeShared];
        if (!entity_vertex_buffers[i] || !entity_index_buffers[i] || !instance_buffers[i]) {
            Com_EPrintf("Metal: could not allocate entity geometry buffers\n");
            return false;
        }
        entity_vertex_buffers[i].label = @"entity vertices";
        entity_index_buffers[i].label = @"entity indices";
        instance_buffers[i].label = @"tlas instances";

        effect_vertex_buffers[i] = [mtl.device newBufferWithLength:sizeof(mtl_packed_float3_t) * MTL_MAX_EFFECT_VERTICES
                                                           options:MTLResourceStorageModeShared];
        effect_attr_buffers[i] = [mtl.device newBufferWithLength:sizeof(MTLEffectVertex) * MTL_MAX_EFFECT_VERTICES
                                                         options:MTLResourceStorageModeShared];
        effect_index_buffers[i] = [mtl.device newBufferWithLength:sizeof(uint32_t) * 3 * MTL_MAX_EFFECT_TRIANGLES
                                                          options:MTLResourceStorageModeShared];
        effect_prim_buffers[i] = [mtl.device newBufferWithLength:sizeof(MTLEffectPrim) * MTL_MAX_EFFECT_TRIANGLES
                                                         options:MTLResourceStorageModeShared];
        effect_instance_buffers[i] = [mtl.device newBufferWithLength:sizeof(MTLAccelerationStructureInstanceDescriptor)
                                                             options:MTLResourceStorageModeShared];
        if (!effect_vertex_buffers[i] || !effect_attr_buffers[i] || !effect_index_buffers[i] ||
            !effect_prim_buffers[i] || !effect_instance_buffers[i]) {
            Com_EPrintf("Metal: could not allocate effect geometry buffers\n");
            return false;
        }
        effect_vertex_buffers[i].label = @"effect vertices";
        effect_attr_buffers[i].label = @"effect attributes";
        effect_index_buffers[i].label = @"effect indices";
        effect_prim_buffers[i].label = @"effect prims";
    }

    cvar_pt_particle_size = Cvar_Get("pt_particle_size", "0.5", 0);
    cvar_pt_beam_width = Cvar_Get("pt_beam_width", "1.0", 0);
    cvar_pt_enable_particles = Cvar_Get("pt_enable_particles", "1", 0);
    cvar_pt_enable_beams = Cvar_Get("pt_enable_beams", "1", 0);
    cvar_pt_enable_sprites = Cvar_Get("pt_enable_sprites", "1", 0);
    cvar_pt_particle_emissive = Cvar_Get("pt_particle_emissive", "10.0", 0);

    cvar_mtl_hidpi = Cvar_Get("mtl_hidpi", "0", CVAR_ARCHIVE);
    cvar_mtl_hidpi->changed = mtl_hidpi_changed;
    mtl_pt_resize(mtl.width, mtl.height);
    apply_resize();
    return true;
}

#define RELEASE_OBJ(t) do { [(t) release]; (t) = nil; } while (0)

void mtl_pt_shutdown(void)
{
    Cmd_RemoveCommand("camera");
    Cmd_RemoveCommand("mtl_find_texture");
    mtl_pt_free_world();
    mtl_bloom_shutdown();
    mtl_tone_mapping_shutdown();
    vkpt_fog_shutdown();

    RELEASE_OBJ(pipeline_primary_rays);
    RELEASE_OBJ(pipeline_reflect_refract);
    RELEASE_OBJ(pipeline_direct_lighting);
    RELEASE_OBJ(pipeline_indirect_lighting);
    RELEASE_OBJ(pipeline_gradient_reproject);
    RELEASE_OBJ(pipeline_gradient_img);
    RELEASE_OBJ(pipeline_gradient_atrous);
    RELEASE_OBJ(pipeline_temporal);
    RELEASE_OBJ(pipeline_atrous_lf);
    RELEASE_OBJ(pipeline_atrous);
    RELEASE_OBJ(pipeline_compositing);
    RELEASE_OBJ(pipeline_interleave);
    RELEASE_OBJ(pipeline_taa);
    RELEASE_OBJ(pipeline_god_rays);
    RELEASE_OBJ(pipeline_god_rays_filter);
    RELEASE_OBJ(pipeline_shadow_map);
    RELEASE_OBJ(shadow_map_depth_state);
    RELEASE_OBJ(tex_shadow_map);
    RELEASE_OBJ(pipeline_fsr_easu);
    RELEASE_OBJ(pipeline_fsr_rcas);

    release_render_targets();

    RELEASE_OBJ(sky_texture);
    RELEASE_OBJ(blue_noise_texture);
    RELEASE_OBJ(dummy_sky_texture);
    RELEASE_OBJ(dummy_buffer);
    RELEASE_OBJ(light_counts_buffer);
    for (int i = 0; i < 2; i++)
        RELEASE_OBJ(image_tables[i]);

    for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++) {
        RELEASE_OBJ(entity_vertex_buffers[i]);
        RELEASE_OBJ(entity_index_buffers[i]);
        RELEASE_OBJ(instance_buffers[i]);
        RELEASE_OBJ(entity_blas[i]);
        RELEASE_OBJ(entity_scratch[i]);
        RELEASE_OBJ(entity_transparent_blas[i]);
        RELEASE_OBJ(entity_transparent_scratch[i]);
        RELEASE_OBJ(viewer_blas[i]);
        RELEASE_OBJ(weapon_blas[i]);
        RELEASE_OBJ(viewer_scratch[i]);
        RELEASE_OBJ(weapon_scratch[i]);
        RELEASE_OBJ(tlas_scratch[i]);
        RELEASE_OBJ(effect_vertex_buffers[i]);
        RELEASE_OBJ(effect_attr_buffers[i]);
        RELEASE_OBJ(effect_index_buffers[i]);
        RELEASE_OBJ(effect_prim_buffers[i]);
        RELEASE_OBJ(effect_instance_buffers[i]);
        RELEASE_OBJ(effect_blas[i]);
        RELEASE_OBJ(effect_tlas[i]);
        RELEASE_OBJ(effect_scratch[i]);
        RELEASE_OBJ(effect_tlas_scratch[i]);
        RELEASE_OBJ(uniform_buffers[i]);
        RELEASE_OBJ(entity_table_buffers[i]);
    }
}

static id<MTLTexture> create_target(int width, int height, MTLPixelFormat format, const char *label)
{
    MTLTextureDescriptor *desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    desc.storageMode = MTLStorageModePrivate;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

    id<MTLTexture> tex = [mtl.device newTextureWithDescriptor:desc];
    tex.label = [NSString stringWithUTF8String:label];
    return tex;
}

static void release_render_targets(void)
{
    for (int i = 0; i < VKPT_NUM_IMAGES; i++)
        RELEASE_OBJ(vkpt_images[i]);
    RELEASE_OBJ(tex_god_rays);
    RELEASE_OBJ(fsr_easu_output);
    RELEASE_OBJ(fsr_rcas_output);
}

// Fills the two image tables: parity p sees textures X_A ^ p as X_A and
// X_B ^ p as X_B (vkpt's qvk_get_current_desc_set_textures()).
static void write_image_tables(void)
{
    for (int parity = 0; parity < 2; parity++) {
        VkptImages *t = (VkptImages *)image_tables[parity].contents;
#define SET_FIELDS_float(field, tex) t->field##_r = t->field##_w = t->field##_s = (tex).gpuResourceID;
#define SET_FIELDS_uint(field, tex)  t->field##_r = t->field##_w = (tex).gpuResourceID;
#define IMG_DO(name, fmt, type, size) SET_FIELDS_##type(name, vkpt_images[VKPT_IMG_##name])
#define IMG_AB(name, fmt, type, size) \
        SET_FIELDS_##type(name##_A, vkpt_images[VKPT_IMG_##name##_A + parity]) \
        SET_FIELDS_##type(name##_B, vkpt_images[VKPT_IMG_##name##_B - parity])
        VKPT_LIST_IMAGES
        VKPT_LIST_IMAGES_A_B
#undef IMG_DO
#undef IMG_AB
#undef SET_FIELDS_float
#undef SET_FIELDS_uint
    }
}

// New images hold whatever the allocator left in memory, possibly NaNs, and
// the first frames read history from them, so they start out zeroed.
static void clear_images(void)
{
    size_t max_bytes = 0;
    for (int i = 0; i < VKPT_NUM_IMAGES; i++) {
        size_t bpp = 16;
        switch (vkpt_images[i].pixelFormat) {
        case MTLPixelFormatR16Uint: case MTLPixelFormatR16Float: case MTLPixelFormatRG8Unorm: bpp = 2; break;
        case MTLPixelFormatR32Uint: case MTLPixelFormatRG16Float: bpp = 4; break;
        case MTLPixelFormatRG32Uint: case MTLPixelFormatRG32Float: case MTLPixelFormatRGBA16Float: bpp = 8; break;
        default: bpp = 16; break;
        }
        max_bytes = max(max_bytes, bpp * vkpt_images[i].width * vkpt_images[i].height);
    }

    id<MTLBuffer> zeros = [mtl.device newBufferWithLength:max(max_bytes, (size_t)16) options:MTLResourceStorageModePrivate];
    id<MTLCommandBuffer> cmd = [mtl.queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit fillBuffer:zeros range:NSMakeRange(0, zeros.length) value:0];
    for (int i = 0; i < VKPT_NUM_IMAGES; i++) {
        id<MTLTexture> t = vkpt_images[i];
        size_t bpp = 16;
        switch (t.pixelFormat) {
        case MTLPixelFormatR16Uint: case MTLPixelFormatR16Float: case MTLPixelFormatRG8Unorm: bpp = 2; break;
        case MTLPixelFormatR32Uint: case MTLPixelFormatRG16Float: bpp = 4; break;
        case MTLPixelFormatRG32Uint: case MTLPixelFormatRG32Float: case MTLPixelFormatRGBA16Float: bpp = 8; break;
        default: break;
        }
        size_t bytes_per_row = bpp * t.width;
        [blit copyFromBuffer:zeros
                sourceOffset:0
           sourceBytesPerRow:bytes_per_row
         sourceBytesPerImage:bytes_per_row * t.height
                  sourceSize:MTLSizeMake(t.width, t.height, 1)
                   toTexture:t
            destinationSlice:0
            destinationLevel:0
           destinationOrigin:MTLOriginMake(0, 0, 0)];
    }
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    [zeros release];
}

// The 3D view's base resolution, before the dynamic scale: the window in
// points unless mtl_hidpi is set. Tracing a Retina backing store at full
// resolution costs about four times the pixels for a sharpness the TAAU /
// FSR / Lanczos upscale to the display mostly recovers anyway.
static void base_resolution(int display_w, int display_h, int *w, int *h)
{
    double scale = (mtl.layer && !cvar_mtl_hidpi->integer) ? mtl.layer.contentsScale : 1.0;
    if (scale < 1.0)
        scale = 1.0;
    *w = max(1, (int)lround(display_w / scale));
    *h = max(1, (int)lround(display_h / scale));
}

// Reallocates the screen images for the current display size. Frames in
// flight still read the old images through the image tables, which do not
// retain them, so the GPU is drained first.
static void apply_resize(void)
{
    resize_pending = false;
    if (display_width <= 0 || display_height <= 0)
        return;

    int width, height;
    base_resolution(display_width, display_height, &width, &height);

    // The checkerboard fields split the width in half, so it stays even.
    int image_width = (width + 1) & ~1;
    bool images_match = vkpt_images[0] && screen_image_width == image_width && screen_image_height == height;
    bool fsr_match = fsr_easu_output && (int)fsr_easu_output.width == display_width && (int)fsr_easu_output.height == display_height;
    unscaled_width = width;
    unscaled_height = height;
    if (images_match && fsr_match)
        return;

    if (vkpt_images[0]) {
        id<MTLCommandBuffer> drain = [mtl.queue commandBuffer];
        [drain commit];
        [drain waitUntilCompleted];
    }

    release_render_targets();

    screen_image_width = image_width;
    screen_image_height = height;
    int grad_width = (image_width + GRAD_DWN - 1) / GRAD_DWN;
    int grad_height = (height + GRAD_DWN - 1) / GRAD_DWN;

#define SIZE_W_FULL image_width
#define SIZE_H_FULL height
#define SIZE_W_GRAD grad_width
#define SIZE_H_GRAD grad_height
#define IMG_DO(name, fmt, type, size) \
    vkpt_images[VKPT_IMG_##name] = create_target(SIZE_W_##size, SIZE_H_##size, MTLPixelFormat##fmt, #name);
#define IMG_AB(name, fmt, type, size) \
    vkpt_images[VKPT_IMG_##name##_A] = create_target(SIZE_W_##size, SIZE_H_##size, MTLPixelFormat##fmt, #name "_A"); \
    vkpt_images[VKPT_IMG_##name##_B] = create_target(SIZE_W_##size, SIZE_H_##size, MTLPixelFormat##fmt, #name "_B");
    VKPT_LIST_IMAGES
    VKPT_LIST_IMAGES_A_B
#undef IMG_DO
#undef IMG_AB
#undef SIZE_W_FULL
#undef SIZE_H_FULL
#undef SIZE_W_GRAD
#undef SIZE_H_GRAD

    // Covers the whole god rays dispatch (8x8 groups per 16x16 pixels).
    tex_god_rays = create_target((image_width + 15) / 16 * 8, (height + 15) / 16 * 8, MTLPixelFormatRGBA16Float,
                                 "god rays intermediate");
    // FSR upscales to the display, past the base resolution.
    fsr_easu_output = create_target(display_width, display_height, MTLPixelFormatRGBA16Float, "fsr easu");
    fsr_rcas_output = create_target(display_width, display_height, MTLPixelFormatRGBA16Float, "fsr rcas");

    for (int i = 0; i < VKPT_NUM_IMAGES; i++) {
        if (!vkpt_images[i]) {
            Com_EPrintf("Metal: could not allocate the screen images\n");
            release_render_targets();
            return;
        }
    }

    write_image_tables();
    clear_images();
    mtl_bloom_resize(width, height);

    temporal_frame_valid = false;
    Com_Printf("Metal: tracing at %dx%d for a %dx%d display\n", width, height, display_width, display_height);
}

// The display (drawable) size changed. Applied at the start of the next
// traced frame, outside of any frame's command recording.
void mtl_pt_resize(int width, int height)
{
    if (width <= 0 || height <= 0)
        return;
    // SDL reports a size change on several events; only a real change needs
    // the GPU drained and the screen images reallocated.
    if (width == display_width && height == display_height)
        return;
    display_width = width;
    display_height = height;
    resize_pending = true;
}

static void mtl_hidpi_changed(cvar_t *self)
{
    resize_pending = true;
}

id<MTLTexture> mtl_pt_output_texture(void)
{
    if (fsr_active)
        return cvar_flt_fsr_rcas->integer ? fsr_rcas_output : fsr_easu_output;
    return vkpt_images[VKPT_IMG_TAA_OUTPUT];
}

//
// World geometry extraction
//

static bool face_is_drawable(const mface_t *surf)
{
    if (!surf->texinfo)
        return false;
    if ((surf->texinfo->c.flags & SURF_NODRAW) && cvar_pt_enable_nodraw->integer)
        return false;
    if (surf->numsurfedges < 3)
        return false;

    return true;
}

static uint32_t material_flags_for(const mtexinfo_t *texinfo)
{
    uint32_t flags = 0;

    if (texinfo->c.flags & SURF_SKY)
        flags |= MTL_MATERIAL_FLAG_SKY;
    if (texinfo->c.flags & SURF_LIGHT)
        flags |= MTL_MATERIAL_FLAG_LIGHT;
    if (texinfo->c.flags & SURF_WARP)
        flags |= MTL_MATERIAL_FLAG_WARP;
    if (texinfo->c.flags & SURF_TRANS_MASK)
        flags |= MTL_MATERIAL_FLAG_TRANS;

    return flags;
}

// The .mat file assigns a kind per texture, but the same texture is reused with
// different surface flags, so the kind has to be reconciled with them. This
// mirrors the fixups vkpt's bsp_mesh.c applies while building its meshes.
static uint32_t material_kind_for(const mtexinfo_t *texinfo, const pbr_material_t *material)
{
    if (!material)
        return MATERIAL_KIND_REGULAR;

    uint32_t id = material->flags;
    uint32_t kind = id & MATERIAL_KIND_MASK;
    int surf_flags = texinfo->c.flags;

    if ((kind == MATERIAL_KIND_WATER || kind == MATERIAL_KIND_SLIME) && !(surf_flags & SURF_WARP))
        kind = MATERIAL_KIND_REGULAR;

    if (kind == MATERIAL_KIND_GLASS && !(surf_flags & SURF_TRANS_MASK))
        kind = MATERIAL_KIND_REGULAR;

    if (surf_flags & SURF_SKY)
        kind = MATERIAL_KIND_SKY;

    if (kind == MATERIAL_KIND_REGULAR && (surf_flags & SURF_TRANS_MASK) &&
        !(id & MATERIAL_FLAG_LIGHT))
        kind = MATERIAL_KIND_TRANSPARENT;

    if (kind == MATERIAL_KIND_SCREEN && (surf_flags & SURF_TRANS_MASK))
        kind = MATERIAL_KIND_GLASS;

    id = (id & ~MATERIAL_KIND_MASK) | kind;

    if (surf_flags & SURF_WARP)
        id |= MATERIAL_FLAG_WARP;
    if (surf_flags & SURF_FLOWING)
        id |= MATERIAL_FLAG_FLOWING;

    return id;
}

static float srgb_to_linear(byte value)
{
    float x = value / 255.0f;
    if (x <= 0.04045f)
        return x / 12.92f;
    return powf((x + 0.055f) / 1.055f, 2.4f);
}

// Average colour of an image in linear space, used to tint the emission of a
// light surface and to derive the sky light radiance.
static void average_image_color(const image_t *image, vec3_t out)
{
    VectorSet(out, 1.0f, 1.0f, 1.0f);

    if (!image || !image->pix_data || image->pixel_format != PF_R8G8B8A8_UNORM)
        return;

    int width = image->upload_width;
    int height = image->upload_height;
    if (width <= 0 || height <= 0)
        return;

    // Sample a bounded grid; light textures are small but this keeps map load
    // time independent of texture resolution.
    const int max_samples = 64;
    int step_x = max(1, width / max_samples);
    int step_y = max(1, height / max_samples);

    double sum[3] = { 0, 0, 0 };
    int count = 0;

    for (int y = 0; y < height; y += step_y) {
        const byte *row = image->pix_data + (size_t)y * width * 4;
        for (int x = 0; x < width; x += step_x) {
            const byte *px = row + (size_t)x * 4;
            // The textures are sRGB encoded; radiance has to be linear.
            for (int c = 0; c < 3; c++)
                sum[c] += srgb_to_linear(px[c]);
            count++;
        }
    }

    if (!count)
        return;

    for (int c = 0; c < 3; c++)
        out[c] = (float)(sum[c] / count);

    // Keep pitch black light textures from killing the light entirely.
    if (VectorLength(out) < 0.01f)
        VectorSet(out, 1.0f, 1.0f, 1.0f);
}

// Registers one material per texinfo and returns the material buffer.
// Registers one PBR material per texinfo, resolving the base and emissive
// textures the material system found for it.
static void build_materials(bsp_t *bsp)
{
    // Two copies of the world materials: [0, numtexinfo) advance with the
    // 0.5 s texture animation (animate_materials.comp), [numtexinfo, 2n) stay
    // on their base frame for inline models, whose texture is picked by the
    // entity frame instead (animate_material()). Entity materials follow.
    world_material_count = bsp->numtexinfo * 2;

    size_t count = (size_t)max(world_material_count, 1) + MTL_MAX_ENTITY_MATERIALS;
    size_t size = sizeof(MTLMaterial) * count;

    for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++) {
        [material_buffers[i] release];
        material_buffers[i] = [mtl.device newBufferWithLength:size options:MTLResourceStorageModeShared];
        material_buffers[i].label = @"materials";
        memset(material_buffers[i].contents, 0, size);
    }

    // The world portion is identical in every slot; only the entity slots that
    // follow it are rewritten per frame.
    MTLMaterial *materials = (MTLMaterial *)material_buffers[0].contents;

    texinfo_inv_size = Z_Mallocz(sizeof(vec2_t) * max(bsp->numtexinfo, 1));
    texinfo_radiance = Z_Mallocz(sizeof(vec3_t) * max(bsp->numtexinfo, 1));

    float radiance_scale = mtl_pt_light_scale->value;

    for (int i = 0; i < bsp->numtexinfo; i++) {
        mtexinfo_t *texinfo = &bsp->texinfo[i];
        MTLMaterial *mat = &materials[i];

        imageflags_t flags = (texinfo->c.flags & SURF_WARP) ? IF_TURBULENT : IF_NONE;

        char path[MAX_QPATH];
        Q_concat(path, sizeof(path), "textures/", texinfo->name, ".wal");
        FS_NormalizePath(path);

        pbr_material_t *material = MAT_Find(path, IT_WALL, flags);
        texinfo->material = material;

        image_t *image = material ? material->image_base : NULL;

        texinfo_inv_size[i][0] = 1.0f / 64.0f;
        texinfo_inv_size[i][1] = 1.0f / 64.0f;
        if (material && material->original_width > 0 && material->original_height > 0) {
            texinfo_inv_size[i][0] = 1.0f / material->original_width;
            texinfo_inv_size[i][1] = 1.0f / material->original_height;
        } else if (image && image->width > 0 && image->height > 0) {
            texinfo_inv_size[i][0] = 1.0f / image->width;
            texinfo_inv_size[i][1] = 1.0f / image->height;
        }

        mat->base_texture = image ? (uint32_t)(image - r_images) : 0;

        // Resolved into bindless indices later, once uploads have happened.
        mat->normal_texture = (material && material->image_normals)
                            ? (uint32_t)(material->image_normals - r_images) : 0;
        mat->emissive_texture = (material && material->image_emissive)
                              ? (uint32_t)(material->image_emissive - r_images) : 0;
        // pad0 carries the alpha test mask (the shader's mask_texture).
        mat->pad0 = (material && material->image_mask)
                  ? (uint32_t)(material->image_mask - r_images) : 0;
        mat->bump_scale = material ? material->bump_scale : 1.0f;
        mat->specular_factor = material ? material->specular_factor : 0.0f;
        mat->kind_flags = material_kind_for(texinfo, material);

        float base_factor = material ? material->base_factor : 1.0f;
        mat->base_color.x = mat->base_color.y = mat->base_color.z = base_factor;
        // Negative means "take roughness from the base texture alpha"; the
        // override only ever raises it, as in the Vulkan backend.
        mat->roughness = material ? material->roughness_override : -1.0f;
        mat->metalness = material ? material->metalness_factor : 0.0f;
        mat->flags = material_flags_for(texinfo);

        // Emission comes from the material's emissive image where one exists.
        // SURF_LIGHT surfaces without one fall back to the average texture
        // colour, except water/slime: vkpt never makes warp surfaces glow
        // unless pt_enable_surface_lights_warp is set.
        vec3_t emissive;
        VectorClear(emissive);

        bool is_warp = (texinfo->c.flags & SURF_WARP) != 0;
        if (material && material->image_emissive) {
            VectorCopy(material->image_emissive->light_color, emissive);
        } else if ((texinfo->c.flags & SURF_LIGHT) && (!is_warp || cvar_pt_enable_surface_lights_warp->integer)) {
            average_image_color(image, emissive);
        }

        if (!VectorEmpty(emissive)) {
            // Port of vkpt's compute_emissive(): only genuine SURF_LIGHT faces
            // use the BSP radiance, everything else uses the material's own
            // default, and the radiance scale applies to the BSP value alone.
            float radiance;
            if (material) {
                radiance = ((texinfo->c.flags & SURF_LIGHT) && material->bsp_radiance)
                         ? (float)texinfo->radiance * radiance_scale
                         : material->default_radiance;
                radiance *= material->emissive_factor;
            } else {
                radiance = (float)texinfo->radiance * radiance_scale;
            }

            VectorScale(emissive, radiance, texinfo_radiance[i]);

            mat->emissive.x = texinfo_radiance[i][0];
            mat->emissive.y = texinfo_radiance[i][1];
            mat->emissive.z = texinfo_radiance[i][2];
            mat->emissive_factor = radiance;
        }
    }
}

// Resolves material texture handles into bindless table indices, then mirrors
// the finished world materials into every frame slot. Must run after the
// textures have actually been uploaded.
static void resolve_material_textures(int numtexinfo)
{
    if (!material_buffers[0])
        return;

    MTLMaterial *materials = (MTLMaterial *)material_buffers[0].contents;
    for (int i = 0; i < numtexinfo; i++) {
        materials[i].base_texture = mtl_texture_index_for_handle((qhandle_t)materials[i].base_texture);
        // A zero handle means the material had no such map; keep it at zero so
        // the shader can tell "absent" from "white".
        materials[i].normal_texture = materials[i].normal_texture
            ? mtl_texture_index_optional((qhandle_t)materials[i].normal_texture) : 0;
        materials[i].emissive_texture = materials[i].emissive_texture
            ? mtl_texture_index_optional((qhandle_t)materials[i].emissive_texture) : 0;
        materials[i].pad0 = materials[i].pad0
            ? mtl_texture_index_optional((qhandle_t)materials[i].pad0) : 0;
    }
    memcpy(materials + numtexinfo, materials, sizeof(MTLMaterial) * numtexinfo);

    for (int i = 1; i < MTL_FRAMES_IN_FLIGHT; i++)
        memcpy(material_buffers[i].contents, materials, sizeof(MTLMaterial) * numtexinfo * 2);

    Z_Free(world_materials_base);
    world_materials_base = Z_Malloc(sizeof(MTLMaterial) * max(numtexinfo, 1));
    memcpy(world_materials_base, materials, sizeof(MTLMaterial) * numtexinfo);

    Z_Free(texinfo_anim_current);
    texinfo_anim_current = Z_Malloc(sizeof(int) * max(numtexinfo, 1));
    for (int i = 0; i < numtexinfo; i++)
        texinfo_anim_current[i] = i;
    world_anim_frame = -1;
    material_slots_dirty = 0;
}

// Steps every animated texinfo to its next frame and pushes the new emissive
// values into the light polygons. The material buffers pick the change up
// over the next MTL_FRAMES_IN_FLIGHT frames so in-flight slots stay intact.
static void advance_material_animation(void)
{
    if (!world_bsp || !texinfo_anim_current)
        return;

    bool changed = false;
    for (int i = 0; i < world_bsp->numtexinfo; i++) {
        const mtexinfo_t *texinfo = &world_bsp->texinfo[i];
        if (texinfo->numframes <= 1)
            continue;
        const mtexinfo_t *current = &world_bsp->texinfo[texinfo_anim_current[i]];
        if (!current->next)
            continue;
        texinfo_anim_current[i] = (int)(current->next - world_bsp->texinfo);
        changed = true;
    }

    if (!changed)
        return;

    material_slots_dirty = MTL_FRAMES_IN_FLIGHT;

    if (light_poly_buffer && light_poly_material) {
        MTLLightPoly *polys = (MTLLightPoly *)light_poly_buffer.contents;
        for (uint32_t p = 0; p < num_light_polys; p++) {
            uint32_t m = light_poly_material[p];
            if (m == ~0u || world_bsp->texinfo[m].numframes <= 1)
                continue;
            const float *radiance = texinfo_radiance[texinfo_anim_current[m]];
            polys[p].radiance.x = radiance[0];
            polys[p].radiance.y = radiance[1];
            polys[p].radiance.z = radiance[2];
        }
    }
}

static void apply_material_animation(uint32_t slot)
{
    if (material_slots_dirty <= 0 || !world_materials_base || !material_buffers[slot])
        return;

    MTLMaterial *materials = (MTLMaterial *)material_buffers[slot].contents;
    for (int i = 0; i < world_bsp->numtexinfo; i++)
        materials[i] = world_materials_base[texinfo_anim_current[i]];

    material_slots_dirty--;
}

// Entity materials need somewhere to live even when no map is loaded (the
// player setup view from the main menu); registering a world replaces this.
static void ensure_entity_materials(void)
{
    if (material_buffers[0])
        return;

    world_material_count = 0;
    size_t size = sizeof(MTLMaterial) * MTL_MAX_ENTITY_MATERIALS;
    for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++) {
        material_buffers[i] = [mtl.device newBufferWithLength:size options:MTLResourceStorageModeShared];
        material_buffers[i].label = @"materials";
        memset(material_buffers[i].contents, 0, size);
    }
}

// Port of vkpt_physical_sky_update_ubo()'s pt_env_scale.
static float mtl_pt_env_scale(void)
{
    float brightness = Q_clipf(cvar_physical_sky_brightness->value, -10.0f, 2.0f);
    return exp2f(brightness - 2.0f);
}

// Gathers every emissive triangle so the tracer can sample map lighting
// directly instead of relying on a bounce ray finding a light surface. Sky
// surfaces are included as area lights, which is how open areas get lit.

static void load_sky_and_lava_clusters(const char *map_name)
{
    num_sky_clusters = 0;
    all_lava_emissive = false;
    have_sky_cluster_list = false;

    // The shareware demo maps reuse the retail data files.
    const char *full_name = map_name;
    if (!strcmp(map_name, "demo1")) full_name = "base1";
    else if (!strcmp(map_name, "demo2")) full_name = "base2";
    else if (!strcmp(map_name, "demo3")) full_name = "base3";

    char filename[MAX_QPATH];
    Q_snprintf(filename, sizeof(filename), "maps/sky/%s.txt", full_name);

    char *filebuf = NULL;
    FS_LoadFile(filename, (void **)&filebuf);
    if (!filebuf) {
        Com_DPrintf("Couldn't read %s\n", filename);
        return;
    }
    have_sky_cluster_list = true;

    const char *ptr = filebuf;
    char linebuf[1024];
    while (sgets(linebuf, sizeof(linebuf), &ptr)) {
        char *t = strchr(linebuf, '#'); if (t) *t = 0;
        t = strchr(linebuf, '\n'); if (t) *t = 0;

        const char *delimiters = " \t\r\n";
        for (const char *word = strtok(linebuf, delimiters); word; word = strtok(NULL, delimiters)) {
            if (!strcmp(word, "!all_lava")) {
                all_lava_emissive = true;
            } else if (num_sky_clusters < MTL_MAX_SKY_CLUSTERS) {
                sky_clusters[num_sky_clusters++] = Q_atoi(word);
            }
        }
    }
    Z_Free(filebuf);
}

// Decides whether one sky/lava triangle becomes an area light; port of the
// test in bsp_mesh.c collect_light_polys(). Without a cluster list for the
// map every sky face is used, which is what lights maps that lack one.
static bool sky_or_lava_tri_is_light(const bsp_t *bsp, const mtexinfo_t *texinfo,
                                     const vec3_t centroid, const vec3_t normal, bool is_sky)
{
    int cluster = BSP_PointLeaf(bsp->nodes, centroid)->cluster;
    if (cluster < 0)
        return false;

    if (!is_sky) {
        // Lava: only downward facing planes when the map asks for all lava.
        return all_lava_emissive && normal[2] < 0.0f;
    }

    if (!have_sky_cluster_list)
        return true;

    for (int i = 0; i < num_sky_clusters; i++)
        if (sky_clusters[i] == cluster)
            return true;

    // SURF_LIGHT sky faces are lights when pt_bsp_sky_lights allows them.
    bool is_light = (texinfo->c.flags & SURF_LIGHT) != 0;
    bool is_nodraw = (texinfo->c.flags & SURF_NODRAW) != 0;
    return cvar_pt_bsp_sky_lights->integer && is_light &&
           (cvar_pt_bsp_sky_lights->integer > 1 || !is_nodraw);
}
static void build_light_polys(bsp_t *bsp, const MTLTriVertex *verts,
                              const uint32_t *indices, uint32_t index_count)
{
    const MTLMaterial *materials = (const MTLMaterial *)material_buffers[0].contents;

    // Radiance of sky area lights. This has to be the same value the camera
    // sees through a sky surface, or the lighting and the visible sky disagree.
    vec3_t sky_radiance;
    VectorSet(sky_radiance, 0.5f, 0.6f, 0.8f);

    if (mtl_physical_sky_active()) {
        mtl_physical_sky_average_color(sky_radiance);
        VectorScale(sky_radiance, cvar_sky_brightness->value * mtl_pt_env_scale(), sky_radiance);
    } else {
        if (!VectorEmpty(sky_average_color))
            VectorCopy(sky_average_color, sky_radiance);
        VectorScale(sky_radiance,
                    cvar_sky_brightness->value * ubo_cvar_pt_envmap_brightness->value * mtl_pt_env_scale(),
                    sky_radiance);
    }

    uint32_t count = 0;
    for (uint32_t i = 0; i + 2 < index_count; i += 3) {
        uint32_t material = verts[indices[i]].material & MTL_VERTEX_MATERIAL_MASK;
        if (material >= (uint32_t)bsp->numtexinfo)
            continue;
        if (materials[material].flags & MTL_MATERIAL_FLAG_SKY) {
            count++;
        } else if (!VectorEmpty(texinfo_radiance[material])) {
            count++;
        }
    }

    num_light_polys = 0;
    if (!count) {
        // Metal still needs a bound buffer, so keep a one element placeholder.
        light_poly_buffer = [mtl.device newBufferWithLength:sizeof(MTLLightPoly)
                                                    options:MTLResourceStorageModeShared];
        light_poly_buffer.label = @"light polys";
        return;
    }

    light_poly_buffer = [mtl.device newBufferWithLength:sizeof(MTLLightPoly) * count
                                                options:MTLResourceStorageModeShared];
    light_poly_buffer.label = @"light polys";

    Z_Free(light_poly_material);
    light_poly_material = Z_Malloc(sizeof(uint32_t) * count);
    Z_Free(light_poly_cluster);
    light_poly_cluster = Z_Malloc(sizeof(int) * count);

    MTLLightPoly *polys = (MTLLightPoly *)light_poly_buffer.contents;

    uint32_t num_sky_polys = 0;

    for (uint32_t i = 0; i + 2 < index_count; i += 3) {
        const MTLTriVertex *a = &verts[indices[i + 0]];
        const MTLTriVertex *b = &verts[indices[i + 1]];
        const MTLTriVertex *c = &verts[indices[i + 2]];

        uint32_t material = a->material & MTL_VERTEX_MATERIAL_MASK;
        if (material >= (uint32_t)bsp->numtexinfo)
            continue;

        bool is_sky = (materials[material].flags & MTL_MATERIAL_FLAG_SKY) != 0;
        if (!is_sky && VectorEmpty(texinfo_radiance[material]))
            continue;

        vec3_t p0 = { a->position.x, a->position.y, a->position.z };
        vec3_t p1 = { b->position.x, b->position.y, b->position.z };
        vec3_t p2 = { c->position.x, c->position.y, c->position.z };

        vec3_t e0, e1, cross;
        VectorSubtract(p1, p0, e0);
        VectorSubtract(p2, p0, e1);
        CrossProduct(e0, e1, cross);
        float area = VectorLength(cross) * 0.5f;

        if (area < 1.0f)
            continue;

        if (is_sky) {
            vec3_t centroid, n;
            VectorAdd(p0, p1, centroid);
            VectorAdd(centroid, p2, centroid);
            VectorScale(centroid, 1.0f / 3.0f, centroid);
            VectorCopy(cross, n);
            VectorNormalize(n);
            // Nudge into the leaf in front of the face (get_triangle_off_center).
            VectorMA(centroid, 1.0f, n, centroid);
            if (!sky_or_lava_tri_is_light(bsp, &bsp->texinfo[material], centroid, n, true))
                continue;
        }

        light_poly_material[num_light_polys] = is_sky ? ~0u : material;
        light_poly_cluster[num_light_polys] = light_poly_cluster_of(bsp, p0, p1, p2);
        // MATERIAL_FLAG_LIGHT per primitive: bounce rays skip these (NEE has them).
        ((MTLTriVertex *)a)->flags |= MTL_VERTEX_FLAG_LIGHT;
        MTLLightPoly *poly = &polys[num_light_polys++];
        poly->v0 = a->position;
        poly->v1 = b->position;
        poly->v2 = c->position;
        poly->area = area;

        const float *radiance = is_sky ? sky_radiance : texinfo_radiance[material];
        poly->radiance.x = radiance[0];
        poly->radiance.y = radiance[1];
        poly->radiance.z = radiance[2];
        poly->pad0 = is_sky ? 1.0f : 0.0f;   // sky lights take their colour from the sky in the sampled direction
        poly->pad1 = poly->pad2 = 0.0f;

        if (is_sky)
            num_sky_polys++;
    }

    have_sky_lights = num_sky_polys > 0;

    Com_Printf("Metal: %u light polys (%u sky)\n", num_light_polys, num_sky_polys);
}

// Emissive triangles of the inline models, kept in model space so they can
// be placed at the entity's pose every frame (vkpt's instance_model_lights).
static void build_submodel_light_polys(const bsp_t *bsp, const MTLTriVertex *verts, const uint32_t *indices)
{
    submodel_lights = Z_Mallocz(sizeof(*submodel_lights) * max(num_submodels, 1u));

    uint32_t capacity = 0;
    for (uint32_t m = 1; m < num_submodels; m++)
        capacity += submodels[m].index_count / 3;
    if (!capacity)
        return;

    submodel_light_polys = Z_Malloc(sizeof(MTLLightPoly) * capacity);
    submodel_light_poly_material = Z_Malloc(sizeof(uint32_t) * capacity);
    num_submodel_light_polys = 0;

    for (uint32_t m = 1; m < num_submodels; m++) {
        submodel_lights[m].first = num_submodel_light_polys;
        uint32_t end = submodels[m].first_index + submodels[m].index_count;
        for (uint32_t i = submodels[m].first_index; i + 2 < end; i += 3) {
            const MTLTriVertex *a = &verts[indices[i + 0]];
            const MTLTriVertex *b = &verts[indices[i + 1]];
            const MTLTriVertex *c = &verts[indices[i + 2]];
            uint32_t material = a->material & MTL_VERTEX_MATERIAL_MASK;
            if (material >= (uint32_t)bsp->numtexinfo || VectorEmpty(texinfo_radiance[material]))
                continue;
            ((MTLTriVertex *)a)->flags |= MTL_VERTEX_FLAG_LIGHT;

            MTLLightPoly *poly = &submodel_light_polys[num_submodel_light_polys];
            poly->v0 = a->position;
            poly->v1 = b->position;
            poly->v2 = c->position;
            poly->area = 0.0f;   // computed after the transform
            poly->radiance.x = texinfo_radiance[material][0];
            poly->radiance.y = texinfo_radiance[material][1];
            poly->radiance.z = texinfo_radiance[material][2];
            poly->pad0 = poly->pad1 = poly->pad2 = 0.0f;
            submodel_light_poly_material[num_submodel_light_polys++] = material;
        }
        submodel_lights[m].count = num_submodel_light_polys - submodel_lights[m].first;
    }
}

static void allocate_frame_light_buffers(void)
{
    frame_light_poly_capacity = num_light_polys + MTL_MAX_DYNAMIC_LIGHT_POLYS;
    for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++) {
        [frame_light_poly_buffers[i] release];
        frame_light_poly_buffers[i] = [mtl.device newBufferWithLength:sizeof(MTLLightPoly) * frame_light_poly_capacity
                                                             options:MTLResourceStorageModeShared];
        frame_light_poly_buffers[i].label = @"frame light polys";
    }
    allocate_light_list_buffers();
}

//
// Per-cluster light lists (bsp_mesh.c collect_cluster_lights and
// vertex_buffer.c inject_model_lights). Each cluster gets the list of light
// polygons that can be seen from it through the PVS and are not entirely
// behind their emitting plane, which is what makes light sampling tractable
// on maps with hundreds of emissive surfaces.
//

typedef struct { vec3_t mins, maxs; } mtl_aabb_t;

static int          num_clusters;
static mtl_aabb_t  *cluster_aabbs;
static uint32_t    *static_cluster_lights;         // compacted lists
static uint32_t    *static_cluster_light_offsets;  // num_clusters + 1
static id<MTLBuffer> sky_visibility_buffer;
static id<MTLBuffer> light_list_offset_buffers[MTL_FRAMES_IN_FLIGHT];
static id<MTLBuffer> light_list_light_buffers[MTL_FRAMES_IN_FLIGHT];
static uint32_t     light_list_capacity;
static int         *dyn_light_clusters;             // per frame scratch
static uint32_t    *cluster_scratch_a, *cluster_scratch_b;

// Cluster of a point just off a triangle's centre (get_triangle_off_center).
static int triangle_cluster(const bsp_t *bsp, const vec3_t p0, const vec3_t p1, const vec3_t p2)
{
    if (!bsp || !bsp->nodes || !bsp->vis)
        return -1;
    vec3_t e1, e2, normal, center;
    VectorSubtract(p1, p0, e1);
    VectorSubtract(p2, p0, e2);
    CrossProduct(e1, e2, normal);
    VectorNormalize(normal);
    VectorAdd(p0, p1, center);
    VectorAdd(center, p2, center);
    VectorScale(center, 1.0f / 3.0f, center);

    static const float offsets[4] = { 0.01f, 1.0f, -0.01f, -1.0f };
    int cluster = -1;
    for (int k = 0; k < 4 && cluster < 0; k++) {
        vec3_t probe;
        VectorMA(center, offsets[k], normal, probe);
        cluster = BSP_PointLeaf(bsp->nodes, probe)->cluster;
    }
    return cluster;
}

static int light_poly_cluster_of(const bsp_t *bsp, const vec3_t p0, const vec3_t p1, const vec3_t p2)
{
    return triangle_cluster(bsp, p0, p1, p2);
}

static int point_cluster(const vec3_t p)
{
    if (!world_bsp || !world_bsp->nodes || !world_bsp->vis)
        return -1;
    return BSP_PointLeaf(world_bsp->nodes, p)->cluster;
}

static void free_light_lists(void)
{
    Z_Free(cluster_aabbs); cluster_aabbs = NULL;
    Z_Free(static_cluster_lights); static_cluster_lights = NULL;
    Z_Free(static_cluster_light_offsets); static_cluster_light_offsets = NULL;
    Z_Free(dyn_light_clusters); dyn_light_clusters = NULL;
    Z_Free(cluster_scratch_a); cluster_scratch_a = NULL;
    Z_Free(cluster_scratch_b); cluster_scratch_b = NULL;
    [sky_visibility_buffer release]; sky_visibility_buffer = nil;
    for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++) {
        [light_list_offset_buffers[i] release]; light_list_offset_buffers[i] = nil;
        [light_list_light_buffers[i] release]; light_list_light_buffers[i] = nil;
    }
    num_clusters = 0;
    light_list_capacity = 0;
}

// Cluster bounding boxes from the world triangles (compute_cluster_aabbs).
static void compute_cluster_aabbs(const MTLTriVertex *verts, const uint32_t *indices, uint32_t index_count)
{
    cluster_aabbs = Z_Malloc(sizeof(mtl_aabb_t) * max(num_clusters, 1));
    for (int c = 0; c < num_clusters; c++) {
        VectorSet(cluster_aabbs[c].mins, FLT_MAX, FLT_MAX, FLT_MAX);
        VectorSet(cluster_aabbs[c].maxs, -FLT_MAX, -FLT_MAX, -FLT_MAX);
    }
    for (uint32_t i = 0; i < index_count; i++) {
        const MTLTriVertex *v = &verts[indices[i]];
        if (v->cluster < 0 || v->cluster >= num_clusters)
            continue;
        vec3_t p = { v->position.x, v->position.y, v->position.z };
        AddPointToBounds(p, cluster_aabbs[v->cluster].mins, cluster_aabbs[v->cluster].maxs);
    }
}

// light_affects_cluster(): false when the whole cluster is behind the light plane.
static bool light_affects_cluster(const MTLLightPoly *light, const mtl_aabb_t *aabb)
{
    if (aabb->mins[0] > aabb->maxs[0])
        return false;

    vec3_t v0 = { light->v0.x, light->v0.y, light->v0.z };
    vec3_t v1 = { light->v1.x, light->v1.y, light->v1.z };
    vec3_t v2 = { light->v2.x, light->v2.y, light->v2.z };
    vec3_t e1, e2, normal;
    VectorSubtract(v1, v0, e1);
    VectorSubtract(v2, v0, e2);
    CrossProduct(e1, e2, normal);
    VectorNormalize(normal);
    float plane_distance = -DotProduct(normal, v0);

    for (int corner = 0; corner < 8; corner++) {
        vec3_t c = {
            (corner & 1) ? aabb->maxs[0] : aabb->mins[0],
            (corner & 2) ? aabb->maxs[1] : aabb->mins[1],
            (corner & 4) ? aabb->maxs[2] : aabb->mins[2],
        };
        if (DotProduct(normal, c) + plane_distance > 0.0f)
            return true;
    }
    return false;
}

#define FOREACH_PVS_CLUSTER(bsp, mask, var) \
    for (int _byte = 0; _byte < (bsp)->visrowsize; _byte++) \
        for (int _bit = 0; _bit < 8; _bit++) \
            if (((mask)[_byte] & (1 << _bit)) && ((var) = _byte * 8 + _bit) < num_clusters)

static void collect_cluster_lights(const bsp_t *bsp)
{
    free_light_lists();
    if (!bsp->vis || !bsp->nodes)
        return;

    num_clusters = bsp->vis->numclusters;
    if (num_clusters <= 0) {
        num_clusters = 0;
        return;
    }

    compute_cluster_aabbs((const MTLTriVertex *)vertex_buffer.contents, (const uint32_t *)index_buffer.contents, num_indices);

    // Sky visibility: clusters that can see a cluster containing sky.
    {
        byte *with_sky = Z_Mallocz((size_t)bsp->visrowsize);
        const MTLTriVertex *verts = (const MTLTriVertex *)vertex_buffer.contents;
        const uint32_t *indices = (const uint32_t *)index_buffer.contents;
        for (uint32_t i = num_sky_first_index; i < num_indices; i += 3) {
            int c = verts[indices[i]].cluster;
            if (c >= 0 && c < num_clusters)
                with_sky[c >> 3] |= 1 << (c & 7);
        }
        uint32_t words = ((uint32_t)bsp->visrowsize + 3) / 4;
        sky_visibility_buffer = [mtl.device newBufferWithLength:max(words, 1u) * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        sky_visibility_buffer.label = @"sky visibility";
        byte *sky_vis = (byte *)sky_visibility_buffer.contents;
        memset(sky_vis, 0, words * sizeof(uint32_t));
        for (int c = 0; c < num_clusters; c++) {
            if (with_sky[c >> 3] & (1 << (c & 7))) {
                const byte *mask = BSP_GetPvs((bsp_t *)bsp, c);
                for (int i = 0; i < bsp->visrowsize; i++)
                    sky_vis[i] |= mask[i];
            }
        }
        Z_Free(with_sky);
    }

    // Static light lists.
    const MTLLightPoly *polys = (const MTLLightPoly *)light_poly_buffer.contents;
    const uint32_t max_per_cluster = 1024;
    uint32_t *lists = Z_Malloc(sizeof(uint32_t) * max_per_cluster * num_clusters);
    uint32_t *counts = Z_Mallocz(sizeof(uint32_t) * num_clusters);

    for (uint32_t n = 0; n < num_light_polys; n++) {
        int light_cluster = light_poly_cluster[n];
        if (light_cluster < 0 || light_cluster >= num_clusters)
            continue;
        const byte *pvs = BSP_GetPvs((bsp_t *)bsp, light_cluster);
        int other;
        FOREACH_PVS_CLUSTER(bsp, pvs, other) {
            if (!light_affects_cluster(&polys[n], &cluster_aabbs[other]))
                continue;
            if (counts[other] < max_per_cluster)
                lists[other * max_per_cluster + counts[other]++] = n;
        }
    }

    uint32_t total = 0;
    for (int c = 0; c < num_clusters; c++)
        total += counts[c];

    static_cluster_lights = Z_Malloc(sizeof(uint32_t) * max(total, 1u));
    static_cluster_light_offsets = Z_Malloc(sizeof(uint32_t) * (num_clusters + 1));
    uint32_t offset = 0;
    for (int c = 0; c < num_clusters; c++) {
        static_cluster_light_offsets[c] = offset;
        memcpy(static_cluster_lights + offset, lists + (size_t)c * max_per_cluster, sizeof(uint32_t) * counts[c]);
        offset += counts[c];
    }
    static_cluster_light_offsets[num_clusters] = offset;

    Z_Free(lists);
    Z_Free(counts);

    dyn_light_clusters = Z_Malloc(sizeof(int) * MTL_MAX_DYNAMIC_LIGHT_POLYS);
    cluster_scratch_a = Z_Malloc(sizeof(uint32_t) * num_clusters);
    cluster_scratch_b = Z_Malloc(sizeof(uint32_t) * num_clusters);

    Com_Printf("Metal: %d clusters, %u static light interactions, %u faces without cluster\n", num_clusters, total, num_unclustered_faces);
}

static void allocate_light_list_buffers(void)
{
    for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++) {
        [light_list_offset_buffers[i] release];
        [light_list_light_buffers[i] release];
        light_list_offset_buffers[i] = nil;
        light_list_light_buffers[i] = nil;
    }
    if (!num_clusters)
        return;

    // Static lists plus room for every dynamic light in every cluster.
    light_list_capacity = static_cluster_light_offsets[num_clusters] + (uint32_t)num_clusters * MTL_MAX_DYNAMIC_LIGHT_POLYS;
    for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++) {
        light_list_offset_buffers[i] = [mtl.device newBufferWithLength:sizeof(uint32_t) * (num_clusters + 1)
                                                               options:MTLResourceStorageModeShared];
        light_list_offset_buffers[i].label = @"light list offsets";
        light_list_light_buffers[i] = [mtl.device newBufferWithLength:sizeof(uint32_t) * max(light_list_capacity, 1u)
                                                              options:MTLResourceStorageModeShared];
        light_list_light_buffers[i].label = @"light list lights";
    }
}

// Builds this frame's lists: the static ones with the dynamic lights
// (inline models, beams) that are visible from each cluster appended.
static void build_frame_light_lists(uint32_t slot, uint32_t num_dynamic)
{
    if (!num_clusters || !light_list_offset_buffers[slot])
        return;

    uint32_t *offsets = (uint32_t *)light_list_offset_buffers[slot].contents;
    uint32_t *lights = (uint32_t *)light_list_light_buffers[slot].contents;
    uint32_t *local_counts = cluster_scratch_a;   // dynamic lights sitting in each cluster
    uint32_t *dyn_counts = cluster_scratch_b;     // dynamic lights visible from each cluster
    memset(local_counts, 0, sizeof(uint32_t) * num_clusters);
    memset(dyn_counts, 0, sizeof(uint32_t) * num_clusters);

    for (uint32_t n = 0; n < num_dynamic; n++) {
        int c = dyn_light_clusters[n];
        if (c >= 0 && c < num_clusters)
            local_counts[c]++;
    }
    for (int c = 0; c < num_clusters; c++) {
        if (!local_counts[c])
            continue;
        const byte *mask = BSP_GetPvs(world_bsp, c);
        int other;
        FOREACH_PVS_CLUSTER(world_bsp, mask, other)
            dyn_counts[other] += local_counts[c];
    }

    // Offsets, then copy the static lists leaving room at the tail of each.
    uint32_t offset = 0;
    for (int c = 0; c < num_clusters; c++) {
        offsets[c] = offset;
        uint32_t s = static_cluster_light_offsets[c + 1] - static_cluster_light_offsets[c];
        memcpy(lights + offset, static_cluster_lights + static_cluster_light_offsets[c], sizeof(uint32_t) * s);
        offset += s + dyn_counts[c];
    }
    offsets[num_clusters] = offset;

    // This frame's list lengths, for the gradient samples of the next frames.
    if (light_counts_buffer) {
        uint32_t history = (pt_frame_counter & 0x7fffu) % LIGHT_COUNT_HISTORY;
        uint32_t *counts = (uint32_t *)light_counts_buffer.contents + history * MAX_LIGHT_LISTS;
        for (int c = 0; c < num_clusters && c < MAX_LIGHT_LISTS; c++)
            counts[c] = offsets[c + 1] - offsets[c];
    }

    // Append the dynamic lights; dyn_counts becomes the per-cluster cursor.
    for (int c = 0; c < num_clusters; c++)
        dyn_counts[c] = static_cluster_light_offsets[c + 1] - static_cluster_light_offsets[c];
    for (uint32_t n = 0; n < num_dynamic; n++) {
        int c = dyn_light_clusters[n];
        if (c < 0 || c >= num_clusters)
            continue;
        const byte *mask = BSP_GetPvs(world_bsp, c);
        int other;
        FOREACH_PVS_CLUSTER(world_bsp, mask, other)
            lights[offsets[other] + dyn_counts[other]++] = num_light_polys + n;
    }
}

static void transform_point(const vec3_t p, const vec3_t origin, vec3_t axis[3], float scale, mtl_float3 *out)
{
    vec3_t l = { p[0] * scale, p[1] * scale, p[2] * scale };
    out->x = origin[0] + l[0] * axis[0][0] + l[1] * axis[1][0] + l[2] * axis[2][0];
    out->y = origin[1] + l[0] * axis[0][1] + l[1] * axis[1][1] + l[2] * axis[2][1];
    out->z = origin[2] + l[0] * axis[0][2] + l[1] * axis[1][2] + l[2] * axis[2][2];
}

static float poly_area(const MTLLightPoly *poly)
{
    vec3_t p0 = { poly->v0.x, poly->v0.y, poly->v0.z };
    vec3_t p1 = { poly->v1.x, poly->v1.y, poly->v1.z };
    vec3_t p2 = { poly->v2.x, poly->v2.y, poly->v2.z };
    vec3_t e0, e1, cross;
    VectorSubtract(p1, p0, e0);
    VectorSubtract(p2, p0, e1);
    CrossProduct(e0, e1, cross);
    return VectorLength(cross) * 0.5f;
}

// Port of vkpt_build_cylinder_light(): a six triangle prism around the beam.
static void append_cylinder_light(MTLLightPoly *polys, uint32_t *count,
                                  const vec3_t begin, const vec3_t end, const vec3_t color, float radius)
{
    vec3_t dir, norm_dir;
    VectorSubtract(end, begin, dir);
    VectorCopy(dir, norm_dir);
    VectorNormalize(norm_dir);

    vec3_t up = { 0.0f, 0.0f, 1.0f };
    vec3_t left = { 1.0f, 0.0f, 0.0f };
    if (fabsf(norm_dir[2]) < 0.9f) {
        CrossProduct(up, norm_dir, left);
        VectorNormalize(left);
        CrossProduct(norm_dir, left, up);
        VectorNormalize(up);
    } else {
        CrossProduct(norm_dir, left, up);
        VectorNormalize(up);
        CrossProduct(up, norm_dir, left);
        VectorNormalize(left);
    }

    vec3_t vertices[6] = {
        { 0.0f, 1.0f, 0.0f }, { 0.866f, -0.5f, 0.0f }, { -0.866f, -0.5f, 0.0f },
        { 0.0f, -1.0f, 1.0f }, { -0.866f, 0.5f, 1.0f }, { 0.866f, 0.5f, 1.0f },
    };
    static const int indices[18] = { 0, 4, 2, 2, 4, 3, 2, 3, 1, 1, 3, 5, 1, 5, 0, 0, 5, 4 };

    for (int v = 0; v < 6; v++) {
        vec3_t t;
        VectorCopy(begin, t);
        VectorMA(t, vertices[v][0] * radius, up, t);
        VectorMA(t, vertices[v][1] * radius, left, t);
        VectorMA(t, vertices[v][2], dir, t);
        VectorCopy(t, vertices[v]);
    }

    for (int tri = 0; tri < 6 && *count < frame_light_poly_capacity; tri++) {
        MTLLightPoly *poly = &polys[(*count)];
        const float *p0 = vertices[indices[tri * 3 + 0]];
        const float *p1 = vertices[indices[tri * 3 + 1]];
        const float *p2 = vertices[indices[tri * 3 + 2]];
        poly->v0.x = p0[0]; poly->v0.y = p0[1]; poly->v0.z = p0[2];
        poly->v1.x = p1[0]; poly->v1.y = p1[1]; poly->v1.z = p1[2];
        poly->v2.x = p2[0]; poly->v2.y = p2[1]; poly->v2.z = p2[2];
        poly->area = poly_area(poly);
        poly->radiance.x = color[0];
        poly->radiance.y = color[1];
        poly->radiance.z = color[2];
        poly->pad0 = poly->pad1 = poly->pad2 = 0.0f;
        if (poly->area > 0.0f) {
            if (dyn_light_clusters) {
                vec3_t mid;
                VectorAdd(begin, end, mid);
                VectorScale(mid, 0.5f, mid);
                dyn_light_clusters[*count - num_light_polys] = point_cluster(mid);
            }
            (*count)++;
        }
    }
}

// Assembles this frame's light list: static world polys, then the emissive
// inline models at their pose, then beam lights (vkpt_build_beam_lights).
static void build_frame_light_polys(const refdef_t *fd, float adapted_luminance)
{
    uint32_t slot = current_frame_slot;
    num_frame_light_polys = 0;
    if (!frame_light_poly_buffers[slot] || !light_poly_buffer)
        return;

    MTLLightPoly *polys = (MTLLightPoly *)frame_light_poly_buffers[slot].contents;
    memcpy(polys, light_poly_buffer.contents, sizeof(MTLLightPoly) * num_light_polys);
    num_frame_light_polys = num_light_polys;

    for (int i = 0; i < fd->num_entities; i++) {
        const entity_t *ent = &fd->entities[i];
        if (!(ent->model & 0x80000000) || !submodel_lights)
            continue;
        uint32_t m = (uint32_t)~ent->model;
        if (m == 0 || m >= num_submodels || !submodel_lights[m].count)
            continue;

        vec3_t origin, axis[3];
        entity_lerped_origin(ent, origin);
        AnglesToAxis(ent->angles, axis);
        float scale = ent->scale > 0.0f ? ent->scale : 1.0f;

        for (uint32_t p = 0; p < submodel_lights[m].count && num_frame_light_polys < frame_light_poly_capacity; p++) {
            uint32_t src = submodel_lights[m].first + p;
            MTLLightPoly *poly = &polys[num_frame_light_polys];
            *poly = submodel_light_polys[src];
            vec3_t v0 = { poly->v0.x, poly->v0.y, poly->v0.z };
            vec3_t v1 = { poly->v1.x, poly->v1.y, poly->v1.z };
            vec3_t v2 = { poly->v2.x, poly->v2.y, poly->v2.z };
            transform_point(v0, origin, axis, scale, &poly->v0);
            transform_point(v1, origin, axis, scale, &poly->v1);
            transform_point(v2, origin, axis, scale, &poly->v2);
            poly->area = poly_area(poly);
            // Animated materials on doors etc. follow the world's frame state.
            uint32_t material = submodel_light_poly_material[src];
            if (texinfo_anim_current) {
                const float *radiance = texinfo_radiance[texinfo_anim_current[material]];
                poly->radiance.x = radiance[0];
                poly->radiance.y = radiance[1];
                poly->radiance.z = radiance[2];
            }
            if (poly->area >= 1.0f) {
                if (dyn_light_clusters) {
                    vec3_t w0 = { poly->v0.x, poly->v0.y, poly->v0.z };
                    vec3_t w1 = { poly->v1.x, poly->v1.y, poly->v1.z };
                    vec3_t w2 = { poly->v2.x, poly->v2.y, poly->v2.z };
                    dyn_light_clusters[num_frame_light_polys - num_light_polys] = triangle_cluster(world_bsp, w0, w1, w2);
                }
                num_frame_light_polys++;
            }
        }
    }

    // Beam lights.
    float hdr_factor = cvar_pt_beam_lights->value * adapted_luminance * 20.0f;
    if (hdr_factor > 0.0f && cvar_pt_enable_beams->integer) {
        int num_beams = 0;
        for (int i = 0; i < fd->num_entities && num_beams < 64; i++) {
            const entity_t *beam = &fd->entities[i];
            if (!(beam->flags & RF_BEAM) || beam->frame == 0)
                continue;
            num_beams++;

            float beam_radius = cvar_pt_beam_width->value * beam->frame * 0.5f;

            vec3_t begin, end, to_end, norm_dir;
            VectorCopy(beam->oldorigin, begin);
            VectorCopy(beam->origin, end);
            VectorSubtract(end, begin, to_end);
            VectorCopy(to_end, norm_dir);
            if (VectorNormalize(norm_dir) < 1e-3f)
                continue;
            VectorMA(begin, -5.0f, norm_dir, begin);
            VectorMA(end, 5.0f, norm_dir, end);

            MTLEffectPrim tmp;
            effect_color(beam->skinnum, &beam->rgba, hdr_factor, &tmp);
            vec3_t color = { tmp.color.x, tmp.color.y, tmp.color.z };

            append_cylinder_light(polys, &num_frame_light_polys, begin, end, color, beam_radius);
        }
    }

    build_frame_light_lists(slot, num_frame_light_polys - num_light_polys);
}

// Appends one BSP face as a triangle fan to the shared geometry buffers.
static uint32_t float_bits(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

// Translucency of a world surface, as vkpt's bsp_mesh.c derives it: only
// MATERIAL_KIND_TRANSPARENT walls use the SURF_TRANS33/66 flags.
static float surface_alpha(const bsp_t *bsp, const mtexinfo_t *texinfo)
{
    if (!material_buffers[0])
        return 1.0f;
    const MTLMaterial *materials = (const MTLMaterial *)material_buffers[0].contents;
    if ((materials[texinfo - bsp->texinfo].kind_flags & MATERIAL_KIND_MASK) != MATERIAL_KIND_TRANSPARENT)
        return 1.0f;
    if (texinfo->c.flags & SURF_TRANS33)
        return 0.33f;
    if (texinfo->c.flags & SURF_TRANS66)
        return 0.66f;
    return 1.0f;
}

// World geometry that shadow rays must look through (filter_static_transparent).
static bool texinfo_is_transparent(const bsp_t *bsp, const mtexinfo_t *texinfo)
{
    // build_materials() has filled slot 0 by the time geometry is built.
    if (!material_buffers[0])
        return false;
    const MTLMaterial *materials = (const MTLMaterial *)material_buffers[0].contents;
    uint32_t kind = materials[texinfo - bsp->texinfo].kind_flags & MATERIAL_KIND_MASK;
    return kind == MATERIAL_KIND_WATER || kind == MATERIAL_KIND_SLIME ||
           kind == MATERIAL_KIND_GLASS || kind == MATERIAL_KIND_TRANSPARENT;
}

static void append_bsp_face(const bsp_t *bsp, const mface_t *surf,
                            MTLTriVertex *verts, uint32_t *indices,
                            uint32_t *vi_inout, uint32_t *ii_inout)
{
    uint32_t vi = *vi_inout, ii = *ii_inout;

    const mtexinfo_t *texinfo = surf->texinfo;
    uint32_t material = (uint32_t)(texinfo - bsp->texinfo);

    vec3_t normal;
    VectorCopy(surf->plane->normal, normal);
    if (surf->drawflags & DSURF_PLANEBACK)
        VectorNegate(normal, normal);

    // The texture axes are in world units; normalize by the image size so
    // the shader can sample with plain [0,1) wrapping.
    float inv_w = texinfo_inv_size[material][0];
    float inv_h = texinfo_inv_size[material][1];

    // texinfo->axis[0] runs along the texture's U direction, which is
    // exactly the tangent a normal map expects.
    vec3_t tangent;
    VectorCopy(texinfo->axis[0], tangent);
    VectorNormalize(tangent);

    uint32_t first_vertex = vi;

    // BSP cluster of the face, from a point just off its centre on the side
    // the face points to (vkpt's get_triangle_off_center).
    int cluster = -1;
    if (surf->numsurfedges >= 3 && bsp->vis && bsp->nodes) {
        vec3_t center = { 0, 0, 0 };
        for (int e = 0; e < surf->numsurfedges; e++) {
            const msurfedge_t *se = &surf->firstsurfedge[e];
            VectorAdd(center, se->edge->v[se->vert]->point, center);
        }
        VectorScale(center, 1.0f / surf->numsurfedges, center);
        static const float offsets[4] = { 0.01f, 1.0f, -0.01f, -1.0f };
        for (int k = 0; k < 4 && cluster < 0; k++) {
            vec3_t probe;
            VectorMA(center, offsets[k], normal, probe);
            cluster = BSP_PointLeaf(bsp->nodes, probe)->cluster;
        }
        if (cluster < 0)
            num_unclustered_faces++;
    }
    // Each camera screen face picks a random camera (bsp_mesh.c), carried in
    // the top bits of the vertex material index.
    uint32_t camera_bits = 0;
    if (material_buffers[0] && num_cameras) {
        const MTLMaterial *materials = (const MTLMaterial *)material_buffers[0].contents;
        if ((materials[material].kind_flags & MATERIAL_KIND_MASK) == MATERIAL_KIND_CAMERA)
            camera_bits = (uint32_t)(Q_rand() % num_cameras) << MTL_VERTEX_CAMERA_SHIFT;
    }

    for (int e = 0; e < surf->numsurfedges; e++) {
        const msurfedge_t *surfedge = &surf->firstsurfedge[e];
        const float *point = surfedge->edge->v[surfedge->vert]->point;

        MTLTriVertex *v = &verts[vi++];
        v->position.x = point[0];
        v->position.y = point[1];
        v->position.z = point[2];
        v->prev_position = v->position;
        v->normal.x = normal[0];
        v->normal.y = normal[1];
        v->normal.z = normal[2];
        v->tangent.x = tangent[0];
        v->tangent.y = tangent[1];
        v->tangent.z = tangent[2];
        v->texcoord.x = (DotProduct(point, texinfo->axis[0]) + texinfo->offset[0]) * inv_w;
        v->texcoord.y = (DotProduct(point, texinfo->axis[1]) + texinfo->offset[1]) * inv_h;
        v->material = material | camera_bits;
        v->alpha_bits = float_bits(surface_alpha(bsp, texinfo));
        v->cluster = cluster;
        v->flags = 0;
    }

    // Light sampling (spherical_tri_area, light_affects_cluster, the spot
    // term) takes cross(v1 - v0, v2 - v0) as the emitting side, so wind every
    // triangle to face along the surface normal like vkpt's create_poly().
    bool flip = false;
    if (surf->numsurfedges >= 3) {
        vec3_t a, b, c, e1, e2, n;
        VectorSet(a, verts[first_vertex].position.x, verts[first_vertex].position.y, verts[first_vertex].position.z);
        VectorSet(b, verts[first_vertex + 1].position.x, verts[first_vertex + 1].position.y, verts[first_vertex + 1].position.z);
        VectorSet(c, verts[first_vertex + 2].position.x, verts[first_vertex + 2].position.y, verts[first_vertex + 2].position.z);
        VectorSubtract(b, a, e1);
        VectorSubtract(c, a, e2);
        CrossProduct(e1, e2, n);
        flip = DotProduct(n, normal) < 0.0f;
    }

    for (int e = 2; e < surf->numsurfedges; e++) {
        indices[ii++] = first_vertex;
        indices[ii++] = first_vertex + (flip ? e : e - 1);
        indices[ii++] = first_vertex + (flip ? e - 1 : e);
    }

    *vi_inout = vi;
    *ii_inout = ii;
}

static void build_world_geometry(bsp_t *bsp)
{
    // Count the triangle fan expansion of every drawable face.
    uint32_t vert_count = 0, index_count = 0;
    for (int i = 0; i < bsp->numfaces; i++) {
        const mface_t *surf = &bsp->faces[i];
        if (!face_is_drawable(surf))
            continue;

        vert_count += surf->numsurfedges;
        index_count += (surf->numsurfedges - 2) * 3;
    }

    if (!index_count) {
        Com_WPrintf("Metal: the world has no drawable faces\n");
        return;
    }

    vertex_buffer = [mtl.device newBufferWithLength:sizeof(MTLTriVertex) * vert_count
                                            options:MTLResourceStorageModeShared];
    index_buffer = [mtl.device newBufferWithLength:sizeof(uint32_t) * index_count
                                           options:MTLResourceStorageModeShared];
    vertex_buffer.label = @"world vertices";
    index_buffer.label = @"world indices";

    MTLTriVertex *verts = (MTLTriVertex *)vertex_buffer.contents;
    uint32_t *indices = (uint32_t *)index_buffer.contents;

    // The world is every face that no inline model (door, platform, button)
    // claims; models[0] does not enumerate them itself. Inline model faces
    // follow the world in the same buffers but are excluded from the world
    // BLAS, and entities instance them per frame at their current pose.
    // Opaque world faces come first, then the transparent ones (water, glass)
    // which get their own BLAS so shadow rays can treat them separately.
    byte *face_is_submodel = Z_Mallocz((size_t)bsp->numfaces);
    for (int m = 1; m < bsp->nummodels; m++) {
        const mmodel_t *model = &bsp->models[m];
        for (int f = 0; f < model->numfaces; f++)
            face_is_submodel[(model->firstface + f) - bsp->faces] = 1;
    }

    uint32_t vi = 0, ii = 0;
    num_unclustered_faces = 0;
    // Order: opaque, masked, transparent, sky (filter_static_opaque/masked/
    // transparent/sky). Masked faces are alpha tested, so they get their own
    // non-opaque structure.
    for (int pass = 0; pass < 4; pass++) {
        for (int i = 0; i < bsp->numfaces; i++) {
            const mface_t *surf = &bsp->faces[i];
            if (face_is_submodel[i] || !face_is_drawable(surf))
                continue;
            const pbr_material_t *mat = surf->texinfo->material;
            int face_pass = (surf->texinfo->c.flags & SURF_SKY) ? 3
                          : (mat && mat->image_mask) ? 1
                          : texinfo_is_transparent(bsp, surf->texinfo) ? 2 : 0;
            if (face_pass != pass)
                continue;
            append_bsp_face(bsp, surf, verts, indices, &vi, &ii);
        }
        if (pass == 0)
            num_masked_first_index = ii;
        if (pass == 1)
            num_opaque_indices = ii;
        if (pass == 2)
            num_sky_first_index = ii;
    }
    Z_Free(face_is_submodel);

    num_vertices = vi;
    num_indices = ii;

    // World bounds drive the fog density falloff of the god rays.
    ClearBounds(world_aabb_min, world_aabb_max);
    for (uint32_t v = 0; v < vi; v++) {
        vec3_t p = { verts[v].position.x, verts[v].position.y, verts[v].position.z };
        AddPointToBounds(p, world_aabb_min, world_aabb_max);
    }

    Z_Free(submodels);
    num_submodels = (uint32_t)bsp->nummodels;
    submodels = Z_Mallocz(sizeof(*submodels) * max(num_submodels, 1u));

    for (uint32_t m = 1; m < num_submodels; m++) {
        const mmodel_t *model = &bsp->models[m];
        submodels[m].first_index = ii;
        for (int f = 0; f < model->numfaces; f++) {
            const mface_t *surf = &model->firstface[f];
            if (face_is_drawable(surf))
                append_bsp_face(bsp, surf, verts, indices, &vi, &ii);
        }
        submodels[m].index_count = ii - submodels[m].first_index;
    }

    build_light_polys(bsp, verts, indices, num_indices);
    build_submodel_light_polys(bsp, verts, indices);
    collect_cluster_lights(bsp);
    allocate_frame_light_buffers();

    Com_Printf("Metal: world geometry %u vertices, %u triangles (%u transparent), %u light polys, %u inline models\n",
               num_vertices, num_indices / 3, (num_indices - num_opaque_indices) / 3, num_light_polys,
               num_submodels ? num_submodels - 1 : 0);
}

static id<MTLAccelerationStructure> build_acceleration_structure(MTLAccelerationStructureDescriptor *desc)
{
    MTLAccelerationStructureSizes sizes = [mtl.device accelerationStructureSizesWithDescriptor:desc];

    id<MTLAccelerationStructure> as =
        [mtl.device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    id<MTLBuffer> scratch = [mtl.device newBufferWithLength:max(sizes.buildScratchBufferSize, 1)
                                                   options:MTLResourceStorageModePrivate];
    if (!as || !scratch) {
        Com_EPrintf("Metal: could not allocate an acceleration structure\n");
        [scratch release];
        [as release];
        return nil;
    }

    id<MTLCommandBuffer> cmd = [mtl.queue commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> enc = [cmd accelerationStructureCommandEncoder];
    [enc buildAccelerationStructure:as descriptor:desc scratchBuffer:scratch scratchBufferOffset:0];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    [scratch release];
    return as;
}

// BLAS over a range of the world index buffer. The range is copied into its
// own buffer for the build.
static id<MTLAccelerationStructure> build_world_range_blas(uint32_t first_index, uint32_t index_count,
                                                          id<MTLBuffer> *range_indices, const char *label, bool opaque)
{
    [*range_indices release];
    *range_indices = nil;
    if (!index_count)
        return nil;

    *range_indices = [mtl.device newBufferWithBytes:(const uint32_t *)index_buffer.contents + first_index
                                             length:sizeof(uint32_t) * index_count
                                            options:MTLResourceStorageModeShared];
    (*range_indices).label = [NSString stringWithUTF8String:label];

    MTLAccelerationStructureTriangleGeometryDescriptor *geo =
        [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    geo.vertexBuffer = vertex_buffer;
    geo.vertexBufferOffset = 0;
    geo.vertexStride = sizeof(MTLTriVertex);
    geo.indexBuffer = *range_indices;
    geo.indexBufferOffset = 0;
    geo.indexType = MTLIndexTypeUInt32;
    geo.triangleCount = index_count / 3;
    geo.opaque = opaque;

    MTLPrimitiveAccelerationStructureDescriptor *desc =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    desc.geometryDescriptors = @[geo];

    id<MTLAccelerationStructure> blas = build_acceleration_structure(desc);
    blas.label = [NSString stringWithUTF8String:label];
    return blas;
}

static bool build_acceleration_structures(void)
{
    world_blas = build_world_range_blas(0, num_masked_first_index, &opaque_index_buffer, "world blas", true);
    world_masked_blas = build_world_range_blas(num_masked_first_index, num_opaque_indices - num_masked_first_index,
                                               &masked_index_buffer, "world masked blas", false);
    world_transparent_blas = build_world_range_blas(num_opaque_indices, num_sky_first_index - num_opaque_indices,
                                                    &transparent_index_buffer, "world transparent blas", true);
    world_sky_blas = build_world_range_blas(num_sky_first_index, num_indices - num_sky_first_index,
                                            &sky_index_buffer, "world sky blas", true);
    return world_blas || world_masked_blas || world_transparent_blas || world_sky_blas;
}

// Loads the six env/*.tga sky faces into a texture array. The slice order
// matches the axis order the shader derives from a ray direction, applying the
// same indirection the classic renderer uses when binding sky faces.
void mtl_pt_set_sky(const char *name, float rotate, int autorotate, const vec3_t axis)
{
    [sky_texture release];
    sky_texture = nil;

    sky_rotate = 0.0f;
    VectorSet(sky_axis, 0.0f, 0.0f, 1.0f);

    if (!name || !*name)
        return;

    static const char suffix[6][3] = { "rt", "bk", "lf", "ft", "up", "dn" };
    static const int axis_to_image[6] = { 0, 2, 1, 3, 4, 5 };

    image_t *faces[6] = { 0 };
    int width = 0, height = 0;

    for (int i = 0; i < 6; i++) {
        char path[MAX_QPATH];
        if (Q_concat(path, sizeof(path), "env/", name, suffix[i], ".tga") >= sizeof(path))
            return;
        FS_NormalizePath(path);

        image_t *image = IMG_Find(path, IT_SKY, IF_SRGB);
        if (!image || image == R_NOTEXTURE || !image->pix_data) {
            Com_DPrintf("Metal: sky face '%s' not found\n", path);
            return;
        }

        if (!width) {
            width = image->upload_width;
            height = image->upload_height;
        } else if (image->upload_width != width || image->upload_height != height) {
            Com_WPrintf("Metal: sky faces have mismatched sizes\n");
            return;
        }

        faces[i] = image;
    }

    if (width <= 0 || height <= 0)
        return;

    MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
    desc.textureType = MTLTextureType2DArray;
    desc.pixelFormat = MTLPixelFormatRGBA8Unorm_sRGB;
    desc.width = width;
    desc.height = height;
    desc.arrayLength = 6;
    desc.mipmapLevelCount = 1;
    desc.storageMode = MTLStorageModeShared;
    desc.usage = MTLTextureUsageShaderRead;

    sky_texture = [mtl.device newTextureWithDescriptor:desc];
    [desc release];

    if (!sky_texture)
        return;

    sky_texture.label = @"skybox";

    for (int a = 0; a < 6; a++) {
        const image_t *image = faces[axis_to_image[a]];
        [sky_texture replaceRegion:MTLRegionMake2D(0, 0, width, height)
                       mipmapLevel:0
                             slice:a
                         withBytes:image->pix_data
                       bytesPerRow:4 * width
                     bytesPerImage:0];
    }

    sky_rotate = rotate;
    sky_autorotate = autorotate != 0;
    if (axis)
        VectorNormalize2((float *)axis, sky_axis);

    // Average the side and top faces for the sky light radiance; the down face
    // is excluded because it never illuminates the level.
    VectorClear(sky_average_color);
    int averaged = 0;
    for (int i = 0; i < 5; i++) {
        vec3_t color;
        average_image_color(faces[i], color);
        VectorAdd(sky_average_color, color, sky_average_color);
        averaged++;
    }
    if (averaged)
        VectorScale(sky_average_color, 1.0f / averaged, sky_average_color);

    Com_Printf("Metal: loaded sky box '%s' (avg %.2f %.2f %.2f)\n", name,
               sky_average_color[0], sky_average_color[1], sky_average_color[2]);
}

static int compare_doubles(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

// GPU time reports from the command buffer completion handler, each tagged
// with the render scale the frame was traced at (0: no world traced). A
// single producer (completion handlers run in order) and a single consumer
// (drs_process on the main thread) share this ring.
#define DRS_REPORT_RING 32
static struct { double ms; int scale; } drs_reports[DRS_REPORT_RING];
static _Atomic uint32_t drs_reports_written;
static int frame_scale_tag;     // scale the current frame traced the world at

// Dynamic resolution. vkpt moves the scale by at most 10% (down: 1%, from a
// swapped clamp) once per five frames, using samples that partly predate the
// previous change, so it reacts over seconds. Here every sample carries the
// scale it was rendered at, only samples at the current scale count, and the
// step jumps to the scale the cost model predicts (GPU time grows with the
// pixel count, i.e. the square of the scale). The same drs_target,
// drs_adjust_up / drs_adjust_down band and min/max scale apply.
#define DRS_SAMPLES 3
static void drs_process(void)
{
    static double samples[DRS_SAMPLES];
    static int num_samples;
    static uint32_t reports_read;

    // Consume new reports, keeping those taken at the current scale.
    uint32_t written = atomic_load(&drs_reports_written);
    if (written - reports_read > DRS_REPORT_RING)
        reports_read = written - DRS_REPORT_RING;

    if (!cvar_drs_enable->integer) {
        reports_read = written;
        num_samples = 0;
        drs_effective_scale = accumulation_active ? max(100, cvar_scr_viewsize->integer) : 0;
        return;
    }

    if (accumulation_active) {
        reports_read = written;
        num_samples = 0;
        drs_effective_scale = max(cvar_drs_minscale->integer, cvar_drs_maxscale->integer);
        return;
    }

    int last_scale = drs_current_scale;
    if (!last_scale)
        last_scale = cvar_drs_last_scale->integer;
    if (!last_scale)
        last_scale = cvar_scr_viewsize->integer;
    last_scale = max(cvar_drs_minscale->integer, min(cvar_drs_maxscale->integer, last_scale));
    drs_effective_scale = last_scale;

    for (; reports_read != written; reports_read++) {
        int i = reports_read % DRS_REPORT_RING;
        if (drs_reports[i].scale != last_scale || drs_reports[i].ms <= 0.0 || drs_reports[i].ms > 1000.0)
            continue;
        if (num_samples < DRS_SAMPLES)
            samples[num_samples++] = drs_reports[i].ms;
    }

    // Menus and the player setup view are not representative.
    if (!drs_last_frame_world || num_samples < DRS_SAMPLES)
        return;
    num_samples = 0;

    qsort(samples, DRS_SAMPLES, sizeof(double), compare_doubles);
    double frame_time = samples[DRS_SAMPLES / 2];
    double target_time = 1000.0 / max(cvar_drs_target->value, 1.0f);

    int scale = last_scale;
    if (frame_time > target_time * cvar_drs_adjust_down->value ||
        frame_time < target_time * cvar_drs_adjust_up->value) {
        // Aim for the middle of the band.
        double goal = target_time * 0.5 * (cvar_drs_adjust_up->value + cvar_drs_adjust_down->value);
        double ideal = last_scale * sqrt(goal / frame_time);
        int step = (int)lrint(ideal) - last_scale;
        // Big steps down so an overloaded frame recovers at once; smaller
        // ones up, since the fixed costs make the model optimistic there.
        step = Q_clip(step, -30, 15);
        if (step == 0)
            step = (frame_time > target_time) ? -1 : 1;
        scale += step;
    }

    drs_current_scale = max(cvar_drs_minscale->integer, min(cvar_drs_maxscale->integer, scale));
    drs_effective_scale = drs_current_scale;
}

// The scale this frame traced the world at, for tagging its GPU time; reset
// so frames that trace nothing report 0.
int mtl_pt_take_frame_scale_tag(void)
{
    int tag = frame_scale_tag;
    frame_scale_tag = 0;
    return tag;
}

// Records the measured GPU time of a completed frame. Called from the
// command buffer completion handler.
void mtl_pt_report_gpu_time(double milliseconds, int scale_tag)
{
    if (!(milliseconds > 0.0 && milliseconds < 1000.0))
        return;
    uint32_t i = atomic_load(&drs_reports_written);
    drs_reports[i % DRS_REPORT_RING].ms = milliseconds;
    drs_reports[i % DRS_REPORT_RING].scale = scale_tag;
    atomic_store(&drs_reports_written, i + 1);
}

int mtl_pt_resolution_scale(void)
{
    return drs_effective_scale ? drs_effective_scale : cvar_scr_viewsize->integer;
}

// Fraction of the screen image the final blit reads.
void mtl_pt_output_uv_scale(float *u_scale, float *v_scale)
{
    if (fsr_active || screen_image_width <= 0 || taa_output_width <= 0) {
        *u_scale = *v_scale = 1.0f;
        return;
    }
    *u_scale = (float)taa_output_width / (float)screen_image_width;
    *v_scale = (float)taa_output_height / (float)screen_image_height;
}

float mtl_pt_tonemap_white_point(void)
{
    return cvar_tm_white_point ? cvar_tm_white_point->value : 10.0f;
}

// View distance of the primary surfaces this frame, in the checkerboard
// field layout, for the depth tested debug lines.
id<MTLTexture> mtl_pt_depth_texture(int *width, int *height)
{
    *width = render_width;
    *height = render_height;
    return vkpt_images[VKPT_IMG_PT_VIEW_DEPTH_A + ((pt_frame_counter - 1) & 1)];
}

bool mtl_pt_water_warp(void)
{
    return last_rdflags_underwater && cvar_pt_waterwarp && cvar_pt_waterwarp->integer;
}

// vkpt uses Lanczos for the final blit whenever the image is upscaled, except
// for an exact 2x nearest-friendly case without DRS.
bool mtl_pt_use_lanczos(void)
{
    if (fsr_active || effective_aa_mode == AA_MODE_UPSCALE)
        return false;
    if (taa_output_width == display_width && taa_output_height == display_height)
        return false;
    if (taa_output_width == display_width / 2 && taa_output_height == display_height / 2 && drs_effective_scale == 0)
        return false;
    return true;
}

float mtl_pt_frame_time(void)
{
    return prev_frame_time;
}

void mtl_pt_output_size(int *width, int *height)
{
    if (fsr_active) {
        *width = display_width;
        *height = display_height;
    } else {
        *width = max(taa_output_width, 1);
        *height = max(taa_output_height, 1);
    }
}

void mtl_pt_register_world(bsp_t *bsp, const char *map_name)
{
    mtl_pt_free_world();

    if (!bsp || !bsp->numfaces)
        return;

    Q_strlcpy(world_map_name, map_name ? map_name : "", sizeof(world_map_name));
    load_sky_and_lava_clusters(world_map_name);
    load_cameras(world_map_name);
    world_bsp = bsp;
    mtl_bloom_reset();
    mtl_tone_mapping_request_reset();
    // Volumes come from the map config, which runs after this.
    vkpt_fog_reset();
    // The sky area lights take their colour from the rendered sky, so it has
    // to exist before the light polygons are collected below.
    mtl_physical_sky_update(0.0f);

    build_materials(bsp);
    build_world_geometry(bsp);

    if (!num_indices)
        return;

    // Wave normal map for water and slime surfaces (vkpt main.c).
    water_normal_image = IMG_Find("textures/water_n.tga", IT_SKIN, IF_PERMANENT);
    if (water_normal_image == R_NOTEXTURE)
        water_normal_image = NULL;

    // Uploaded textures must exist before material indices are resolved.
    mtl_textures_update();
    resolve_material_textures(bsp->numtexinfo);

    world_ready = build_acceleration_structures();
    if (!world_ready)
        Com_EPrintf("Metal: failed to build the world acceleration structure\n");

    // Light statistics: hit and miss counters per cluster, static light and
    // surface orientation (vertex_buffer.c), in NUM_LIGHT_STATS_BUFFERS slices.
    uint64_t num_stats = (uint64_t)num_clusters * num_light_polys * 6 * 2;
    uint64_t stats_bytes = max(num_stats, 1ull) * sizeof(uint32_t) * NUM_LIGHT_STATS_BUFFERS;
    if (stats_bytes <= 1024ull * 1024 * 1024) {
        light_stats_buffer = [mtl.device newBufferWithLength:stats_bytes options:MTLResourceStorageModePrivate];
        light_stats_buffer.label = @"light stats";
    }
    light_stats_size = light_stats_buffer ? (uint32_t)max(num_stats, 1ull) : 0;
    if (light_stats_buffer) {
        id<MTLCommandBuffer> cmd = [mtl.queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit fillBuffer:light_stats_buffer range:NSMakeRange(0, stats_bytes) value:0];
        [blit endEncoding];
        [cmd commit];
    } else {
        Com_WPrintf("Metal: light statistics disabled (%llu MB needed)\n", stats_bytes >> 20);
    }
    temporal_frame_valid = false;
}

void mtl_pt_free_world(void)
{
    // The instance buffers are allocated once at init and outlive the map.
    for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++) {
        [tlas[i] release];
        tlas[i] = nil;
        [material_buffers[i] release];
        material_buffers[i] = nil;
    }
    [world_blas release];
    world_blas = nil;
    [world_transparent_blas release];
    world_transparent_blas = nil;
    [world_sky_blas release];
    world_sky_blas = nil;
    [world_masked_blas release];
    world_masked_blas = nil;
    [masked_index_buffer release];
    masked_index_buffer = nil;
    [transparent_index_buffer release];
    transparent_index_buffer = nil;
    [opaque_index_buffer release];
    opaque_index_buffer = nil;
    [sky_index_buffer release];
    sky_index_buffer = nil;
    [vertex_buffer release];
    vertex_buffer = nil;
    [index_buffer release];
    index_buffer = nil;
    [light_poly_buffer release];
    light_poly_buffer = nil;
    for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++) {
        [frame_light_poly_buffers[i] release];
        frame_light_poly_buffers[i] = nil;
    }
    frame_light_poly_capacity = 0;
    Z_Free(submodel_light_polys);
    submodel_light_polys = NULL;
    Z_Free(submodel_light_poly_material);
    submodel_light_poly_material = NULL;
    Z_Free(submodel_lights);
    submodel_lights = NULL;
    num_submodel_light_polys = 0;

    Z_Free(texinfo_inv_size);
    texinfo_inv_size = NULL;
    Z_Free(texinfo_radiance);
    texinfo_radiance = NULL;
    Z_Free(submodels);
    submodels = NULL;
    num_submodels = 0;
    Z_Free(world_materials_base);
    world_materials_base = NULL;
    Z_Free(texinfo_anim_current);
    texinfo_anim_current = NULL;
    Z_Free(light_poly_material);
    light_poly_material = NULL;
    Z_Free(light_poly_cluster);
    light_poly_cluster = NULL;
    free_light_lists();
    [light_stats_buffer release];
    light_stats_buffer = nil;
    light_stats_size = 0;

    num_vertices = num_indices = 0;
    num_light_polys = 0;
    have_sky_lights = false;
    temporal_frame_valid = false;
    world_ready = false;
    world_bsp = NULL;
}

//
// Per-frame dispatch
//

//
// Entity geometry
//

// Adds one entity skin to the per-frame material slots and returns its index.
static uint32_t register_entity_material(const image_t *skin,
                                         const pbr_material_t *material,
                                         int entity_flags)
{
    if (num_entity_materials >= MTL_MAX_ENTITY_MATERIALS)
        return 0;

    uint32_t index = (uint32_t)world_material_count + num_entity_materials++;

    MTLMaterial *materials = (MTLMaterial *)material_buffers[current_frame_slot].contents;
    MTLMaterial *mat = &materials[index];
    memset(mat, 0, sizeof(*mat));

    mat->base_texture = skin ? mtl_texture_index_for_handle((qhandle_t)(skin - r_images))
                             : mtl_texture_index_for_handle((qhandle_t)MTL_TEXNUM_WHITE);
    mat->base_color.x = mat->base_color.y = mat->base_color.z = 1.0f;
    mat->roughness = -1.0f;
    mat->metalness = 0.0f;
    mat->specular_factor = 0.0f;
    mat->bump_scale = 1.0f;

    if (material) {
        if (material->image_normals) {
            mat->normal_texture = mtl_texture_index_optional(
                (qhandle_t)(material->image_normals - r_images));
        }
        if (material->image_emissive) {
            mat->emissive_texture = mtl_texture_index_optional(
                (qhandle_t)(material->image_emissive - r_images));
            // Model triangles carry emissive_factor 1 in vkpt.
            mat->emissive_factor = material->emissive_factor;
        }
        mat->bump_scale = material->bump_scale;
        mat->specular_factor = material->specular_factor;
        mat->metalness = material->metalness_factor;
        mat->roughness = material->roughness_override;
        // Reflective model skins get the model specific chrome kind, which is
        // what makes weapons and armour mirror the scene.
        uint32_t kind_flags = material->flags;
        if ((kind_flags & MATERIAL_KIND_MASK) == MATERIAL_KIND_CHROME)
            kind_flags = (kind_flags & ~MATERIAL_KIND_MASK) | MATERIAL_KIND_CHROME_MODEL;
        mat->kind_flags = kind_flags;
    }
    if (entity_flags & RF_WEAPONMODEL)
        mat->kind_flags |= MATERIAL_FLAG_WEAPON;
    if (entity_flags & RF_MTL_HANDEDNESS)
        mat->kind_flags |= MATERIAL_FLAG_HANDEDNESS;

    // Power-up shells (vkpt compute_mesh_material). IR goggles force red.
    if ((mat->kind_flags & MATERIAL_KIND_MASK) != MATERIAL_KIND_GLASS) {
        if ((entity_flags & RF_IR_VISIBLE) && (current_rdflags & RDF_IRGOGGLES)) {
            mat->shell |= SHELL_RED;
        } else {
            if (entity_flags & RF_SHELL_HALF_DAM) mat->shell |= SHELL_HALF_DAM;
            if (entity_flags & RF_SHELL_DOUBLE)   mat->shell |= SHELL_DOUBLE;
            if (entity_flags & RF_SHELL_RED)      mat->shell |= SHELL_RED;
            if (entity_flags & RF_SHELL_GREEN)    mat->shell |= SHELL_GREEN;
            if (entity_flags & RF_SHELL_BLUE)     mat->shell |= SHELL_BLUE;
        }
    }

    // RF_FULLBRIGHT is deliberately ignored, as in vkpt: a constant emissive
    // turns the model into a flat unshaded silhouette (the player setup view
    // marks its model fullbright and lights it with its own dlights).

    return index;
}

// Identity of the entity whose triangles are being written, stored above the
// vertex flag bits; the gradient reprojection maps last frame's triangles of
// an entity onto this frame's through the entity tables.
static uint32_t current_entity_id;

static uint32_t entity_table_id(const entity_t *ent)
{
    return ent->id > 0 ? ((uint32_t)ent->id & 0xffffffu) : 0u;
}

// Entity origin interpolated the same way vkpt's create_entity_matrix does.
static void entity_lerped_origin(const entity_t *ent, vec3_t out)
{
    float backlerp = Q_clipf(ent->backlerp, 0.0f, 1.0f);
    LerpVector(ent->origin, ent->oldorigin, backlerp, out);
}

// Instances a door/platform/button: copies its triangles out of the shared
// world buffers into this frame's entity geometry at the entity's pose.
// vkpt sorts dynamic geometry into an opaque and a transparent instance
// (process_regular_entity's MESH_FILTER, is_model_transparent for brush
// models): shadow and last-bounce refraction rays skip the transparent one,
// which is what lets them pass through glass. 0 = everything, 1 = opaque
// only, 2 = transparent only.
static int entity_mesh_filter;

static bool material_kind_is_transparent(uint32_t kind_flags)
{
    uint32_t kind = kind_flags & MATERIAL_KIND_MASK;
    return kind == MATERIAL_KIND_SLIME || kind == MATERIAL_KIND_WATER || kind == MATERIAL_KIND_GLASS ||
           kind == MATERIAL_KIND_TRANSPARENT || kind == MATERIAL_KIND_TRANSP_MODEL;
}

static bool entity_filter_rejects(bool transparent)
{
    return (entity_mesh_filter == 1 && transparent) || (entity_mesh_filter == 2 && !transparent);
}

static void add_inline_model(const entity_t *ent)
{
    uint32_t model_index = (uint32_t)~ent->model;
    if (model_index == 0 || model_index >= num_submodels || !vertex_buffer)
        return;

    const mtl_submodel_t *sub = &submodels[model_index];
    if (!sub->index_count)
        return;

    if (entity_mesh_filter) {
        // is_model_transparent(): every face water, slime, glass or
        // transparent; or the whole model translucent.
        const MTLTriVertex *sv = (const MTLTriVertex *)vertex_buffer.contents;
        const uint32_t *si = (const uint32_t *)index_buffer.contents + sub->first_index;
        bool transparent = sub->index_count > 0;
        for (uint32_t i = 0; i < sub->index_count && transparent; i += 3) {
            uint32_t m = sv[si[i]].material & MTL_VERTEX_MATERIAL_MASK;
            if (!world_bsp || m >= (uint32_t)world_bsp->numtexinfo || !texinfo_is_transparent(world_bsp, &world_bsp->texinfo[m]))
                transparent = false;
        }
        if ((ent->flags & RF_TRANSLUCENT) && ent->alpha < 1.0f)
            transparent = true;
        if (entity_filter_rejects(transparent))
            return;
    }
    if (num_entity_vertices + sub->index_count > MTL_MAX_ENTITY_VERTICES ||
        num_entity_indices + sub->index_count > MTL_MAX_ENTITY_INDICES)
        return;

    vec3_t origin, axis[3];
    entity_lerped_origin(ent, origin);
    AnglesToAxis(ent->angles, axis);
    float scale = ent->scale > 0.0f ? ent->scale : 1.0f;

    // Previous pose for motion vectors.
    vec3_t prev_angles, prev_origin, prev_axis[3];
    int prev_frame, prev_oldframe;
    float prev_backlerp, prev_scale;
    entity_prev_pose(ent, prev_origin, prev_angles, &prev_frame, &prev_oldframe, &prev_backlerp, &prev_scale);
    AnglesToAxis(prev_angles, prev_axis);
    prev_scale = prev_scale > 0.0f ? prev_scale : 1.0f;

    const MTLTriVertex *src_verts = (const MTLTriVertex *)vertex_buffer.contents;
    const uint32_t *src_indices = (const uint32_t *)index_buffer.contents + sub->first_index;

    // process_bsp_entity(): the cluster at the model's transformed bounding
    // box center, or failing that at one of its corners (a pushed button can
    // sink into a wall). The entity origin is usually (0 0 0) for brush
    // models, which picked some unrelated cluster and its light list, so
    // e.g. breakable floors came out dark.
    int ent_cluster = -1;
    {
        vec3_t mins, maxs;
        ClearBounds(mins, maxs);
        for (uint32_t i = 0; i < sub->index_count; i++) {
            const MTLTriVertex *s = &src_verts[src_indices[i]];
            vec3_t p = { s->position.x, s->position.y, s->position.z };
            AddPointToBounds(p, mins, maxs);
        }
        for (int corner = -1; corner < 8 && ent_cluster < 0; corner++) {
            vec3_t local;
            for (int c = 0; c < 3; c++)
                local[c] = (corner < 0 ? (mins[c] + maxs[c]) * 0.5f : ((corner >> c) & 1) ? maxs[c] : mins[c]) * scale;
            vec3_t world;
            for (int c = 0; c < 3; c++)
                world[c] = origin[c] + local[0] * axis[0][c] + local[1] * axis[1][c] + local[2] * axis[2][c];
            ent_cluster = point_cluster(world);
        }
    }

    MTLTriVertex *verts = (MTLTriVertex *)entity_vertex_buffers[current_frame_slot].contents;
    uint32_t *indices = (uint32_t *)entity_index_buffers[current_frame_slot].contents;

    for (uint32_t i = 0; i < sub->index_count; i++) {
        const MTLTriVertex *s = &src_verts[src_indices[i]];
        MTLTriVertex *v = &verts[num_entity_vertices];
        *v = *s;

        // Frame-based texture animation (animate_material): walk the texinfo
        // chain ent->frame steps, and use the static material block.
        uint32_t base_material = s->material & MTL_VERTEX_MATERIAL_MASK;
        uint32_t anim_material = base_material;
        if (ent->frame > 0 && world_bsp && base_material < (uint32_t)world_bsp->numtexinfo) {
            const mtexinfo_t *ti = &world_bsp->texinfo[base_material];
            if (ti->numframes > 1) {
                int steps = ent->frame % ti->numframes;
                while (steps-- > 0 && ti->next)
                    ti = ti->next;
                anim_material = (uint32_t)(ti - world_bsp->texinfo);
            }
        }
        v->material = (s->material & ~MTL_VERTEX_MATERIAL_MASK) | (anim_material + (uint32_t)world_bsp->numtexinfo);
        v->cluster = ent_cluster;
        v->flags = (s->flags & ((1u << MTL_VERTEX_ENTITY_SHIFT) - 1u)) | (current_entity_id << MTL_VERTEX_ENTITY_SHIFT);

        vec3_t local = { s->position.x * scale, s->position.y * scale, s->position.z * scale };
        vec3_t plocal = { s->position.x * prev_scale, s->position.y * prev_scale, s->position.z * prev_scale };
        vec3_t n = { s->normal.x, s->normal.y, s->normal.z };
        vec3_t t = { s->tangent.x, s->tangent.y, s->tangent.z };

        v->position.x = origin[0] + local[0] * axis[0][0] + local[1] * axis[1][0] + local[2] * axis[2][0];
        v->position.y = origin[1] + local[0] * axis[0][1] + local[1] * axis[1][1] + local[2] * axis[2][1];
        v->position.z = origin[2] + local[0] * axis[0][2] + local[1] * axis[1][2] + local[2] * axis[2][2];
        v->prev_position.x = prev_origin[0] + plocal[0] * prev_axis[0][0] + plocal[1] * prev_axis[1][0] + plocal[2] * prev_axis[2][0];
        v->prev_position.y = prev_origin[1] + plocal[0] * prev_axis[0][1] + plocal[1] * prev_axis[1][1] + plocal[2] * prev_axis[2][1];
        v->prev_position.z = prev_origin[2] + plocal[0] * prev_axis[0][2] + plocal[1] * prev_axis[1][2] + plocal[2] * prev_axis[2][2];
        v->normal.x = n[0] * axis[0][0] + n[1] * axis[1][0] + n[2] * axis[2][0];
        v->normal.y = n[0] * axis[0][1] + n[1] * axis[1][1] + n[2] * axis[2][1];
        v->normal.z = n[0] * axis[0][2] + n[1] * axis[1][2] + n[2] * axis[2][2];
        v->tangent.x = t[0] * axis[0][0] + t[1] * axis[1][0] + t[2] * axis[2][0];
        v->tangent.y = t[0] * axis[0][1] + t[1] * axis[1][1] + t[2] * axis[2][1];
        v->tangent.z = t[0] * axis[0][2] + t[1] * axis[1][2] + t[2] * axis[2][2];

        indices[num_entity_indices++] = num_entity_vertices++;
    }
}

// The camera the view weapon was drawn with last frame, so its previous
// vertex positions go through last frame's gun fov transform the way vkpt's
// transform_prev keeps last frame's view weapon matrix.
typedef struct {
    bool   valid;
    vec3_t origin, forward, right, up;
    float  adjust_x, adjust_y;
} mtl_gun_view_t;
static mtl_gun_view_t gun_view_prev, gun_view_curr;

// create_viewweapon_matrix(): scale factors that make the weapon appear
// rendered with cl_gunfov. Returns false when cl_gunfov is off.
static bool gun_fov_view(const refdef_t *fd, mtl_gun_view_t *out)
{
    float gunfov = Cvar_VariableValue("cl_gunfov");
    if (gunfov <= 0.0f)
        return false;
    float gunfov_x = Q_clipf(gunfov, 30.0f, 160.0f), gunfov_y;
    if (Cvar_VariableInteger("cl_adjustfov")) {
        gunfov_y = V_CalcFov(gunfov_x, 4, 3);
        gunfov_x = V_CalcFov(gunfov_y, fd->height, fd->width);
    } else {
        gunfov_y = V_CalcFov(gunfov_x, fd->width, fd->height);
    }
    out->adjust_x = tanf(DEG2RAD(fd->fov_x) * 0.5f) / tanf(DEG2RAD(gunfov_x) * 0.5f);
    out->adjust_y = tanf(DEG2RAD(fd->fov_y) * 0.5f) / tanf(DEG2RAD(gunfov_y) * 0.5f);
    VectorCopy(fd->vieworg, out->origin);
    AngleVectors(fd->viewangles, out->forward, out->right, out->up);
    out->valid = true;
    return true;
}

static void gun_fov_apply(const mtl_gun_view_t *v, float p[3])
{
    vec3_t d;
    VectorSubtract(p, v->origin, d);
    float x = DotProduct(d, v->right) * v->adjust_x, y = DotProduct(d, v->up) * v->adjust_y, z = DotProduct(d, v->forward);
    for (int c = 0; c < 3; c++)
        p[c] = v->origin[c] + x * v->right[c] + y * v->up[c] + z * v->forward[c];
}

static void add_entity_mesh(const entity_t *ent, const model_t *model, const maliasmesh_t *mesh, const refdef_t *fd)
{
    if (mesh->numverts <= 0 || mesh->numindices <= 0)
        return;

    if (entity_mesh_filter) {
        const pbr_material_t *mdef = NULL;
        if (mesh->numskins > 0)
            mdef = mesh->materials[Q_clip(ent->skinnum, 0, mesh->numskins - 1)];
        bool transparent = (mdef && material_kind_is_transparent(mdef->flags)) ||
                           ((ent->flags & RF_TRANSLUCENT) && ent->alpha < 1.0f);
        if (entity_filter_rejects(transparent))
            return;
    }

    if (num_entity_vertices + mesh->numverts > MTL_MAX_ENTITY_VERTICES)
        return;
    if (num_entity_indices + mesh->numindices > MTL_MAX_ENTITY_INDICES)
        return;

    int numframes = max(model->numframes, 1);
    int frame = Q_clip(ent->frame, 0, numframes - 1);
    int oldframe = Q_clip(ent->oldframe, 0, numframes - 1);
    float lerp = 1.0f - Q_clipf(ent->backlerp, 0.0f, 1.0f);

    vec3_t forward, right, up;
    AngleVectors(ent->angles, forward, right, up);

    vec3_t origin;
    entity_lerped_origin(ent, origin);

    float scale = ent->scale > 0.0f ? ent->scale : 1.0f;

    // View weapon: mirrored for left-handed players and, with cl_gunfov, drawn
    // as if projected with the gun fov (create_viewweapon_matrix).
    bool is_weapon = (ent->flags & RF_WEAPONMODEL) != 0;
    float mirror = (is_weapon && Cvar_VariableInteger("hand") == 1) ? -1.0f : 1.0f;
    bool gun_adjust = is_weapon && fd && gun_fov_view(fd, &gun_view_curr);
    const mtl_gun_view_t *gun_prev = gun_view_prev.valid ? &gun_view_prev : &gun_view_curr;

    // Previous pose (origin, angles and animation frame) for motion vectors.
    vec3_t prev_angles, prev_origin, prev_forward, prev_right, prev_up;
    int prev_frame_i, prev_oldframe_i;
    float prev_backlerp, prev_scale;
    entity_prev_pose(ent, prev_origin, prev_angles, &prev_frame_i, &prev_oldframe_i, &prev_backlerp, &prev_scale);
    AngleVectors(prev_angles, prev_forward, prev_right, prev_up);
    int prev_frame = Q_clip(prev_frame_i, 0, numframes - 1);
    int prev_oldframe = Q_clip(prev_oldframe_i, 0, numframes - 1);
    float prev_lerp = 1.0f - Q_clipf(prev_backlerp, 0.0f, 1.0f);
    prev_scale = prev_scale > 0.0f ? prev_scale : 1.0f;

    const image_t *skin = NULL;
    const pbr_material_t *material_def = NULL;
    if (mesh->numskins > 0) {
        int skinnum = Q_clip(ent->skinnum, 0, mesh->numskins - 1);
        skin = mesh->skins[skinnum];
        material_def = mesh->materials[skinnum];
    }
    if (ent->skin)
        skin = IMG_ForHandle(ent->skin);

    uint32_t material = register_entity_material(skin, material_def, ent->flags | (mesh->handedness ? RF_MTL_HANDEDNESS : 0));
    int ent_cluster = point_cluster(origin);

    MTLTriVertex *verts = (MTLTriVertex *)entity_vertex_buffers[current_frame_slot].contents;
    uint32_t *indices = (uint32_t *)entity_index_buffers[current_frame_slot].contents;

    uint32_t first_vertex = num_entity_vertices;

    for (int i = 0; i < mesh->numverts; i++) {
        const float *p0 = mesh->positions[i * numframes + oldframe];
        const float *p1 = mesh->positions[i * numframes + frame];
        const float *n0 = mesh->normals[i * numframes + oldframe];
        const float *n1 = mesh->normals[i * numframes + frame];

        vec3_t local, normal;
        for (int c = 0; c < 3; c++) {
            local[c] = (p0[c] + (p1[c] - p0[c]) * lerp) * scale;
            normal[c] = n0[c] + (n1[c] - n0[c]) * lerp;
        }

        // Quake models are authored with +X forward and +Y left, so the right
        // vector is subtracted rather than added.
        vec3_t world, world_normal;
        for (int c = 0; c < 3; c++) {
            world[c] = origin[c] + local[0] * forward[c] - mirror * local[1] * right[c] + local[2] * up[c];
            world_normal[c] = normal[0] * forward[c] - mirror * normal[1] * right[c] + normal[2] * up[c];
        }
        VectorNormalize(world_normal);

        // Same vertex, last frame's animation frame and pose.
        const float *pp0 = mesh->positions[i * numframes + prev_oldframe];
        const float *pp1 = mesh->positions[i * numframes + prev_frame];
        vec3_t plocal, prev_world;
        for (int c = 0; c < 3; c++)
            plocal[c] = (pp0[c] + (pp1[c] - pp0[c]) * prev_lerp) * prev_scale;
        for (int c = 0; c < 3; c++)
            prev_world[c] = prev_origin[c] + plocal[0] * prev_forward[c] - mirror * plocal[1] * prev_right[c] + plocal[2] * prev_up[c];

        if (gun_adjust) {
            // Scale in view space so the gun appears rendered with cl_gunfov;
            // the previous pose with the previous frame's view.
            gun_fov_apply(&gun_view_curr, world);
            gun_fov_apply(gun_prev, prev_world);
        }

        MTLTriVertex *v = &verts[num_entity_vertices++];
        v->position.x = world[0];
        v->position.y = world[1];
        v->position.z = world[2];
        v->prev_position.x = prev_world[0];
        v->prev_position.y = prev_world[1];
        v->prev_position.z = prev_world[2];
        v->normal.x = world_normal[0];
        v->normal.y = world_normal[1];
        v->normal.z = world_normal[2];
        if (mesh->tangents) {
            const float *t0 = mesh->tangents[i * numframes + oldframe];
            const float *t1 = mesh->tangents[i * numframes + frame];
            vec3_t t, wt;
            for (int c = 0; c < 3; c++)
                t[c] = t0[c] + (t1[c] - t0[c]) * lerp;
            for (int c = 0; c < 3; c++)
                wt[c] = t[0] * forward[c] - mirror * t[1] * right[c] + t[2] * up[c];
            VectorNormalize(wt);
            v->tangent.x = wt[0];
            v->tangent.y = wt[1];
            v->tangent.z = wt[2];
        } else {
            // No tangents: the shader falls back to a generated basis.
            v->tangent.x = v->tangent.y = v->tangent.z = 0.0f;
        }
        v->texcoord.x = mesh->tex_coords[i * numframes + frame][0];
        v->texcoord.y = mesh->tex_coords[i * numframes + frame][1];
        v->material = material;
        v->alpha_bits = float_bits((ent->flags & RF_TRANSLUCENT) ? ent->alpha : 1.0f);
        v->cluster = ent_cluster;
        v->flags = current_entity_id << MTL_VERTEX_ENTITY_SHIFT;
    }

    for (int i = 0; i < mesh->numindices; i++)
        indices[num_entity_indices++] = first_vertex + (uint32_t)mesh->indices[i];
}

// Records the triangle range just written for current_entity_id.
static void record_entity_prims(VkptEntitySlot *table, uint32_t first_prim)
{
    uint32_t count = num_entity_indices / 3 - first_prim;
    if (!current_entity_id || !count)
        return;
    VkptEntitySlot *slot = &table[current_entity_id % VKPT_ENTITY_TABLE_SIZE];
    if (slot->id)
        return;   // hash collision: the first entity keeps the slot
    slot->id = current_entity_id;
    slot->first_prim = first_prim;
    slot->prim_count = count;
}

static void build_entity_geometry(const refdef_t *fd)
{
    num_entity_vertices = 0;
    num_entity_indices = 0;
    num_entity_materials = 0;
    entity_transparent_first_index = entity_viewer_first_index = entity_weapon_first_index = 0;

    if (!entity_vertex_buffers[current_frame_slot] || !material_buffers[current_frame_slot])
        return;

    VkptEntitySlot *table = (VkptEntitySlot *)entity_table_buffers[current_frame_slot].contents;
    memset(table, 0, sizeof(VkptEntitySlot) * VKPT_ENTITY_TABLE_SIZE);
    gun_view_curr.valid = false;

    // Same grouping as vkpt_pt_create_all_dynamic: regular entities, then the
    // player's own model (only in first person mode, seen in mirrors), then
    // the view weapon. Each group gets its own BLAS/instance mask.
    bool first_person_model = Cvar_VariableInteger("cl_player_model") == CL_PLAYER_MODEL_FIRST_PERSON;

    // Passes: regular opaque, regular transparent, viewer model, view weapon.
    for (int pass = 0; pass < 4; pass++) {
        if (pass == 1)
            entity_transparent_first_index = num_entity_indices;
        if (pass == 2)
            entity_viewer_first_index = num_entity_indices;
        if (pass == 3)
            entity_weapon_first_index = num_entity_indices;
        entity_mesh_filter = pass == 0 ? 1 : pass == 1 ? 2 : 0;

        for (int i = 0; i < fd->num_entities; i++) {
            const entity_t *ent = &fd->entities[i];

            if (!ent->model)
                continue;

            uint32_t first_prim = num_entity_indices / 3;
            current_entity_id = entity_table_id(ent);

            // Inline BSP models are referenced by their bitwise-inverted index.
            if (ent->model & 0x80000000) {
                if (pass <= 1) {
                    add_inline_model(ent);
                    record_entity_prims(table, first_prim);
                }
                continue;
            }

            int ent_pass = (ent->flags & RF_VIEWERMODEL) ? 2 : (ent->flags & RF_WEAPONMODEL) ? 3 : 0;
            if (ent_pass != pass && !(ent_pass == 0 && pass == 1))
                continue;
            if (pass == 2 && !first_person_model)
                continue;

            const model_t *model = MOD_ForHandle(ent->model);
            if (!model || model->type != MOD_ALIAS || !model->meshes)
                continue;

            // Explosions and muzzle flashes are additive effects, not opaque.
            if (model->model_class == MCLASS_EXPLOSION || model->model_class == MCLASS_FLASH)
                continue;

            for (int m = 0; m < model->nummeshes; m++)
                add_entity_mesh(ent, model, &model->meshes[m], fd);
            record_entity_prims(table, first_prim);
        }
    }

    gun_view_prev = gun_view_curr;
}

// Builds one BLAS over a range of the per-frame entity index buffer.
static bool build_entity_range_blas(id<MTLCommandBuffer> cmd, id<MTLAccelerationStructure> *blas, id<MTLBuffer> *scratch,
                                    uint32_t first_index, uint32_t index_count, const char *label)
{
    uint32_t slot = current_frame_slot;
    if (!index_count) {
        [*blas release];
        *blas = nil;
        return false;
    }

    MTLAccelerationStructureTriangleGeometryDescriptor *geo =
        [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    geo.vertexBuffer = entity_vertex_buffers[slot];
    geo.vertexStride = sizeof(MTLTriVertex);
    geo.indexBuffer = entity_index_buffers[slot];
    geo.indexBufferOffset = sizeof(uint32_t) * first_index;
    geo.indexType = MTLIndexTypeUInt32;
    geo.triangleCount = index_count / 3;
    geo.opaque = YES;

    MTLPrimitiveAccelerationStructureDescriptor *desc =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    desc.geometryDescriptors = @[geo];
    // vkpt builds dynamic BLASes with PREFER_FAST_BUILD; on Apple GPUs the
    // default (trace optimized) build costs ~0.1 ms more but saves ~0.8 ms of
    // traversal per frame at 1080p, so it is kept.

    MTLAccelerationStructureSizes sizes = [mtl.device accelerationStructureSizesWithDescriptor:desc];

    if (!*blas || (*blas).size < sizes.accelerationStructureSize) {
        [*blas release];
        *blas = [mtl.device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
        (*blas).label = [NSString stringWithUTF8String:label];
    }

    if (!*scratch || (*scratch).length < sizes.buildScratchBufferSize) {
        [*scratch release];
        *scratch = [mtl.device newBufferWithLength:max(sizes.buildScratchBufferSize, 1)
                                           options:MTLResourceStorageModePrivate];
    }

    if (!*blas || !*scratch)
        return false;

    id<MTLAccelerationStructureCommandEncoder> enc = mtl_profiler_accel_encoder(cmd, MTL_PROFILER_ENTITY_BLAS);
    [enc buildAccelerationStructure:*blas descriptor:desc scratchBuffer:*scratch scratchBufferOffset:0];
    [enc endEncoding];
    return true;
}

// Rebuilds the entity acceleration structures in place each frame.
static bool refresh_entity_blas(id<MTLCommandBuffer> cmd)
{
    uint32_t slot = current_frame_slot;
    uint32_t regular_count = entity_transparent_first_index;
    uint32_t transparent_count = entity_viewer_first_index - entity_transparent_first_index;
    uint32_t viewer_count = entity_weapon_first_index - entity_viewer_first_index;
    uint32_t weapon_count = num_entity_indices - entity_weapon_first_index;

    bool a = build_entity_range_blas(cmd, &entity_blas[slot], &entity_scratch[slot], 0, regular_count, "entity blas");
    bool b = build_entity_range_blas(cmd, &viewer_blas[slot], &viewer_scratch[slot], entity_viewer_first_index, viewer_count, "viewer model blas");
    bool c = build_entity_range_blas(cmd, &weapon_blas[slot], &weapon_scratch[slot], entity_weapon_first_index, weapon_count, "viewer weapon blas");
    bool d = build_entity_range_blas(cmd, &entity_transparent_blas[slot], &entity_transparent_scratch[slot],
                                     entity_transparent_first_index, transparent_count, "transparent entity blas");
    return a || b || c || d;
}

static void set_identity_instance(MTLAccelerationStructureInstanceDescriptor *instance, uint32_t as_index, uint32_t mask,
                                  bool opaque)
{
    memset(instance, 0, sizeof(*instance));
    instance->transformationMatrix.columns[0] = (MTLPackedFloat3){ 1, 0, 0 };
    instance->transformationMatrix.columns[1] = (MTLPackedFloat3){ 0, 1, 0 };
    instance->transformationMatrix.columns[2] = (MTLPackedFloat3){ 0, 0, 1 };
    instance->transformationMatrix.columns[3] = (MTLPackedFloat3){ 0, 0, 0 };
    instance->options = opaque ? MTLAccelerationStructureInstanceOptionOpaque : MTLAccelerationStructureInstanceOptionNone;
    instance->mask = mask;
    instance->accelerationStructureIndex = as_index;
}

// The top level structure is rebuilt every frame because the entity instances
// only exist when something dynamic is visible. Instance ids are fixed
// (see INSTANCE_* in path_tracer.metal): 0 opaque world, 1 entities,
// 2 transparent world, 3 player model, 4 view weapon, 5 sky, 6 alpha tested
// world; absent ones point at the first present structure with mask 0.
// Masks follow vkpt's AS_FLAG_* bits. Without the world (player setup view)
// only the entity groups are instanced. Returns false when there is nothing
// to trace.
static bool refresh_tlas(id<MTLCommandBuffer> cmd, bool include_world)
{
    uint32_t slot = current_frame_slot;

    id<MTLAccelerationStructure> parts[MTL_TLAS_INSTANCES] = {
        include_world ? world_blas : nil,
        entity_blas[slot],
        include_world ? world_transparent_blas : nil,
        viewer_blas[slot],
        weapon_blas[slot],
        include_world ? world_sky_blas : nil,
        include_world ? world_masked_blas : nil,
        entity_transparent_blas[slot],
    };
    static const uint32_t masks[MTL_TLAS_INSTANCES] = {
        AS_FLAG_OPAQUE, AS_FLAG_OPAQUE, AS_FLAG_TRANSPARENT, AS_FLAG_VIEWER_MODELS,
        AS_FLAG_VIEWER_WEAPON, AS_FLAG_SKY, AS_FLAG_MASKED, AS_FLAG_TRANSPARENT,
    };

    NSMutableArray *structures = [NSMutableArray array];
    MTLAccelerationStructureInstanceDescriptor *instances =
        (MTLAccelerationStructureInstanceDescriptor *)instance_buffers[slot].contents;

    for (int i = 0; i < MTL_TLAS_INSTANCES; i++) {
        if (parts[i]) {
            set_identity_instance(&instances[i], (uint32_t)structures.count, masks[i], i != 6);
            [structures addObject:parts[i]];
        }
    }
    if (!structures.count)
        return false;
    for (int i = 0; i < MTL_TLAS_INSTANCES; i++) {
        if (!parts[i])
            set_identity_instance(&instances[i], 0, 0x0, true);
    }

    MTLInstanceAccelerationStructureDescriptor *desc =
        [MTLInstanceAccelerationStructureDescriptor descriptor];
    desc.instancedAccelerationStructures = structures;
    desc.instanceCount = MTL_TLAS_INSTANCES;
    desc.instanceDescriptorBuffer = instance_buffers[slot];

    MTLAccelerationStructureSizes sizes = [mtl.device accelerationStructureSizesWithDescriptor:desc];

    if (!tlas[slot] || tlas[slot].size < sizes.accelerationStructureSize) {
        [tlas[slot] release];
        tlas[slot] = [mtl.device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
        tlas[slot].label = @"tlas";
    }

    if (!tlas_scratch[slot] || tlas_scratch[slot].length < sizes.buildScratchBufferSize) {
        [tlas_scratch[slot] release];
        tlas_scratch[slot] = [mtl.device newBufferWithLength:max(sizes.buildScratchBufferSize, 1)
                                                    options:MTLResourceStorageModePrivate];
    }

    if (!tlas[slot] || !tlas_scratch[slot])
        return false;

    id<MTLAccelerationStructureCommandEncoder> enc = mtl_profiler_accel_encoder(cmd, MTL_PROFILER_BVH_UPDATE);
    [enc buildAccelerationStructure:tlas[slot]
                         descriptor:desc
                      scratchBuffer:tlas_scratch[slot]
                scratchBufferOffset:0];
    [enc endEncoding];
    return true;
}

// Marks every bottom level structure the current TLAS can reach as used.
static void use_scene_structures(id<MTLComputeCommandEncoder> enc, uint32_t slot, bool include_world)
{
    id<MTLAccelerationStructure> parts[] = {
        include_world ? world_blas : nil,
        include_world ? world_transparent_blas : nil,
        include_world ? world_sky_blas : nil,
        include_world ? world_masked_blas : nil,
        entity_blas[slot], viewer_blas[slot], weapon_blas[slot], entity_transparent_blas[slot],
    };
    for (size_t i = 0; i < sizeof(parts) / sizeof(parts[0]); i++) {
        if (parts[i])
            [enc useResource:parts[i] usage:MTLResourceUsageRead];
    }
}

//
// Transparent effects (port of transparency.c)
//

static void effect_color(int color_index, const color_t *rgba, float hdr_factor, MTLEffectPrim *prim)
{
    color_t color;
    if (color_index < 0)
        color.u32 = rgba->u32;
    else
        color.u32 = d_8to24table[color_index & 0xff];

    float c[3];
    for (int i = 0; i < 3; i++) {
        float s = color.u8[i] / 255.0f;
        c[i] = hdr_factor * ((s <= 0.04045f) ? s / 12.92f : powf((s + 0.055f) / 1.055f, 2.4f));
    }
    prim->color.x = c[0];
    prim->color.y = c[1];
    prim->color.z = c[2];
}

// Effect geometry writer state for the current frame.
typedef struct {
    mtl_packed_float3_t *positions;
    MTLEffectVertex     *attrs;
    uint32_t            *indices;
    MTLEffectPrim       *prims;
} effect_writer_t;

static bool effect_room(uint32_t vertices, uint32_t triangles)
{
    return num_effect_vertices + vertices <= MTL_MAX_EFFECT_VERTICES &&
           num_effect_triangles + triangles <= MTL_MAX_EFFECT_TRIANGLES;
}

// Emits a quad as two triangles sharing one MTLEffectPrim description. Corner
// texture coordinates follow the sprite convention: left/down = (0,1),
// left/up = (0,0), right/up = (1,0), right/down = (1,1).
static void write_quad(effect_writer_t *w, const MTLEffectPrim *prim,
                       const vec3_t p0, const vec3_t p1, const vec3_t p2, const vec3_t p3)
{
    if (!effect_room(4, 2))
        return;

    static const float uvs[4][2] = { { 0, 1 }, { 0, 0 }, { 1, 0 }, { 1, 1 } };
    const float *pts[4] = { p0, p1, p2, p3 };

    uint32_t base = num_effect_vertices;
    for (int i = 0; i < 4; i++) {
        mtl_packed_float3_t *v = &w->positions[base + i];
        v->x = pts[i][0]; v->y = pts[i][1]; v->z = pts[i][2];
        MTLEffectVertex *a = &w->attrs[base + i];
        memset(a, 0, sizeof(*a));
        a->texcoord.x = uvs[i][0];
        a->texcoord.y = uvs[i][1];
    }

    uint32_t *idx = &w->indices[num_effect_triangles * 3];
    idx[0] = base + 0; idx[1] = base + 1; idx[2] = base + 2;
    idx[3] = base + 0; idx[4] = base + 2; idx[5] = base + 3;

    w->prims[num_effect_triangles + 0] = *prim;
    w->prims[num_effect_triangles + 1] = *prim;

    num_effect_vertices += 4;
    num_effect_triangles += 2;
}

// Explosion and muzzle flash alias models are additive transparent geometry in
// the Vulkan backend (path_tracer_explosion.rahit), so they go here rather
// than into the opaque entity structure.
static void write_effect_model(effect_writer_t *w, const entity_t *ent, const model_t *model)
{
    int numframes = max(model->numframes, 1);
    int frame = Q_clip(ent->frame, 0, numframes - 1);
    int oldframe = Q_clip(ent->oldframe, 0, numframes - 1);
    float lerp = 1.0f - Q_clipf(ent->backlerp, 0.0f, 1.0f);

    vec3_t forward, right, up, origin;
    AngleVectors(ent->angles, forward, right, up);
    entity_lerped_origin(ent, origin);
    float scale = ent->scale > 0.0f ? ent->scale : 1.0f;

    MTLEffectPrim prim;
    memset(&prim, 0, sizeof(prim));
    prim.type = (model->model_class == MCLASS_EXPLOSION) ? MTL_EFFECT_EXPLOSION : MTL_EFFECT_ADDITIVE;
    prim.color.w = (ent->flags & RF_TRANSLUCENT) ? ent->alpha : 1.0f;

    for (int m = 0; m < model->nummeshes; m++) {
        const maliasmesh_t *mesh = &model->meshes[m];
        if (mesh->numverts <= 0 || mesh->numindices <= 0)
            continue;
        if (!effect_room((uint32_t)mesh->numverts, (uint32_t)mesh->numindices / 3))
            return;

        const image_t *skin = NULL;
        if (mesh->numskins > 0)
            skin = mesh->skins[Q_clip(ent->skinnum, 0, mesh->numskins - 1)];
        if (ent->skin)
            skin = IMG_ForHandle(ent->skin);
        prim.texture = skin ? mtl_texture_index_for_handle((qhandle_t)(skin - r_images))
                            : mtl_texture_index_for_handle((qhandle_t)MTL_TEXNUM_WHITE);

        uint32_t base = num_effect_vertices;
        for (int i = 0; i < mesh->numverts; i++) {
            const float *p0 = mesh->positions[i * numframes + oldframe];
            const float *p1 = mesh->positions[i * numframes + frame];
            const float *n0 = mesh->normals[i * numframes + oldframe];
            const float *n1 = mesh->normals[i * numframes + frame];

            vec3_t local, normal, world, world_normal;
            for (int c = 0; c < 3; c++) {
                local[c] = (p0[c] + (p1[c] - p0[c]) * lerp) * scale;
                normal[c] = n0[c] + (n1[c] - n0[c]) * lerp;
            }
            for (int c = 0; c < 3; c++) {
                world[c] = origin[c] + local[0] * forward[c] - local[1] * right[c] + local[2] * up[c];
                world_normal[c] = normal[0] * forward[c] - normal[1] * right[c] + normal[2] * up[c];
            }
            VectorNormalize(world_normal);

            mtl_packed_float3_t *v = &w->positions[base + i];
            v->x = world[0]; v->y = world[1]; v->z = world[2];
            MTLEffectVertex *a = &w->attrs[base + i];
            a->texcoord.x = mesh->tex_coords[i * numframes + frame][0];
            a->texcoord.y = mesh->tex_coords[i * numframes + frame][1];
            a->normal.x = world_normal[0]; a->normal.y = world_normal[1]; a->normal.z = world_normal[2];
            a->pad = 0;
        }

        uint32_t tris = (uint32_t)mesh->numindices / 3;
        uint32_t *idx = &w->indices[num_effect_triangles * 3];
        for (uint32_t i = 0; i < tris * 3; i++)
            idx[i] = base + (uint32_t)mesh->indices[i];
        for (uint32_t t = 0; t < tris; t++)
            w->prims[num_effect_triangles + t] = prim;

        num_effect_vertices += (uint32_t)mesh->numverts;
        num_effect_triangles += tris;
    }
}

static bool model_is_effect(const model_t *model)
{
    return model && model->type == MOD_ALIAS && model->meshes &&
           (model->model_class == MCLASS_EXPLOSION || model->model_class == MCLASS_FLASH);
}

static void build_effect_geometry(const refdef_t *fd, const vec3_t view_right, const vec3_t view_up)
{
    num_effect_vertices = 0;
    num_effect_triangles = 0;

    uint32_t slot = current_frame_slot;
    if (!effect_vertex_buffers[slot] || !effect_prim_buffers[slot])
        return;

    effect_writer_t w = {
        .positions = (mtl_packed_float3_t *)effect_vertex_buffers[slot].contents,
        .attrs = (MTLEffectVertex *)effect_attr_buffers[slot].contents,
        .indices = (uint32_t *)effect_index_buffers[slot].contents,
        .prims = (MTLEffectPrim *)effect_prim_buffers[slot].contents,
    };

    const vec3_t world_up = { 0.0f, 0.0f, 1.0f };

    // Particles: write_particle_geometry().
    if (cvar_pt_enable_particles->integer) {
        float particle_size = cvar_pt_particle_size->value;

        for (int i = 0; i < fd->num_particles && effect_room(4, 2); i++) {
            const particle_t *particle = &fd->particles[i];
            MTLEffectPrim prim;
            memset(&prim, 0, sizeof(prim));

            effect_color(particle->color, &particle->rgba, particle->brightness, &prim);
            prim.color.w = particle->alpha;
            prim.type = MTL_EFFECT_PARTICLE;

            vec3_t z_axis, x_axis, y_axis;
            VectorSubtract(fd->vieworg, particle->origin, z_axis);
            VectorNormalize(z_axis);
            CrossProduct(z_axis, view_up, x_axis);
            CrossProduct(x_axis, z_axis, y_axis);

            float size_factor = powf(particle->alpha, 0.05f);
            float size = (particle->radius == 0.0f) ? particle_size * size_factor : particle->radius;
            VectorScale(x_axis, size, x_axis);
            VectorScale(y_axis, size, y_axis);

            vec3_t p0, p1, p2, p3, tmp;
            VectorSubtract(particle->origin, x_axis, tmp); VectorAdd(tmp, y_axis, p0);
            VectorAdd(particle->origin, x_axis, tmp);      VectorAdd(tmp, y_axis, p1);
            VectorAdd(particle->origin, x_axis, tmp);      VectorSubtract(tmp, y_axis, p2);
            VectorSubtract(particle->origin, x_axis, tmp); VectorSubtract(tmp, y_axis, p3);
            write_quad(&w, &prim, p0, p1, p2, p3);
        }
    }

    for (int i = 0; i < fd->num_entities; i++) {
        const entity_t *e = &fd->entities[i];

        if (e->flags & RF_BEAM) {
            // write_beam_geometry(): the quad is a camera facing strip covering
            // the capsule; the shader evaluates the analytic capsule inside it.
            if (!cvar_pt_enable_beams->integer || e->frame == 0)
                continue;

            float beam_radius = cvar_pt_beam_width->value * e->frame * 0.5f;

            MTLEffectPrim prim;
            memset(&prim, 0, sizeof(prim));
            effect_color(e->skinnum, &e->rgba, cvar_pt_particle_emissive->value, &prim);
            prim.color.w = e->alpha;
            prim.type = MTL_EFFECT_BEAM;

            vec3_t begin, end, to_end, dir;
            VectorCopy(e->oldorigin, begin);
            VectorCopy(e->origin, end);
            VectorSubtract(end, begin, to_end);
            float length = VectorLength(to_end);
            if (length < 1e-3f)
                continue;
            VectorScale(to_end, 1.0f / length, dir);

            vec3_t bx, by;
            MakeNormalVectors(dir, bx, by);

            prim.radius = beam_radius;
            prim.length = length;
            prim.world_to_beam[0].x = bx[0];  prim.world_to_beam[0].y = bx[1];  prim.world_to_beam[0].z = bx[2];
            prim.world_to_beam[0].w = -DotProduct(begin, bx);
            prim.world_to_beam[1].x = by[0];  prim.world_to_beam[1].y = by[1];  prim.world_to_beam[1].z = by[2];
            prim.world_to_beam[1].w = -DotProduct(begin, by);
            prim.world_to_beam[2].x = dir[0]; prim.world_to_beam[2].y = dir[1]; prim.world_to_beam[2].z = dir[2];
            prim.world_to_beam[2].w = -DotProduct(begin, dir);

            // Strip perpendicular to both the beam and the view direction.
            vec3_t mid, to_cam, side;
            VectorMA(begin, 0.5f, to_end, mid);
            VectorSubtract(fd->vieworg, mid, to_cam);
            CrossProduct(dir, to_cam, side);
            if (VectorNormalize(side) < 1e-4f)
                VectorCopy(bx, side);
            VectorScale(side, beam_radius, side);

            vec3_t a, b, p0, p1, p2, p3;
            VectorMA(begin, -beam_radius, dir, a);
            VectorMA(end, beam_radius, dir, b);
            VectorSubtract(a, side, p0);
            VectorAdd(a, side, p1);
            VectorAdd(b, side, p2);
            VectorSubtract(b, side, p3);
            write_quad(&w, &prim, p0, p1, p2, p3);
            continue;
        }

        if ((e->model & 0x80000000) || !e->model)
            continue;

        const model_t *model = MOD_ForHandle(e->model);
        if (!model)
            continue;

        if (model_is_effect(model)) {
            write_effect_model(&w, e, model);
            continue;
        }

        if (!cvar_pt_enable_sprites->integer || model->type != MOD_SPRITE || !model->numframes)
            continue;

        // write_sprite_geometry(), reference GL_DrawSpriteModel.
        const mspriteframe_t *frame = &model->spriteframes[e->frame % model->numframes];
        if (!frame->image)
            continue;

        MTLEffectPrim prim;
        memset(&prim, 0, sizeof(prim));
        prim.type = MTL_EFFECT_SPRITE;
        prim.texture = mtl_texture_index_for_handle((qhandle_t)(frame->image - r_images));
        prim.color.w = (e->flags & RF_TRANSLUCENT) ? e->alpha : 1.0f;

        vec3_t up, down, left, right;
        VectorScale(view_right, frame->origin_x, left);
        VectorScale(view_right, frame->origin_x - frame->width, right);
        if (model->sprite_vertical) {
            VectorScale(world_up, -frame->origin_y, down);
            VectorScale(world_up, frame->height - frame->origin_y, up);
        } else {
            VectorScale(view_up, -frame->origin_y, down);
            VectorScale(view_up, frame->height - frame->origin_y, up);
        }

        vec3_t p0, p1, p2, p3;
        VectorAdd3(e->origin, down, left, p0);
        VectorAdd3(e->origin, up, left, p1);
        VectorAdd3(e->origin, up, right, p2);
        VectorAdd3(e->origin, down, right, p3);
        write_quad(&w, &prim, p0, p1, p2, p3);
    }
}

// Builds the effects BLAS and its single instance TLAS for this frame.
static bool refresh_effect_accel(id<MTLCommandBuffer> cmd)
{
    if (!num_effect_triangles)
        return false;

    uint32_t slot = current_frame_slot;

    MTLAccelerationStructureTriangleGeometryDescriptor *geo =
        [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    geo.vertexBuffer = effect_vertex_buffers[slot];
    geo.vertexStride = sizeof(mtl_packed_float3_t);
    geo.indexBuffer = effect_index_buffers[slot];
    geo.indexType = MTLIndexTypeUInt32;
    geo.triangleCount = num_effect_triangles;
    geo.opaque = YES;

    MTLPrimitiveAccelerationStructureDescriptor *desc =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    desc.geometryDescriptors = @[geo];

    MTLAccelerationStructureSizes sizes = [mtl.device accelerationStructureSizesWithDescriptor:desc];

    if (!effect_blas[slot] || effect_blas[slot].size < sizes.accelerationStructureSize) {
        [effect_blas[slot] release];
        effect_blas[slot] = [mtl.device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
        effect_blas[slot].label = @"effects blas";
    }
    if (!effect_scratch[slot] || effect_scratch[slot].length < sizes.buildScratchBufferSize) {
        [effect_scratch[slot] release];
        effect_scratch[slot] = [mtl.device newBufferWithLength:max(sizes.buildScratchBufferSize, 1)
                                                      options:MTLResourceStorageModePrivate];
    }
    if (!effect_blas[slot] || !effect_scratch[slot])
        return false;

    MTLAccelerationStructureInstanceDescriptor *instance =
        (MTLAccelerationStructureInstanceDescriptor *)effect_instance_buffers[slot].contents;
    set_identity_instance(instance, 0, 0xff, true);

    MTLInstanceAccelerationStructureDescriptor *tdesc =
        [MTLInstanceAccelerationStructureDescriptor descriptor];
    tdesc.instancedAccelerationStructures = @[effect_blas[slot]];
    tdesc.instanceCount = 1;
    tdesc.instanceDescriptorBuffer = effect_instance_buffers[slot];

    MTLAccelerationStructureSizes tsizes = [mtl.device accelerationStructureSizesWithDescriptor:tdesc];

    if (!effect_tlas[slot] || effect_tlas[slot].size < tsizes.accelerationStructureSize) {
        [effect_tlas[slot] release];
        effect_tlas[slot] = [mtl.device newAccelerationStructureWithSize:tsizes.accelerationStructureSize];
        effect_tlas[slot].label = @"effects tlas";
    }
    if (!effect_tlas_scratch[slot] || effect_tlas_scratch[slot].length < tsizes.buildScratchBufferSize) {
        [effect_tlas_scratch[slot] release];
        effect_tlas_scratch[slot] = [mtl.device newBufferWithLength:max(tsizes.buildScratchBufferSize, 1)
                                                           options:MTLResourceStorageModePrivate];
    }
    if (!effect_tlas[slot] || !effect_tlas_scratch[slot])
        return false;

    id<MTLAccelerationStructureCommandEncoder> enc = mtl_profiler_accel_encoder(cmd, MTL_PROFILER_EFFECTS_BLAS);
    [enc buildAccelerationStructure:effect_blas[slot]
                         descriptor:desc
                      scratchBuffer:effect_scratch[slot]
                scratchBufferOffset:0];
    [enc buildAccelerationStructure:effect_tlas[slot]
                         descriptor:tdesc
                      scratchBuffer:effect_tlas_scratch[slot]
                scratchBufferOffset:0];
    [enc endEncoding];

    return true;
}

static void draw_shadowed_string(int x, int y, int flags, size_t maxlen, const char *s)
{
    R_SetColor(0xff000000u);
    SCR_DrawStringEx(x + 1, y + 1, flags, maxlen, s, SCR_GetFont());
    R_SetColor(~0u);
    SCR_DrawStringEx(x, y, flags, maxlen, s, SCR_GetFont());
}

// Port of evaluate_reference_mode(): while the game is paused, accumulate a
// converged image instead of denoising, with a progress HUD.
static void evaluate_reference_mode(void)
{
    bool active = cl_paused->integer == 2 && sv_paused->integer && cvar_pt_accumulation_rendering->integer > 0;

    if (!active) {
        num_accumulated_frames = 0;
        accumulation_active = false;
        temporal_blend_factor = 0.0f;
        return;
    }

    if (reset_accumulation_requested) {
        num_accumulated_frames = 0;
        reset_accumulation_requested = false;
    }
    num_accumulated_frames++;

    const int num_warmup_frames = 5;
    const int num_frames_to_accumulate = max(128, cvar_pt_accumulation_rendering_framenum->integer);

    accumulation_active = true;
    temporal_blend_factor = 1.0f / min(max(1, num_accumulated_frames - num_warmup_frames), num_frames_to_accumulate);

    switch (cvar_pt_accumulation_rendering->integer) {
    case 1: {
        char text[MAX_QPATH];
        float percentage = powf(max(0.0f, (num_accumulated_frames - num_warmup_frames) / (float)num_frames_to_accumulate), 0.5f);
        Q_snprintf(text, sizeof(text), "Photo mode: accumulating samples... %d%%", (int)(min(1.0f, percentage) * 100.0f));

        int frames_after = num_accumulated_frames - num_warmup_frames - num_frames_to_accumulate;
        float hud_alpha = max(0.0f, min(1.0f, (50 - frames_after) * 0.02f));

        int x = r_config.width / 4;
        int y = 30;
        R_SetScale(0.5f);
        R_SetAlphaScale(hud_alpha);
        draw_shadowed_string(x, y, UI_CENTER, MAX_QPATH, text);

        if (cvar_pt_dof->integer) {
            x = 5;
            y = r_config.height / 2 - 55;
            Q_snprintf(text, sizeof(text), "Focal Distance: %.1f", cvar_pt_focus->value);
            draw_shadowed_string(x, y, UI_LEFT, MAX_QPATH, text);
            y += 10;
            Q_snprintf(text, sizeof(text), "Aperture: %.2f", cvar_pt_aperture->value);
            draw_shadowed_string(x, y, UI_LEFT, MAX_QPATH, text);
            y += 10;
            draw_shadowed_string(x, y, UI_LEFT, MAX_QPATH, "Use Mouse Wheel, Shift, Ctrl to adjust");
        }

        R_SetAlphaScale(1.0f);
        R_SetScale(1.0f);
        SCR_SetHudAlpha(hud_alpha);
        break;
    }
    case 2:
        SCR_SetHudAlpha(0.0f);
        break;
    }
}

//
// Camera matrices (matrix.c), column major like the GLSL side.
//

static void mtl_create_projection_matrix(float matrix[16], float znear, float zfar, float fov_x, float fov_y)
{
    float ymax = znear * tanf(fov_y * (float)M_PI / 360.0f);
    float ymin = -ymax;
    float xmax = znear * tanf(fov_x * (float)M_PI / 360.0f);
    float xmin = -xmax;

    float width = xmax - xmin;
    float height = ymax - ymin;
    float depth = zfar - znear;

    matrix[0] = 2 * znear / width;
    matrix[4] = 0;
    matrix[8] = (xmax + xmin) / width;
    matrix[12] = 0;

    matrix[1] = 0;
    matrix[5] = -2 * znear / height;
    matrix[9] = (ymax + ymin) / height;
    matrix[13] = 0;

    matrix[2] = 0;
    matrix[6] = 0;
    matrix[10] = (zfar + znear) / depth;
    matrix[14] = 2 * zfar * znear / depth;

    matrix[3] = 0;
    matrix[7] = 0;
    matrix[11] = 1;
    matrix[15] = 0;
}

static void mtl_create_view_matrix(float matrix[16], const refdef_t *fd)
{
    vec3_t viewaxis[3];
    AnglesToAxis(fd->viewangles, viewaxis);

    matrix[0]  = -viewaxis[1][0];
    matrix[4]  = -viewaxis[1][1];
    matrix[8]  = -viewaxis[1][2];
    matrix[12] = DotProduct(viewaxis[1], fd->vieworg);

    matrix[1]  = viewaxis[2][0];
    matrix[5]  = viewaxis[2][1];
    matrix[9]  = viewaxis[2][2];
    matrix[13] = -DotProduct(viewaxis[2], fd->vieworg);

    matrix[2]  = viewaxis[0][0];
    matrix[6]  = viewaxis[0][1];
    matrix[10] = viewaxis[0][2];
    matrix[14] = -DotProduct(viewaxis[0], fd->vieworg);

    matrix[3]  = 0;
    matrix[7]  = 0;
    matrix[11] = 0;
    matrix[15] = 1;
}

static void mtl_inverse(const float m[16], float inv[16])
{
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] +
             m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] -
             m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] +
             m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] -
             m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] +
             m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] -
             m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] +
             m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] -
             m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] +
              m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] -
              m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] -
              m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] +
              m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] -
              m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

    float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    det = 1.0f / det;
    for (int i = 0; i < 16; i++)
        inv[i] *= det;
}

static void mtl_mult_matrix_matrix(float p[16], const float a[16], const float b[16])
{
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            p[i * 4 + j] = a[0 * 4 + j] * b[i * 4 + 0] + a[1 * 4 + j] * b[i * 4 + 1] +
                           a[2 * 4 + j] * b[i * 4 + 2] + a[3 * 4 + j] * b[i * 4 + 3];
        }
    }
}

static uint32_t float_to_half(float f)
{
    _Float16 h = (_Float16)f;
    uint16_t u;
    memcpy(&u, &h, sizeof(u));
    return u;
}

static inline ubo_vec3 ubo_vec3_from(const float *v)
{
    return (ubo_vec3){ v[0], v[1], v[2] };
}

// main.c add_dlights()
static void add_dlights(const refdef_t *fd, GlobalUbo *ubo)
{
    ubo->num_dyn_lights = 0;

    for (int i = 0; i < fd->num_dlights && ubo->num_dyn_lights < MAX_LIGHT_SOURCES; i++) {
        const dlight_t *light = &fd->dlights[i];
        VkptDynLight *d = &ubo->dyn_light_data[ubo->num_dyn_lights];
        memset(d, 0, sizeof(*d));

        d->center = ubo_vec3_from(light->origin);
        vec3_t color;
        VectorScale(light->color, light->intensity / 25.0f, color);
        d->color = ubo_vec3_from(color);
        d->radius = light->radius;

        switch (light->light_type) {
        case DLIGHT_SPHERE:
            d->type = DYNLIGHT_SPHERE;
            break;
        case DLIGHT_SPOT:
            d->type = DYNLIGHT_SPOT;
            d->spot_direction = ubo_vec3_from(light->spot.direction);
            switch (light->spot.emission_profile) {
            case DLIGHT_SPOT_EMISSION_PROFILE_FALLOFF:
                d->type |= DYNLIGHT_SPOT_EMISSION_PROFILE_FALLOFF << 16;
                d->spot_data = float_to_half(light->spot.cos_total_width) |
                               (float_to_half(light->spot.cos_falloff_start) << 16);
                break;
            case DLIGHT_SPOT_EMISSION_PROFILE_AXIS_ANGLE_TEXTURE:
                d->type |= DYNLIGHT_SPOT_EMISSION_PROFILE_AXIS_ANGLE_TEXTURE << 16;
                d->spot_data = float_to_half(light->spot.total_width) |
                               (mtl_texture_index_for_handle(light->spot.texture) << 16);
                break;
            }
            break;
        }

        ubo->num_dyn_lights++;
    }
}

// Rotates a direction into the sky box frame the way env_map() does.
static void rotate_to_envmap(const vec3_t dir, float angle, const vec3_t axis, vec3_t out)
{
    float c = cosf(angle), s = sinf(angle);
    vec3_t cr;
    CrossProduct(axis, dir, cr);
    float d = DotProduct(axis, dir);
    for (int i = 0; i < 3; i++)
        out[i] = dir[i] * c + cr[i] * s + axis[i] * d * (1.0f - c);
}

typedef struct {
    bool  render_world;
    bool  include_world;
    bool  fsr_enabled;      // vkpt_fsr_is_enabled()
    bool  enable_denoiser;
    bool  enable_accumulation;
    float num_bounce_rays;
    int   reflect_refract;
} frame_mode_t;

// main.c prepare_ubo() with vkpt_physical_sky_update_ubo() and
// vkpt_god_rays_prepare_ubo() folded in, into the persistent host copy.
static void prepare_ubo(const refdef_t *fd, const frame_mode_t *mode, const mtl_sun_light_t *sun)
{
    GlobalUbo *ubo = &ubo_host;

    ubo->V_prev = ubo->V;
    ubo->P_prev = ubo->P;
    ubo->invP_prev = ubo->invP;
    ubo->cylindrical_hfov_prev = ubo->cylindrical_hfov;
    ubo->prev_taa_output_width = ubo->taa_output_width;
    ubo->prev_taa_output_height = ubo->taa_output_height;

    float P[16], V[16];
    {
        float raw_proj[16];
        mtl_create_projection_matrix(raw_proj, 1.0f, 4096.0f, fd->fov_x, fd->fov_y);

        // The player setup view covers part of the screen: a projection
        // adjustment maps it onto its rectangle (viewport_proj).
        float viewport_proj[16] = {
            [0] = (float)fd->width / (float)display_width,
            [12] = (float)(fd->x * 2 + fd->width - display_width) / (float)display_width,
            [5] = (float)fd->height / (float)display_height,
            [13] = -(float)(fd->y * 2 + fd->height - display_height) / (float)display_height,
            [10] = 1.0f,
            [15] = 1.0f,
        };
        mtl_mult_matrix_matrix(P, viewport_proj, raw_proj);
    }
    mtl_create_view_matrix(V, fd);
    memcpy(ubo->V.m, V, sizeof(V));
    memcpy(ubo->P.m, P, sizeof(P));
    mtl_inverse(V, ubo->invV.m);
    mtl_inverse(P, ubo->invP.m);

    float vfov = fd->fov_y * (float)M_PI / 180.0f;
    float unscaled_aspect = (float)unscaled_width / (float)unscaled_height;
    float fov_scale[2] = { 0.0f, 0.0f };

    switch (cvar_pt_projection->integer) {
    case PROJECTION_PANINI:
        fov_scale[1] = tanf(vfov / 2.0f);
        fov_scale[0] = fov_scale[1] * unscaled_aspect;
        break;
    case PROJECTION_STEREOGRAPHIC:
        fov_scale[1] = tanf(vfov / 2.0f * STEREOGRAPHIC_ANGLE);
        fov_scale[0] = fov_scale[1] * unscaled_aspect;
        break;
    case PROJECTION_CYLINDRICAL: {
        float rad_per_pixel = atanf(tanf(fd->fov_y * (float)M_PI / 360.0f) / ((float)unscaled_height * 0.5f));
        ubo->cylindrical_hfov = rad_per_pixel * (float)unscaled_width;
        break;
    }
    case PROJECTION_EQUIRECTANGULAR:
        fov_scale[1] = vfov / 2.0f;
        fov_scale[0] = fov_scale[1] * unscaled_aspect;
        break;
    case PROJECTION_MERCATOR:
        fov_scale[1] = logf(tanf((float)M_PI * 0.25f + (vfov / 2.0f) * 0.5f));
        fov_scale[0] = fov_scale[1] * unscaled_aspect;
        break;
    }

    ubo->projection_fov_scale_prev = ubo->projection_fov_scale;
    ubo->projection_fov_scale = (ubo_vec2){ fov_scale[0], fov_scale[1] };
    // Always rectilinear for the player setup view.
    ubo->pt_projection = mode->render_world ? cvar_pt_projection->integer : 0;
    ubo->current_frame_idx = (int)pt_frame_counter;
    ubo->width = render_width;
    ubo->height = render_height;
    ubo->prev_width = prev_render_width ? prev_render_width : render_width;
    ubo->prev_height = prev_render_height ? prev_render_height : render_height;
    ubo->inv_width = 1.0f / (float)render_width;
    ubo->inv_height = 1.0f / (float)render_height;
    ubo->unscaled_width = unscaled_width;
    ubo->unscaled_height = unscaled_height;
    ubo->taa_image_width = screen_image_width;
    ubo->taa_image_height = screen_image_height;
    ubo->taa_output_width = taa_output_width;
    ubo->taa_output_height = taa_output_height;
    ubo->current_gpu_slice_width = render_width;
    ubo->prev_gpu_slice_width = ubo->prev_width;
    ubo->screen_image_width = screen_image_width;
    ubo->screen_image_height = screen_image_height;
    ubo->water_normal_texture = water_normal_image
        ? (int)mtl_texture_index_optional((qhandle_t)(water_normal_image - r_images)) : 0;
    ubo->pt_swap_checkerboard = 0;
    ubo->ui_color_scale = 1.0f;

    int contents = 0;
    if (mode->include_world && world_bsp && world_bsp->nodes) {
        const mleaf_t *leaf = BSP_PointLeaf(world_bsp->nodes, fd->vieworg);
        contents = leaf ? leaf->contents : 0;
    }
    if (contents & CONTENTS_WATER)
        ubo->medium = MEDIUM_WATER;
    else if (contents & CONTENTS_SLIME)
        ubo->medium = MEDIUM_SLIME;
    else if (contents & CONTENTS_LAVA)
        ubo->medium = MEDIUM_LAVA;
    else
        ubo->medium = MEDIUM_NONE;

    ubo->time = fd->time;
    ubo->num_static_primitives = mode->include_world ? num_sky_first_index / 3 : 0;
    ubo->num_static_lights = mode->include_world ? (int)num_light_polys : 0;

    vkpt_fog_upload(ubo->fog_volumes);

#define UBO_CVAR_DO(name, default_value) ubo->name = ubo_cvar_##name->value;
    VKPT_UBO_CVAR_LIST
#undef UBO_CVAR_DO

    if (!mode->enable_denoiser) {
        // No fake specular without the denoiser; it looks too dark without.
        ubo->pt_fake_roughness_threshold = 1.0f;

        // Swap the checkerboard fields every frame so that every pixel
        // accumulates both reflection and refraction.
        ubo->pt_swap_checkerboard = (int)(pt_frame_counter & 1);

        if (mode->enable_accumulation) {
            ubo->pt_texture_lod_bias = -log2f(sqrtf((float)max(128, cvar_pt_accumulation_rendering_framenum->integer)));

            // Disable the other stabilization hacks.
            ubo->pt_specular_anti_flicker = 0.0f;
            ubo->pt_sun_bounce_range = 10000.0f;
            ubo->pt_ndf_trim = 1.0f;
        }
    } else if (mode->fsr_enabled || effective_aa_mode == AA_MODE_UPSCALE) {
        // Negative texture LOD bias to match the resolution scale.
        float resolution_scale = (drs_effective_scale != 0) ? (float)drs_effective_scale : (float)cvar_scr_viewsize->integer;
        resolution_scale = Q_clipf(resolution_scale * 0.01f, 0.1f, 1.0f);
        ubo->pt_texture_lod_bias = ubo_cvar_pt_texture_lod_bias->value + log2f(resolution_scale);
    }

    {
        bool enable_dof;
        switch (cvar_pt_dof->integer) {
        case 0:  enable_dof = false; break;
        case 1:  enable_dof = mode->enable_accumulation; break;
        case 2:  enable_dof = !mode->enable_denoiser; break;
        default: enable_dof = true; break;
        }
        // No physical meaning with the other projections.
        if (cvar_pt_projection->integer != 0)
            enable_dof = false;
        if (!enable_dof)
            ubo->pt_aperture = 0.0f;
    }

    ubo->pt_aperture_type = roundf(ubo->pt_aperture_type);

    ubo->temporal_blend_factor = temporal_blend_factor;
    ubo->flt_enable = mode->enable_denoiser;
    ubo->flt_taa = (float)effective_aa_mode;
    ubo->pt_num_bounce_rays = mode->num_bounce_rays;
    ubo->pt_reflect_refract = (float)mode->reflect_refract;

    if (mode->num_bounce_rays < 1.0f)
        ubo->pt_specular_mis = 0;   // no MIS without specular rays

    ubo->pt_min_log_sky_luminance = exp2f(ubo->pt_min_log_sky_luminance);
    ubo->pt_max_log_sky_luminance = exp2f(ubo->pt_max_log_sky_luminance);

    ubo->cam_pos = (ubo_vec4){ fd->vieworg[0], fd->vieworg[1], fd->vieworg[2], 0.0f };
    ubo->cluster_debug_index = -1;

    if (!temporal_frame_valid) {
        ubo->flt_temporal_lf = 0;
        ubo->flt_temporal_hf = 0;
        ubo->flt_temporal_spec = 0;
        ubo->flt_taa = 0;
    }

    if (effective_aa_mode == AA_MODE_UPSCALE) {
        int taa_index = (int)(pt_frame_counter % NUM_TAA_SAMPLES);
        ubo->sub_pixel_jitter = (ubo_vec2){ taa_samples[taa_index][0], taa_samples[taa_index][1] };
    } else {
        ubo->sub_pixel_jitter = (ubo_vec2){ 0.0f, 0.0f };
    }

    ubo->first_person_model = Cvar_VariableInteger("cl_player_model") == CL_PLAYER_MODEL_FIRST_PERSON;
    ubo->weapon_left_handed = Cvar_VariableInteger("hand") == 1;

    add_dlights(fd, ubo);

    if (num_cameras > 0 && mode->include_world) {
        for (uint32_t n = 0; n < num_cameras; n++)
            prepare_camera(cameras[n].pos, cameras[n].dir, (mtl_float4 *)ubo->security_camera_data[n].m);
    } else {
        ubo->pt_cameras = 0;
    }
    ubo->num_cameras = mode->include_world ? (int)num_cameras : 0;

    // vkpt_physical_sky_update_ubo()
    ubo->pt_env_scale = cvar_physical_sky_space->integer ? 0.3f : mtl_pt_env_scale();
    ubo->sun_bounce_scale = Cvar_VariableValue("sun_bounce");

    float sky_angle = DEG2RAD((sky_autorotate ? fd->time : 1.0f) * sky_rotate);
    ubo->sky_rotate = sky_angle;
    ubo->sky_axis = ubo_vec3_from(sky_axis);

    mtl_sun_light_t light;
    memset(&light, 0, sizeof(light));
    if (mode->render_world && sun)
        light = *sun;

    ubo->sun_tan_half_angle = tanf(light.angular_size_rad * 0.5f);
    ubo->sun_cos_half_angle = cosf(light.angular_size_rad * 0.5f);
    ubo->sun_solid_angle = 2.0f * (float)M_PI * (float)(1.0 - cos(light.angular_size_rad * 0.5));
    ubo->sun_color = ubo_vec3_from(light.color);
    ubo->sun_direction = ubo_vec3_from(light.direction);
    {
        vec3_t envmap_dir;
        rotate_to_envmap(light.direction, sky_angle, sky_axis, envmap_dir);
        ubo->sun_direction_envmap = ubo_vec3_from(envmap_dir);
    }

    if (light.direction[2] >= 0.99f) {
        ubo->sun_tangent = (ubo_vec3){ 1.0f, 0.0f, 0.0f };
        ubo->sun_bitangent = (ubo_vec3){ 0.0f, 1.0f, 0.0f };
    } else {
        vec3_t up = { 0.0f, 0.0f, 1.0f }, tangent, bitangent;
        CrossProduct(light.direction, up, tangent);
        VectorNormalize(tangent);
        CrossProduct(light.direction, tangent, bitangent);
        VectorNormalize(bitangent);
        ubo->sun_tangent = ubo_vec3_from(tangent);
        ubo->sun_bitangent = ubo_vec3_from(bitangent);
    }

    if (!mode->render_world)
        ubo->environment_type = ENVIRONMENT_NONE;
    else if (light.use_physical_sky)
        ubo->environment_type = ENVIRONMENT_DYNAMIC;
    else
        ubo->environment_type = ENVIRONMENT_STATIC;
    ubo->sun_visible = light.use_physical_sky && light.visible;

    {
        vec3_t avg;
        mtl_physical_sky_average_color(avg);
        ubo->sky_luminance = avg[0] * 0.299f + avg[1] * 0.587f + avg[2] * 0.114f;
    }

    // vkpt_god_rays_prepare_ubo()
    {
        vec3_t center, size;
        VectorAdd(world_aabb_min, world_aabb_max, center);
        VectorScale(center, 0.5f, center);
        VectorSubtract(world_aabb_max, world_aabb_min, size);
        ubo->world_center = (ubo_vec4){ center[0], center[1], center[2], 0.0f };
        ubo->world_size = (ubo_vec4){ size[0], size[1], size[2], 0.0f };
        ubo->world_half_size_inv = (ubo_vec4){ 2.0f / max(size[0], 1.0f), 2.0f / max(size[1], 1.0f),
                                               2.0f / max(size[2], 1.0f), 0.0f };
        ubo->god_rays_intensity = max(0.0f, cvar_gr_intensity->value);
        ubo->god_rays_eccentricity = cvar_gr_eccentricity->value;
    }

    // Metal additions.
    ubo->world_transparent_first_prim = num_opaque_indices / 3;
    ubo->world_sky_first_prim = num_sky_first_index / 3;
    ubo->world_masked_first_prim = num_masked_first_index / 3;
    ubo->entity_transparent_first_prim = entity_transparent_first_index / 3;
    ubo->entity_viewer_first_prim = entity_viewer_first_index / 3;
    ubo->entity_weapon_first_prim = entity_weapon_first_index / 3;
    ubo->num_clusters = mode->include_world ? (uint32_t)num_clusters : 0;
    ubo->light_stats_size = light_stats_size;
    if (!light_stats_size)
        ubo->pt_light_stats = 0.0f;
    ubo->has_skybox = sky_texture ? 1 : 0;
    ubo->has_physical_sky = mtl_physical_sky_active() ? 1 : 0;
    ubo->viewport_scale = (ubo_vec2){ 1.0f, 1.0f };
    ubo->viewport_offset = (ubo_vec2){ 0.0f, 0.0f };
    ubo->lava_emissive = cvar_lava_emissive->value;
    ubo->has_world = mode->include_world ? 1 : 0;
    ubo->has_masked_world = (mode->include_world && world_masked_blas) ? 1 : 0;
    ubo->material_filter = (uint32_t)Q_clip(cvar_pt_nearest->integer, 0, 2);
}

//
// Pass recording
//

typedef struct {
    id<MTLCommandBuffer> cmd;
    uint32_t slot;
    uint32_t prev_slot;     // frame slot of the previous traced frame
    int      parity;        // image table
    bool     include_world;
    bool     have_effects;
} frame_ctx_t;

#define OR_DUMMY(b) ((b) ? (b) : dummy_buffer)

// The scene bindings of PT_KERNEL_PARAMS.
static void bind_scene(id<MTLComputeCommandEncoder> enc, const frame_ctx_t *fc)
{
    uint32_t slot = fc->slot;

    [enc setBuffer:mtl_texture_argument_buffer() offset:0 atIndex:VKPT_BUF_TEXTURES];
    [enc setBuffer:OR_DUMMY(vertex_buffer) offset:0 atIndex:VKPT_BUF_WORLD_VERTICES];
    [enc setBuffer:OR_DUMMY(index_buffer) offset:0 atIndex:VKPT_BUF_WORLD_INDICES];
    [enc setBuffer:entity_vertex_buffers[slot] offset:0 atIndex:VKPT_BUF_ENT_VERTICES];
    [enc setBuffer:entity_index_buffers[slot] offset:0 atIndex:VKPT_BUF_ENT_INDICES];
    [enc setBuffer:OR_DUMMY(material_buffers[slot]) offset:0 atIndex:VKPT_BUF_MATERIALS];
    [enc setBuffer:OR_DUMMY(frame_light_poly_buffers[slot] ? frame_light_poly_buffers[slot] : light_poly_buffer)
            offset:0 atIndex:VKPT_BUF_LIGHT_POLYS];
    [enc setBuffer:OR_DUMMY(light_list_offset_buffers[slot]) offset:0 atIndex:VKPT_BUF_LIGHT_OFFSETS];
    [enc setBuffer:OR_DUMMY(light_list_light_buffers[slot]) offset:0 atIndex:VKPT_BUF_LIGHT_LIGHTS];
    [enc setBuffer:OR_DUMMY(sky_visibility_buffer) offset:0 atIndex:VKPT_BUF_SKY_VISIBILITY];
    [enc setBuffer:light_counts_buffer offset:0 atIndex:VKPT_BUF_LIGHT_COUNTS];
    [enc setBuffer:OR_DUMMY(light_stats_buffer) offset:0 atIndex:VKPT_BUF_LIGHT_STATS];
    [enc setBuffer:effect_prim_buffers[slot] offset:0 atIndex:VKPT_BUF_EFFECT_PRIMS];
    [enc setBuffer:effect_attr_buffers[slot] offset:0 atIndex:VKPT_BUF_EFFECT_VERTS];
    [enc setBuffer:effect_index_buffers[slot] offset:0 atIndex:VKPT_BUF_EFFECT_INDICES];
    [enc setBuffer:entity_table_buffers[slot] offset:0 atIndex:VKPT_BUF_ENTITY_TABLE];
    [enc setBuffer:entity_table_buffers[fc->prev_slot] offset:0 atIndex:VKPT_BUF_ENTITY_TABLE_PREV];

    [enc setAccelerationStructure:tlas[slot] atBufferIndex:VKPT_BUF_ACCEL];
    use_scene_structures(enc, slot, fc->include_world);
    if (fc->have_effects) {
        [enc setAccelerationStructure:effect_tlas[slot] atBufferIndex:VKPT_BUF_EFFECTS_ACCEL];
        [enc useResource:effect_blas[slot] usage:MTLResourceUsageRead];
    } else {
        // A structure must be bound even when nothing is traced through it.
        [enc setAccelerationStructure:tlas[slot] atBufferIndex:VKPT_BUF_EFFECTS_ACCEL];
    }

    [enc setTexture:sky_texture ? sky_texture : dummy_sky_texture atIndex:VKPT_TEX_SKY];
    [enc setTexture:mtl_physical_sky_texture() atIndex:VKPT_TEX_PHYSICAL_SKY];
    mtl_textures_encode_use_compute(enc);
}

// An encoder with the UBO, the screen images and the blue noise bound, plus
// the scene for the passes that trace rays.
static id<MTLComputeCommandEncoder> begin_pass(const frame_ctx_t *fc, int profiler_entry, NSString *label, bool scene)
{
    id<MTLComputeCommandEncoder> enc = mtl_profiler_compute_encoder(fc->cmd, profiler_entry);
    enc.label = label;
    [enc setBuffer:uniform_buffers[fc->slot] offset:0 atIndex:VKPT_BUF_UBO];
    [enc setBuffer:image_tables[fc->parity] offset:0 atIndex:VKPT_BUF_IMAGES];
    [enc setTexture:blue_noise_texture atIndex:VKPT_TEX_BLUE_NOISE];
    [enc useResources:(const id<MTLResource> *)vkpt_images
                count:VKPT_NUM_IMAGES
                usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    if (scene)
        bind_scene(enc, fc);
    return enc;
}

// The equivalent of vkpt's BARRIER_COMPUTE between dependent dispatches.
static void barrier(id<MTLComputeCommandEncoder> enc)
{
    [enc memoryBarrierWithScope:MTLBarrierScopeTextures | MTLBarrierScopeBuffers];
}

static void dispatch(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                     int bounce_index, uint32_t iteration, MTLSize groups, MTLSize threads)
{
    VkptPush push = { .bounce_index = bounce_index, .iteration = iteration };
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&push length:sizeof(push) atIndex:VKPT_BUF_PUSH];
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

// path_tracer.c dispatch_rays(): half the width per checkerboard field, the
// two fields in z.
static void dispatch_rays(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                          int bounce_index, uint32_t iteration, int height)
{
    dispatch(enc, pipeline, bounce_index, iteration,
             MTLSizeMake((render_width / 2 + 7) / 8, (height + 7) / 8, 2), MTLSizeMake(8, 8, 1));
}

static inline MTLSize groups_16(int width, int height)
{
    return MTLSizeMake((width + 15) / 16, (height + 15) / 16, 1);
}

static void record_god_rays_trace(const frame_ctx_t *fc, int profiler_entry, const MTLGodRaysUniforms *g, uint32_t pass)
{
    id<MTLComputeCommandEncoder> enc = begin_pass(fc, profiler_entry, pass ? @"god rays reflect/refract" : @"god rays", false);
    [enc setBytes:g length:sizeof(*g) atIndex:VKPT_BUF_PUSH + 1];
    [enc setTexture:tex_shadow_map atIndex:3];
    [enc setTexture:tex_god_rays atIndex:4];
    dispatch(enc, pipeline_god_rays, 0, pass,
             MTLSizeMake((render_width + 15) / 16, (render_height + 15) / 16, 1), MTLSizeMake(8, 8, 1));
    [enc endEncoding];
}

static void record_god_rays_filter(const frame_ctx_t *fc)
{
    id<MTLComputeCommandEncoder> enc = begin_pass(fc, MTL_PROFILER_GOD_RAYS_FILTER, @"god rays filter", false);
    [enc setTexture:tex_god_rays atIndex:4];
    [enc setTexture:vkpt_images[VKPT_IMG_PT_TRANSPARENT] atIndex:5];
    dispatch(enc, pipeline_god_rays_filter, 0, 0,
             MTLSizeMake((render_width + 7) / 8, (render_height + 7) / 8, 1), MTLSizeMake(8, 8, 1));
    [enc endEncoding];
}

// asvgf.c vkpt_asvgf_filter()
static void record_asvgf_filter(const frame_ctx_t *fc, bool enable_lf)
{
    MTLSize grad_groups = groups_16(render_width / GRAD_DWN, render_height / GRAD_DWN);
    MTLSize full_groups = groups_16(render_width, render_height);
    MTLSize threads_16 = MTLSizeMake(16, 16, 1);

    id<MTLComputeCommandEncoder> enc = begin_pass(fc, MTL_PROFILER_DENOISE_GRADIENT, @"asvgf gradient", false);
    dispatch(enc, pipeline_gradient_img, 0, 0, grad_groups, threads_16);
    for (uint32_t i = 0; i < 7; i++) {
        barrier(enc);
        dispatch(enc, pipeline_gradient_atrous, 0, i, grad_groups, threads_16);
    }
    [enc endEncoding];

    enc = begin_pass(fc, MTL_PROFILER_DENOISE_TEMPORAL, @"asvgf temporal", false);
    dispatch(enc, pipeline_temporal, 0, 0,
             MTLSizeMake((render_width + 14) / 15, (render_height + 14) / 15, 1), MTLSizeMake(15, 15, 1));
    [enc endEncoding];

    enc = begin_pass(fc, MTL_PROFILER_DENOISE_ATROUS, @"asvgf atrous", false);
    for (uint32_t i = 0; i < 4; i++) {
        if (enable_lf) {
            dispatch(enc, pipeline_atrous_lf, 0, i, grad_groups, threads_16);
            barrier(enc);
        }
        dispatch(enc, pipeline_atrous, enable_lf ? 1 : 0, i, full_groups, threads_16);
        barrier(enc);
    }
    [enc endEncoding];
}

void mtl_pt_render(id<MTLCommandBuffer> cmd, const refdef_t *fd_in)
{
    if (resize_pending)
        apply_resize();
    if (!pipeline_primary_rays || !vkpt_images[0])
        return;

    // The freecam edits the view, so work on a copy of the refdef.
    refdef_t fd_copy = *fd_in;
    refdef_t *fd = &fd_copy;
    camera_cmd_refdef = fd_in;

    // RDF_NOWORLDMODEL: the player setup view, models only, lit by the refdef
    // dlights, in a sub-rectangle of the screen (vkpt render_world == false).
    bool render_world = (fd->rdflags & RDF_NOWORLDMODEL) == 0;

    if (render_world && !world_ready) {
        drs_last_frame_world = false;
        return;
    }

    uint32_t slot = mtl.frame_index;
    current_frame_slot = slot;
    last_rdflags_underwater = (fd->rdflags & RDF_UNDERWATER) != 0;
    current_rdflags = fd->rdflags;

    float frame_time = (prev_frame_time > 0.0f) ? Q_clipf(fd->time - prev_frame_time, 0.0f, 1.0f) : 0.0f;

    if (render_world) {
        if (mtl_freecam_update(fd, frame_time > 0.0f ? frame_time : 1.0f / 60.0f))
            reset_accumulation_requested = true;
    }

    // Debug aid: `mtl_view_override "x y z pitch yaw"` renders from a fixed
    // camera, which makes scripted screenshots of a spot reproducible.
    if (render_world && cvar_mtl_view_override->string[0]) {
        float v[5];
        if (sscanf(cvar_mtl_view_override->string, "%f %f %f %f %f", &v[0], &v[1], &v[2], &v[3], &v[4]) == 5) {
            VectorSet(fd->vieworg, v[0], v[1], v[2]);
            VectorSet(fd->viewangles, v[3], v[4], 0.0f);
        }
    }

    // evaluate_reference_mode()
    evaluate_reference_mode();
    frame_mode_t mode;
    mode.render_world = render_world;
    mode.include_world = render_world && world_ready;
    mode.enable_accumulation = accumulation_active;
    if (accumulation_active) {
        mode.enable_denoiser = false;
        mode.num_bounce_rays = 2.0f;
        mode.reflect_refract = max(4, ubo_cvar_pt_reflect_refract->integer);
    } else {
        mode.enable_denoiser = ubo_cvar_flt_enable->integer != 0;
        if (ubo_cvar_pt_num_bounce_rays->value == 0.5f)
            mode.num_bounce_rays = 0.5f;
        else
            mode.num_bounce_rays = (float)max(0, min(2, (int)roundf(ubo_cvar_pt_num_bounce_rays->value)));
        mode.reflect_refract = max(0, ubo_cvar_pt_reflect_refract->integer);
    }
    mode.reflect_refract = min(10, mode.reflect_refract);

    drs_process();
    drs_last_frame_world = render_world;

    // get_render_extent(): the width stays even for the checkerboard fields.
    {
        int scale;
        if (drs_effective_scale) {
            scale = drs_effective_scale;
        } else {
            scale = cvar_scr_viewsize->integer;
            if (cvar_drs_enable->integer)
                scale = min(cvar_drs_maxscale->integer, scale);
        }
        // The screen images are allocated at the window size.
        scale = max(25, min(100, scale));
        render_width = ((int)((float)unscaled_width * (float)scale / 100.0f) + 1) & ~1;
        render_height = (int)((float)unscaled_height * (float)scale / 100.0f);
        render_width = max(2, min(render_width, screen_image_width));
        render_height = max(1, min(render_height, screen_image_height));
    }

    // vkpt_fsr_is_enabled() and evaluate_taa_settings().
    bool menu_mode = render_world && cl_paused->integer == 1 && uis.menuDepth > 0;
    bool fsr_enabled = false;
    if (cvar_flt_fsr_enable->integer) {
        bool upscaling = render_width < display_width || render_height < display_height;
        if ((cvar_flt_fsr_enable->integer != 1 || upscaling) &&
            (cvar_flt_fsr_easu->integer || cvar_flt_fsr_rcas->integer))
            fsr_enabled = true;
    }
    fsr_active = fsr_enabled && !menu_mode;
    mode.fsr_enabled = fsr_enabled;

    effective_aa_mode = AA_MODE_OFF;
    taa_output_width = render_width;
    taa_output_height = render_height;
    if (mode.enable_denoiser) {
        bool force_upscaling = fsr_enabled && !cvar_flt_fsr_easu->integer;
        int flt_taa = force_upscaling ? AA_MODE_UPSCALE : ubo_cvar_flt_taa->integer;
        if (flt_taa == AA_MODE_TAA) {
            effective_aa_mode = AA_MODE_TAA;
        } else if (flt_taa == AA_MODE_UPSCALE) {
            if (render_width > unscaled_width || render_height > unscaled_height) {
                effective_aa_mode = AA_MODE_TAA;
            } else {
                effective_aa_mode = AA_MODE_UPSCALE;
                if (!fsr_enabled || force_upscaling) {
                    taa_output_width = unscaled_width;
                    taa_output_height = unscaled_height;
                }
            }
        }
    }

    // Switching between the world and the player setup view swaps the whole
    // scene, so nothing in the history applies.
    if (render_world != prev_render_world)
        temporal_frame_valid = false;
    prev_render_world = render_world;

    // Animated materials advance twice a second, as in the Vulkan backend.
    {
        int new_world_anim_frame = (int)(fd->time * 2);
        if (new_world_anim_frame != world_anim_frame) {
            if (world_anim_frame >= 0)
                advance_material_animation();
            world_anim_frame = new_world_anim_frame;
        }
        apply_material_animation(slot);
    }

    //
    // Dynamic geometry, then the acceleration structures.
    //
    vec3_t view_forward, view_right, view_up;
    AngleVectors(fd->viewangles, view_forward, view_right, view_up);

    bool have_effects = false;
    if (!render_world)
        ensure_entity_materials();
    build_entity_geometry(fd);
    refresh_entity_blas(cmd);
    tlas_valid = refresh_tlas(cmd, mode.include_world);
    if (mode.include_world) {
        build_effect_geometry(fd, view_right, view_up);
        have_effects = refresh_effect_accel(cmd);
    }
    if (!tlas_valid)
        return;

    // Effects are scaled by last frame's adapted luminance like vkpt does.
    {
        float adapted = mtl_tone_mapping_enabled() ? mtl_tone_mapping_adapted_luminance() : 0.0f;
        if (adapted > 0.0f && adapted != 1.0f)
            ubo_host.prev_adapted_luminance = adapted;
        if (ubo_host.prev_adapted_luminance <= 0.0f)
            ubo_host.prev_adapted_luminance = 0.005f;
    }

    if (mode.include_world)
        build_frame_light_polys(fd, ubo_host.prev_adapted_luminance);
    else
        num_frame_light_polys = 0;

    const mtl_sun_light_t *sun = mtl_physical_sky_sun();
    prepare_ubo(fd, &mode, sun);
    ubo_host.num_effect_triangles = have_effects ? num_effect_triangles : 0;
    memcpy(uniform_buffers[slot].contents, &ubo_host, sizeof(ubo_host));

    frame_ctx_t fc = {
        .cmd = cmd,
        .slot = slot,
        .prev_slot = prev_traced_slot,
        .parity = (int)(pt_frame_counter & 1),
        .include_world = mode.include_world,
        .have_effects = have_effects,
    };

    // This frame's light statistics slice starts empty.
    if (light_stats_buffer) {
        uint32_t slice = pt_frame_counter % NUM_LIGHT_STATS_BUFFERS;
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit fillBuffer:light_stats_buffer
                   range:NSMakeRange(sizeof(uint32_t) * light_stats_size * slice, sizeof(uint32_t) * light_stats_size)
                   value:0];
        [blit endEncoding];
    }

    bool god_rays_enabled = cvar_gr_enable->integer && cvar_gr_intensity->value > 0.0f &&
                            sun->use_physical_sky && sun->visible && !cvar_physical_sky_space->integer &&
                            mode.include_world;
    MTLGodRaysUniforms god_rays;
    memset(&god_rays, 0, sizeof(god_rays));
    if (god_rays_enabled) {
        god_rays.max_steps = (uint32_t)max(4, cvar_gr_max_steps->integer);
        shadow_map_setup(sun->direction, &god_rays);
        render_shadow_map(cmd, slot, &god_rays);
    }

    id<MTLComputeCommandEncoder> enc;

    //
    // Primary rays, then god rays and the reflection/refraction passes
    // interleaved the way R_RenderFrame_RTX orders them.
    //
    enc = begin_pass(&fc, MTL_PROFILER_PRIMARY_RAYS, @"primary rays", true);
    dispatch_rays(enc, pipeline_primary_rays, 0, 0, render_height);
    [enc endEncoding];

    if (god_rays_enabled)
        record_god_rays_trace(&fc, MTL_PROFILER_GOD_RAYS, &god_rays, 0);

    if (mode.reflect_refract > 0) {
        enc = begin_pass(&fc, MTL_PROFILER_REFLECT_REFRACT_1, @"reflect refract 1", true);
        dispatch_rays(enc, pipeline_reflect_refract, 0, 0, render_height);
        [enc endEncoding];
    }

    if (god_rays_enabled) {
        if (mode.reflect_refract > 0)
            record_god_rays_trace(&fc, MTL_PROFILER_GOD_RAYS_REFLECT_REFRACT, &god_rays, 1);
        record_god_rays_filter(&fc);
    }

    if (mode.reflect_refract > 1) {
        enc = begin_pass(&fc, MTL_PROFILER_REFLECT_REFRACT_2, @"reflect refract 2", true);
        for (int pass = 0; pass < mode.reflect_refract - 1; pass++) {
            if (pass)
                barrier(enc);
            dispatch_rays(enc, pipeline_reflect_refract, pass + 1, 0, render_height);
        }
        [enc endEncoding];
    }

    if (mode.enable_denoiser) {
        enc = begin_pass(&fc, MTL_PROFILER_DENOISE_GRADIENT_REPROJECT, @"asvgf gradient reproject", true);
        dispatch(enc, pipeline_gradient_reproject, 0, 0,
                 MTLSizeMake((render_width + 23) / 24, (render_height + 23) / 24, 1), MTLSizeMake(24, 24, 1));
        [enc endEncoding];
    }

    //
    // Lighting (vkpt_pt_trace_lighting).
    //
    enc = begin_pass(&fc, MTL_PROFILER_DIRECT_LIGHTING, @"direct lighting", true);
    dispatch_rays(enc, pipeline_direct_lighting, 0, cvar_pt_caustics->value != 0.0f ? 1 : 0, render_height);
    [enc endEncoding];

    if (mode.num_bounce_rays > 0.0f) {
        int height = (mode.num_bounce_rays == 0.5f) ? render_height / 2 : render_height;
        // One encoder per bounce, so the profiler times them separately
        // (vkpt's INDIRECT_LIGHTING_0/1).
        for (int bounce_ray = 0; bounce_ray < (int)ceilf(mode.num_bounce_rays); bounce_ray++) {
            enc = begin_pass(&fc, MTL_PROFILER_INDIRECT_LIGHTING_0 + min(bounce_ray, 1), @"indirect lighting", true);
            dispatch_rays(enc, pipeline_indirect_lighting, bounce_ray, 0, height);
            [enc endEncoding];
        }
    }

    //
    // Denoising or plain compositing, checkerboard interleave, TAA(U).
    //
    if (mode.enable_denoiser) {
        record_asvgf_filter(&fc, ubo_cvar_pt_num_bounce_rays->value >= 0.5f);
    } else {
        enc = begin_pass(&fc, MTL_PROFILER_COMPOSITING, @"compositing", false);
        dispatch(enc, pipeline_compositing, 0, 0, groups_16(render_width, render_height), MTLSizeMake(16, 16, 1));
        [enc endEncoding];
    }

    enc = begin_pass(&fc, MTL_PROFILER_INTERLEAVE, @"checkerboard interleave", false);
    // vkpt covers the whole image to clear the unused area, but nothing reads
    // past the render size (TAA clamps to it), and at a 4K output with a low
    // DRS scale the full-size dispatch was a fixed cost of most of a ms.
    dispatch(enc, pipeline_interleave, 0, 0, groups_16(render_width, render_height), MTLSizeMake(16, 16, 1));
    [enc endEncoding];

    {
        int dispatch_width = taa_output_width, dispatch_height = taa_output_height;
        if (dispatch_width < screen_image_width)
            dispatch_width += 8;
        if (dispatch_height < screen_image_height)
            dispatch_height += 8;
        enc = begin_pass(&fc, MTL_PROFILER_TAA, @"taau", false);
        dispatch(enc, pipeline_taa, 0, 0, groups_16(dispatch_width, dispatch_height), MTLSizeMake(16, 16, 1));
        [enc endEncoding];
    }

    prev_render_width = render_width;
    prev_render_height = render_height;
    prev_traced_slot = slot;
    if (render_world)
        frame_scale_tag = drs_effective_scale;
    temporal_frame_valid = mode.enable_denoiser;
    pt_frame_counter++;

    //
    // Bloom, tone mapping and FSR on the TAA output (scene radiance).
    //
    id<MTLTexture> post = vkpt_images[VKPT_IMG_TAA_OUTPUT];
    prev_frame_time = fd->time;

    mtl_bloom_update(frame_time, ubo_host.medium != MEDIUM_NONE, menu_mode);
    if (mtl_bloom_wanted(menu_mode))
        mtl_bloom_record(cmd, post, taa_output_width, taa_output_height);

    if (mtl_tone_mapping_enabled()) {
        mtl_tone_mapping_record(cmd, post, taa_output_width, taa_output_height,
                                frame_time > 0.0f ? frame_time : 1.0f / 60.0f, fd,
                                mtl_bloom_hdr_clamp_strength());
    }

    // Skipped in menu mode like vkpt, since the image is blurred there anyway.
    if (fsr_active) {
        uint32_t out_w = (uint32_t)display_width;
        uint32_t out_h = (uint32_t)display_height;
        float in_w = (float)taa_output_width, in_h = (float)taa_output_height;
        float cont_w = (float)post.width, cont_h = (float)post.height;

        MTLFsrUniforms f;
        memset(&f, 0, sizeof(f));
        // FsrEasuCon()
        f.easu_con0.x = in_w / (float)out_w;
        f.easu_con0.y = in_h / (float)out_h;
        f.easu_con0.z = 0.5f * in_w / (float)out_w - 0.5f;
        f.easu_con0.w = 0.5f * in_h / (float)out_h - 0.5f;
        f.easu_con1.x = 1.0f / cont_w;
        f.easu_con1.y = 1.0f / cont_h;
        f.easu_con1.z = 1.0f / cont_w;
        f.easu_con1.w = -1.0f / cont_h;
        f.easu_con2.x = -1.0f / cont_w;
        f.easu_con2.y = 2.0f / cont_h;
        f.easu_con2.z = 1.0f / cont_w;
        f.easu_con2.w = 2.0f / cont_h;
        f.easu_con3.x = 0.0f;
        f.easu_con3.y = 4.0f / cont_h;
        // FsrRcasCon()
        f.rcas_sharpness = exp2f(-cvar_flt_fsr_sharpness->value);
        f.container_inv_width = 1.0f / cont_w;
        f.container_inv_height = 1.0f / cont_h;
        f.is_hdr = mtl.is_hdr ? 1 : 0;
        f.input_width = (uint32_t)taa_output_width;
        f.input_height = (uint32_t)taa_output_height;
        f.output_width = out_w;
        f.output_height = out_h;
        f.easu_to_display = cvar_flt_fsr_rcas->integer ? 0 : 1;
        f.rcas_after_easu = cvar_flt_fsr_easu->integer ? 1 : 0;

        // Each 64 thread group covers a 16x16 tile.
        MTLSize fsr_groups = MTLSizeMake((out_w + 15) / 16, (out_h + 15) / 16, 1);
        MTLSize fsr_threads = MTLSizeMake(64, 1, 1);

        if (cvar_flt_fsr_easu->integer) {
            enc = mtl_profiler_compute_encoder(cmd, MTL_PROFILER_FSR_EASU);
            enc.label = @"fsr easu";
            [enc setComputePipelineState:pipeline_fsr_easu];
            [enc setTexture:post atIndex:0];
            [enc setTexture:fsr_easu_output atIndex:1];
            [enc setBytes:&f length:sizeof(f) atIndex:0];
            [enc dispatchThreadgroups:fsr_groups threadsPerThreadgroup:fsr_threads];
            [enc endEncoding];
        }
        if (cvar_flt_fsr_rcas->integer) {
            enc = mtl_profiler_compute_encoder(cmd, MTL_PROFILER_FSR_RCAS);
            enc.label = @"fsr rcas";
            [enc setComputePipelineState:pipeline_fsr_rcas];
            [enc setTexture:cvar_flt_fsr_easu->integer ? fsr_easu_output : post atIndex:0];
            [enc setTexture:fsr_rcas_output atIndex:1];
            [enc setBytes:&f length:sizeof(f) atIndex:0];
            [enc dispatchThreadgroups:fsr_groups threadsPerThreadgroup:fsr_threads];
            [enc endEncoding];
        }
    }
}
