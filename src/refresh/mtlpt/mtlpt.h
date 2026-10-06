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

// Public (C-visible) interface of the Metal path tracing backend.
// This header must stay free of Metal/Objective-C types so it can be included
// from the plain C parts of the engine.

#pragma once

#include "shared/shared.h"
#include "shared/list.h"
#include "common/bsp.h"
#include "common/cmd.h"
#include "common/common.h"
#include "common/cvar.h"
#include "common/files.h"
#include "client/client.h"
#include "client/video.h"
#include "refresh/refresh.h"
#include "refresh/images.h"
#include "refresh/models.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel texture handle meaning "flat white", used by 2D fills.
#define MTL_TEXNUM_WHITE (~0)

// Maximum number of 2D UI quads queued per frame.
#define MTL_MAX_STRETCH_PICS (1 << 14)

// Working limits for the model loaders.
#define TESS_MAX_VERTICES 16384
#define TESS_MAX_INDICES  (3 * TESS_MAX_VERTICES)

typedef struct maliasframe_s {
    vec3_t scale;
    vec3_t translate;
    vec3_t bounds[2];
    vec_t  radius;
} maliasframe_t;

// The Metal backend keeps plain image skins alongside the PBR material records
// so that the path tracer can read roughness, metalness and material kind.
typedef struct maliasmesh_s {
    int        numverts;
    int        numtris;
    int        numindices;
    int        numskins;
    int        tri_offset;
    int       *indices;
    vec3_t    *positions;
    vec3_t    *normals;
    vec2_t    *tex_coords;
    vec3_t    *tangents;
    uint32_t  *blend_indices;   // iqm only
    uint32_t  *blend_weights;   // iqm only
    image_t  **skins;
    struct pbr_material_s **materials;
    bool       handedness;
} maliasmesh_t;

//
// Renderer entry points, installed into the global function pointers by
// R_RegisterFunctionsMTL().
//
ref_type_t R_Init_MTL(bool total);
void       R_Shutdown_MTL(bool total);
void       R_BeginRegistration_MTL(const char *name);
void       R_EndRegistration_MTL(void);
void       R_SetSky_MTL(const char *name, float rotate, int autorotate, const vec3_t axis);
void       R_RenderFrame_MTL(refdef_t *fd);
void       R_LightPoint_MTL(const vec3_t origin, vec3_t light);
void       R_ClearColor_MTL(void);
void       R_SetAlpha_MTL(float alpha);
void       R_SetAlphaScale_MTL(float alpha);
void       R_SetColor_MTL(uint32_t color);
void       R_SetClipRect_MTL(const clipRect_t *clip);
void       R_SetScale_MTL(float scale);
void       R_DrawChar_MTL(int x, int y, int flags, int c, qhandle_t font);
int        R_DrawString_MTL(int x, int y, int flags, size_t maxlen, const char *s, qhandle_t font);
void       R_DrawPic_MTL(int x, int y, qhandle_t pic);
void       R_DrawStretchPic_MTL(int x, int y, int w, int h, qhandle_t pic);
void       R_DrawKeepAspectPic_MTL(int x, int y, int w, int h, qhandle_t pic);
void       R_DrawStretchRaw_MTL(int x, int y, int w, int h);
void       R_UpdateRawPic_MTL(int pic_w, int pic_h, const uint32_t *pic);
void       R_DiscardRawPic_MTL(void);
void       R_TileClear_MTL(int x, int y, int w, int h, qhandle_t pic);
void       R_DrawFill8_MTL(int x, int y, int w, int h, int c);
void       R_DrawFill32_MTL(int x, int y, int w, int h, uint32_t color);
void       R_BeginFrame_MTL(void);
void       R_EndFrame_MTL(void);
void       R_ModeChanged_MTL(int width, int height, int flags);
void       R_AddDecal_MTL(decal_t *d);
bool       R_InterceptKey_MTL(unsigned key, bool down);
bool       R_IsHDR_MTL(void);

void IMG_Load_MTL(image_t *image, byte *pic);
void IMG_Unload_MTL(image_t *image);
void IMG_ReadPixels_MTL(screenshot_t *s);
void IMG_ReadPixelsHDR_MTL(screenshot_t *s);

int  MOD_LoadMD2_MTL(model_t *model, const void *rawdata, size_t length, const char *mod_name);
int  MOD_LoadMD3_MTL(model_t *model, const void *rawdata, size_t length, const char *mod_name);
int  MOD_LoadIQM_MTL(model_t *model, const void *rawdata, size_t length, const char *mod_name);
void MOD_Reference_MTL(model_t *model);

//
// device.mm - Metal device, CAMetalLayer and per-frame command buffer plumbing.
//
bool        mtl_device_init(void);
void        mtl_device_shutdown(void);
void        mtl_device_mode_changed(int width, int height);
bool        mtl_device_supports_raytracing(void);
const char *mtl_device_name(void);

//
// draw.mm - 2D user interface rendering.
//
bool mtl_draw_init(void);
void mtl_draw_shutdown(void);
void mtl_draw_clear_stretch_pics(void);

//
// textures.mm - image_t -> MTLTexture management.
//
bool mtl_textures_init(void);
void mtl_textures_shutdown(void);
void mtl_textures_update(void);
void mtl_textures_invalidate(const image_t *image);

//
// path_tracer.mm - acceleration structures and the ray tracing kernels.
//
bool mtl_pt_init(void);
void mtl_pt_shutdown(void);
void mtl_pt_resize(int width, int height);
void mtl_pt_register_world(bsp_t *bsp, const char *map_name);
void mtl_pt_free_world(void);
void mtl_pt_set_sky(const char *name, float rotate, int autorotate, const vec3_t axis);
void mtl_pt_report_gpu_time(double milliseconds, int scale_tag);
int  mtl_pt_take_frame_scale_tag(void);
int  mtl_pt_resolution_scale(void);
void mtl_pt_output_uv_scale(float *u_scale, float *v_scale);
float mtl_pt_tonemap_white_point(void);
// Final blit parameters for the frame just rendered.
bool  mtl_pt_water_warp(void);
bool  mtl_pt_use_lanczos(void);
float mtl_pt_frame_time(void);
void  mtl_pt_output_size(int *width, int *height);

// material_support.m
void mtl_material_init_cvars(void);

#ifdef __cplusplus
}
#endif
