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

// Objective-C++ only internals shared between the Metal backend source files.
// Compiled without ARC, so Metal objects held in C structs are retained and
// released manually.

#pragma once

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <QuartzCore/CAMetalLayer.h>
#import <dispatch/dispatch.h>

#include "mtlpt.h"
#include "mtlpt_shared.h"

#define MTL_FRAMES_IN_FLIGHT 2   // vkpt MAX_FRAMES_IN_FLIGHT; 3 added a frame of input lag

typedef struct {
    id<MTLDevice>           device;
    id<MTLCommandQueue>     queue;
    id<MTLLibrary>          library;
    CAMetalLayer           *layer;

    // Per-frame state, valid between R_BeginFrame_MTL and R_EndFrame_MTL.
    id<MTLCommandBuffer>    cmd;
    id<CAMetalDrawable>     drawable;
    NSAutoreleasePool      *frame_pool;

    dispatch_semaphore_t    frame_sem;
    uint32_t                frame_index;    // [0, MTL_FRAMES_IN_FLIGHT)
    uint64_t                frame_counter;

    // Drawable dimensions in pixels.
    int                     width;
    int                     height;

    MTLPixelFormat          drawable_format;
    bool                    is_hdr;
    bool                    supports_raytracing;
    bool                    initialized;
} mtl_state_t;

extern mtl_state_t mtl;

// Convenience: log a Metal error and return false.
bool mtl_log_error(const char *what, NSError *error);

// Looks a shader function up in the shader library (or, with
// Q2RTX_MTL_SHADER_DIR set, in the libraries compiled from source).
id<MTLFunction> mtl_new_function(NSString *name);

//
// draw.mm
//
void mtl_draw_submit(id<MTLCommandBuffer> cmd, id<MTLTexture> target);
// scRGB scale that maps UI white to ui_hdr_nits (1.0 when not HDR).
float mtl_ui_color_scale(void);

//
// textures.mm
//
// Returns the argument-buffer index of the texture for a qhandle, uploading it
// on demand. Returns the white texture index for invalid handles.
uint32_t          mtl_texture_index_for_handle(qhandle_t pic);
// Same, but reports 0 for a texture that is not resident. Optional maps must
// not fall back to the white texture: the shader reads metalness and roughness
// out of their alpha, and white would mean "fully metallic, fully rough".
uint32_t          mtl_texture_index_optional(qhandle_t pic);
id<MTLTexture>    mtl_texture_for_index(uint32_t index);
id<MTLBuffer>     mtl_texture_argument_buffer(void);
void              mtl_textures_encode_use(id<MTLRenderCommandEncoder> encoder);
void              mtl_textures_encode_use_compute(id<MTLComputeCommandEncoder> encoder);

//
// path_tracer.mm
//
void           mtl_pt_render(id<MTLCommandBuffer> cmd, const refdef_t *fd);
id<MTLTexture> mtl_pt_output_texture(void);
// Primary normal/depth G-buffer (depth in .w) and its live size.
id<MTLTexture> mtl_pt_depth_texture(int *width, int *height);

//
// debug.m
//
bool mtl_debug_init(void);
void mtl_debug_shutdown(void);
void mtl_debug_set_view(const refdef_t *fd);
bool mtl_debug_have_lines(void);
void mtl_debug_draw(id<MTLRenderCommandEncoder> enc, id<MTLTexture> normal_depth,
                    int render_width, int render_height, float color_scale);

//
// bloom.m
//
bool mtl_bloom_init(void);
void mtl_bloom_shutdown(void);
void mtl_bloom_resize(int width, int height);
void mtl_bloom_reset(void);
void mtl_bloom_update(float frame_time, bool under_water, bool menu_mode);
bool mtl_bloom_wanted(bool menu_mode);
// Blends bloom into the live width x height corner of `color` in place.
void mtl_bloom_record(id<MTLCommandBuffer> cmd, id<MTLTexture> color, int width, int height);
// 0..1 while a menu fades in over a paused game; clamps HDR colour for legibility.
float mtl_bloom_hdr_clamp_strength(void);

//
// freecam.m
//
void mtl_freecam_init(void);
void mtl_freecam_reset(void);
// Overrides the view in `fd` while the game is paused; true if it moved.
bool mtl_freecam_update(refdef_t *fd, float frame_time);
extern cvar_t *cvar_pt_dof;
extern cvar_t *cvar_pt_aperture;
extern cvar_t *cvar_pt_focus;
extern cvar_t *cvar_pt_freecam;

//
// tone_mapping.m
//
bool  mtl_tone_mapping_init(void);
void  mtl_tone_mapping_shutdown(void);
void  mtl_tone_mapping_request_reset(void);
bool  mtl_tone_mapping_enabled(void);
float mtl_tone_mapping_adapted_luminance(void);
// Tone maps the live width x height corner of `color` in place to display range.
void  mtl_tone_mapping_record(id<MTLCommandBuffer> cmd, id<MTLTexture> color, int width, int height,
                              float frame_time, const refdef_t *fd, float hdr_clamp_strength);
