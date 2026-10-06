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

// Entry points and frame lifecycle of the Metal path tracing backend.

#include "mtlpt_metal.h"
#import <AppKit/NSScreen.h>
#include "mtlpt_profiler.h"
#include "material_compat.h"
#include "physical_sky.h"
#include "../vkpt/material.h"
#include "refresh/debug.h"

static bsp_t *bsp_world_model;

bool R_InterceptKey_MTL(unsigned key, bool down);

static id<MTLRenderPipelineState> pipeline_blit;
static id<MTLBuffer>              blit_uniforms;

// The frame is composed here and then copied to the drawable, so screenshots
// have a stable source that still contains the UI. One per frame slot: the
// present command buffer of the previous frame waits for its drawable (up to
// a refresh interval with vsync) before copying, and a shared target made the
// next frame's render buffer wait for that copy, which the GPU timing then
// counted as render time and the dynamic resolution controller reacted to.
static id<MTLTexture>             frame_targets[MTL_FRAMES_IN_FLIGHT];
static id<MTLTexture>             frame_target;   // this frame's slot

// Presentation runs off the main thread, on its own command queue. Waiting
// for a drawable (nextDrawable blocks until the display frees one, at a
// vsync) on the main thread kept the next frame from being encoded, so the
// GPU ran dry between frames, its clocks dropped, every pass got slower, and
// the dynamic resolution controller followed the inflated GPU times down to
// its minimum scale, worst at 60 Hz. Now the main thread commits the render
// and moves on; a serial dispatch queue acquires the drawable and presents,
// ordered after the render by an event. Frames in flight stay bounded by
// frame_sem, which the present signals when it completes.
static dispatch_queue_t           present_dispatch;
static id<MTLCommandQueue>        present_queue;
static id<MTLEvent>               render_done_event;
static uint64_t                   render_done_value;

static bool frame_started;
static bool world_rendered;
// The world view of this frame, traced in R_EndFrame (see R_RenderFrame_MTL).
static refdef_t *pending_world_fd;

extern cvar_t *mtl_vsync;
extern cvar_t *mtl_hdr;

static void create_frame_target(int width, int height)
{
    if (width <= 0 || height <= 0)
        return;

    for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++) {
        id<MTLTexture> t = frame_targets[i];
        if (t && (int)t.width == width && (int)t.height == height)
            continue;

        [t release];

        MTLTextureDescriptor *desc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:mtl.drawable_format
                                                               width:width
                                                              height:height
                                                           mipmapped:NO];
        desc.storageMode = MTLStorageModePrivate;
        desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

        frame_targets[i] = [mtl.device newTextureWithDescriptor:desc];
        frame_targets[i].label = @"frame target";
    }
    frame_target = frame_targets[mtl.frame_index % MTL_FRAMES_IN_FLIGHT];
}

static bool create_blit_pipeline(void)
{
    NSError *err = nil;
    id<MTLFunction> vs = mtl_new_function(@"final_blit_vertex");
    id<MTLFunction> fs = mtl_new_function(@"final_blit_fragment");
    if (!vs || !fs) {
        Com_EPrintf("Metal: final_blit shader functions missing from the library\n");
        return false;
    }

    MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.label = @"final blit";
    desc.vertexFunction = vs;
    desc.fragmentFunction = fs;
    desc.colorAttachments[0].pixelFormat = mtl.drawable_format;

    pipeline_blit = [mtl.device newRenderPipelineStateWithDescriptor:desc error:&err];
    [desc release];
    [vs release];
    [fs release];

    if (!pipeline_blit)
        return mtl_log_error("newRenderPipelineStateWithDescriptor(final blit)", err);

    blit_uniforms = [mtl.device newBufferWithLength:sizeof(MTLDrawUniforms)
                                            options:MTLResourceStorageModeShared];
    return blit_uniforms != nil;
}

ref_type_t R_Init_MTL(bool total)
{
    registration_sequence = 1;

    if (!vid.init(GAPI_METAL)) {
        Com_EPrintf("Metal: video initialization failed\n");
        return REF_TYPE_NONE;
    }

    IMG_Init();
    IMG_GetPalette();
    MOD_Init();
    mtl_material_init_cvars();
    MAT_Init();

    if (!mtl_device_init())
        goto fail;
    if (!mtl_textures_init())
        goto fail;
    if (!mtl_draw_init())
        goto fail;
    if (!create_blit_pipeline())
        goto fail;
    if (!mtl_debug_init())
        goto fail;
    if (!mtl_profiler_init())
        goto fail;
    if (!mtl_physical_sky_init())
        goto fail;
    if (!mtl_pt_init())
        goto fail;

    Com_Printf("Metal renderer initialized.\n");
    return REF_TYPE_MTLPT;

fail:
    R_Shutdown_MTL(true);
    return REF_TYPE_NONE;
}

void R_Shutdown_MTL(bool total)
{
    if (bsp_world_model) {
        BSP_Free(bsp_world_model);
        bsp_world_model = NULL;
    }

    mtl_pt_shutdown();
    mtl_physical_sky_shutdown();

    for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++) {
        [frame_targets[i] release];
        frame_targets[i] = nil;
    }
    frame_target = nil;
    if (present_dispatch) {
        dispatch_sync(present_dispatch, ^{});   // drain pending presents
        dispatch_release(present_dispatch);
        present_dispatch = NULL;
    }
    [present_queue release];
    present_queue = nil;
    [render_done_event release];
    render_done_event = nil;
    [pipeline_blit release];
    pipeline_blit = nil;
    [blit_uniforms release];
    blit_uniforms = nil;

    mtl_draw_shutdown();
    mtl_textures_shutdown();
    mtl_device_shutdown();

    IMG_FreeAll();
    MOD_FreeAll();
    MAT_Shutdown();
    IMG_Shutdown();
    MOD_Shutdown();

    vid.shutdown();
}

void R_BeginRegistration_MTL(const char *name)
{
    registration_sequence++;

    Com_Printf("loading %s\n", name);

    Com_AddConfigFile("maps/default.cfg", 0);
    Com_AddConfigFile(va("maps/%s.cfg", name), 0);

    // Loads the map specific material file on top of the global one.
    MAT_ChangeMap(name);

    mtl_pt_free_world();

    if (bsp_world_model) {
        BSP_Free(bsp_world_model);
        bsp_world_model = NULL;
    }

    char bsp_path[MAX_QPATH];
    Q_concat(bsp_path, sizeof(bsp_path), "maps/", name, ".bsp");

    bsp_t *bsp;
    int ret = BSP_Load(bsp_path, &bsp);
    if (!bsp)
        Com_Error(ERR_DROP, "%s: couldn't load %s: %s", __func__, bsp_path, Q_ErrorString(ret));

    bsp_world_model = bsp;
    mtl_pt_register_world(bsp, name);
}

void R_EndRegistration_MTL(void)
{
    IMG_FreeUnused();
    MOD_FreeUnused();
    MAT_FreeUnused();
    mtl_textures_update();
}

void R_SetSky_MTL(const char *name, float rotate, int autorotate, const vec3_t axis)
{
    mtl_pt_set_sky(name, rotate, autorotate, axis);
}

void R_AddDecal_MTL(decal_t *d)
{
}

bool R_IsHDR_MTL(void)
{
    return mtl.is_hdr;
}

void R_ModeChanged_MTL(int width, int height, int flags)
{
    r_config.width = width;
    r_config.height = height;
    r_config.flags = flags;

    mtl_device_mode_changed(width, height);
    mtl_pt_resize(width, height);
    create_frame_target(width, height);
}

// Frame pacing. With vsync a frame is shown at the first refresh after it is
// done, so a frame time just under a multiple of the refresh interval (e.g.
// 31.5 ms with DRS at 30 fps on a 60 Hz display) shows most frames for two
// refreshes and every few frames one for a single refresh: visible judder.
// When the GPU time is within 12% under such a multiple, every frame is held
// for exactly that many refreshes instead. Faster frame rates are untouched.
static volatile double smoothed_gpu_ms;

static CFTimeInterval present_pacing_interval(void)
{
    if (!mtl_vsync || !mtl_vsync->integer)
        return 0.0;

    NSInteger hz = [NSScreen mainScreen].maximumFramesPerSecond;
    if (hz <= 0)
        return 0.0;
    double refresh_ms = 1000.0 / (double)hz;
    double t = smoothed_gpu_ms;
    if (t <= refresh_ms)
        return 0.0;

    double n = ceil(t / refresh_ms);
    if (n * refresh_ms - t > 0.12 * t)
        return 0.0;
    // A little under the multiple, so the hold lands on that refresh.
    return (n * refresh_ms - 1.0) * 0.001;
}

void R_BeginFrame_MTL(void)
{
    if (!mtl.initialized)
        return;

    // Metal hands back autoreleased command buffers and drawables. Without a
    // pool that is drained every frame the drawable pool runs dry and
    // nextDrawable blocks forever.
    mtl.frame_pool = [[NSAutoreleasePool alloc] init];

    dispatch_semaphore_wait(mtl.frame_sem, DISPATCH_TIME_FOREVER);

    // vid_vsync changes apply right away, as in vkpt.
    bool vsync = mtl_vsync && mtl_vsync->integer != 0;
    if (mtl.layer && mtl.layer.displaySyncEnabled != vsync)
        mtl.layer.displaySyncEnabled = vsync;

    mtl.frame_index = (uint32_t)(mtl.frame_counter % MTL_FRAMES_IN_FLIGHT);

    // This slot's previous command buffer has completed (the semaphore above),
    // so its timestamps can be read.
    mtl_profiler_next_frame();

    mtl.cmd = [[mtl.queue commandBuffer] retain];
    mtl.cmd.label = @"frame";

    // The drawable is acquired in R_EndFrame, only for the final copy: holding
    // it across the frame makes the CPU wait for the display up front.

    create_frame_target(mtl.width, mtl.height);

    mtl_textures_update();
    mtl_draw_clear_stretch_pics();

    frame_started = true;
    world_rendered = false;
    pending_world_fd = NULL;
}

static void trace_view(refdef_t *fd)
{
    // The shared material system reads the current refdef through this.
    vkpt_refdef.fd = fd;

    if (!(fd->rdflags & RDF_NOWORLDMODEL)) {
        // Re-renders the sky cube map only when the sun or the preset changed.
        mtl_physical_sky_update(fd->time);
        mtl_debug_set_view(fd);
    }

    mtl_pt_render(mtl.cmd, fd);
    world_rendered = true;

    mtl_profiler_draw(Cvar_VariableInteger("flt_enable") != 0);
}

// The path tracer owns the whole frame: like vkpt, a later view replaces an
// earlier one. The player setup menu draws its model view (RDF_NOWORLDMODEL)
// after the world view of the same frame, so the world is not traced right
// away but at the end of the frame, and only if no such view replaced it.
// That keeps an open player setup menu from tracing the scene twice per frame
// and from the two views overwriting each other's denoiser and TAA history.
void R_RenderFrame_MTL(refdef_t *fd)
{
    if (!frame_started || !mtl.cmd)
        return;

    vkpt_refdef.fd = fd;

    fd->feedback.viewcluster = 0;
    fd->feedback.lookatcluster = 0;
    fd->feedback.num_light_polys = 0;
    fd->feedback.resolution_scale = mtl_pt_resolution_scale();
    fd->feedback.adapted_luminance = mtl_tone_mapping_adapted_luminance();

    if (fd->rdflags & RDF_NOWORLDMODEL) {
        pending_world_fd = NULL;
        trace_view(fd);
        return;
    }

    // Several world views in one frame (timerefresh) are all traced.
    if (pending_world_fd)
        trace_view(pending_world_fd);

    // The client's refdef and the entity/light arrays it points to stay valid
    // until its next V_RenderView, so the pointer can be kept for R_EndFrame.
    pending_world_fd = fd;
}

void R_EndFrame_MTL(void)
{
    if (!frame_started)
        return;

    frame_started = false;

    if (pending_world_fd && mtl.cmd) {
        trace_view(pending_world_fd);
        pending_world_fd = NULL;
    }

    if (!mtl.cmd) {
        dispatch_semaphore_signal(mtl.frame_sem);
        [mtl.frame_pool drain];
        mtl.frame_pool = nil;
        return;
    }

    if (frame_target) {
        // Resolve the traced image into the frame target. When nothing was
        // traced this frame (menus, loading screens) just clear it.
        MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = frame_target;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;

        id<MTLRenderCommandEncoder> enc = [mtl.cmd renderCommandEncoderWithDescriptor:pass];
        enc.label = @"resolve";

        if (world_rendered && mtl_pt_output_texture()) {
            float uv_scale_x = 1.0f, uv_scale_y = 1.0f;
            mtl_pt_output_uv_scale(&uv_scale_x, &uv_scale_y);

            // Per draw copy: a shared buffer is rewritten by the next frame
            // while this one may still be queued, which drew a frame with the
            // following frame's UV scale (the image shrinking into a corner
            // when leaving the menu or when the resolution scale changes).
            MTLDrawUniforms blit;
            memset(&blit, 0, sizeof(blit));
            MTLDrawUniforms *u = &blit;
            u->hdr_color_scale = 1.0f;
            u->hdr_saturation_scale = 1.0f;
            u->is_hdr = mtl.is_hdr ? 1 : 0;
            u->tonemapped = mtl_tone_mapping_enabled() ? 1 : 0;
            u->uv_scale_x = uv_scale_x;
            u->uv_scale_y = uv_scale_y;
            u->tm_white_point = mtl_pt_tonemap_white_point();
            u->water_warp = mtl_pt_water_warp() ? 1 : 0;
            u->filter_lanczos = mtl_pt_use_lanczos() ? 1 : 0;
            u->time = mtl_pt_frame_time();
            int in_w, in_h;
            mtl_pt_output_size(&in_w, &in_h);
            u->input_width = (float)in_w;
            u->input_height = (float)in_h;

            [enc setRenderPipelineState:pipeline_blit];
            [enc setFragmentBytes:&blit length:sizeof(blit) atIndex:MTL_BUF_DRAW_UNIFORMS];
            [enc setFragmentTexture:mtl_pt_output_texture() atIndex:0];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];

            if (mtl_debug_have_lines()) {
                int dw, dh;
                id<MTLTexture> depth = mtl_pt_depth_texture(&dw, &dh);
                mtl_debug_draw(enc, depth, dw, dh, mtl.is_hdr ? mtl_ui_color_scale() : 1.0f);
            }
        }

        [enc endEncoding];

        R_ExpireDebugLines();

        // The UI is drawn on top of the resolved 3D image.
        mtl_draw_submit(mtl.cmd, frame_target);
    }

    // Rendering and presentation go in separate command buffers (see
    // present_dispatch), so the GPU times below are the frame's own work.
    // The DRS controller only trusts times measured at its current scale.
    int scale_tag = mtl_pt_take_frame_scale_tag();
    [mtl.cmd addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        // GPU timestamps drive the dynamic resolution controller. Apple GPUs
        // overlap consecutive command buffers, so End - Start of one buffer
        // also counts the tail of the previous frame and overstates the cost
        // (often close to 2x when GPU bound). Only the time after the previous
        // frame finished is this frame's own. Completion handlers of one
        // queue run in order, so a static is enough.
        // GPU faults and timeouts are otherwise silent. This runs off the
        // main thread, so stderr rather than the console.
        static int errors_reported;
        if (buffer.error && errors_reported < 10) {
            errors_reported++;
            fprintf(stderr, "Metal: frame command buffer failed: %s\n", buffer.error.localizedDescription.UTF8String);
        }

        static CFTimeInterval prev_end;
        CFTimeInterval start = buffer.GPUStartTime, end = buffer.GPUEndTime;
        double gpu_ms = (end - max(start, prev_end)) * 1000.0;
        if (end > prev_end)
            prev_end = end;
        mtl_pt_report_gpu_time(gpu_ms, scale_tag);
        // Q2RTX_MTL_TIMING=1: every frame's GPU time on stderr, for scripted runs.
        static int print_timing = -1;
        if (print_timing < 0)
            print_timing = getenv("Q2RTX_MTL_TIMING") != NULL;
        if (print_timing)
            fprintf(stderr, "gpu %.2f ms scale %d\n", gpu_ms, scale_tag);
        mtl_profiler_set_frame_time(gpu_ms);
        smoothed_gpu_ms = smoothed_gpu_ms > 0.0 ? smoothed_gpu_ms * 0.9 + gpu_ms * 0.1 : gpu_ms;
    }];
    if (!present_queue) {
        present_queue = [mtl.device newCommandQueue];
        present_queue.label = @"present";
        render_done_event = [mtl.device newEvent];
        present_dispatch = dispatch_queue_create("q2rtx.present", DISPATCH_QUEUE_SERIAL);
    }
    [mtl.cmd encodeSignalEvent:render_done_event value:++render_done_value];
    [mtl.cmd commit];
    [mtl.cmd release];
    mtl.cmd = nil;

    dispatch_semaphore_t sem = mtl.frame_sem;
    if (!frame_target) {
        mtl_draw_clear_stretch_pics();
        dispatch_semaphore_signal(sem);
    } else {
        id<MTLTexture> target = [frame_target retain];
        id<MTLCommandQueue> queue = present_queue;
        id<MTLEvent> event = render_done_event;
        uint64_t value = render_done_value;
        CAMetalLayer *layer = mtl.layer;
        CFTimeInterval pace = present_pacing_interval();
        dispatch_async(present_dispatch, ^{
            @autoreleasepool {
                id<MTLCommandBuffer> present = [queue commandBuffer];
                present.label = @"present";
                [present encodeWaitForEvent:event value:value];
                id<CAMetalDrawable> drawable = [layer nextDrawable];
                bool signal_on_present = false;
                if (drawable && drawable.texture.width == target.width && drawable.texture.height == target.height) {
                    id<MTLBlitCommandEncoder> blit = [present blitCommandEncoder];
                    [blit copyFromTexture:target toTexture:drawable.texture];
                    [blit endEncoding];
                    if (pace > 0.0) {
                        // Paced: like a Vulkan FIFO swapchain, the frame's
                        // slot frees up once its image reached the screen,
                        // so held frames never pile up as input lag.
                        [drawable addPresentedHandler:^(id<MTLDrawable> d) {
                            dispatch_semaphore_signal(sem);
                        }];
                        [present presentDrawable:drawable afterMinimumDuration:pace];
                        signal_on_present = true;
                    } else {
                        [present presentDrawable:drawable];
                    }
                } else if (getenv("Q2RTX_MTL_TIMING")) {
                    fprintf(stderr, "present skipped: drawable %dx%d, frame %dx%d, layer bounds %.0fx%.0f scale %.1f\n",
                            drawable ? (int)drawable.texture.width : 0, drawable ? (int)drawable.texture.height : 0,
                            (int)target.width, (int)target.height,
                            layer.bounds.size.width, layer.bounds.size.height, layer.contentsScale);
                }
                // Otherwise the frame's slot is free once its present completed.
                if (!signal_on_present) {
                    [present addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
                        dispatch_semaphore_signal(sem);
                    }];
                }
                [present commit];
                [target release];
            }
        });
    }

    mtl.frame_counter++;

    [mtl.frame_pool drain];
    mtl.frame_pool = nil;
}

//
// Screenshots
//

static float tonemap_aces_channel(float x)
{
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    float v = (x * (a * x + b)) / (x * (c * x + d) + e);
    return Q_clipf(v, 0.0f, 1.0f);
}

static float linear_to_srgb(float x)
{
    if (x <= 0.0031308f)
        return x * 12.92f;
    return 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
}

// IEEE 754 binary16 -> binary32, for reading back RGBA16Float render targets.
static float half_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1f;
    uint32_t mantissa = h & 0x3ff;
    uint32_t bits;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            // Subnormal: renormalize into a binary32 normal value.
            exponent = 127 - 15 + 1;
            while (!(mantissa & 0x400)) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3ff;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1f) {
        bits = sign | 0x7f800000 | (mantissa << 13);
    } else {
        bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }

    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

// Reads the composed frame back into host memory. The drawable cannot be read
// after presentation, so the frame target is the stable source, and it already
// contains the tone mapped image plus the UI.
static void *read_frame_target(int *width_p, int *height_p, size_t *row_bytes_p)
{
    if (!frame_target)
        return NULL;

    NSUInteger width = frame_target.width;
    NSUInteger height = frame_target.height;
    NSUInteger pixel_size = (mtl.drawable_format == MTLPixelFormatRGBA16Float) ? 8 : 4;
    NSUInteger row_bytes = width * pixel_size;

    id<MTLBuffer> staging = [mtl.device newBufferWithLength:row_bytes * height
                                                    options:MTLResourceStorageModeShared];
    if (!staging)
        return NULL;

    id<MTLCommandBuffer> cmd = [mtl.queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromTexture:frame_target
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(width, height, 1)
                 toBuffer:staging
        destinationOffset:0
   destinationBytesPerRow:row_bytes
 destinationBytesPerImage:row_bytes * height];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    void *pixels = Z_Malloc(row_bytes * height);
    memcpy(pixels, staging.contents, row_bytes * height);
    [staging release];

    *width_p = (int)width;
    *height_p = (int)height;
    *row_bytes_p = row_bytes;
    return pixels;
}

void IMG_ReadPixels_MTL(screenshot_t *s)
{
    int width, height;
    size_t src_row_bytes;
    void *raw = read_frame_target(&width, &height, &src_row_bytes);
    if (!raw)
        return;

    byte *pixels = IMG_AllocPixels((size_t)width * height * 3);

    for (int y = 0; y < height; y++) {
        byte *dst = pixels + (size_t)y * width * 3;
        // The engine writes screenshots with stbi_flip_vertically_on_write, so
        // rows are handed over bottom-up like the GL and Vulkan backends do.
        int src_y = height - 1 - y;

        if (mtl.drawable_format == MTLPixelFormatRGBA16Float) {
            const uint16_t *src = (const uint16_t *)((byte *)raw + (size_t)src_y * src_row_bytes);
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < 3; c++) {
                    float value = tonemap_aces_channel(half_to_float(src[x * 4 + c]));
                    dst[x * 3 + c] = (byte)(Q_clipf(linear_to_srgb(value), 0.0f, 1.0f) * 255.0f + 0.5f);
                }
            }
        } else {
            // BGRA8, already sRGB encoded.
            const byte *src = (const byte *)raw + (size_t)src_y * src_row_bytes;
            for (int x = 0; x < width; x++) {
                dst[x * 3 + 0] = src[x * 4 + 2];
                dst[x * 3 + 1] = src[x * 4 + 1];
                dst[x * 3 + 2] = src[x * 4 + 0];
            }
        }
    }

    Z_Free(raw);

    s->pixels = pixels;
    s->width = width;
    s->height = height;
    s->rowbytes = width * 3;
    s->bpp = 8;
}

void IMG_ReadPixelsHDR_MTL(screenshot_t *s)
{
    int width, height;
    size_t src_row_bytes;
    void *raw = read_frame_target(&width, &height, &src_row_bytes);
    if (!raw)
        return;

    if (mtl.drawable_format != MTLPixelFormatRGBA16Float) {
        Z_Free(raw);
        Com_WPrintf("Metal: HDR screenshots require an HDR drawable\n");
        return;
    }

    byte *pixels = IMG_AllocPixels((size_t)width * height * 3 * sizeof(float));
    float *dst_base = (float *)pixels;

    for (int y = 0; y < height; y++) {
        // Bottom-up, matching what the engine's screenshot writer expects.
        const uint16_t *src = (const uint16_t *)((byte *)raw + (size_t)(height - 1 - y) * src_row_bytes);
        float *dst = dst_base + (size_t)y * width * 3;

        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++)
                dst[x * 3 + c] = half_to_float(src[x * 4 + c]);
        }
    }

    Z_Free(raw);

    s->pixels = pixels;
    s->width = width;
    s->height = height;
    s->rowbytes = width * 3 * sizeof(float);
    s->bpp = 32;
}

static bool R_SupportsDebugLines_MTL(void)
{
    return true;
}

static void R_AddDebugText_MTL(const vec3_t origin, const vec3_t angles, const char *text,
                               float size, uint32_t color, uint32_t time, bool depth_test)
{
    if (vkpt_refdef.fd)
        R_AddDebugText_Lines(vkpt_refdef.fd->vieworg, origin, angles, text, size, color, time, depth_test);
}

void R_RegisterFunctionsMTL(void)
{
    R_Init = R_Init_MTL;
    R_Shutdown = R_Shutdown_MTL;
    R_BeginRegistration = R_BeginRegistration_MTL;
    R_EndRegistration = R_EndRegistration_MTL;
    R_SetSky = R_SetSky_MTL;
    R_RenderFrame = R_RenderFrame_MTL;
    R_LightPoint = R_LightPoint_MTL;
    R_ClearColor = R_ClearColor_MTL;
    R_SetAlpha = R_SetAlpha_MTL;
    R_SetAlphaScale = R_SetAlphaScale_MTL;
    R_SetColor = R_SetColor_MTL;
    R_SetClipRect = R_SetClipRect_MTL;
    R_SetScale = R_SetScale_MTL;
    R_DrawChar = R_DrawChar_MTL;
    R_DrawString = R_DrawString_MTL;
    R_DrawPic = R_DrawPic_MTL;
    R_DrawStretchPic = R_DrawStretchPic_MTL;
    R_DrawKeepAspectPic = R_DrawKeepAspectPic_MTL;
    R_DrawStretchRaw = R_DrawStretchRaw_MTL;
    R_UpdateRawPic = R_UpdateRawPic_MTL;
    R_DiscardRawPic = R_DiscardRawPic_MTL;
    R_TileClear = R_TileClear_MTL;
    R_DrawFill8 = R_DrawFill8_MTL;
    R_DrawFill32 = R_DrawFill32_MTL;
    R_BeginFrame = R_BeginFrame_MTL;
    R_EndFrame = R_EndFrame_MTL;
    R_ModeChanged = R_ModeChanged_MTL;
    R_AddDecal = R_AddDecal_MTL;
    R_InterceptKey = R_InterceptKey_MTL;
    R_IsHDR = R_IsHDR_MTL;
    R_SupportsDebugLines = R_SupportsDebugLines_MTL;
    R_AddDebugText_ = R_AddDebugText_MTL;
    IMG_Load = IMG_Load_MTL;
    IMG_Unload = IMG_Unload_MTL;
    IMG_ReadPixels = IMG_ReadPixels_MTL;
    IMG_ReadPixelsHDR = IMG_ReadPixelsHDR_MTL;
    MOD_LoadMD2 = MOD_LoadMD2_MTL;
#if USE_MD3
    MOD_LoadMD3 = MOD_LoadMD3_MTL;
#endif
    MOD_LoadIQM = MOD_LoadIQM_MTL;
    MOD_Reference = MOD_Reference_MTL;
}
