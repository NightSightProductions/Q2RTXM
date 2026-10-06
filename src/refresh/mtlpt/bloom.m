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

// Bloom post process, the Metal counterpart of vkpt/bloom.c.

#include "mtlpt_metal.h"
#include "mtlpt_profiler.h"
#include "system/system.h"

static id<MTLComputePipelineState> pipeline_downscale;
static id<MTLComputePipelineState> pipeline_blur;
static id<MTLComputePipelineState> pipeline_composite;

// Quarter resolution ping-pong pair.
static id<MTLTexture> bloom_tex[2];

cvar_t *cvar_bloom_enable;
static cvar_t *cvar_bloom_debug;
static cvar_t *cvar_bloom_sigma;
static cvar_t *cvar_bloom_intensity;
static cvar_t *cvar_bloom_sigma_water;
static cvar_t *cvar_bloom_intensity_water;

static float bloom_intensity;
static float bloom_sigma;
static float under_water_animation;
static float menu_sigma;
static float effective_intensity;
static float hdr_clamp_strength;
static uint32_t menu_start_ms;

static id<MTLComputePipelineState> make_pipeline(const char *name)
{
    NSError *err = nil;
    id<MTLFunction> fn = mtl_new_function([NSString stringWithUTF8String:name]);
    if (!fn) {
        Com_EPrintf("Metal: kernel '%s' missing from the library\n", name);
        return nil;
    }
    id<MTLComputePipelineState> p = [mtl.device newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    if (!p)
        mtl_log_error(name, err);
    return p;
}

bool mtl_bloom_init(void)
{
    cvar_bloom_enable = Cvar_Get("bloom_enable", "1", 0);
    cvar_bloom_debug = Cvar_Get("bloom_debug", "0", 0);
    cvar_bloom_sigma = Cvar_Get("bloom_sigma", "0.037", 0); // relative to screen height
    cvar_bloom_intensity = Cvar_Get("bloom_intensity", "0.002", 0);
    cvar_bloom_sigma_water = Cvar_Get("bloom_sigma_water", "0.037", 0);
    cvar_bloom_intensity_water = Cvar_Get("bloom_intensity_water", "0.2", 0);

    pipeline_downscale = make_pipeline("bloom_downscale");
    pipeline_blur = make_pipeline("bloom_blur");
    pipeline_composite = make_pipeline("bloom_composite");

    mtl_bloom_reset();

    return pipeline_downscale && pipeline_blur && pipeline_composite;
}

void mtl_bloom_shutdown(void)
{
    [pipeline_downscale release];
    pipeline_downscale = nil;
    [pipeline_blur release];
    pipeline_blur = nil;
    [pipeline_composite release];
    pipeline_composite = nil;
    for (int i = 0; i < 2; i++) {
        [bloom_tex[i] release];
        bloom_tex[i] = nil;
    }
}

void mtl_bloom_resize(int width, int height)
{
    for (int i = 0; i < 2; i++) {
        [bloom_tex[i] release];

        MTLTextureDescriptor *desc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                               width:max(1, width / 4)
                                                              height:max(1, height / 4)
                                                           mipmapped:NO];
        desc.storageMode = MTLStorageModePrivate;
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        bloom_tex[i] = [mtl.device newTextureWithDescriptor:desc];
        bloom_tex[i].label = i ? @"bloom b" : @"bloom a";
    }
}

void mtl_bloom_reset(void)
{
    bloom_intensity = cvar_bloom_intensity->value;
    bloom_sigma = cvar_bloom_sigma->value;
    under_water_animation = 0.0f;
    menu_start_ms = 0;
}

static float lerpf(float a, float b, float s)
{
    return a * (1.0f - s) + b * s;
}

// Mirrors vkpt_bloom_update(): fades the water settings in and out and, while
// a menu is up over a paused game, blurs the whole frame for legibility.
void mtl_bloom_update(float frame_time, bool under_water, bool menu_mode)
{
    if (under_water) {
        under_water_animation = min(1.0f, under_water_animation + frame_time * 3.0f);
        bloom_intensity = cvar_bloom_intensity_water->value;
        bloom_sigma = cvar_bloom_sigma->value;
    } else {
        under_water_animation = max(0.0f, under_water_animation - frame_time * 3.0f);
        bloom_intensity = lerpf(cvar_bloom_intensity->value, cvar_bloom_intensity_water->value, under_water_animation);
        bloom_sigma = lerpf(cvar_bloom_sigma->value, cvar_bloom_sigma_water->value, under_water_animation);
    }

    if (menu_mode) {
        if (menu_start_ms == 0)
            menu_start_ms = Sys_Milliseconds();
        float phase = Q_clipf((float)(Sys_Milliseconds() - menu_start_ms) / 150.0f, 0.0f, 1.0f);
        hdr_clamp_strength = phase;
        phase = powf(phase, 0.25f);
        menu_sigma = phase * 0.03f;
        effective_intensity = 1.0f;
    } else {
        menu_start_ms = 0;
        menu_sigma = -1.0f;
        effective_intensity = bloom_intensity;
        hdr_clamp_strength = 0.0f;
    }
}

float mtl_bloom_hdr_clamp_strength(void)
{
    return hdr_clamp_strength;
}

bool mtl_bloom_wanted(bool menu_mode)
{
    return cvar_bloom_enable->integer != 0 || menu_mode;
}

void mtl_bloom_record(id<MTLCommandBuffer> cmd, id<MTLTexture> color, int width, int height)
{
    if (!bloom_tex[0] || !color || width <= 0 || height <= 0)
        return;

    float sigma = menu_sigma >= 0.0f ? menu_sigma : bloom_sigma;
    float effective_sigma = Q_clipf(sigma * (float)height * 0.25f, 1.0f, 100.0f);

    MTLBloomUniforms u = {
        .output_width = (uint32_t)width,
        .output_height = (uint32_t)height,
        .pixstep_x = 1.0f,
        .pixstep_y = 0.0f,
        .argument_scale = -1.0f / (2.0f * effective_sigma * effective_sigma),
        .normalization_scale = 1.0f / (sqrtf(2.0f * (float)M_PI) * effective_sigma),
        .num_samples = (uint32_t)roundf(effective_sigma * 4.0f),
        .intensity = effective_intensity,
    };

    MTLSize threads = MTLSizeMake(16, 16, 1);
    MTLSize quarter_groups = MTLSizeMake((width / 4 + 15) / 16, (height / 4 + 15) / 16, 1);
    MTLSize full_groups = MTLSizeMake((width + 15) / 16, (height + 15) / 16, 1);

    id<MTLComputeCommandEncoder> enc = mtl_profiler_compute_encoder(cmd, MTL_PROFILER_BLOOM);
    enc.label = @"bloom";

    [enc setComputePipelineState:pipeline_downscale];
    [enc setTexture:color atIndex:0];
    [enc setTexture:bloom_tex[0] atIndex:1];
    [enc setBytes:&u length:sizeof(u) atIndex:0];
    [enc dispatchThreadgroups:quarter_groups threadsPerThreadgroup:threads];

    // Horizontal blur a -> b.
    [enc setComputePipelineState:pipeline_blur];
    [enc setTexture:bloom_tex[0] atIndex:0];
    [enc setTexture:bloom_tex[1] atIndex:1];
    [enc setBytes:&u length:sizeof(u) atIndex:0];
    [enc dispatchThreadgroups:quarter_groups threadsPerThreadgroup:threads];

    // Vertical blur b -> a.
    u.pixstep_x = 0.0f;
    u.pixstep_y = 1.0f;
    [enc setTexture:bloom_tex[1] atIndex:0];
    [enc setTexture:bloom_tex[0] atIndex:1];
    [enc setBytes:&u length:sizeof(u) atIndex:0];
    [enc dispatchThreadgroups:quarter_groups threadsPerThreadgroup:threads];

    if (cvar_bloom_debug->integer)
        u.intensity = 1.0f;

    [enc setComputePipelineState:pipeline_composite];
    [enc setTexture:bloom_tex[0] atIndex:0];
    [enc setTexture:color atIndex:1];
    [enc setBytes:&u length:sizeof(u) atIndex:0];
    [enc dispatchThreadgroups:full_groups threadsPerThreadgroup:threads];

    [enc endEncoding];
}
