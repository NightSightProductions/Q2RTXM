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

// Histogram based tone mapper, the Metal counterpart of vkpt/tone_mapping.c.

#include "mtlpt_metal.h"
#include "mtlpt_profiler.h"

static id<MTLComputePipelineState> pipeline_histogram;
static id<MTLComputePipelineState> pipeline_curve;
static id<MTLComputePipelineState> pipeline_apply;

static id<MTLBuffer> tonemap_buffer;
static id<MTLBuffer> uniform_buffers[MTL_FRAMES_IN_FLIGHT];

static bool reset_required = true;

// Same names and defaults as the UBO_CVAR_DO(tm_*) list in global_ubo.h.
static cvar_t *cvar_tm_enable;
static cvar_t *cvar_tm_debug;
static cvar_t *cvar_tm_blend_enable;
static cvar_t *cvar_tm_dyn_range_stops;
static cvar_t *cvar_tm_exposure_bias;
static cvar_t *cvar_tm_exposure_speed_up;
static cvar_t *cvar_tm_exposure_speed_down;
static cvar_t *cvar_tm_blend_scale_border;
static cvar_t *cvar_tm_blend_scale_center;
static cvar_t *cvar_tm_blend_scale_fade_exp;
static cvar_t *cvar_tm_blend_distance_factor;
static cvar_t *cvar_tm_blend_max_alpha;
static cvar_t *cvar_tm_high_percentile;
static cvar_t *cvar_tm_low_percentile;
static cvar_t *cvar_tm_knee_start;
static cvar_t *cvar_tm_max_luminance;
static cvar_t *cvar_tm_min_luminance;
static cvar_t *cvar_tm_noise_blend;
static cvar_t *cvar_tm_noise_stops;
static cvar_t *cvar_tm_reinhard;
static cvar_t *cvar_tm_slope_blur_sigma;
static cvar_t *cvar_tm_white_point;
static cvar_t *cvar_tm_hdr_peak_nits;
static cvar_t *cvar_tm_hdr_saturation_scale;
static cvar_t *cvar_ui_hdr_nits;

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

bool mtl_tone_mapping_init(void)
{
    cvar_tm_enable = Cvar_Get("tm_enable", "1", 0);
    cvar_tm_debug = Cvar_Get("tm_debug", "0", 0);
    cvar_tm_blend_enable = Cvar_Get("tm_blend_enable", "1", CVAR_ARCHIVE);
    cvar_tm_dyn_range_stops = Cvar_Get("tm_dyn_range_stops", "7.0", 0);
    cvar_tm_exposure_bias = Cvar_Get("tm_exposure_bias", "-1.0", CVAR_ARCHIVE);
    cvar_tm_exposure_speed_up = Cvar_Get("tm_exposure_speed_up", "2", 0);
    cvar_tm_exposure_speed_down = Cvar_Get("tm_exposure_speed_down", "1", 0);
    cvar_tm_blend_scale_border = Cvar_Get("tm_blend_scale_border", "1", 0);
    cvar_tm_blend_scale_center = Cvar_Get("tm_blend_scale_center", "0", 0);
    cvar_tm_blend_scale_fade_exp = Cvar_Get("tm_blend_scale_fade_exp", "4", 0);
    cvar_tm_blend_distance_factor = Cvar_Get("tm_blend_distance_factor", "1.2", 0);
    cvar_tm_blend_max_alpha = Cvar_Get("tm_blend_max_alpha", "0.2", 0);
    cvar_tm_high_percentile = Cvar_Get("tm_high_percentile", "90", 0);
    cvar_tm_low_percentile = Cvar_Get("tm_low_percentile", "70", 0);
    cvar_tm_knee_start = Cvar_Get("tm_knee_start", "0.6", 0);
    cvar_tm_max_luminance = Cvar_Get("tm_max_luminance", "1.0", 0);
    cvar_tm_min_luminance = Cvar_Get("tm_min_luminance", "0.0002", 0);
    cvar_tm_noise_blend = Cvar_Get("tm_noise_blend", "0.5", 0);
    cvar_tm_noise_stops = Cvar_Get("tm_noise_stops", "-12", 0);
    cvar_tm_reinhard = Cvar_Get("tm_reinhard", "0.5", 0);
    cvar_tm_slope_blur_sigma = Cvar_Get("tm_slope_blur_sigma", "12.0", 0);
    cvar_tm_white_point = Cvar_Get("tm_white_point", "10.0", CVAR_ARCHIVE);
    cvar_tm_hdr_peak_nits = Cvar_Get("tm_hdr_peak_nits", "800.0", 0);
    cvar_tm_hdr_saturation_scale = Cvar_Get("tm_hdr_saturation_scale", "100", 0);
    cvar_ui_hdr_nits = Cvar_Get("ui_hdr_nits", "300", 0);

    pipeline_histogram = make_pipeline("tone_mapping_histogram");
    pipeline_curve = make_pipeline("tone_mapping_curve");
    pipeline_apply = make_pipeline("tone_mapping_apply");

    // Shared so the adapted luminance can be read back for the HUD feedback.
    tonemap_buffer = [mtl.device newBufferWithLength:sizeof(MTLToneMapBuffer)
                                             options:MTLResourceStorageModeShared];
    tonemap_buffer.label = @"tone mapping";
    memset(tonemap_buffer.contents, 0, sizeof(MTLToneMapBuffer));

    for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++) {
        uniform_buffers[i] = [mtl.device newBufferWithLength:sizeof(MTLToneMapUniforms)
                                                     options:MTLResourceStorageModeShared];
        if (!uniform_buffers[i])
            return false;
    }

    return pipeline_histogram && pipeline_curve && pipeline_apply && tonemap_buffer;
}

void mtl_tone_mapping_shutdown(void)
{
    [pipeline_histogram release];
    pipeline_histogram = nil;
    [pipeline_curve release];
    pipeline_curve = nil;
    [pipeline_apply release];
    pipeline_apply = nil;
    [tonemap_buffer release];
    tonemap_buffer = nil;
    for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++) {
        [uniform_buffers[i] release];
        uniform_buffers[i] = nil;
    }
}

void mtl_tone_mapping_request_reset(void)
{
    reset_required = true;
}

bool mtl_tone_mapping_enabled(void)
{
    return cvar_tm_enable && cvar_tm_enable->integer != 0;
}

float mtl_tone_mapping_adapted_luminance(void)
{
    if (!tonemap_buffer)
        return 0.0f;
    return ((const MTLToneMapBuffer *)tonemap_buffer.contents)->adapted_luminance;
}

void mtl_tone_mapping_record(id<MTLCommandBuffer> cmd, id<MTLTexture> color, int width, int height,
                             float frame_time, const refdef_t *fd, float hdr_clamp_strength)
{
    if (!color || width <= 0 || height <= 0)
        return;

    if (reset_required) {
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit fillBuffer:tonemap_buffer range:NSMakeRange(0, sizeof(MTLToneMapBuffer)) value:0];
        [blit endEncoding];
    }

    MTLToneMapUniforms *u = (MTLToneMapUniforms *)uniform_buffers[mtl.frame_index].contents;
    memset(u, 0, sizeof(*u));

    u->width = (uint32_t)width;
    u->height = (uint32_t)height;
    u->unscaled_height = (uint32_t)max(1, mtl.height);
    u->frame_index = (uint32_t)mtl.frame_counter;

    if (cvar_tm_blend_enable->integer && fd) {
        u->fs_blend_color.x = fd->blend[0];
        u->fs_blend_color.y = fd->blend[1];
        u->fs_blend_color.z = fd->blend[2];
        u->fs_blend_color.w = fd->blend[3];
    }
    if (fd && (fd->rdflags & RDF_IRGOGGLES)) {
        u->fs_colorize.x = 1.0f;
        u->fs_colorize.w = 0.8f;
    }

    u->reset_curve = reset_required ? 1.0f : 0.0f;
    u->frame_time = frame_time;
    u->is_hdr = mtl.is_hdr ? 1 : 0;
    u->tm_debug = (uint32_t)max(0, cvar_tm_debug->integer);
    u->hdr_clamp_strength = hdr_clamp_strength;
    // An scRGB luminance of 1.0 is 80 nits.
    u->ui_color_scale = mtl.is_hdr ? cvar_ui_hdr_nits->value * 0.0125f : 1.0f;

    u->tm_exposure_bias = cvar_tm_exposure_bias->value;
    u->tm_dyn_range_stops = cvar_tm_dyn_range_stops->value;
    u->tm_exposure_speed_up = cvar_tm_exposure_speed_up->value;
    u->tm_exposure_speed_down = cvar_tm_exposure_speed_down->value;
    u->tm_low_percentile = cvar_tm_low_percentile->value;
    u->tm_high_percentile = cvar_tm_high_percentile->value;
    u->tm_min_luminance = cvar_tm_min_luminance->value;
    u->tm_max_luminance = cvar_tm_max_luminance->value;
    u->tm_noise_stops = cvar_tm_noise_stops->value;
    u->tm_noise_blend = cvar_tm_noise_blend->value;
    u->tm_reinhard = cvar_tm_reinhard->value;
    u->tm_white_point = cvar_tm_white_point->value;
    u->tm_knee_start = cvar_tm_knee_start->value;
    u->tm_hdr_peak_nits = cvar_tm_hdr_peak_nits->value;
    u->tm_hdr_saturation_scale = cvar_tm_hdr_saturation_scale->value;
    u->tm_blend_scale_border = cvar_tm_blend_scale_border->value;
    u->tm_blend_scale_center = cvar_tm_blend_scale_center->value;
    u->tm_blend_scale_fade_exp = cvar_tm_blend_scale_fade_exp->value;
    u->tm_blend_distance_factor = cvar_tm_blend_distance_factor->value;
    u->tm_blend_max_alpha = cvar_tm_blend_max_alpha->value;

    // Half of a symmetric Gaussian used to blur the tone curve slopes.
    float sigma = max(cvar_tm_slope_blur_sigma->value, 1e-3f);
    float gaussian_sum = 0.0f;
    for (int i = 0; i < 14; i++) {
        float k = expf(-(float)(i * i) / (2.0f * sigma * sigma));
        gaussian_sum += k * (i == 0 ? 1.0f : 2.0f);
        u->weights[i] = k;
    }
    for (int i = 0; i < 14; i++)
        u->weights[i] /= gaussian_sum;

    // Knee: y(x) = (w x + a) / (x + b) that joins the identity at knee_start
    // and reaches 1 at the white point.
    float knee_start = cvar_tm_knee_start->value;
    float knee_white_point = max(cvar_tm_white_point->value, 1.0001f);
    u->knee_w = (knee_start * (knee_start - 2.0f) + knee_white_point) / (knee_white_point - 1.0f);
    u->knee_a = -knee_start * knee_start;
    u->knee_b = u->knee_w - 2.0f * knee_start;

    MTLSize threads = MTLSizeMake(16, 16, 1);
    MTLSize groups = MTLSizeMake((width + 15) / 16, (height + 15) / 16, 1);

    id<MTLComputeCommandEncoder> enc = mtl_profiler_compute_encoder(cmd, MTL_PROFILER_TONE_MAPPING);
    enc.label = @"tone mapping";

    [enc setComputePipelineState:pipeline_histogram];
    [enc setTexture:color atIndex:0];
    [enc setBuffer:uniform_buffers[mtl.frame_index] offset:0 atIndex:0];
    [enc setBuffer:tonemap_buffer offset:0 atIndex:1];
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];

    // The curve kernel is written for exactly one 128 thread group.
    [enc setComputePipelineState:pipeline_curve];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(HISTOGRAM_BINS, 1, 1)];

    [enc setComputePipelineState:pipeline_apply];
    [enc setTexture:color atIndex:0];
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];

    [enc endEncoding];

    reset_required = false;
}
