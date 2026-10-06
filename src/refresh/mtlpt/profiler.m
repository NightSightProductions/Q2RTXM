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

// GPU pass profiler, the Metal counterpart of vkpt/profiler.c. Metal has no
// arbitrary timestamp writes inside an encoder, so timings are taken at
// encoder boundaries with MTLCounterSampleBuffer stage sampling; every
// profiled pass therefore gets its own encoder through
// mtl_profiler_compute_encoder().

#include "mtlpt_metal.h"
#include "mtlpt_profiler.h"
#include <ctype.h>

extern cvar_t *cvar_flt_fsr_enable;
static cvar_t *cvar_pt_reflect_refract;

const char *mtl_profiler_labels[MTL_PROFILER_NUM_ENTRIES] = {
#define PROFILER_DO(a, ...) #a,
    MTL_PROFILER_LIST
#undef PROFILER_DO
};

typedef struct {
    uint64_t *data;
    size_t    num_samples;
    size_t    next_idx;
    uint64_t  accumulated;
} sample_ring_t;

static struct {
    id<MTLCounterSampleBuffer> buffers[MTL_FRAMES_IN_FLIGHT];
    bool     used[MTL_FRAMES_IN_FLIGHT][MTL_PROFILER_NUM_ENTRIES];
    double   results_ms[MTL_PROFILER_NUM_ENTRIES];
    double   ticks_to_ms;
    bool     supported;
    volatile double pending_frame_ms;

    size_t        allocated_samples;
    sample_ring_t samples[MTL_PROFILER_NUM_ENTRIES];
} prof;

cvar_t *cvar_profiler;
cvar_t *cvar_profiler_samples;
cvar_t *cvar_profiler_scale;

static void set_sample_count(size_t count)
{
    for (int i = 0; i < MTL_PROFILER_NUM_ENTRIES; i++) {
        Z_Free(prof.samples[i].data);
        prof.samples[i].data = Z_Mallocz(sizeof(uint64_t) * count);
        prof.samples[i].num_samples = 0;
        prof.samples[i].next_idx = 0;
        prof.samples[i].accumulated = 0;
    }
    prof.allocated_samples = count;
}

static void reset_samples(int idx)
{
    prof.samples[idx].num_samples = 0;
    prof.samples[idx].next_idx = 0;
    prof.samples[idx].accumulated = 0;
}

static void record_sample(int idx, uint64_t value)
{
    sample_ring_t *s = &prof.samples[idx];
    if (s->num_samples == prof.allocated_samples)
        s->accumulated -= s->data[s->next_idx];
    else
        s->num_samples++;
    s->data[s->next_idx] = value;
    s->accumulated += value;
    s->next_idx = (s->next_idx + 1) % prof.allocated_samples;
}

// Dev aid: prints every pass's average over the sample window.
static void Profiler_Dump_f(void)
{
    for (int i = 0; i < MTL_PROFILER_NUM_ENTRIES; i++) {
        const sample_ring_t *r = &prof.samples[i];
        if (!r->num_samples)
            continue;
        double avg = (double)r->accumulated / (double)r->num_samples * prof.ticks_to_ms;
        Com_Printf("prof %-40s %7.3f ms\n", mtl_profiler_labels[i] + 13, avg);
    }
}

bool mtl_profiler_init(void)
{
    Cmd_AddCommand("profiler_dump", Profiler_Dump_f);
    cvar_profiler = Cvar_Get("profiler", "0", 0);
    cvar_profiler_samples = Cvar_Get("profiler_samples", "60", CVAR_ARCHIVE);
    cvar_profiler_scale = Cvar_Get("profiler_scale", "1", CVAR_ARCHIVE);
    cvar_pt_reflect_refract = Cvar_Get("pt_reflect_refract", "2", CVAR_ARCHIVE);

    memset(&prof, 0, sizeof(prof));
    set_sample_count(60);

    id<MTLCounterSet> timestamps = nil;
    for (id<MTLCounterSet> set in mtl.device.counterSets) {
        if ([set.name isEqualToString:MTLCommonCounterSetTimestamp])
            timestamps = set;
    }
    if (!timestamps || ![mtl.device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
        Com_Printf("Metal: GPU timestamp sampling not supported, profiler disabled\n");
        return true;
    }

    MTLCounterSampleBufferDescriptor *desc = [[MTLCounterSampleBufferDescriptor alloc] init];
    desc.counterSet = timestamps;
    desc.storageMode = MTLStorageModeShared;
    desc.sampleCount = MTL_PROFILER_NUM_ENTRIES * 2;

    for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++) {
        NSError *err = nil;
        desc.label = [NSString stringWithFormat:@"profiler %d", i];
        prof.buffers[i] = [mtl.device newCounterSampleBufferWithDescriptor:desc error:&err];
        if (!prof.buffers[i]) {
            mtl_log_error("newCounterSampleBufferWithDescriptor", err);
            [desc release];
            return true;
        }
    }
    [desc release];

    // GPU timestamps are in device ticks; correlate two host/GPU samples to
    // find the tick period.
    MTLTimestamp cpu0, gpu0, cpu1, gpu1;
    [mtl.device sampleTimestamps:&cpu0 gpuTimestamp:&gpu0];
    usleep(20000);
    [mtl.device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
    if (gpu1 > gpu0 && cpu1 > cpu0)
        prof.ticks_to_ms = (double)(cpu1 - cpu0) / (double)(gpu1 - gpu0) * 1e-6;
    else
        prof.ticks_to_ms = 1e-6;

    prof.supported = true;
    return true;
}

void mtl_profiler_shutdown(void)
{
    Cmd_RemoveCommand("profiler_dump");
    for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++) {
        [prof.buffers[i] release];
        prof.buffers[i] = nil;
    }
    for (int i = 0; i < MTL_PROFILER_NUM_ENTRIES; i++) {
        Z_Free(prof.samples[i].data);
        prof.samples[i].data = NULL;
    }
    prof.supported = false;
}

bool mtl_profiler_enabled(void)
{
    return prof.supported && cvar_profiler && cvar_profiler->integer;
}

// Reads back the slot about to be reused (its command buffer has completed).
void mtl_profiler_next_frame(void)
{
    size_t new_samples = (size_t)max(cvar_profiler_samples->integer, 1);
    if (prof.allocated_samples != new_samples)
        set_sample_count(new_samples);

    if (!prof.supported)
        return;

    double frame_ms = prof.pending_frame_ms;
    if (frame_ms > 0.0) {
        prof.results_ms[MTL_PROFILER_FRAME_TIME] = frame_ms;
        record_sample(MTL_PROFILER_FRAME_TIME, (uint64_t)(frame_ms / prof.ticks_to_ms));
    }

    uint32_t slot = mtl.frame_index;
    id<MTLCounterSampleBuffer> buffer = prof.buffers[slot];

    NSData *data = [buffer resolveCounterRange:NSMakeRange(0, MTL_PROFILER_NUM_ENTRIES * 2)];
    const MTLCounterResultTimestamp *ts = (const MTLCounterResultTimestamp *)data.bytes;
    bool have = data && data.length >= sizeof(MTLCounterResultTimestamp) * MTL_PROFILER_NUM_ENTRIES * 2;

    for (int i = 0; i < MTL_PROFILER_NUM_ENTRIES; i++) {
        if (i == MTL_PROFILER_FRAME_TIME)
            continue;
        uint64_t begin = have ? ts[i * 2 + 0].timestamp : 0;
        uint64_t end = have ? ts[i * 2 + 1].timestamp : 0;
        bool valid = prof.used[slot][i] && begin != 0 && end != 0 &&
                     begin != MTLCounterErrorValue && end != MTLCounterErrorValue && end >= begin;
        if (valid) {
            prof.results_ms[i] = (double)(end - begin) * prof.ticks_to_ms;
            record_sample(i, end - begin);
        } else {
            prof.results_ms[i] = 0.0;
            if (!prof.used[slot][i])
                reset_samples(i);
        }
        prof.used[slot][i] = false;
    }

    // vkpt's parent rows time a range around their sub-passes. Here every
    // pass has its own encoder, so a parent is the sum of its children.
    static const struct { int parent, children[4]; } groups[] = {
        { MTL_PROFILER_INDIRECT_LIGHTING, { MTL_PROFILER_INDIRECT_LIGHTING_0, MTL_PROFILER_INDIRECT_LIGHTING_1, -1, -1 } },
        { MTL_PROFILER_DENOISE_FULL, { MTL_PROFILER_DENOISE_GRADIENT, MTL_PROFILER_DENOISE_TEMPORAL,
                                       MTL_PROFILER_DENOISE_ATROUS, MTL_PROFILER_TAA } },
        { MTL_PROFILER_FSR, { MTL_PROFILER_FSR_EASU, MTL_PROFILER_FSR_RCAS, -1, -1 } },
        // Dynamic geometry: vkpt instances it on the GPU then builds BLASes
        // in "bvh update"; Metal builds entity and effect BLASes up front.
        { MTL_PROFILER_INSTANCE_GEOMETRY, { -1, -1, -1, -1 } },
        { MTL_PROFILER_BVH_TOTAL, { MTL_PROFILER_ENTITY_BLAS, MTL_PROFILER_EFFECTS_BLAS, MTL_PROFILER_BVH_UPDATE, -1 } },
    };
    for (size_t g = 0; g < sizeof(groups) / sizeof(groups[0]); g++) {
        double ms = 0.0;
        for (int c = 0; c < 4; c++)
            if (groups[g].children[c] >= 0)
                ms += prof.results_ms[groups[g].children[c]];
        int parent = groups[g].parent;
        prof.results_ms[parent] = ms;
        if (ms > 0.0)
            record_sample(parent, (uint64_t)(ms / prof.ticks_to_ms));
        else
            reset_samples(parent);
    }
}

void mtl_profiler_set_frame_time(double gpu_ms)
{
    // Called from the command buffer completion handler; only publish.
    prof.pending_frame_ms = gpu_ms;
}

id<MTLComputeCommandEncoder> mtl_profiler_compute_encoder(id<MTLCommandBuffer> cmd, int entry)
{
    if (!mtl_profiler_enabled() || entry < 0 || entry >= MTL_PROFILER_NUM_ENTRIES)
        return [cmd computeCommandEncoder];

    uint32_t slot = mtl.frame_index;
    MTLComputePassDescriptor *desc = [MTLComputePassDescriptor computePassDescriptor];
    desc.sampleBufferAttachments[0].sampleBuffer = prof.buffers[slot];
    desc.sampleBufferAttachments[0].startOfEncoderSampleIndex = entry * 2;
    desc.sampleBufferAttachments[0].endOfEncoderSampleIndex = entry * 2 + 1;
    prof.used[slot][entry] = true;
    return [cmd computeCommandEncoderWithDescriptor:desc];
}

id<MTLAccelerationStructureCommandEncoder> mtl_profiler_accel_encoder(id<MTLCommandBuffer> cmd, int entry)
{
    if (!mtl_profiler_enabled() || entry < 0 || entry >= MTL_PROFILER_NUM_ENTRIES)
        return [cmd accelerationStructureCommandEncoder];

    uint32_t slot = mtl.frame_index;
    MTLAccelerationStructurePassDescriptor *desc = [MTLAccelerationStructurePassDescriptor accelerationStructurePassDescriptor];
    desc.sampleBufferAttachments[0].sampleBuffer = prof.buffers[slot];
    desc.sampleBufferAttachments[0].startOfEncoderSampleIndex = entry * 2;
    desc.sampleBufferAttachments[0].endOfEncoderSampleIndex = entry * 2 + 1;
    prof.used[slot][entry] = true;
    return [cmd accelerationStructureCommandEncoderWithDescriptor:desc];
}

// vkpt's draw_query(), with the vkpt row label passed in. idx < 0 is a row
// vkpt has but Metal does not time; it reads N/A like any idle vkpt row.
static void draw_query(int x, int y, qhandle_t font, const char *label, int idx)
{
    char buf[256];

    R_DrawString(x, y, 0, 128, label, font);

    double ms = idx >= 0 ? prof.results_ms[idx] : 0.0;
    double avg_ms = (idx >= 0 && prof.samples[idx].num_samples)
                  ? (double)prof.samples[idx].accumulated / (double)prof.samples[idx].num_samples * prof.ticks_to_ms
                  : 0.0;

    if (ms > 0.005)
        snprintf(buf, sizeof buf, "%8.2f ms %8.2f ms", ms, avg_ms);
    else if (avg_ms > 0.005)
        snprintf(buf, sizeof buf, "       N/A  %8.2f ms", avg_ms);
    else
        snprintf(buf, sizeof buf, "       N/A");

    R_DrawString(x + 256, y, 0, 128, buf, font);
}

void mtl_profiler_draw(bool denoiser_enabled)
{
    if (!cvar_profiler->integer)
        return;

    float profiler_scale = R_ClampScale(cvar_profiler_scale);
    int x = (int)(500 * profiler_scale);
    int y = (int)(100 * profiler_scale);

    qhandle_t font = R_RegisterFont("conchars");
    if (!font)
        return;

    R_SetScale(profiler_scale);

    R_DrawString(x + 256, y - 16, 0, 128, "    imm         avg", font);

// vkpt's rows, order and labels (draw_profiler).
#define PROFILER_DO(label, idx) \
    draw_query(x, y, font, label, idx); y += 10;

    int reflect_refract = cvar_pt_reflect_refract ? cvar_pt_reflect_refract->integer : 2;

    PROFILER_DO("frame time", MTL_PROFILER_FRAME_TIME);
    PROFILER_DO("instance geometry", MTL_PROFILER_INSTANCE_GEOMETRY);
    PROFILER_DO("bvh update", MTL_PROFILER_BVH_TOTAL);
    PROFILER_DO("update environment", MTL_PROFILER_UPDATE_ENVIRONMENT);
    PROFILER_DO("shadow map", MTL_PROFILER_SHADOW_MAP);
    PROFILER_DO("primary rays", MTL_PROFILER_PRIMARY_RAYS);
    if (reflect_refract > 0) { PROFILER_DO("reflect refract 1", MTL_PROFILER_REFLECT_REFRACT_1); }
    if (reflect_refract > 1) { PROFILER_DO("reflect refract 2", MTL_PROFILER_REFLECT_REFRACT_2); }
    if (denoiser_enabled) {
        PROFILER_DO("asvgf gradient reproject", MTL_PROFILER_DENOISE_GRADIENT_REPROJECT);
    }
    PROFILER_DO("direct lighting", MTL_PROFILER_DIRECT_LIGHTING);
    PROFILER_DO("indirect lighting", MTL_PROFILER_INDIRECT_LIGHTING);
    PROFILER_DO("indirect lighting 0", MTL_PROFILER_INDIRECT_LIGHTING_0);
    PROFILER_DO("indirect lighting 1", MTL_PROFILER_INDIRECT_LIGHTING_1);
    PROFILER_DO("god rays", MTL_PROFILER_GOD_RAYS);
    PROFILER_DO("god rays reflect refract", MTL_PROFILER_GOD_RAYS_REFLECT_REFRACT);
    PROFILER_DO("god rays filter", MTL_PROFILER_GOD_RAYS_FILTER);
    if (denoiser_enabled) {
        PROFILER_DO("asvgf full", MTL_PROFILER_DENOISE_FULL);
        PROFILER_DO("asvgf reconstruct gradient", MTL_PROFILER_DENOISE_GRADIENT);
        PROFILER_DO("asvgf temporal", MTL_PROFILER_DENOISE_TEMPORAL);
        PROFILER_DO("asvgf atrous", MTL_PROFILER_DENOISE_ATROUS);
        PROFILER_DO("asvgf taa", MTL_PROFILER_TAA);
    } else {
        PROFILER_DO("compositing", MTL_PROFILER_COMPOSITING);
    }
    PROFILER_DO("interleave", MTL_PROFILER_INTERLEAVE);
    PROFILER_DO("bloom", MTL_PROFILER_BLOOM);
    PROFILER_DO("tone mapping", MTL_PROFILER_TONE_MAPPING);
    if (cvar_flt_fsr_enable && cvar_flt_fsr_enable->integer) {
        PROFILER_DO("fsr", MTL_PROFILER_FSR);
        PROFILER_DO("fsr easu", MTL_PROFILER_FSR_EASU);
        PROFILER_DO("fsr rcas", MTL_PROFILER_FSR_RCAS);
    }
#undef PROFILER_DO

    R_SetScale(1.0f);
}
