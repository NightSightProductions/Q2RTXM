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

#pragma once

#import <Metal/Metal.h>
#include <stdbool.h>

// Profiled GPU passes. Names double as the overlay labels.
#define MTL_PROFILER_LIST \
    PROFILER_DO(MTL_PROFILER_FRAME_TIME) \
    PROFILER_DO(MTL_PROFILER_ENTITY_BLAS) \
    PROFILER_DO(MTL_PROFILER_EFFECTS_BLAS) \
    PROFILER_DO(MTL_PROFILER_BVH_UPDATE) \
    PROFILER_DO(MTL_PROFILER_UPDATE_ENVIRONMENT) \
    PROFILER_DO(MTL_PROFILER_SHADOW_MAP) \
    PROFILER_DO(MTL_PROFILER_PRIMARY_RAYS) \
    PROFILER_DO(MTL_PROFILER_GOD_RAYS) \
    PROFILER_DO(MTL_PROFILER_REFLECT_REFRACT_1) \
    PROFILER_DO(MTL_PROFILER_GOD_RAYS_REFLECT_REFRACT) \
    PROFILER_DO(MTL_PROFILER_GOD_RAYS_FILTER) \
    PROFILER_DO(MTL_PROFILER_REFLECT_REFRACT_2) \
    PROFILER_DO(MTL_PROFILER_DENOISE_GRADIENT_REPROJECT) \
    PROFILER_DO(MTL_PROFILER_DIRECT_LIGHTING) \
    PROFILER_DO(MTL_PROFILER_INDIRECT_LIGHTING) \
    PROFILER_DO(MTL_PROFILER_INDIRECT_LIGHTING_0) \
    PROFILER_DO(MTL_PROFILER_INDIRECT_LIGHTING_1) \
    PROFILER_DO(MTL_PROFILER_DENOISE_FULL) \
    PROFILER_DO(MTL_PROFILER_DENOISE_GRADIENT) \
    PROFILER_DO(MTL_PROFILER_DENOISE_TEMPORAL) \
    PROFILER_DO(MTL_PROFILER_DENOISE_ATROUS) \
    PROFILER_DO(MTL_PROFILER_COMPOSITING) \
    PROFILER_DO(MTL_PROFILER_INTERLEAVE) \
    PROFILER_DO(MTL_PROFILER_TAA) \
    PROFILER_DO(MTL_PROFILER_BLOOM) \
    PROFILER_DO(MTL_PROFILER_TONE_MAPPING) \
    PROFILER_DO(MTL_PROFILER_FSR) \
    PROFILER_DO(MTL_PROFILER_FSR_EASU) \
    PROFILER_DO(MTL_PROFILER_FSR_RCAS) \
    PROFILER_DO(MTL_PROFILER_INSTANCE_GEOMETRY) \
    PROFILER_DO(MTL_PROFILER_BVH_TOTAL)

enum {
#define PROFILER_DO(a) a,
    MTL_PROFILER_LIST
#undef PROFILER_DO
    MTL_PROFILER_NUM_ENTRIES
};

bool mtl_profiler_init(void);
void mtl_profiler_shutdown(void);
bool mtl_profiler_enabled(void);
// Call once per frame after the slot's previous command buffer completed.
void mtl_profiler_next_frame(void);
void mtl_profiler_set_frame_time(double gpu_ms);
void mtl_profiler_draw(bool denoiser_enabled);

// Encoders whose start/end timestamps land in the given profiler entry. Any
// pass that wants to show up in the overlay must use its own encoder.
id<MTLComputeCommandEncoder> mtl_profiler_compute_encoder(id<MTLCommandBuffer> cmd, int entry);
id<MTLAccelerationStructureCommandEncoder> mtl_profiler_accel_encoder(id<MTLCommandBuffer> cmd, int entry);
