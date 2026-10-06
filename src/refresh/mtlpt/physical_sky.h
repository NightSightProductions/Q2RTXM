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

// Port of the Vulkan backend's physical sky: an atmospheric scattering model
// evaluated into an environment cube map, plus the sun light that goes with it.

#ifndef MTLPT_PHYSICAL_SKY_H
#define MTLPT_PHYSICAL_SKY_H

#include "shared/shared.h"

typedef struct {
    vec3_t direction;       // towards the sun, world space
    vec3_t color;           // radiance, already scaled by sun_brightness
    float  angular_size_rad;
    bool   use_physical_sky;
    bool   visible;
} mtl_sun_light_t;

bool mtl_physical_sky_init(void);
void mtl_physical_sky_shutdown(void);

// Recomputes the sun position and, when anything changed, re-renders the sky
// cube map. Safe to call every frame.
void mtl_physical_sky_update(float time);

// True when the active preset draws an atmosphere rather than the map's sky box.
bool mtl_physical_sky_active(void);

const mtl_sun_light_t *mtl_physical_sky_sun(void);

// Average radiance of the rendered sky, used for the sky area lights.
void mtl_physical_sky_average_color(vec3_t out);

#ifdef __OBJC__
#import <Metal/Metal.h>
// The rendered environment cube map. Never nil once init succeeded, so the
// path tracer can bind it unconditionally.
id<MTLTexture> mtl_physical_sky_texture(void);
#endif

#endif // MTLPT_PHYSICAL_SKY_H
