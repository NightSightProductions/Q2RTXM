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

// Lets refresh/vkpt/material.c compile against the Metal backend by supplying
// the handful of declarations it would otherwise pick up from vkpt.h. The
// material system itself is renderer agnostic, so the file is shared rather
// than duplicated.

#pragma once

#include "shared/shared.h"
#include "shared/list.h"
#include "common/cmd.h"
#include "common/common.h"
#include "common/cvar.h"
#include "common/files.h"
#include "common/prompt.h"
#include "refresh/refresh.h"
#include "refresh/images.h"
#include "refresh/models.h"

// Material kind and flag bits, shared with the Vulkan backend's shaders.
#include "../vkpt/shader/constants.h"

// material.c reads the material index the view is currently looking at in order
// to refresh geometry after a live material edit.
typedef struct {
    refdef_t *fd;
} mtl_refdef_shim_t;

extern mtl_refdef_shim_t vkpt_refdef;

// Implemented in refresh/mtlpt/material_support.m. The names match the Vulkan
// backend so the shared material.c needs no further changes.
image_t *vkpt_fake_emissive_texture(image_t *image, int bright_threshold_int);
void     vkpt_extract_emissive_texture_info(image_t *image);
void     vkpt_vertex_buffer_invalidate_static_model_vbos(int material_index);

// fgets() over an in-memory string, used by the material file parser.
char *sgets(char *str, int num, char const **input);
