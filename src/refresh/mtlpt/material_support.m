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

// Support routines that the shared material system expects the renderer to
// provide. These are ports of the equivalents in refresh/vkpt/textures.c and
// operate purely on CPU side pixel data.

#include "mtlpt_metal.h"
#include "material_compat.h"
#include "../vkpt/material.h"

// Emissive textures are authored slightly hot; this pulls them back so that a
// surface which is merely bright does not read as a light source.
#define EMISSIVE_TRANSFORM_BIAS (-0.001f)

mtl_refdef_shim_t vkpt_refdef;

// Owned by the Vulkan backend when both are compiled in.
cvar_t *cvar_pt_surface_lights_fake_emissive_algo = NULL;
cvar_t *cvar_pt_surface_lights_threshold = NULL;

void mtl_material_init_cvars(void)
{
    cvar_pt_surface_lights_fake_emissive_algo =
        Cvar_Get("pt_surface_lights_fake_emissive_algo", "1", CVAR_FILES);
    cvar_pt_surface_lights_threshold =
        Cvar_Get("pt_surface_lights_threshold", "215", CVAR_FILES);
}

char *sgets(char *str, int num, char const **input)
{
    char const *next = *input;
    int numread = 0;

    while (numread + 1 < num && *next) {
        int isnewline = (*next == '\n');
        *str++ = *next++;
        numread++;
        if (isnewline)
            break;
    }

    if (numread == 0)
        return NULL;

    *str = '\0';
    *input = next;
    return str;
}

static float decode_srgb(byte value)
{
    float x = value / 255.0f;
    if (x <= 0.04045f)
        return x / 12.92f;
    return powf((x + 0.055f) / 1.055f, 2.4f);
}

static byte encode_srgb(float value)
{
    float x = value <= 0.0031308f ? value * 12.92f
                                  : 1.055f * powf(value, 1.0f / 2.4f) - 0.055f;
    return (byte)(Q_clipf(x, 0.0f, 1.0f) * 255.0f + 0.5f);
}

// Keeps only the pixels brighter than the threshold, so a diffuse texture can
// stand in for a missing emissive map.
static void apply_fake_emissive_threshold(image_t *image, int bright_threshold_int)
{
    if (!image->pix_data)
        return;

    float threshold = bright_threshold_int / 255.0f;
    bool use_luminance = cvar_pt_surface_lights_fake_emissive_algo &&
                         cvar_pt_surface_lights_fake_emissive_algo->integer != 0;

    byte *pixel = image->pix_data;
    int count = image->upload_width * image->upload_height;

    for (int i = 0; i < count; i++, pixel += 4) {
        float r = decode_srgb(pixel[0]);
        float g = decode_srgb(pixel[1]);
        float b = decode_srgb(pixel[2]);

        float measure = use_luminance ? LUMINANCE(r, g, b) : max(r, max(g, b));

        if (measure < threshold) {
            pixel[0] = pixel[1] = pixel[2] = 0;
            continue;
        }

        pixel[0] = encode_srgb(r);
        pixel[1] = encode_srgb(g);
        pixel[2] = encode_srgb(b);
    }
}

image_t *vkpt_fake_emissive_texture(image_t *image, int bright_threshold_int)
{
    if (!image)
        return NULL;

    if (image->upload_width == 1 && image->upload_height == 1)
        return image;

    // The fake extension is required by the image lookup logic.
    const char emissive_image_suffix[] = "*E.wal";
    char emissive_image_name[MAX_QPATH];

    Q_strlcpy(emissive_image_name, image->name, sizeof(emissive_image_name));
    size_t pos = strlen(emissive_image_name) - 4;
    if (pos + sizeof(emissive_image_suffix) > sizeof(emissive_image_name))
        pos = sizeof(emissive_image_name) - sizeof(emissive_image_suffix);
    Q_strlcpy(emissive_image_name + pos, emissive_image_suffix,
              sizeof(emissive_image_name) - pos);

    image_t *prev_image = IMG_FindExisting(emissive_image_name, image->type);
    if (prev_image != R_NOTEXTURE) {
        prev_image->registration_sequence = registration_sequence;
        return prev_image;
    }

    image_t *new_image = IMG_Clone(image, emissive_image_name);
    if (new_image == R_NOTEXTURE)
        return image;

    new_image->flags |= IF_FAKE_EMISSIVE |
                        (Q_clip_uint8(bright_threshold_int) << IF_FAKE_EMISSIVE_THRESH_SHIFT);
    apply_fake_emissive_threshold(new_image, bright_threshold_int);

    mtl_textures_invalidate(new_image);
    return new_image;
}

void vkpt_extract_emissive_texture_info(image_t *image)
{
    int w = image->upload_width;
    int h = image->upload_height;

    if (!image->pix_data || w <= 0 || h <= 0) {
        VectorSet(image->light_color, 0.f, 0.f, 0.f);
        image->processing_complete = true;
        return;
    }

    byte *current_pixel = image->pix_data;
    vec3_t emissive_color;
    VectorClear(emissive_color);

    int min_x = w, max_x = -1;
    int min_y = h, max_y = -1;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (current_pixel[0] + current_pixel[1] + current_pixel[2] > 0) {
                vec3_t color;
                color[0] = max(0.f, decode_srgb(current_pixel[0]) + EMISSIVE_TRANSFORM_BIAS);
                color[1] = max(0.f, decode_srgb(current_pixel[1]) + EMISSIVE_TRANSFORM_BIAS);
                color[2] = max(0.f, decode_srgb(current_pixel[2]) + EMISSIVE_TRANSFORM_BIAS);

                VectorAdd(emissive_color, color, emissive_color);

                min_x = min(min_x, x);
                min_y = min(min_y, y);
                max_x = max(max_x, x);
                max_y = max(max_y, y);
            }

            current_pixel += 4;
        }
    }

    if (min_x <= max_x && min_y <= max_y) {
        float normalization = 1.f / (float)((max_x - min_x + 1) * (max_y - min_y + 1));
        VectorScale(emissive_color, normalization, image->light_color);
    } else {
        VectorSet(image->light_color, 0.f, 0.f, 0.f);
    }

    image->min_light_texcoord[0] = (float)min_x / (float)w;
    image->min_light_texcoord[1] = (float)min_y / (float)h;
    image->max_light_texcoord[0] = (float)(max_x + 1) / (float)w;
    image->max_light_texcoord[1] = (float)(max_y + 1) / (float)h;

    image->entire_texture_emissive =
        (min_x == 0) && (min_y == 0) && (max_x == w - 1) && (max_y == h - 1);

    image->processing_complete = true;
}

// The Metal backend rebuilds world geometry on map load rather than caching
// per-material vertex buffers, so a live material edit needs no invalidation.
void vkpt_vertex_buffer_invalidate_static_model_vbos(int material_index)
{
}
