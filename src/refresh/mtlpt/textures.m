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

// Maps the engine's image_t records onto MTLTextures and exposes them to the
// shaders as a Metal 3 bindless table of MTLResourceIDs.

#include "mtlpt_metal.h"

#define MTL_WHITE_TEXTURE_INDEX MAX_RIMAGES
#define MTL_TEXTURE_TABLE_SIZE  (MAX_RIMAGES + 1)

static id<MTLTexture>  textures[MTL_TEXTURE_TABLE_SIZE];
static id<MTLBuffer>   texture_table;
static id<MTLTexture> *resident_list;
static NSUInteger      resident_count;
static bool            table_dirty;
static bool            pending_mipmaps;

// Image backing cinematic playback, registered through the normal image cache.
static image_t        *raw_image;

static void write_table_entry(uint32_t index, id<MTLTexture> texture)
{
    if (!texture_table)
        return;

    MTLResourceID *slots = (MTLResourceID *)texture_table.contents;
    slots[index] = texture.gpuResourceID;
    table_dirty = true;
}

static id<MTLTexture> create_white_texture(void)
{
    MTLTextureDescriptor *desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:1
                                                          height:1
                                                       mipmapped:NO];
    desc.storageMode = MTLStorageModeShared;
    desc.usage = MTLTextureUsageShaderRead;

    id<MTLTexture> tex = [mtl.device newTextureWithDescriptor:desc];
    const uint32_t white = 0xffffffffu;
    [tex replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
           mipmapLevel:0
             withBytes:&white
           bytesPerRow:4];
    tex.label = @"white";
    return tex;
}

bool mtl_textures_init(void)
{
    texture_table = [mtl.device newBufferWithLength:sizeof(MTLResourceID) * MTL_TEXTURE_TABLE_SIZE
                                            options:MTLResourceStorageModeShared];
    if (!texture_table) {
        Com_EPrintf("Metal: could not allocate the texture table\n");
        return false;
    }
    texture_table.label = @"texture table";

    resident_list = Z_Mallocz(sizeof(id<MTLTexture>) * MTL_TEXTURE_TABLE_SIZE);
    resident_count = 0;

    textures[MTL_WHITE_TEXTURE_INDEX] = create_white_texture();
    if (!textures[MTL_WHITE_TEXTURE_INDEX])
        return false;

    // Point every slot at the white texture so an unregistered handle renders
    // as an untextured quad instead of sampling a stale resource.
    for (uint32_t i = 0; i < MTL_TEXTURE_TABLE_SIZE; i++)
        write_table_entry(i, textures[MTL_WHITE_TEXTURE_INDEX]);

    resident_list[resident_count++] = textures[MTL_WHITE_TEXTURE_INDEX];
    return true;
}

void mtl_textures_shutdown(void)
{
    for (int i = 0; i < MTL_TEXTURE_TABLE_SIZE; i++) {
        [textures[i] release];
        textures[i] = nil;
    }

    raw_image = NULL;

    [texture_table release];
    texture_table = nil;

    Z_Free(resident_list);
    resident_list = NULL;
    resident_count = 0;
}

static MTLPixelFormat pixel_format_for(const image_t *image)
{
    if (image->pixel_format == PF_R16_UNORM)
        return MTLPixelFormatR16Unorm;

    return image->is_srgb ? MTLPixelFormatRGBA8Unorm_sRGB : MTLPixelFormatRGBA8Unorm;
}

// normalize_normal_map.comp: vkpt renormalizes the base level of every
// linear normal map before generating its mips. The maps are not stored unit
// length, and the shortfall reads as normal variance to the Toksvig roughness
// filter: without this, distant surfaces turned much rougher than near ones
// and smooth metal showed a hard line, moving with the camera, where its
// roughness crossed the direct specular threshold.
static byte *normalized_normal_map(const image_t *image, int width, int height)
{
    size_t count = (size_t)width * height;
    byte *out = Z_Malloc(count * 4);
    const byte *in = image->pix_data;

    for (size_t i = 0; i < count; i++) {
        float x = in[i * 4 + 0] / 255.0f * 2.0f - 1.0f;
        float y = in[i * 4 + 1] / 255.0f * 2.0f - 1.0f;
        float z = in[i * 4 + 2] / 255.0f;
        float len = sqrtf(x * x + y * y + z * z);
        if (len == 0.0f) {
            x = 0.0f; y = 0.0f; z = 1.0f;
        } else {
            x /= len; y /= len; z /= len;
        }
        out[i * 4 + 0] = (byte)Q_clip((int)lrintf((x * 0.5f + 0.5f) * 255.0f), 0, 255);
        out[i * 4 + 1] = (byte)Q_clip((int)lrintf((y * 0.5f + 0.5f) * 255.0f), 0, 255);
        out[i * 4 + 2] = (byte)Q_clip((int)lrintf(z * 255.0f), 0, 255);
        out[i * 4 + 3] = in[i * 4 + 3];
    }
    return out;
}

static void upload_image(uint32_t index, image_t *image)
{
    int width = image->upload_width;
    int height = image->upload_height;

    if (width <= 0 || height <= 0 || !image->pix_data)
        return;

    MTLTextureDescriptor *desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format_for(image)
                                                           width:width
                                                          height:height
                                                       mipmapped:YES];
    desc.storageMode = MTLStorageModeShared;
    desc.usage = MTLTextureUsageShaderRead;

    id<MTLTexture> tex = [mtl.device newTextureWithDescriptor:desc];
    if (!tex) {
        Com_EPrintf("Metal: out of memory uploading '%s'\n", image->name);
        return;
    }

    NSUInteger bytes_per_pixel = (image->pixel_format == PF_R16_UNORM) ? 2 : 4;
    bool normalize = (image->flags & IF_NORMAL_MAP) && !image->is_srgb && bytes_per_pixel == 4;
    byte *normalized = normalize ? normalized_normal_map(image, width, height) : NULL;
    [tex replaceRegion:MTLRegionMake2D(0, 0, width, height)
           mipmapLevel:0
             withBytes:normalized ? normalized : image->pix_data
           bytesPerRow:bytes_per_pixel * width];
    Z_Free(normalized);

    tex.label = [NSString stringWithUTF8String:image->name];

    [textures[index] release];
    textures[index] = tex;
    write_table_entry(index, tex);
    pending_mipmaps = true;

    image->processing_complete = true;
}

static void rebuild_resident_list(void)
{
    resident_count = 0;
    for (uint32_t i = 0; i < MTL_TEXTURE_TABLE_SIZE; i++) {
        if (textures[i])
            resident_list[resident_count++] = textures[i];
    }
}

// Mip levels are allocated at upload time but have to be filled on the GPU.
static void generate_mipmaps(void)
{
    if (!pending_mipmaps)
        return;

    id<MTLCommandBuffer> cmd = [mtl.queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];

    for (uint32_t i = 0; i < MTL_TEXTURE_TABLE_SIZE; i++) {
        if (textures[i] && textures[i].mipmapLevelCount > 1)
            [blit generateMipmapsForTexture:textures[i]];
    }

    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    pending_mipmaps = false;
}

void mtl_textures_update(void)
{
    bool changed = false;

    for (int i = 0; i < r_numImages; i++) {
        image_t *image = &r_images[i];

        if (!image->registration_sequence)
            continue;
        if (textures[i] || !image->pix_data)
            continue;

        upload_image(i, image);
        changed = true;
    }

    if (changed)
        rebuild_resident_list();

    generate_mipmaps();

    if (table_dirty) {
        // Shared storage on Apple Silicon needs no explicit flush, but blit
        // mipmap generation does need a command buffer.
        table_dirty = false;
    }
}

void mtl_textures_invalidate(const image_t *image)
{
    ptrdiff_t index = image - r_images;
    if (index < 0 || index >= MAX_RIMAGES)
        return;

    if (!textures[index])
        return;

    [textures[index] release];
    textures[index] = nil;
    write_table_entry((uint32_t)index, textures[MTL_WHITE_TEXTURE_INDEX]);
    rebuild_resident_list();
}

uint32_t mtl_texture_index_for_handle(qhandle_t pic)
{
    if (pic == (qhandle_t)MTL_TEXNUM_WHITE)
        return MTL_WHITE_TEXTURE_INDEX;
    if (pic < 0 || pic >= MAX_RIMAGES)
        return MTL_WHITE_TEXTURE_INDEX;
    if (!textures[pic])
        return MTL_WHITE_TEXTURE_INDEX;

    return (uint32_t)pic;
}

uint32_t mtl_texture_index_optional(qhandle_t pic)
{
    if (pic <= 0 || pic >= MAX_RIMAGES)
        return 0;
    if (!textures[pic])
        return 0;

    return (uint32_t)pic;
}

id<MTLTexture> mtl_texture_for_index(uint32_t index)
{
    if (index >= MTL_TEXTURE_TABLE_SIZE)
        return textures[MTL_WHITE_TEXTURE_INDEX];
    return textures[index];
}

id<MTLBuffer> mtl_texture_argument_buffer(void)
{
    return texture_table;
}

void mtl_textures_encode_use(id<MTLRenderCommandEncoder> encoder)
{
    if (resident_count)
        [encoder useResources:resident_list
                        count:resident_count
                        usage:MTLResourceUsageRead
                       stages:MTLRenderStageFragment];
}

void mtl_textures_encode_use_compute(id<MTLComputeCommandEncoder> encoder)
{
    if (resident_count)
        [encoder useResources:resident_list count:resident_count usage:MTLResourceUsageRead];
}

//
// Engine entry points
//

void IMG_Load_MTL(image_t *image, byte *pic)
{
    image->pix_data = pic;
    image->processing_complete = false;
}

void IMG_Unload_MTL(image_t *image)
{
    mtl_textures_invalidate(image);

    if (image->pix_data) {
        Z_Free(image->pix_data);
        image->pix_data = NULL;
    }
}

void R_UpdateRawPic_MTL(int pic_w, int pic_h, const uint32_t *pic)
{
    if (raw_image)
        R_UnregisterImage(raw_image - r_images);

    size_t raw_size = (size_t)pic_w * pic_h * 4;
    byte *raw_data = Z_Malloc(raw_size);
    memcpy(raw_data, pic, raw_size);

    static int raw_id;
    raw_image = r_images + R_RegisterRawImage(va("**raw[%d]**", raw_id++), pic_w, pic_h,
                                              raw_data, IT_SPRITE, IF_SRGB);
}

void R_DiscardRawPic_MTL(void)
{
    if (raw_image) {
        R_UnregisterImage(raw_image - r_images);
        raw_image = NULL;
    }
}

// Handle used by R_DrawStretchRaw_MTL to reference the cinematic image.
qhandle_t mtl_raw_pic_handle(void)
{
    return raw_image ? (qhandle_t)(raw_image - r_images) : (qhandle_t)-1;
}
