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

// 2D user interface rendering. Draw calls issued by the engine during a frame
// are accumulated into a queue and submitted as a single instanced draw once
// the 3D image is done, matching the structure of the Vulkan backend.

#include "mtlpt_metal.h"

typedef struct {
    color_t colors[2];  // 0 - actual color, 1 - alternate color for text
    float   scale;
    float   alpha_scale;
} draw_static_t;

static draw_static_t draw = {
    .scale = 1.0f,
    .alpha_scale = 1.0f
};

static MTLStretchPic stretch_pic_queue[MTL_MAX_STRETCH_PICS];
static int           num_stretch_pics;

static clipRect_t    clip_rect;
static bool          clip_enable;

static id<MTLRenderPipelineState> pipeline;
static id<MTLBuffer>              pic_buffers[MTL_FRAMES_IN_FLIGHT];
static id<MTLBuffer>              uniform_buffers[MTL_FRAMES_IN_FLIGHT];
static id<MTLSamplerState>        sampler_linear;

static cvar_t *mtl_ui_hdr_nits;

// Declared in textures.m
qhandle_t mtl_raw_pic_handle(void);

bool mtl_draw_init(void)
{
    mtl_ui_hdr_nits = Cvar_Get("ui_hdr_nits", "300", 0);   // the HDR menu's "UI brightness"

    NSError *err = nil;
    id<MTLFunction> vs = mtl_new_function(@"stretch_pic_vertex");
    id<MTLFunction> fs = mtl_new_function(@"stretch_pic_fragment");
    if (!vs || !fs) {
        Com_EPrintf("Metal: stretch_pic shader functions missing from the library\n");
        return false;
    }

    MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.label = @"stretch pic";
    desc.vertexFunction = vs;
    desc.fragmentFunction = fs;
    desc.colorAttachments[0].pixelFormat = mtl.drawable_format;
    desc.colorAttachments[0].blendingEnabled = YES;
    desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

    pipeline = [mtl.device newRenderPipelineStateWithDescriptor:desc error:&err];
    [desc release];
    [vs release];
    [fs release];

    if (!pipeline)
        return mtl_log_error("newRenderPipelineStateWithDescriptor(stretch pic)", err);

    for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++) {
        pic_buffers[i] = [mtl.device newBufferWithLength:sizeof(stretch_pic_queue)
                                                 options:MTLResourceStorageModeShared];
        uniform_buffers[i] = [mtl.device newBufferWithLength:sizeof(MTLDrawUniforms)
                                                     options:MTLResourceStorageModeShared];
        if (!pic_buffers[i] || !uniform_buffers[i]) {
            Com_EPrintf("Metal: could not allocate UI buffers\n");
            return false;
        }
    }

    MTLSamplerDescriptor *sd = [[MTLSamplerDescriptor alloc] init];
    sd.minFilter = MTLSamplerMinMagFilterLinear;
    sd.magFilter = MTLSamplerMinMagFilterLinear;
    sd.mipFilter = MTLSamplerMipFilterNotMipmapped;
    sd.sAddressMode = MTLSamplerAddressModeRepeat;
    sd.tAddressMode = MTLSamplerAddressModeRepeat;
    sampler_linear = [mtl.device newSamplerStateWithDescriptor:sd];
    [sd release];

    draw.colors[0].u32 = U32_WHITE;
    draw.colors[1].u32 = U32_WHITE;
    return true;
}

void mtl_draw_shutdown(void)
{
    [pipeline release];
    pipeline = nil;

    for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++) {
        [pic_buffers[i] release];
        pic_buffers[i] = nil;
        [uniform_buffers[i] release];
        uniform_buffers[i] = nil;
    }

    [sampler_linear release];
    sampler_linear = nil;

    num_stretch_pics = 0;
}

void mtl_draw_clear_stretch_pics(void)
{
    num_stretch_pics = 0;
}

static void enqueue_stretch_pic(float x, float y, float w, float h,
                                float s1, float t1, float s2, float t2,
                                uint32_t color, qhandle_t tex_handle)
{
    if (draw.alpha_scale == 0.0f)
        return;

    if (num_stretch_pics == MTL_MAX_STRETCH_PICS) {
        Com_EPrintf("Metal: stretch pic queue full\n");
        return;
    }

    if (clip_enable) {
        if (x >= clip_rect.right || x + w <= clip_rect.left ||
            y >= clip_rect.bottom || y + h <= clip_rect.top)
            return;

        if (x < clip_rect.left) {
            float dw = clip_rect.left - x;
            s1 += dw / w * (s2 - s1);
            w -= dw;
            x = clip_rect.left;
            if (w <= 0) return;
        }
        if (x + w > clip_rect.right) {
            float dw = x + w - clip_rect.right;
            s2 -= dw / w * (s2 - s1);
            w -= dw;
            if (w <= 0) return;
        }
        if (y < clip_rect.top) {
            float dh = clip_rect.top - y;
            t1 += dh / h * (t2 - t1);
            h -= dh;
            y = clip_rect.top;
            if (h <= 0) return;
        }
        if (y + h > clip_rect.bottom) {
            float dh = y + h - clip_rect.bottom;
            t2 -= dh / h * (t2 - t1);
            h -= dh;
            if (h <= 0) return;
        }
    }

    float width = r_config.width * draw.scale;
    float height = r_config.height * draw.scale;
    if (width <= 0.0f || height <= 0.0f)
        return;

    MTLStretchPic *sp = &stretch_pic_queue[num_stretch_pics++];

    sp->x = 2.0f * x / width - 1.0f;
    sp->y = 2.0f * y / height - 1.0f;
    sp->w = 2.0f * w / width;
    sp->h = 2.0f * h / height;

    sp->s = s1;
    sp->t = t1;
    sp->w_s = s2 - s1;
    sp->h_t = t2 - t1;

    if (draw.alpha_scale < 1.0f) {
        float alpha = (color >> 24) & 0xff;
        alpha *= draw.alpha_scale;
        alpha = max(0.0f, min(255.0f, alpha));
        color = (color & 0xffffff) | ((uint32_t)alpha << 24);
    }

    sp->color = color;
    sp->tex_index = mtl_texture_index_for_handle(tex_handle);
    sp->pad0 = sp->pad1 = 0;
}

float mtl_ui_color_scale(void)
{
    // An scRGB luminance of 1.0 is 80 nits.
    return mtl.is_hdr ? mtl_ui_hdr_nits->value * 0.0125f : 1.0f;
}

void mtl_draw_submit(id<MTLCommandBuffer> cmd, id<MTLTexture> target)
{
    if (num_stretch_pics == 0 || !pipeline)
        return;

    uint32_t frame = mtl.frame_index;

    memcpy(pic_buffers[frame].contents, stretch_pic_queue,
           sizeof(MTLStretchPic) * num_stretch_pics);

    MTLDrawUniforms *ubo = (MTLDrawUniforms *)uniform_buffers[frame].contents;
    ubo->hdr_color_scale = mtl_ui_color_scale();
    ubo->hdr_saturation_scale = 1.0f;
    ubo->is_hdr = mtl.is_hdr ? 1 : 0;
    ubo->tonemapped = 0;
    ubo->uv_scale_x = 1.0f;
    ubo->uv_scale_y = 1.0f;

    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:pass];
    enc.label = @"ui";

    [enc setRenderPipelineState:pipeline];
    [enc setVertexBuffer:pic_buffers[frame] offset:0 atIndex:MTL_BUF_STRETCH_PICS];
    [enc setFragmentBuffer:uniform_buffers[frame] offset:0 atIndex:MTL_BUF_DRAW_UNIFORMS];
    [enc setFragmentBuffer:mtl_texture_argument_buffer() offset:0 atIndex:MTL_BUF_TEXTURE_TABLE];
    [enc setFragmentSamplerState:sampler_linear atIndex:0];
    mtl_textures_encode_use(enc);

    [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip
            vertexStart:0
            vertexCount:4
          instanceCount:num_stretch_pics];

    [enc endEncoding];

    num_stretch_pics = 0;
}

//
// Engine entry points
//

void R_SetClipRect_MTL(const clipRect_t *clip)
{
    if (!clip) {
        clip_enable = false;
        return;
    }

    clip_rect = *clip;
    clip_enable = true;
}

void R_ClearColor_MTL(void)
{
    draw.colors[0].u32 = U32_WHITE;
    draw.colors[1].u32 = U32_WHITE;
}

void R_SetAlpha_MTL(float alpha)
{
    draw.colors[0].u8[3] = alpha * 255;
    draw.colors[1].u8[3] = alpha * 255;
}

void R_SetAlphaScale_MTL(float alpha)
{
    draw.alpha_scale = alpha;
}

void R_SetColor_MTL(uint32_t color)
{
    draw.colors[0].u32 = color;
    draw.colors[1].u8[3] = draw.colors[0].u8[3];
}

void R_SetScale_MTL(float scale)
{
    draw.scale = scale;
}

void R_LightPoint_MTL(const vec3_t origin, vec3_t light)
{
    VectorSet(light, 1, 1, 1);
}

void R_DrawStretchPic_MTL(int x, int y, int w, int h, qhandle_t pic)
{
    enqueue_stretch_pic(x, y, w, h, 0.0f, 0.0f, 1.0f, 1.0f, draw.colors[0].u32, pic);
}

void R_DrawPic_MTL(int x, int y, qhandle_t pic)
{
    image_t *image = IMG_ForHandle(pic);
    R_DrawStretchPic_MTL(x, y, image->width, image->height, pic);
}

void R_DrawStretchRaw_MTL(int x, int y, int w, int h)
{
    qhandle_t raw = mtl_raw_pic_handle();
    if (raw < 0)
        return;

    R_DrawStretchPic_MTL(x, y, w, h, raw);
}

void R_DrawKeepAspectPic_MTL(int x, int y, int w, int h, qhandle_t pic)
{
    image_t *image = IMG_ForHandle(pic);

    if (image->flags & IF_SCRAP) {
        R_DrawStretchPic_MTL(x, y, w, h, pic);
        return;
    }

    float scale_w = w;
    float scale_h = h * image->aspect;
    float scale = max(scale_w, scale_h);

    float s = (1.0f - scale_w / scale) * 0.5f;
    float t = (1.0f - scale_h / scale) * 0.5f;

    enqueue_stretch_pic(x, y, w, h, s, t, 1.0f - s, 1.0f - t, draw.colors[0].u32, pic);
}

#define DIV64 (1.0f / 64.0f)

void R_TileClear_MTL(int x, int y, int w, int h, qhandle_t pic)
{
    enqueue_stretch_pic(x, y, w, h,
                        x * DIV64, y * DIV64, (x + w) * DIV64, (y + h) * DIV64,
                        U32_WHITE, pic);
}

void R_DrawFill8_MTL(int x, int y, int w, int h, int c)
{
    if (!w || !h)
        return;

    enqueue_stretch_pic(x, y, w, h, 0.0f, 0.0f, 1.0f, 1.0f,
                        d_8to24table[c & 0xff], (qhandle_t)MTL_TEXNUM_WHITE);
}

void R_DrawFill32_MTL(int x, int y, int w, int h, uint32_t color)
{
    if (!w || !h)
        return;

    enqueue_stretch_pic(x, y, w, h, 0.0f, 0.0f, 1.0f, 1.0f,
                        color, (qhandle_t)MTL_TEXNUM_WHITE);
}

static void draw_char(int x, int y, int flags, int c, qhandle_t font)
{
    if ((c & 127) == 32)
        return;

    if (flags & UI_ALTCOLOR)
        c |= 0x80;
    if (flags & UI_XORCOLOR)
        c ^= 0x80;

    float s = (c & 15) * 0.0625f;
    float t = (c >> 4) * 0.0625f;

    const float eps = 1e-5f; // keeps neighbouring glyphs from bleeding in

    enqueue_stretch_pic(x, y, CHAR_WIDTH, CHAR_HEIGHT,
                        s + eps, t + eps, s + 0.0625f - eps, t + 0.0625f - eps,
                        draw.colors[c >> 7].u32, font);
}

void R_DrawChar_MTL(int x, int y, int flags, int c, qhandle_t font)
{
    draw_char(x, y, flags, c & 255, font);
}

int R_DrawString_MTL(int x, int y, int flags, size_t maxlen, const char *s, qhandle_t font)
{
    while (maxlen-- && *s) {
        byte c = *s++;
        draw_char(x, y, flags, c, font);
        x += CHAR_WIDTH;
    }

    return x;
}
