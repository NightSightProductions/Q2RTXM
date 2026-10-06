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

// Debug line drawing, the Metal counterpart of vkpt/debug.c. The shared
// line list lives in refresh/debug.c; this converts it to view space and
// draws it over the resolved frame.

#include "mtlpt_metal.h"
#include "refresh/debug.h"

typedef struct {
    float    view_pos[3];
    uint32_t color;
} mtl_debug_vertex_t;

static id<MTLRenderPipelineState> pipeline_lines;
static id<MTLBuffer>              vertex_buffers[MTL_FRAMES_IN_FLIGHT];
static cvar_t                    *cvar_pt_debug_linewidth;
static cvar_t                    *cvar_pt_debug_distfrac;

// Camera of the frame being drawn, captured from the refdef.
static vec3_t view_origin, view_forward, view_right, view_up;
static float  tan_half_fov_x, tan_half_fov_y;
static bool   have_view;

bool mtl_debug_init(void)
{
    cvar_pt_debug_linewidth = Cvar_Get("pt_debug_linewidth", "2", 0);
    cvar_pt_debug_distfrac = Cvar_Get("pt_debug_distfrac", "0.004", 0);

    NSError *err = nil;
    id<MTLFunction> vs = mtl_new_function(@"debug_line_vertex");
    id<MTLFunction> fs = mtl_new_function(@"debug_line_fragment");
    if (!vs || !fs) {
        Com_EPrintf("Metal: debug_line shader functions missing from the library\n");
        return false;
    }

    MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.label = @"debug lines";
    desc.vertexFunction = vs;
    desc.fragmentFunction = fs;
    desc.colorAttachments[0].pixelFormat = mtl.drawable_format;
    desc.colorAttachments[0].blendingEnabled = YES;
    desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorZero;

    pipeline_lines = [mtl.device newRenderPipelineStateWithDescriptor:desc error:&err];
    [desc release];
    [vs release];
    [fs release];
    if (!pipeline_lines)
        return mtl_log_error("newRenderPipelineStateWithDescriptor(debug lines)", err);

    for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++) {
        vertex_buffers[i] = [mtl.device newBufferWithLength:sizeof(mtl_debug_vertex_t) * MAX_DEBUG_VERTICES
                                                    options:MTLResourceStorageModeShared];
        if (!vertex_buffers[i])
            return false;
        vertex_buffers[i].label = @"debug line vertices";
    }
    return true;
}

void mtl_debug_shutdown(void)
{
    [pipeline_lines release];
    pipeline_lines = nil;
    for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++) {
        [vertex_buffers[i] release];
        vertex_buffers[i] = nil;
    }
}

void mtl_debug_set_view(const refdef_t *fd)
{
    VectorCopy(fd->vieworg, view_origin);
    AngleVectors(fd->viewangles, view_forward, view_right, view_up);
    tan_half_fov_x = tanf(DEG2RAD(fd->fov_x) * 0.5f);
    tan_half_fov_y = tanf(DEG2RAD(fd->fov_y) * 0.5f);
    have_view = true;
}

static void to_view(const vec3_t world, float out[3])
{
    vec3_t d;
    VectorSubtract(world, view_origin, d);
    out[0] = DotProduct(d, view_right);
    out[1] = DotProduct(d, view_up);
    out[2] = DotProduct(d, view_forward);
}

// Clips the segment against the near plane; false when fully behind it.
static bool clip_near(float a[3], float b[3])
{
    const float clip_z = 0.1f;
    bool a_vis = a[2] >= clip_z;
    bool b_vis = b[2] >= clip_z;
    if (a_vis == b_vis)
        return a_vis;

    float t = (a[2] - clip_z) / (a[2] - b[2]);
    float p[3] = { a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, clip_z };
    if (!a_vis)
        memcpy(a, p, sizeof(p));
    else
        memcpy(b, p, sizeof(p));
    return true;
}

bool mtl_debug_have_lines(void)
{
    return !LIST_EMPTY(&r_debug_lines_active);
}

void mtl_debug_draw(id<MTLRenderCommandEncoder> enc, id<MTLTexture> normal_depth,
                    int render_width, int render_height, float color_scale)
{
    if (!pipeline_lines || !have_view || !normal_depth || LIST_EMPTY(&r_debug_lines_active))
        return;

    mtl_debug_vertex_t *verts = (mtl_debug_vertex_t *)vertex_buffers[mtl.frame_index].contents;
    uint32_t numverts = 0;

    r_debug_line_t *l, *next;
    LIST_FOR_EACH_SAFE(r_debug_line_t, l, next, &r_debug_lines_active, entry) {
        if (numverts + 2 > MAX_DEBUG_VERTICES)
            break;

        float a[3], b[3];
        to_view(l->start, a);
        to_view(l->end, b);
        if (!clip_near(a, b))
            continue;
        if (!l->depth_test) {
            a[2] = -a[2];
            b[2] = -b[2];
        }
        memcpy(verts[numverts].view_pos, a, sizeof(a));
        verts[numverts].color = l->color.u32;
        memcpy(verts[numverts + 1].view_pos, b, sizeof(b));
        verts[numverts + 1].color = l->color.u32;
        numverts += 2;

        if (!l->time) {
            List_Remove(&l->entry);
            List_Insert(&r_debug_lines_free, &l->entry);
        }
    }

    if (!numverts)
        return;

    MTLDebugLineUniforms u = {
        .tan_half_fov_x = tan_half_fov_x,
        .tan_half_fov_y = tan_half_fov_y,
        .depth_scale_x = (float)render_width / (float)max(1, mtl.width),
        .depth_scale_y = (float)render_height / (float)max(1, mtl.height),
        .depth_width = (uint32_t)max(1, render_width),
        .depth_height = (uint32_t)max(1, render_height),
        .color_scale = color_scale,
    };

    [enc setRenderPipelineState:pipeline_lines];
    [enc setVertexBuffer:vertex_buffers[mtl.frame_index] offset:0 atIndex:0];
    [enc setVertexBytes:&u length:sizeof(u) atIndex:1];
    [enc setFragmentBytes:&u length:sizeof(u) atIndex:1];
    [enc setFragmentTexture:normal_depth atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeLine vertexStart:0 vertexCount:numverts];
}
