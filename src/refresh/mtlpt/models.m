/*
Copyright (C) 1997-2001 Id Software, Inc.
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

// Model loading for the Metal backend. Geometry is decoded into the same CPU
// side layout the Vulkan backend uses; each skin resolves to both a plain image
// and the PBR material record the path tracer shades with.

#include "mtlpt_metal.h"
#include "../vkpt/material.h"

#include "common/files.h"
#include "common/math.h"
#include "system/hunk.h"
#include "format/md2.h"
#if USE_MD3
#include "format/md3.h"
#endif
#include <errno.h>

#define CHECK(x) if (!(x)) { ret = Q_ERR(ENOMEM); goto fail; }

static image_t *find_skin(const char *name)
{
    char normalized[MAX_QPATH];

    Q_strlcpy(normalized, name, sizeof(normalized));
    FS_NormalizePath(normalized);

    if (!normalized[0])
        return R_NOTEXTURE;

    return IMG_ForHandle(R_RegisterSkin(normalized));
}

static pbr_material_t *find_skin_material(const char *name)
{
    char normalized[MAX_QPATH];

    Q_strlcpy(normalized, name, sizeof(normalized));
    FS_NormalizePath(normalized);

    if (!normalized[0])
        return NULL;

    return MAT_Find(normalized, IT_SKIN, IF_NONE);
}

//
// MD2
//

// Port of vkpt's compute_missing_model_tangents() for the [vertex][frame]
// layout used here. Tangents follow the texture U direction so normal maps
// on models line up; handedness records mirrored UV layouts.
static int compute_missing_model_tangents(model_t *model)
{
    int ret = Q_ERR_SUCCESS;
    int nf = max(model->numframes, 1);

    for (int mesh_idx = 0; mesh_idx < model->nummeshes; mesh_idx++) {
        maliasmesh_t *mesh = &model->meshes[mesh_idx];
        if (mesh->tangents || !mesh->positions || !mesh->tex_coords)
            continue;

        size_t count = (size_t)mesh->numverts * nf;
        CHECK(mesh->tangents = MOD_Malloc(count * sizeof(vec3_t)));
        memset(mesh->tangents, 0, count * sizeof(vec3_t));

        int handedness = 0;
        for (int frame = 0; frame < nf; frame++) {
            for (int tri = 0; tri < mesh->numtris; tri++) {
                int iA = mesh->indices[tri * 3 + 0] * nf + frame;
                int iB = mesh->indices[tri * 3 + 1] * nf + frame;
                int iC = mesh->indices[tri * 3 + 2] * nf + frame;

                vec3_t dP0, dP1;
                VectorSubtract(mesh->positions[iB], mesh->positions[iA], dP0);
                VectorSubtract(mesh->positions[iC], mesh->positions[iA], dP1);

                vec2_t dt0, dt1;
                Vector2Subtract(mesh->tex_coords[iB], mesh->tex_coords[iA], dt0);
                Vector2Subtract(mesh->tex_coords[iC], mesh->tex_coords[iA], dt1);

                float inv_r = dt0[0] * dt1[1] - dt1[0] * dt0[1];
                if (inv_r == 0.0f)
                    continue;
                float r = 1.0f / inv_r;

                vec3_t tangent = {
                    (dt1[1] * dP0[0] - dt0[1] * dP1[0]) * r,
                    (dt1[1] * dP0[1] - dt0[1] * dP1[1]) * r,
                    (dt1[1] * dP0[2] - dt0[1] * dP1[2]) * r };
                VectorNormalize(tangent);

                VectorAdd(mesh->tangents[iA], tangent, mesh->tangents[iA]);
                VectorAdd(mesh->tangents[iB], tangent, mesh->tangents[iB]);
                VectorAdd(mesh->tangents[iC], tangent, mesh->tangents[iC]);

                if (handedness == 0) {
                    vec3_t bitangent = {
                        (dt0[0] * dP1[0] - dt1[0] * dP0[0]) * r,
                        (dt0[0] * dP1[1] - dt1[0] * dP0[1]) * r,
                        (dt0[0] * dP1[2] - dt1[0] * dP0[2]) * r };
                    VectorNormalize(bitangent);
                    vec3_t cross;
                    CrossProduct(mesh->normals[iA], tangent, cross);
                    float dot = DotProduct(cross, bitangent);
                    if (dot < 0.0f)
                        handedness = -1;
                    else if (dot > 0.0f)
                        handedness = 1;
                }
            }
        }

        for (size_t v = 0; v < count; v++)
            VectorNormalize(mesh->tangents[v]);

        mesh->handedness = (handedness < 0);
    }

fail:
    return ret;
}

int MOD_LoadMD2_MTL(model_t *model, const void *rawdata, size_t length, const char *mod_name)
{
    dmd2header_t   header;
    dmd2frame_t   *src_frame;
    dmd2trivertx_t *src_vert;
    dmd2triangle_t *src_tri;
    dmd2stvert_t  *src_tc;
    char          *src_skin;
    maliasframe_t *dst_frame;
    maliasmesh_t  *dst_mesh;
    uint16_t       remap[TESS_MAX_INDICES];
    uint16_t       vertIndices[TESS_MAX_INDICES];
    uint16_t       tcIndices[TESS_MAX_INDICES];
    uint16_t       finalIndices[TESS_MAX_INDICES];
    int            numverts, numindices;
    char           skinname[MAX_QPATH];
    vec_t          scale_s, scale_t;
    vec3_t         mins, maxs;
    int            ret;
    const char    *err;

    if (length < sizeof(header))
        return Q_ERR_FILE_TOO_SMALL;

    LittleBlock(&header, rawdata, sizeof(header));

    if (header.ident != MD2_IDENT)
        return Q_ERR_UNKNOWN_FORMAT;
    if (header.version != MD2_VERSION)
        return Q_ERR_UNKNOWN_FORMAT;

    if (header.num_tris < 1 || header.num_st < 3 || header.num_xyz < 3 || header.num_frames < 1) {
        model->type = MOD_EMPTY;
        return Q_ERR_SUCCESS;
    }

    err = MOD_ValidateMD2(&header, length);
    if (err) {
        Com_SetLastError(err);
        return Q_ERR_INVALID_FORMAT;
    }

    if (header.num_tris * 3 > TESS_MAX_INDICES) {
        Com_SetLastError("too many triangles");
        return Q_ERR_INVALID_FORMAT;
    }

    // Collect the triangle indices, dropping the malformed triangles some
    // community models contain.
    numindices = 0;
    src_tri = (dmd2triangle_t *)((byte *)rawdata + header.ofs_tris);
    for (int i = 0; i < header.num_tris; i++) {
        int good = 1;
        for (int j = 0; j < 3; j++) {
            uint16_t idx_xyz = LittleShort(src_tri->index_xyz[j]);
            uint16_t idx_st = LittleShort(src_tri->index_st[j]);

            if (idx_xyz >= header.num_xyz || idx_st >= header.num_st) {
                good = 0;
                break;
            }

            vertIndices[numindices + j] = idx_xyz;
            tcIndices[numindices + j] = idx_st;
        }
        if (good)
            numindices += 3;
        src_tri++;
    }

    if (numindices < 3) {
        Com_SetLastError("too few valid indices");
        return Q_ERR_INVALID_FORMAT;
    }

    // Some models (players/w_*.md2) store a single normal index for every
    // vertex, in which case the normals have to be regenerated and vertices
    // must not be merged.
    bool all_normals_same = true;
    int same_normal = -1;

    src_frame = (dmd2frame_t *)((byte *)rawdata + header.ofs_frames);
    for (int i = 0; i < numindices; i++) {
        int normal = src_frame->verts[vertIndices[i]].lightnormalindex;
        if (same_normal < 0)
            same_normal = normal;
        else if (normal != same_normal)
            all_normals_same = false;
    }

    for (int i = 0; i < numindices; i++)
        remap[i] = 0xFFFF;

    numverts = 0;
    src_tc = (dmd2stvert_t *)((byte *)rawdata + header.ofs_st);
    for (int i = 0; i < numindices; i++) {
        if (remap[i] != 0xFFFF)
            continue;

        if (!all_normals_same) {
            for (int j = i + 1; j < numindices; j++) {
                if (vertIndices[i] == vertIndices[j] &&
                    src_tc[tcIndices[i]].s == src_tc[tcIndices[j]].s &&
                    src_tc[tcIndices[i]].t == src_tc[tcIndices[j]].t) {
                    remap[j] = i;
                    finalIndices[j] = numverts;
                }
            }
        }

        remap[i] = i;
        finalIndices[i] = numverts++;
    }

    Hunk_Begin(&model->hunk, 50u << 20);
    model->type = MOD_ALIAS;
    model->nummeshes = 1;
    model->numframes = header.num_frames;
    CHECK(model->meshes = MOD_Malloc(sizeof(model->meshes[0])));
    CHECK(model->frames = MOD_Malloc(header.num_frames * sizeof(model->frames[0])));

    dst_mesh = model->meshes;
    memset(dst_mesh, 0, sizeof(*dst_mesh));
    dst_mesh->numtris = numindices / 3;
    dst_mesh->numindices = numindices;
    dst_mesh->numverts = numverts;
    dst_mesh->numskins = header.num_skins;

    CHECK(dst_mesh->positions  = MOD_Malloc(numverts * header.num_frames * sizeof(dst_mesh->positions[0])));
    CHECK(dst_mesh->normals    = MOD_Malloc(numverts * header.num_frames * sizeof(dst_mesh->normals[0])));
    CHECK(dst_mesh->tex_coords = MOD_Malloc(numverts * header.num_frames * sizeof(dst_mesh->tex_coords[0])));
    CHECK(dst_mesh->indices    = MOD_Malloc(numindices * sizeof(dst_mesh->indices[0])));
    CHECK(dst_mesh->skins      = MOD_Malloc(max(header.num_skins, 1) * sizeof(dst_mesh->skins[0])));
    CHECK(dst_mesh->materials  = MOD_Malloc(max(header.num_skins, 1) * sizeof(dst_mesh->materials[0])));

    for (int i = 0; i < numindices; i++)
        dst_mesh->indices[i] = finalIndices[i];

    src_skin = (char *)rawdata + header.ofs_skins;
    for (int i = 0; i < header.num_skins; i++) {
        if (!Q_memccpy(skinname, src_skin, 0, sizeof(skinname))) {
            ret = Q_ERR_STRING_TRUNCATED;
            goto fail;
        }

        dst_mesh->skins[i] = find_skin(skinname);
        dst_mesh->materials[i] = find_skin_material(skinname);
        src_skin += MD2_MAX_SKINNAME;
    }

    scale_s = 1.0f / header.skinwidth;
    scale_t = 1.0f / header.skinheight;

    src_frame = (dmd2frame_t *)((byte *)rawdata + header.ofs_frames);
    dst_frame = model->frames;
    for (int j = 0; j < header.num_frames; j++) {
        LittleVector(src_frame->scale, dst_frame->scale);
        LittleVector(src_frame->translate, dst_frame->translate);

        ClearBounds(mins, maxs);

        for (int i = 0; i < numindices; i++) {
            if (remap[i] != i)
                continue;

            src_vert = &src_frame->verts[vertIndices[i]];
            vec3_t *pos = &dst_mesh->positions[header.num_frames * finalIndices[i] + j];

            (*pos)[0] = src_vert->v[0] * dst_frame->scale[0] + dst_frame->translate[0];
            (*pos)[1] = src_vert->v[1] * dst_frame->scale[1] + dst_frame->translate[1];
            (*pos)[2] = src_vert->v[2] * dst_frame->scale[2] + dst_frame->translate[2];

            vec3_t *normal = &dst_mesh->normals[header.num_frames * finalIndices[i] + j];
            int k = src_vert->lightnormalindex;
            if (k < NUMVERTEXNORMALS && !all_normals_same)
                VectorCopy(bytedirs[k], *normal);
            else
                VectorClear(*normal);

            vec2_t *tc = &dst_mesh->tex_coords[header.num_frames * finalIndices[i] + j];
            (*tc)[0] = (int16_t)LittleShort(src_tc[tcIndices[i]].s) * scale_s;
            (*tc)[1] = (int16_t)LittleShort(src_tc[tcIndices[i]].t) * scale_t;

            AddPointToBounds(*pos, mins, maxs);
        }

        if (all_normals_same) {
            // Rebuild per-vertex normals by averaging the face normals.
            for (int i = 0; i + 2 < numindices; i += 3) {
                uint16_t i0 = finalIndices[i + 0];
                uint16_t i1 = finalIndices[i + 1];
                uint16_t i2 = finalIndices[i + 2];

                const float *p0 = dst_mesh->positions[header.num_frames * i0 + j];
                const float *p1 = dst_mesh->positions[header.num_frames * i1 + j];
                const float *p2 = dst_mesh->positions[header.num_frames * i2 + j];

                vec3_t e0, e1, face_normal;
                VectorSubtract(p1, p0, e0);
                VectorSubtract(p2, p0, e1);
                CrossProduct(e1, e0, face_normal);
                VectorNormalize(face_normal);

                VectorAdd(dst_mesh->normals[header.num_frames * i0 + j], face_normal,
                          dst_mesh->normals[header.num_frames * i0 + j]);
                VectorAdd(dst_mesh->normals[header.num_frames * i1 + j], face_normal,
                          dst_mesh->normals[header.num_frames * i1 + j]);
                VectorAdd(dst_mesh->normals[header.num_frames * i2 + j], face_normal,
                          dst_mesh->normals[header.num_frames * i2 + j]);
            }

            for (int i = 0; i < numverts; i++)
                VectorNormalize(dst_mesh->normals[header.num_frames * i + j]);
        }

        VectorCopy(mins, dst_frame->bounds[0]);
        VectorCopy(maxs, dst_frame->bounds[1]);
        dst_frame->radius = RadiusFromBounds(mins, maxs);

        src_frame = (dmd2frame_t *)((byte *)src_frame + header.framesize);
        dst_frame++;
    }

    ret = compute_missing_model_tangents(model);
    if (ret)
        goto fail;

    Hunk_End(&model->hunk);
    return Q_ERR_SUCCESS;

fail:
    Hunk_Free(&model->hunk);
    return ret;
}

//
// MD3
//

#if USE_MD3
static inline float md3_tab_sin(unsigned x)
{
    return sinf((float)(x & 255) * (float)(2.0 * M_PI / 255.0));
}

static int load_md3_mesh(model_t *model, maliasmesh_t *mesh,
                         const byte *rawdata, size_t length, size_t *offset_p)
{
    dmd3mesh_t   header;
    const byte  *src_vert, *src_tc, *src_idx;
    char         skinname[MAX_QPATH];
    const char  *err;
    int          ret;

    if (length < sizeof(header))
        return Q_ERR_FILE_TOO_SMALL;

    LittleBlock(&header, rawdata, sizeof(header));

    err = MOD_ValidateMD3Mesh(model, &header, length);
    if (err) {
        Com_SetLastError(err);
        return Q_ERR_INVALID_FORMAT;
    }

    memset(mesh, 0, sizeof(*mesh));
    mesh->numtris = header.num_tris;
    mesh->numindices = header.num_tris * 3;
    mesh->numverts = header.num_verts;
    mesh->numskins = header.num_skins;

    CHECK(mesh->positions  = MOD_Malloc(sizeof(mesh->positions[0]) * header.num_verts * model->numframes));
    CHECK(mesh->normals    = MOD_Malloc(sizeof(mesh->normals[0]) * header.num_verts * model->numframes));
    CHECK(mesh->tex_coords = MOD_Malloc(sizeof(mesh->tex_coords[0]) * header.num_verts * model->numframes));
    CHECK(mesh->indices    = MOD_Malloc(sizeof(mesh->indices[0]) * header.num_tris * 3));
    CHECK(mesh->skins      = MOD_Malloc(sizeof(mesh->skins[0]) * max(header.num_skins, 1)));
    CHECK(mesh->materials  = MOD_Malloc(sizeof(mesh->materials[0]) * max(header.num_skins, 1)));

    // Skins
    const byte *src_skin = rawdata + header.ofs_skins;
    for (int i = 0; i < header.num_skins; i++) {
        if (!Q_memccpy(skinname, src_skin, 0, sizeof(skinname)))
            return Q_ERR_STRING_TRUNCATED;

        mesh->skins[i] = find_skin(skinname);
        mesh->materials[i] = find_skin_material(skinname);
        src_skin += MD3_MAX_PATH;
    }

    // Texture coordinates are shared across frames.
    src_tc = rawdata + header.ofs_tcs;
    for (int i = 0; i < header.num_verts; i++) {
        for (int j = 0; j < model->numframes; j++) {
            mesh->tex_coords[i * model->numframes + j][0] = LittleFloat(((const float *)src_tc)[0]);
            mesh->tex_coords[i * model->numframes + j][1] = LittleFloat(((const float *)src_tc)[1]);
        }
        src_tc += sizeof(dmd3coord_t);
    }

    // Vertices
    src_vert = rawdata + header.ofs_verts;
    for (int j = 0; j < model->numframes; j++) {
        for (int i = 0; i < header.num_verts; i++) {
            const dmd3vertex_t *v = (const dmd3vertex_t *)src_vert;

            vec3_t *pos = &mesh->positions[i * model->numframes + j];
            (*pos)[0] = (int16_t)LittleShort(v->point[0]) * MD3_XYZ_SCALE;
            (*pos)[1] = (int16_t)LittleShort(v->point[1]) * MD3_XYZ_SCALE;
            (*pos)[2] = (int16_t)LittleShort(v->point[2]) * MD3_XYZ_SCALE;

            // vkpt's decode: byte angles through its sine table, where
            // TAB_SIN(x) = sin(x * 2pi / 255) and TAB_COS(x) = TAB_SIN(x + 64).
            unsigned lat = v->norm[0];
            unsigned lng = v->norm[1];
            vec3_t *normal = &mesh->normals[i * model->numframes + j];
            (*normal)[0] = md3_tab_sin(lat) * md3_tab_sin(lng + 64);
            (*normal)[1] = md3_tab_sin(lat) * md3_tab_sin(lng);
            (*normal)[2] = md3_tab_sin(lat + 64);
            VectorNormalize(*normal);

            src_vert += sizeof(dmd3vertex_t);
        }
    }

    // Indices
    src_idx = rawdata + header.ofs_indexes;
    for (int i = 0; i < header.num_tris * 3; i++) {
        mesh->indices[i] = (int)LittleLong(((const uint32_t *)src_idx)[0]);
        src_idx += sizeof(uint32_t);
    }

    *offset_p = header.meshsize;
    return Q_ERR_SUCCESS;

fail:
    return ret;
}

int MOD_LoadMD3_MTL(model_t *model, const void *rawdata, size_t length, const char *mod_name)
{
    dmd3header_t   header;
    const byte    *src_mesh;
    maliasframe_t *dst_frame;
    const byte    *src_frame;
    size_t         offset, remaining;
    const char    *err;
    int            ret;

    if (length < sizeof(header))
        return Q_ERR_FILE_TOO_SMALL;

    LittleBlock(&header, rawdata, sizeof(header));

    if (header.ident != MD3_IDENT)
        return Q_ERR_UNKNOWN_FORMAT;
    if (header.version != MD3_VERSION)
        return Q_ERR_UNKNOWN_FORMAT;

    err = MOD_ValidateMD3(&header, length);
    if (err) {
        Com_SetLastError(err);
        return Q_ERR_INVALID_FORMAT;
    }

    Hunk_Begin(&model->hunk, 0x4000000);
    model->type = MOD_ALIAS;
    model->numframes = header.num_frames;
    model->nummeshes = header.num_meshes;

    CHECK(model->meshes = MOD_Malloc(sizeof(model->meshes[0]) * header.num_meshes));
    CHECK(model->frames = MOD_Malloc(sizeof(model->frames[0]) * header.num_frames));

    src_frame = (const byte *)rawdata + header.ofs_frames;
    dst_frame = model->frames;
    for (int i = 0; i < header.num_frames; i++) {
        const dmd3frame_t *f = (const dmd3frame_t *)src_frame;

        LittleVector(f->translate, dst_frame->translate);
        VectorSet(dst_frame->scale, MD3_XYZ_SCALE, MD3_XYZ_SCALE, MD3_XYZ_SCALE);
        LittleVector(f->mins, dst_frame->bounds[0]);
        LittleVector(f->maxs, dst_frame->bounds[1]);
        dst_frame->radius = LittleFloat(f->radius);

        src_frame += sizeof(dmd3frame_t);
        dst_frame++;
    }

    src_mesh = (const byte *)rawdata + header.ofs_meshes;
    remaining = length - header.ofs_meshes;
    for (int i = 0; i < header.num_meshes; i++) {
        ret = load_md3_mesh(model, &model->meshes[i], src_mesh, remaining, &offset);
        if (ret)
            goto fail;

        src_mesh += offset;
        remaining -= offset;
    }

    ret = compute_missing_model_tangents(model);
    if (ret)
        goto fail;

    Hunk_End(&model->hunk);
    return Q_ERR_SUCCESS;

fail:
    Hunk_Free(&model->hunk);
    return ret;
}
#endif // USE_MD3

//
// IQM
//

int MOD_LoadIQM_MTL(model_t *model, const void *rawdata, size_t length, const char *mod_name)
{
    int ret;

    Hunk_Begin(&model->hunk, 0x4000000);
    model->type = MOD_ALIAS;

    int res = MOD_LoadIQM_Base(model, rawdata, length, mod_name);
    if (res != Q_ERR_SUCCESS) {
        Hunk_Free(&model->hunk);
        return res;
    }

    CHECK(model->meshes = MOD_Malloc(sizeof(model->meshes[0]) * model->iqmData->num_meshes));
    model->nummeshes = (int)model->iqmData->num_meshes;
    model->numframes = 1; // baked frames, the uploader makes one copy

    for (unsigned model_idx = 0; model_idx < model->iqmData->num_meshes; model_idx++) {
        iqm_mesh_t *iqm_mesh = &model->iqmData->meshes[model_idx];
        maliasmesh_t *mesh = &model->meshes[model_idx];

        memset(mesh, 0, sizeof(*mesh));
        mesh->indices = (int *)(model->iqmData->indices + iqm_mesh->first_triangle * 3);
        mesh->positions = (vec3_t *)(model->iqmData->positions + iqm_mesh->first_vertex * 3);
        mesh->normals = (vec3_t *)(model->iqmData->normals + iqm_mesh->first_vertex * 3);
        mesh->tex_coords = (vec2_t *)(model->iqmData->texcoords + iqm_mesh->first_vertex * 2);
        mesh->tangents = (vec3_t *)(model->iqmData->tangents + iqm_mesh->first_vertex * 3);
        mesh->numindices = (int)(iqm_mesh->num_triangles * 3);
        mesh->numverts = (int)iqm_mesh->num_vertexes;
        mesh->numtris = (int)iqm_mesh->num_triangles;

        if (model->iqmData->blend_indices) {
            mesh->blend_indices = (uint32_t *)(model->iqmData->blend_indices + iqm_mesh->first_vertex * 4);
            mesh->blend_weights = (uint32_t *)(model->iqmData->blend_weights + iqm_mesh->first_vertex * 4);
        }

        // The IQM indices are relative to the whole model, rebase them.
        for (int i = 0; i < mesh->numindices; i++)
            mesh->indices[i] -= iqm_mesh->first_vertex;

        CHECK(mesh->skins = MOD_Malloc(sizeof(mesh->skins[0])));
        CHECK(mesh->materials = MOD_Malloc(sizeof(mesh->materials[0])));
        mesh->numskins = 1;
        mesh->skins[0] = find_skin(iqm_mesh->material);
        mesh->materials[0] = find_skin_material(iqm_mesh->material);
    }

    Hunk_End(&model->hunk);
    return Q_ERR_SUCCESS;

fail:
    Hunk_Free(&model->hunk);
    return ret;
}

void MOD_Reference_MTL(model_t *model)
{
    switch (model->type) {
    case MOD_ALIAS:
        for (int mesh_idx = 0; mesh_idx < model->nummeshes; mesh_idx++) {
            maliasmesh_t *mesh = &model->meshes[mesh_idx];
            for (int skin_idx = 0; skin_idx < mesh->numskins; skin_idx++) {
                if (mesh->skins[skin_idx])
                    mesh->skins[skin_idx]->registration_sequence = registration_sequence;
                if (mesh->materials[skin_idx])
                    MAT_UpdateRegistration(mesh->materials[skin_idx]);
            }
        }
        break;

    case MOD_SPRITE:
        for (int frame_idx = 0; frame_idx < model->numframes; frame_idx++) {
            if (model->spriteframes[frame_idx].image)
                model->spriteframes[frame_idx].image->registration_sequence = registration_sequence;
        }
        break;

    case MOD_EMPTY:
        break;

    default:
        Com_Error(ERR_FATAL, "%s: bad model type", __func__);
    }

    model->registration_sequence = registration_sequence;
}
