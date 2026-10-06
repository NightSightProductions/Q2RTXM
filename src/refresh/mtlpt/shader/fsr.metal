/*
Copyright (C) 2021, AMD. FidelityFX Super Resolution 1.0 (MIT licensed).
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

// FidelityFX Super Resolution 1.0: EASU upscale and RCAS sharpen, a direct
// MSL translation of the FP32 paths in vkpt/fsr/ffx_fsr1.h as driven by
// fsr_easu.glsl / fsr_rcas.glsl. See fsr.c in the Vulkan backend for the
// overview.

#include <metal_stdlib>
#include "../mtlpt_shared.h"

using namespace metal;

// ffx_a.h approximations.
static inline float aprx_lo_rcp(float a) { return as_type<float>(0x7ef07ebbu - as_type<uint>(a)); }
static inline float aprx_lo_rsq(float a) { return as_type<float>(0x5f347d74u - (as_type<uint>(a) >> 1)); }
static inline float aprx_med_rcp(float a)
{
    float b = as_type<float>(0x7ef19fffu - as_type<uint>(a));
    return b * (-b * a + 2.0);
}

// Reversible tone mapper so HDR input can be filtered in [0,1].
static inline float3 srtm(float3 c)     { return c * (1.0 / (max3(c.r, c.g, c.b) + 1.0)); }
static inline float3 srtm_inv(float3 c) { return c * (1.0 / max(1.0 / 32768.0, 1.0 - max3(c.r, c.g, c.b))); }

// Remap a 64 thread group into an 8x8 swizzle, as ARmp8x8 does:
// x = bits 1..3, y = bits 4..5 above bit 0 (ABfiM(ABfe(a, 3, 3), a, 1)).
// Every thread must land on a distinct pixel of the 8x8 tile, otherwise
// some output pixels are written twice and others never.
static inline uint2 rmp8x8(uint a)
{
    return uint2(extract_bits(a, 1, 3), (extract_bits(a, 3, 3) & ~1u) | (a & 1u));
}

//
// EASU
//

static void easu_tap(thread float3 &ac, thread float &aw, float2 off, float2 dir, float2 len,
                     float lob, float clp, float3 c)
{
    float2 v;
    v.x = off.x * dir.x + off.y * dir.y;
    v.y = off.x * -dir.y + off.y * dir.x;
    v *= len;
    float d2 = min(v.x * v.x + v.y * v.y, clp);
    float wB = 2.0 / 5.0 * d2 - 1.0;
    float wA = lob * d2 - 1.0;
    wB *= wB;
    wA *= wA;
    wB = 25.0 / 16.0 * wB - (25.0 / 16.0 - 1.0);
    float w = wB * wA;
    ac += c * w;
    aw += w;
}

static void easu_set(thread float2 &dir, thread float &len, float2 pp,
                     bool biS, bool biT, bool biU, bool biV,
                     float lA, float lB, float lC, float lD, float lE)
{
    float w = 0.0;
    if (biS) w = (1.0 - pp.x) * (1.0 - pp.y);
    if (biT) w = pp.x * (1.0 - pp.y);
    if (biU) w = (1.0 - pp.x) * pp.y;
    if (biV) w = pp.x * pp.y;

    float dc = lD - lC;
    float cb = lC - lB;
    float lenX = max(abs(dc), abs(cb));
    lenX = aprx_lo_rcp(lenX);
    float dirX = lD - lB;
    dir.x += dirX * w;
    lenX = saturate(abs(dirX) * lenX);
    lenX *= lenX;
    len += lenX * w;

    float ec = lE - lC;
    float ca = lC - lA;
    float lenY = max(abs(ec), abs(ca));
    lenY = aprx_lo_rcp(lenY);
    float dirY = lE - lA;
    dir.y += dirY * w;
    lenY = saturate(abs(dirY) * lenY);
    lenY *= lenY;
    len += lenY * w;
}

struct EasuGather {
    float4 r, g, b;
};

static EasuGather easu_gather(texture2d<float> tex, sampler s, float2 p, constant MTLFsrUniforms &u)
{
    // Never fetch pixels outside the rendered region of the container texture.
    float2 inv = float2(u.container_inv_width, u.container_inv_height);
    p = clamp(p, 0.5 * inv, float2(1.0) - 1.5 * inv);

    EasuGather g;
    g.r = tex.gather(s, p, int2(0), component::x);
    g.g = tex.gather(s, p, int2(0), component::y);
    g.b = tex.gather(s, p, int2(0), component::z);

    if (u.is_hdr != 0u) {
        for (uint i = 0; i < 4; i++) {
            float3 c = srtm(float3(g.r[i], g.g[i], g.b[i]));
            g.r[i] = c.r; g.g[i] = c.g; g.b[i] = c.b;
        }
    }
    return g;
}

static float3 easu(texture2d<float> tex, sampler s, uint2 ip, constant MTLFsrUniforms &u)
{
    float4 con0 = float4(u.easu_con0), con1 = float4(u.easu_con1);
    float4 con2 = float4(u.easu_con2), con3 = float4(u.easu_con3);

    float2 pp = float2(ip) * con0.xy + con0.zw;
    float2 fp = floor(pp);
    pp -= fp;

    float2 p0 = fp * con1.xy + con1.zw;
    float2 p1 = p0 + con2.xy;
    float2 p2 = p0 + con2.zw;
    float2 p3 = p0 + con3.xy;

    EasuGather bczz = easu_gather(tex, s, p0, u);
    EasuGather ijfe = easu_gather(tex, s, p1, u);
    EasuGather klhg = easu_gather(tex, s, p2, u);
    EasuGather zzon = easu_gather(tex, s, p3, u);

    float4 bczzL = bczz.b * 0.5 + (bczz.r * 0.5 + bczz.g);
    float4 ijfeL = ijfe.b * 0.5 + (ijfe.r * 0.5 + ijfe.g);
    float4 klhgL = klhg.b * 0.5 + (klhg.r * 0.5 + klhg.g);
    float4 zzonL = zzon.b * 0.5 + (zzon.r * 0.5 + zzon.g);

    float bL = bczzL.x, cL = bczzL.y;
    float iL = ijfeL.x, jL = ijfeL.y, fL = ijfeL.z, eL = ijfeL.w;
    float kL = klhgL.x, lL = klhgL.y, hL = klhgL.z, gL = klhgL.w;
    float oL = zzonL.z, nL = zzonL.w;

    float2 dir = float2(0.0);
    float len = 0.0;
    easu_set(dir, len, pp, true,  false, false, false, bL, eL, fL, gL, jL);
    easu_set(dir, len, pp, false, true,  false, false, cL, fL, gL, hL, kL);
    easu_set(dir, len, pp, false, false, true,  false, fL, iL, jL, kL, nL);
    easu_set(dir, len, pp, false, false, false, true,  gL, jL, kL, lL, oL);

    float2 dir2 = dir * dir;
    float dirR = dir2.x + dir2.y;
    bool zro = dirR < 1.0 / 32768.0;
    dirR = aprx_lo_rsq(dirR);
    dirR = zro ? 1.0 : dirR;
    dir.x = zro ? 1.0 : dir.x;
    dir *= dirR;

    len = len * 0.5;
    len *= len;
    float stretch = (dir.x * dir.x + dir.y * dir.y) * aprx_lo_rcp(max(abs(dir.x), abs(dir.y)));
    float2 len2 = float2(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
    float lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
    float clp = aprx_lo_rcp(lob);

    float3 f_c = float3(ijfe.r.z, ijfe.g.z, ijfe.b.z);
    float3 g_c = float3(klhg.r.w, klhg.g.w, klhg.b.w);
    float3 j_c = float3(ijfe.r.y, ijfe.g.y, ijfe.b.y);
    float3 k_c = float3(klhg.r.x, klhg.g.x, klhg.b.x);
    float3 min4 = min(min(min(f_c, g_c), j_c), k_c);
    float3 max4 = max(max(max(f_c, g_c), j_c), k_c);

    float3 ac = float3(0.0);
    float aw = 0.0;
    easu_tap(ac, aw, float2( 0.0, -1.0) - pp, dir, len2, lob, clp, float3(bczz.r.x, bczz.g.x, bczz.b.x)); // b
    easu_tap(ac, aw, float2( 1.0, -1.0) - pp, dir, len2, lob, clp, float3(bczz.r.y, bczz.g.y, bczz.b.y)); // c
    easu_tap(ac, aw, float2(-1.0,  1.0) - pp, dir, len2, lob, clp, float3(ijfe.r.x, ijfe.g.x, ijfe.b.x)); // i
    easu_tap(ac, aw, float2( 0.0,  1.0) - pp, dir, len2, lob, clp, j_c);                                    // j
    easu_tap(ac, aw, float2( 0.0,  0.0) - pp, dir, len2, lob, clp, f_c);                                    // f
    easu_tap(ac, aw, float2(-1.0,  0.0) - pp, dir, len2, lob, clp, float3(ijfe.r.w, ijfe.g.w, ijfe.b.w)); // e
    easu_tap(ac, aw, float2( 1.0,  1.0) - pp, dir, len2, lob, clp, k_c);                                    // k
    easu_tap(ac, aw, float2( 2.0,  1.0) - pp, dir, len2, lob, clp, float3(klhg.r.y, klhg.g.y, klhg.b.y)); // l
    easu_tap(ac, aw, float2( 2.0,  0.0) - pp, dir, len2, lob, clp, float3(klhg.r.z, klhg.g.z, klhg.b.z)); // h
    easu_tap(ac, aw, float2( 1.0,  0.0) - pp, dir, len2, lob, clp, g_c);                                    // g
    easu_tap(ac, aw, float2( 1.0,  2.0) - pp, dir, len2, lob, clp, float3(zzon.r.z, zzon.g.z, zzon.b.z)); // o
    easu_tap(ac, aw, float2( 0.0,  2.0) - pp, dir, len2, lob, clp, float3(zzon.r.w, zzon.g.w, zzon.b.w)); // n

    return min(max4, max(min4, ac * (1.0 / aw)));
}

static void easu_store(texture2d<float, access::write> out, uint2 gxy, float3 color, constant MTLFsrUniforms &u)
{
    if (gxy.x >= u.output_width || gxy.y >= u.output_height)
        return;
    // Leave the colour tone mapped when RCAS follows; undo it for display.
    if (u.easu_to_display != 0u && u.is_hdr != 0u)
        color = srtm_inv(color);
    out.write(float4(color, 1.0), gxy);
}

kernel void fsr_easu(
    uint local_index                                 [[thread_index_in_threadgroup]],
    uint2 group_id                                   [[threadgroup_position_in_grid]],
    texture2d<float>                in_color         [[texture(0)]],
    texture2d<float, access::write> out_color        [[texture(1)]],
    constant MTLFsrUniforms &u                       [[buffer(0)]])
{
    constexpr sampler s(filter::linear, address::clamp_to_edge, coord::normalized);

    uint2 gxy = rmp8x8(local_index) + uint2(group_id.x << 4, group_id.y << 4);

    easu_store(out_color, gxy, easu(in_color, s, gxy, u), u);
    gxy.x += 8;
    easu_store(out_color, gxy, easu(in_color, s, gxy, u), u);
    gxy.y += 8;
    easu_store(out_color, gxy, easu(in_color, s, gxy, u), u);
    gxy.x -= 8;
    easu_store(out_color, gxy, easu(in_color, s, gxy, u), u);
}

//
// RCAS
//

#define FSR_RCAS_LIMIT (0.25 - (1.0 / 16.0))

static float3 rcas_load(texture2d<float> tex, int2 p, constant MTLFsrUniforms &u)
{
    if (u.rcas_after_easu != 0u)
        return tex.read(uint2(clamp(p, int2(0), int2(u.output_width - 1, u.output_height - 1)))).rgb;

    // Sharpening the TAA output directly (EASU disabled from the console).
    p = clamp(p, int2(0), int2(u.input_width - 1, u.input_height - 1));
    float3 c = tex.read(uint2(p)).rgb;
    return (u.is_hdr != 0u) ? srtm(c) : c;
}

static float3 rcas(texture2d<float> tex, uint2 ip, constant MTLFsrUniforms &u)
{
    int2 sp = int2(ip);
    float3 b = rcas_load(tex, sp + int2( 0, -1), u);
    float3 d = rcas_load(tex, sp + int2(-1,  0), u);
    float3 e = rcas_load(tex, sp, u);
    float3 f = rcas_load(tex, sp + int2( 1,  0), u);
    float3 h = rcas_load(tex, sp + int2( 0,  1), u);

    float bL = b.b * 0.5 + (b.r * 0.5 + b.g);
    float dL = d.b * 0.5 + (d.r * 0.5 + d.g);
    float eL = e.b * 0.5 + (e.r * 0.5 + e.g);
    float fL = f.b * 0.5 + (f.r * 0.5 + f.g);
    float hL = h.b * 0.5 + (h.r * 0.5 + h.g);

    float3 mn4 = min(min(b, d), min(f, h));
    float3 mx4 = max(max(b, d), max(f, h));
    float2 peakC = float2(1.0, -4.0);

    float3 hitMin = mn4 / (4.0 * mx4);
    float3 hitMax = (peakC.x - mx4) / (4.0 * mn4 + peakC.y);
    float3 lobeRGB = max(-hitMin, hitMax);
    float lobe = max(-FSR_RCAS_LIMIT, min(max3(lobeRGB.r, lobeRGB.g, lobeRGB.b), 0.0)) * u.rcas_sharpness;

    // Noise detection (FSR_RCAS_DENOISE is off in the Vulkan backend too).
    (void)eL;

    float rcpL = aprx_med_rcp(4.0 * lobe + 1.0);
    return (lobe * b + lobe * d + lobe * h + lobe * f + e) * rcpL;
}

kernel void fsr_rcas(
    uint local_index                                 [[thread_index_in_threadgroup]],
    uint2 group_id                                   [[threadgroup_position_in_grid]],
    texture2d<float>                in_color         [[texture(0)]],
    texture2d<float, access::write> out_color        [[texture(1)]],
    constant MTLFsrUniforms &u                       [[buffer(0)]])
{
    uint2 gxy = rmp8x8(local_index) + uint2(group_id.x << 4, group_id.y << 4);

    for (uint i = 0; i < 4; i++) {
        uint2 p = gxy + uint2((i == 1 || i == 2) ? 8 : 0, (i >= 2) ? 8 : 0);
        if (p.x < u.output_width && p.y < u.output_height) {
            float3 c = rcas(in_color, p, u);
            if (u.is_hdr != 0u)
                c = srtm_inv(c);
            out_color.write(float4(c, 1.0), p);
        }
    }
}
