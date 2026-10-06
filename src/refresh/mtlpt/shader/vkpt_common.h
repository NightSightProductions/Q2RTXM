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

// Shared shader code for the path tracer and the denoiser: line by line ports
// of vkpt's utils.glsl, brdf.glsl, water.glsl (extinction), projection.glsl
// and asvgf.glsl.
//
// vkpt shaders read `global_ubo`, the screen images and `rng_seed` as globals.
// Metal has no global resources, so every function takes a context; the
// macros below keep the ported code close to the GLSL originals.

#ifndef VKPT_COMMON_H
#define VKPT_COMMON_H

#include <metal_stdlib>
#include "../mtlpt_vkpt.h"

using namespace metal;

#define global_ubo (*ctx.ubo)

// Minimal context for the post-tracing passes; the path tracer extends it.
struct VkptCtx {
    constant GlobalUbo *ubo;
    device const VkptImages *img;
};

// Image access with GLSL texelFetch/imageStore semantics: reads outside the
// image return zero, writes outside it are dropped. Metal leaves both
// undefined, and several vkpt filters rely on the robust behavior.
template <typename T>
static inline vec<T, 4> img_load(texture2d<T, access::read> t, int2 p)
{
    if (any(p < 0) || p.x >= int(t.get_width()) || p.y >= int(t.get_height()))
        return vec<T, 4>(0);
    return t.read(uint2(p));
}

template <typename T>
static inline void img_store(texture2d<T, access::write> t, int2 p, vec<T, 4> v)
{
    if (any(p < 0) || p.x >= int(t.get_width()) || p.y >= int(t.get_height()))
        return;
    t.write(v, uint2(p));
}

#define IMG_LOAD(name, p)       img_load(ctx.img->name##_r, int2(p))
#define IMG_STORE(name, p, v)   img_store(ctx.img->name##_w, int2(p), v)
#define TEX_SAMPLE(name, uv)    ctx.img->name##_s.sample(vkpt_linear_clamp, float2(uv), level(0.0))

constexpr sampler vkpt_linear_clamp(filter::linear, address::clamp_to_edge, coord::normalized);
constexpr sampler vkpt_material_sampler(filter::linear, mip_filter::linear, address::repeat, max_anisotropy(1));

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

static inline float square(float x) { return x * x; }

//
// utils.glsl
//

static inline float luminance(float3 color)
{
    return dot(color, float3(0.299, 0.587, 0.114));
}

static float3 decode_normal(uint enc)
{
    uint2 u = uint2(enc & 0xffffu, enc >> 16);
    float2 p = float2(u) / float(0xffff);
    p = p * 2.0 - 1.0;

    float3 n = float3(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    float t = max(0.0, -n.z);
    n.xy += select(float2(t), float2(-t), n.xy >= float2(0.0));
    return normalize(n);
}

static uint encode_normal(float3 normal)
{
    float invL1Norm = 1.0 / (abs(normal.x) + abs(normal.y) + abs(normal.z));
    float2 p = normal.xy * invL1Norm;
    p = (normal.z < 0.0) ? (1.0 - abs(p.yx)) * select(float2(-1.0), float2(1.0), p.xy >= float2(0.0)) : p;
    p = clamp(p.xy * 0.5 + 0.5, float2(0.0), float2(1.0));
    uint2 u = uint2(p * float(0xffffu));
    return u.x | (u.y << 16);
}

static void BicubicCatmullRom(float2 UV, float2 texSize, thread float2 *Sample, thread float2 *Weight)
{
    const float2 invTexSize = 1.0 / texSize;
    float2 tc = floor(UV - 0.5) + 0.5;
    float2 f = UV - tc;
    float2 f2 = f * f;
    float2 f3 = f2 * f;

    float2 w0 = f2 - 0.5 * (f3 + f);
    float2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    float2 w3 = 0.5 * (f3 - f2);
    float2 w2 = 1.0 - w0 - w1 - w3;

    Weight[0] = w0;
    Weight[1] = w1 + w2;
    Weight[2] = w3;

    Sample[0] = tc - 1.0;
    Sample[1] = tc + w2 / Weight[1];
    Sample[2] = tc + 2.0;

    Sample[0] *= invTexSize;
    Sample[1] *= invTexSize;
    Sample[2] *= invTexSize;
}

// uv is in pixel coordinates
static float4 sample_texture_catmull_rom(texture2d<float, access::sample> tex, float2 uv)
{
    float4 sum = float4(0.0);
    float2 sampleLoc[3], sampleWeight[3];
    BicubicCatmullRom(uv, float2(tex.get_width(), tex.get_height()), sampleLoc, sampleWeight);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float2 suv = float2(sampleLoc[j].x, sampleLoc[i].y);
            float4 c = tex.sample(vkpt_linear_clamp, suv, level(0.0));
            sum += c * float4(sampleWeight[j].x * sampleWeight[i].y);
        }
    }
    return sum;
}

static inline float4 alpha_blend(float4 top, float4 bottom)
{
    return float4(top.rgb + bottom.rgb * (1.0 - top.a) * bottom.a, 1.0 - (1.0 - top.a) * (1.0 - bottom.a));
}

static inline float4 alpha_blend_premultiplied(float4 top, float4 bottom)
{
    return float4(top.rgb + bottom.rgb * (1.0 - top.a), 1.0 - (1.0 - top.a) * (1.0 - bottom.a));
}

static float3x3 construct_ONB_frisvad(float3 normal)
{
    float3x3 ret;
    ret[1] = normal;
    if (normal.z < -0.999805696) {
        ret[0] = float3(0.0, -1.0, 0.0);
        ret[2] = float3(-1.0, 0.0, 0.0);
    } else {
        float a = 1.0 / (1.0 + normal.z);
        float b = -normal.x * normal.y * a;
        ret[0] = float3(1.0 - normal.x * normal.x * a, b, -normal.x);
        ret[2] = float3(b, 1.0 - normal.y * normal.y * a, -normal.y);
    }
    return ret;
}

static inline float2 sample_disk(float2 uv)
{
    float theta = 2.0 * M_PI * uv.x;
    float r = sqrt(uv.y);
    return float2(cos(theta), sin(theta)) * r;
}

static inline float3 sample_triangle(float2 xi)
{
    float sqrt_xi = sqrt(xi.x);
    return float3(1.0 - sqrt_xi, sqrt_xi * (1.0 - xi.y), sqrt_xi * xi.y);
}

static inline float3 sample_cos_hemisphere(float2 uv)
{
    float2 disk = sample_disk(uv);
    return float3(disk.x, sqrt(max(0.0, 1.0 - dot(disk, disk))), disk.y);
}

#define HEMISPHERE_COSINE 0.5
#define HEMISPHERE_UNIFORMISH 0.4

static inline float3 sample_cos_hemisphere_multi(float sample_index, float sample_count, float2 uv, float y_power)
{
    float strata_angle = 2.0 * M_PI / sample_count;
    float azimuth = strata_angle * (sample_index + uv.x);
    float2 azimuthal_direction = float2(cos(azimuth), sin(azimuth)) * pow(uv.y, y_power);
    float normal_direction = sqrt(max(0.0, 1.0 - dot(azimuthal_direction, azimuthal_direction)));
    return float3(azimuthal_direction.x, normal_direction, azimuthal_direction.y);
}

static float3 get_explosion_color(float3 normal, float3 direction)
{
    float d = abs(dot(direction, normal)) * 2.0;
    const float3 c0 = float3(0.5, 0.05, 0.0);
    const float3 c1 = float3(1.0, 0.8, 0.1);
    const float3 c2 = float3(1.0, 0.9, 0.6) * 2.0;
    return (d > 1.0) ? mix(c1, c2, d - 1.0) : mix(c0, c1, d);
}

struct SH {
    float4 shY;
    float2 CoCg;
};

static float3 project_SH_irradiance(SH sh, float3 N)
{
    float d = dot(sh.shY.xyz, N);
    float Y = 2.0 * (1.023326 * d + 0.886226 * sh.shY.w);
    Y = max(Y, 0.0);

    sh.CoCg *= Y * 0.282095 / (sh.shY.w + 1e-6);

    float T = Y - sh.CoCg.y * 0.5;
    float G = sh.CoCg.y + T;
    float B = T - sh.CoCg.x * 0.5;
    float R = B + sh.CoCg.x;
    return max(float3(R, G, B), float3(0.0));
}

static SH irradiance_to_SH(float3 color, float3 dir)
{
    SH result;
    float Co = color.r - color.b;
    float t = color.b + Co * 0.5;
    float Cg = color.g - t;
    float Y = max(t + Cg * 0.5, 0.0);
    result.CoCg = float2(Co, Cg);

    float L00 = 0.282095;
    float L1_1 = 0.488603 * dir.y;
    float L10 = 0.488603 * dir.z;
    float L11 = 0.488603 * dir.x;
    result.shY = float4(L11, L1_1, L10, L00) * Y;
    return result;
}

static float3 SH_to_irradiance(SH sh)
{
    float Y = sh.shY.w / 0.282095;
    float T = Y - sh.CoCg.y * 0.5;
    float G = sh.CoCg.y + T;
    float B = T - sh.CoCg.x * 0.5;
    float R = B + sh.CoCg.x;
    return max(float3(R, G, B), float3(0.0));
}

static inline SH init_SH()
{
    SH result;
    result.shY = float4(0.0);
    result.CoCg = float2(0.0);
    return result;
}

static inline void accumulate_SH(thread SH &accum, SH b, float scale)
{
    accum.shY += b.shY * scale;
    accum.CoCg += b.CoCg * scale;
}

static inline SH mix_SH(SH a, SH b, float s)
{
    SH result;
    result.shY = mix(a.shY, b.shY, float4(s));
    result.CoCg = mix(a.CoCg, b.CoCg, float2(s));
    return result;
}

#define LOAD_SH(img_shY, img_CoCg, p) make_SH(IMG_LOAD(img_shY, p), IMG_LOAD(img_CoCg, p).xy)
#define STORE_SH(img_shY, img_CoCg, p, sh) { IMG_STORE(img_shY, p, (sh).shY); IMG_STORE(img_CoCg, p, float4((sh).CoCg, 0.0, 0.0)); }

static inline SH make_SH(float4 shY, float2 CoCg)
{
    SH r;
    r.shY = shY;
    r.CoCg = CoCg;
    return r;
}

static inline uint packHalf2x16(float2 v) { return as_type<uint>(half2(v)); }
static inline float2 unpackHalf2x16(uint v) { return float2(as_type<half2>(v)); }
static inline uint2 packHalf4x16(float4 v) { return uint2(packHalf2x16(v.xy), packHalf2x16(v.zw)); }
static inline float4 unpackHalf4x16(uint2 v) { return float4(unpackHalf2x16(v.x), unpackHalf2x16(v.y)); }

static uint packRGBE(float3 v)
{
    float3 va = max(float3(0.0), v);
    float max_abs = max(va.r, max(va.g, va.b));
    if (max_abs == 0.0)
        return 0u;

    float exponent = floor(log2(max_abs));

    uint result;
    result = uint(clamp(exponent + 20.0, 0.0, 31.0)) << 27;

    float scale = pow(2.0, -exponent) * 256.0;
    uint3 vu = min(uint3(511), uint3(round(va * scale)));
    result |= vu.r;
    result |= vu.g << 9;
    result |= vu.b << 18;
    return result;
}

static float3 unpackRGBE(uint x)
{
    int exponent = int(x >> 27) - 20;
    float scale = pow(2.0, float(exponent)) / 256.0;

    float3 v;
    v.r = float(x & 0x1ffu) * scale;
    v.g = float((x >> 9) & 0x1ffu) * scale;
    v.b = float((x >> 18) & 0x1ffu) * scale;
    return v;
}

static inline uint get_primary_direction(float3 dir)
{
    float3 adir = abs(dir);
    if (adir.x > adir.y && adir.x > adir.z)
        return (dir.x < 0.0) ? 1u : 0u;
    if (adir.y > adir.z)
        return (dir.y < 0.0) ? 3u : 2u;
    return (dir.z < 0.0) ? 5u : 4u;
}

static inline float2 lava_uv_warp(float2 uv, float time)
{
    return uv + sin(fract(uv.yx * 0.5 + time * 20.0 / 128.0) * 2.0 * M_PI) * 0.125;
}

static inline void perturb_tex_coord(uint material_id, float time, thread float2 &tex_coord)
{
    if ((material_id & MATERIAL_FLAG_FLOWING) != 0u)
        tex_coord.x -= time * 0.5;
    if ((material_id & MATERIAL_FLAG_WARP) != 0u)
        tex_coord = lava_uv_warp(tex_coord, time);
}

static inline int popcount_int(int v) { return int(popcount(uint(v))); }

//
// brdf.glsl
//

static inline float RoughnessSquareToSpecPower(float alpha)
{
    return max(0.01, 2.0 / (square(alpha) + 1e-4) - 2.0);
}

static inline float SpecPowerToRoughnessSquare(float s)
{
    return clamp(sqrt(max(0.0, 2.0 / (s + 2.0))), 0.0, 1.0);
}

static inline float G1_Smith(float roughness, float NdotL)
{
    float alpha = square(roughness);
    return 2.0 * NdotL / (NdotL + sqrt(square(alpha) + (1.0 - square(alpha)) * square(NdotL)));
}

static inline float G_Smith_over_4_NdotV(float roughness, float NdotV, float NdotL)
{
    float alpha = square(roughness);
    float g1 = NdotL / (NdotL + sqrt(square(alpha) + (1.0 - square(alpha)) * square(NdotL)));
    float g2 = 1.0 / (NdotV + sqrt(square(alpha) + (1.0 - square(alpha)) * square(NdotV)));
    return g1 * g2;
}

static inline float3 schlick_fresnel(float3 F0, float HdotV, float specular_factor)
{
    float3 F = F0 + (float3(1.0) - F0) * pow(1.0 - HdotV, 5.0);
    F *= specular_factor;
    return clamp(F, float3(0.0), float3(1.0));
}

static float3 GGX_times_NdotL(float3 V, float3 L, float3 N, float roughness, float3 F0, float NoH_offset,
                              float specular_factor, thread float3 &F)
{
    float3 H = normalize(L - V);
    float NoL = max(0.0, dot(N, L));
    float VoH = max(0.0, -dot(V, H));
    float NoV = max(0.0, -dot(N, V));
    float NoH = clamp(dot(N, H) + NoH_offset, 0.0, 1.0);

    F = schlick_fresnel(F0, VoH, specular_factor);

    if (NoL > 0.0 && VoH > 0.0) {
        float G = G_Smith_over_4_NdotV(roughness, NoV, NoL);
        float alpha = square(max(roughness, 0.02));
        float D = square(alpha) / (M_PI * square(square(NoH) * square(alpha) + (1.0 - square(NoH))));
        return F * (D * G);
    }
    return float3(0.0);
}

static inline float ImportanceSampleGGX_VNDF_PDF(float roughness, float3 N, float3 V, float3 L)
{
    float3 H = normalize(L + V);
    float NoH = clamp(dot(N, H), 0.0, 1.0);
    float VoH = clamp(dot(V, H), 0.0, 1.0);
    float alpha = square(roughness);
    float D = square(alpha) / (M_PI * square(square(NoH) * square(alpha) + (1.0 - square(NoH))));
    return (VoH > 0.0) ? D / (4.0 * VoH) : 0.0;
}

static inline float phong(float3 N, float3 L, float3 V, float phong_exp)
{
    float3 H = normalize(L - V);
    return pow(max(0.0, dot(H, N)), phong_exp);
}

static inline void get_reflectivity(float3 base_color, float metallic, thread float3 &o_albedo, thread float3 &o_base_reflectivity)
{
    const float dielectric_specular = 0.04;
    o_albedo = mix(base_color * (1.0 - dielectric_specular), float3(0.0), metallic);
    o_base_reflectivity = mix(float3(dielectric_specular), base_color, metallic);
}

//
// asvgf.glsl
//

#define STRATUM_OFFSET_SHIFT 3
#define STRATUM_OFFSET_MASK ((1 << STRATUM_OFFSET_SHIFT) - 1)

constant float wavelet_factor = 0.5;
constant float wavelet_kernel[2][2] = {
    { 1.0, wavelet_factor },
    { wavelet_factor, wavelet_factor * wavelet_factor }
};

// Field layout (checkerboard_interleave.comp / asvgf_temporal.comp): the two
// checkerboard fields are rendered de-interleaved into the left and right
// halves of the screen images.
static inline int2 checker_to_flat(int2 pos, int width)
{
    int half_width = width / 2;
    bool is_even_checkerboard = pos.x < half_width;
    return int2(is_even_checkerboard
                    ? (pos.x * 2) + (pos.y & 1)
                    : ((pos.x - half_width) * 2) + ((pos.y & 1) ^ 1),
                pos.y);
}

static inline int2 flat_to_checker(int2 pos, int width)
{
    int half_width = width / 2;
    bool is_even_checkerboard = (pos.x & 1) == (pos.y & 1);
    return int2((pos.x / 2) + (is_even_checkerboard ? 0 : half_width), pos.y);
}

//
// projection.glsl
//

static inline void view_to_lonlat(float3 view, thread float &lon, thread float &lat)
{
    lon = atan2(view.x, view.z);
    lat = atan2(-view.y, sqrt(view.x * view.x + view.z * view.z));
}

static inline float3 lonlat_to_view(float lon, float lat)
{
    return float3(sin(lon) * cos(lat), -sin(lat), cos(lon) * cos(lat));
}

template <typename C>
static inline float2 get_projection_fov_scale(thread const C &ctx, bool previous)
{
    return previous ? float2(global_ubo.projection_fov_scale_prev) : float2(global_ubo.projection_fov_scale);
}

template <typename C>
static bool projection_view_to_screen(thread const C &ctx, float3 view_pos, thread float2 &screen_pos,
                                      thread float &distance, bool previous)
{
    switch (global_ubo.pt_projection) {
    default:
    case PROJECTION_RECTILINEAR: {
        float4 clip_pos = (previous ? global_ubo.P_prev : global_ubo.P) * float4(view_pos, 1.0);
        float3 normalized = clip_pos.xyz / clip_pos.w;
        screen_pos = normalized.xy * 0.5 + 0.5;
        distance = length(view_pos);
        return screen_pos.y > 0.0 && screen_pos.y < 1.0 && screen_pos.x > 0.0 && screen_pos.x < 1.0 && view_pos.z > 0.0;
    }
    case PROJECTION_PANINI: {
        float lat, lon;
        distance = length(view_pos);
        view_to_lonlat(normalize(view_pos), lon, lat);
        float S = (PANINI_D + 1.0) / (PANINI_D + cos(lon));
        screen_pos = float2(S * sin(lon), S * tan(lat));
        screen_pos = screen_pos / get_projection_fov_scale(ctx, previous) * 0.5 + 0.5;
        return true;
    }
    case PROJECTION_STEREOGRAPHIC: {
        distance = length(view_pos);
        float3 v = normalize(view_pos);
        float x = v.x, y = -v.y, z = v.z;
        float theta = acos(z);
        if (theta == 0.0) {
            screen_pos = float2(0.5);
        } else {
            float r = tan(theta * STEREOGRAPHIC_ANGLE);
            float c = r / sqrt(x * x + y * y);
            screen_pos = float2(x * c, y * c);
        }
        screen_pos = screen_pos / get_projection_fov_scale(ctx, previous) * 0.5 + 0.5;
        return true;
    }
    case PROJECTION_CYLINDRICAL: {
        float cylindrical_hfov = previous ? global_ubo.cylindrical_hfov_prev : global_ubo.cylindrical_hfov;
        float y = view_pos.y / length(view_pos.xz);
        y *= previous ? global_ubo.P_prev[1][1] : global_ubo.P[1][1];
        screen_pos.y = y * 0.5 + 0.5;
        float angle = atan2(view_pos.x, view_pos.z);
        screen_pos.x = (angle / cylindrical_hfov) + 0.5;
        distance = length(view_pos);
        return screen_pos.y > 0.0 && screen_pos.y < 1.0 && screen_pos.x > 0.0 && screen_pos.x < 1.0;
    }
    case PROJECTION_EQUIRECTANGULAR: {
        float lat, lon;
        distance = length(view_pos);
        view_to_lonlat(normalize(view_pos), lon, lat);
        screen_pos = float2(lon, lat) / get_projection_fov_scale(ctx, previous) * 0.5 + 0.5;
        return true;
    }
    case PROJECTION_MERCATOR: {
        float lat, lon;
        distance = length(view_pos);
        view_to_lonlat(normalize(view_pos), lon, lat);
        screen_pos = float2(lon, log(tan(M_PI * 0.25 + lat * 0.5)));
        screen_pos = screen_pos / get_projection_fov_scale(ctx, previous) * 0.5 + 0.5;
        return true;
    }
    }
}

template <typename C>
static float3 projection_screen_to_view(thread const C &ctx, float2 screen_pos, float distance, bool previous)
{
    switch (global_ubo.pt_projection) {
    default:
    case PROJECTION_RECTILINEAR: {
        float4 clip_pos = float4(screen_pos * 2.0 - 1.0, 1.0, 1.0);
        float3 view_dir = normalize(((previous ? global_ubo.invP_prev : global_ubo.invP) * clip_pos).xyz);
        return view_dir * distance;
    }
    case PROJECTION_PANINI: {
        const float c_D = PANINI_D;
        float2 s = (screen_pos * 2.0 - 1.0) * get_projection_fov_scale(ctx, previous);
        float k = s.x * s.x / ((c_D + 1.0) * (c_D + 1.0));
        float dscr = k * k * c_D * c_D - (k + 1.0) * (k * c_D * c_D - 1.0);
        float clon = (-k * c_D + sqrt(dscr)) / (k + 1.0);
        float S = (c_D + 1.0) / (c_D + clon);
        float lon = atan2(s.x, S * clon);
        float lat = atan2(s.y, S);
        return lonlat_to_view(lon, lat) * distance;
    }
    case PROJECTION_STEREOGRAPHIC: {
        float2 s = (screen_pos * 2.0 - 1.0) * get_projection_fov_scale(ctx, previous);
        float r = sqrt(s.x * s.x + s.y * s.y);
        float theta = atan(r) / STEREOGRAPHIC_ANGLE;
        float sn = sin(theta);
        float3 view_dir = float3(s.x / r * sn, -s.y / r * sn, cos(theta));
        return view_dir * distance;
    }
    case PROJECTION_CYLINDRICAL: {
        float cylindrical_hfov = previous ? global_ubo.cylindrical_hfov_prev : global_ubo.cylindrical_hfov;
        float4 clip_pos = float4(0.0, screen_pos.y * 2.0 - 1.0, 1.0, 1.0);
        float3 view_dir = ((previous ? global_ubo.invP_prev : global_ubo.invP) * clip_pos).xyz;
        float xangle = (screen_pos.x - 0.5) * cylindrical_hfov;
        view_dir.x = sin(xangle);
        view_dir.z = cos(xangle);
        return normalize(view_dir) * distance;
    }
    case PROJECTION_EQUIRECTANGULAR: {
        float2 s = (screen_pos * 2.0 - 1.0) * get_projection_fov_scale(ctx, previous);
        return lonlat_to_view(s.x, s.y) * distance;
    }
    case PROJECTION_MERCATOR: {
        float2 s = (screen_pos * 2.0 - 1.0) * get_projection_fov_scale(ctx, previous);
        return lonlat_to_view(s.x, atan(sinh(s.y))) * distance;
    }
    }
}

#endif // VKPT_COMMON_H
