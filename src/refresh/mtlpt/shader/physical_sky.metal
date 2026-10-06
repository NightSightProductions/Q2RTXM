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

// Port of refresh/vkpt/shader/precomputed_sky.glsl and physical_sky.comp.
// Based on E. Bruneton and F. Neyret, "Precomputed Atmospheric Scattering".

#include <metal_stdlib>
#include "../mtlpt_shared.h"

using namespace metal;

constant float SM_PI = 3.1415926535897932384626433832795;

//
// Table dimensions and the half texel insets used to address them.
//

constant float TRANSMITTANCE_TEXTURE_WIDTH = 256.0;
constant float TRANSMITTANCE_TEXTURE_HEIGHT = 64.0;
constant float SCATTERING_TEXTURE_R_SIZE = 32.0;
constant float SCATTERING_TEXTURE_MU_S_SIZE = 32.0;
constant float SCATTERING_TEXTURE_NU_SIZE = 8.0;
constant float IRRADIANCE_TEXTURE_WIDTH = 64.0;
constant float IRRADIANCE_TEXTURE_HEIGHT = 16.0;
constant float SCATTERING_TEXTURE_MU_SIZE_HALF = 64.0;

static inline float ranged(float val, float size)
{
    return val * (size - 1.0) / size + 0.5 / size;
}

constant float SKY_LUM_SCALE = 0.001;
constant float SUN_LUM_SCALE = 0.00001;

constant float3 SKY_SPECTRAL_RADIANCE_TO_LUMINANCE =
    float3(683.0 * SKY_LUM_SCALE);
constant float3 SUN_SPECTRAL_RADIANCE_TO_LUMINANCE =
    float3(98242.786222 * SUN_LUM_SCALE,
           69954.398112 * SUN_LUM_SCALE,
           66475.012354 * SUN_LUM_SCALE);

constant float SKY_IRRADIANCE_TO_RADIANCE = 0.5 / SM_PI;

//
// Scattering table lookups
//

static float clamp_radius(constant MTLAtmosphereParams &a, float point_height)
{
    return clamp(point_height, a.planet_surface_radius, a.planet_atmosphere_radius);
}

static bool ray_intersects_ground(constant MTLAtmosphereParams &a,
                                  float point_height, float view_angle_cos)
{
    return view_angle_cos < 0.0 &&
           point_height * point_height * (view_angle_cos * view_angle_cos - 1.0) +
           a.planet_surface_radius * a.planet_surface_radius >= 0.0;
}

static float distance_to_top_atmosphere_boundary(constant MTLAtmosphereParams &a,
                                                 float planet_radius, float view_angle_cos)
{
    float d = planet_radius * planet_radius * (view_angle_cos * view_angle_cos - 1.0) +
              a.planet_atmosphere_radius * a.planet_atmosphere_radius;

    return max(0.0, -planet_radius * view_angle_cos + sqrt(max(0.0, d)));
}

static float2 get_transmittance_uv(constant MTLAtmosphereParams &a,
                                   float point_height, float view_angle_cos)
{
    float x0 = sqrt(a.sq_distance_to_horizontal_boundary);
    float dh = sqrt(max(0.0, point_height * point_height -
                             a.planet_surface_radius * a.planet_surface_radius));
    float dH = distance_to_top_atmosphere_boundary(a, point_height, view_angle_cos);
    float x_top = a.planet_atmosphere_radius - point_height;
    float xh = dh + x0;

    float u = (dH - x_top) / (xh - x_top);
    float v = dh / x0;

    return float2(ranged(u, TRANSMITTANCE_TEXTURE_WIDTH),
                  ranged(v, TRANSMITTANCE_TEXTURE_HEIGHT));
}

static float3 get_transmittance_to_top(constant MTLAtmosphereParams &a,
                                       texture2d<float> transmittance,
                                       sampler s,
                                       float point_height, float view_angle_cos)
{
    float2 uv = get_transmittance_uv(a, point_height, view_angle_cos);
    return transmittance.sample(s, uv).rgb;
}

static float4 get_scattering_uvwz(constant MTLAtmosphereParams &a,
                                  float point_height, float view_angle_cos,
                                  float sun_zenith_cos, float sun_view_cos,
                                  bool intersects_ground)
{
    float square_height = point_height * point_height;
    float square_view_sin = 1.0 - view_angle_cos * view_angle_cos;
    float h = sqrt(a.sq_distance_to_horizontal_boundary);
    float horizon_distance = sqrt(max(0.0, square_height -
                                           a.planet_surface_radius * a.planet_surface_radius));

    float u_height = ranged(horizon_distance / h, SCATTERING_TEXTURE_R_SIZE);

    float discriminant = -square_height * square_view_sin +
                         a.planet_surface_radius * a.planet_surface_radius;
    float u_view;

    if (intersects_ground) {
        float d = -point_height * view_angle_cos - sqrt(max(0.0, discriminant));
        float d_min = point_height - a.planet_surface_radius;
        float d_max = horizon_distance;
        float du = (d_max == d_min) ? 0.0 : (d - d_min) / (d_max - d_min);
        du = ranged(du, SCATTERING_TEXTURE_MU_SIZE_HALF);
        u_view = 0.5 - 0.5 * du;
    } else {
        float d = -point_height * view_angle_cos + sqrt(max(0.0, discriminant + h * h));
        float d_min = a.planet_atmosphere_radius - point_height;
        float d_max = horizon_distance + h;
        float du = (d - d_min) / (d_max - d_min);
        du = ranged(du, SCATTERING_TEXTURE_MU_SIZE_HALF);
        u_view = 0.5 + 0.5 * du;
    }

    float d = distance_to_top_atmosphere_boundary(a, a.planet_surface_radius, sun_zenith_cos);
    float d_min = a.atmosphere_height;
    float d_max = h;
    float alpha = (d - d_min) / (d_max - d_min);
    float A = 0.41582 * a.planet_surface_radius / (d_max - d_min);
    float dy = max(1.0 - alpha / A, 0.0) / (1.0 + alpha);
    float u_sun_zenith = ranged(dy, SCATTERING_TEXTURE_MU_S_SIZE);

    float u_sun_view = (sun_view_cos + 1.0) / 2.0;

    return float4(u_sun_view, u_sun_zenith, u_view, u_height);
}

// The Mie term is packed into the Rayleigh sample; see part 4 of the paper.
static float3 get_mie_from_float4(constant MTLAtmosphereParams &a, float4 c)
{
    if (c.r == 0.0)
        return float3(0.0);

    return c.rgb * c.a / c.r *
           (float3(a.rayleigh_scattering).r / float3(a.mie_scattering).r) *
           (float3(a.mie_scattering) / float3(a.rayleigh_scattering));
}

static float3 sample_4d(constant MTLAtmosphereParams &a,
                        texture3d<float> scattering,
                        sampler s,
                        float point_height, float view_angle_cos,
                        float sun_zenith_cos, float sun_view_cos,
                        bool intersects_ground,
                        thread float3 &out_mie)
{
    float4 uvwz = get_scattering_uvwz(a, point_height, view_angle_cos,
                                      sun_zenith_cos, sun_view_cos, intersects_ground);

    float ux = uvwz.x * (SCATTERING_TEXTURE_NU_SIZE - 1.0);
    float offset = floor(ux);
    float t = fract(ux);

    float3 uvw0 = float3((offset + uvwz.y) / SCATTERING_TEXTURE_NU_SIZE, uvwz.z, uvwz.w);
    float3 uvw1 = float3((offset + 1.0 + uvwz.y) / SCATTERING_TEXTURE_NU_SIZE, uvwz.z, uvwz.w);

    float4 interpolated = scattering.sample(s, uvw0) * (1.0 - t) +
                          scattering.sample(s, uvw1) * t;

    out_mie = get_mie_from_float4(a, interpolated);
    return interpolated.xyz;
}

static float rayleigh_phase(float nu)
{
    return (3.0 / (16.0 * SM_PI)) * (1.0 + nu * nu);
}

static float mie_phase(float g, float nu)
{
    float k = 3.0 / (8.0 * SM_PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + nu * nu) / pow(1.0 + g * g - 2.0 * g * nu, 1.5);
}

static void get_parameters(constant MTLAtmosphereParams &a,
                           float3 view_ray, float3 camera,
                           thread float &point_height,
                           thread float &dot_view_angle_cos,
                           thread bool &intersects_atmosphere)
{
    point_height = length(camera);
    dot_view_angle_cos = dot(camera, view_ray);

    float t = -dot_view_angle_cos -
              sqrt(dot_view_angle_cos * dot_view_angle_cos - point_height * point_height +
                   a.planet_atmosphere_radius * a.planet_atmosphere_radius);

    if (t > 0.0) {
        // The viewer is in space; move it to the atmosphere boundary.
        point_height = a.planet_atmosphere_radius;
        dot_view_angle_cos += t;
        intersects_atmosphere = true;
    } else {
        intersects_atmosphere = false;
    }
}

static float3 get_sky_radiance(constant MTLAtmosphereParams &a,
                               texture2d<float> transmittance,
                               texture3d<float> scattering,
                               sampler s,
                               float3 camera, float3 view_ray, float3 sun_direction,
                               thread float3 &out_transmittance)
{
    out_transmittance = float3(1.0);

    float point_height, dot_view_angle_cos;
    bool intersects_atmosphere;
    get_parameters(a, view_ray, camera, point_height, dot_view_angle_cos, intersects_atmosphere);

    if (!intersects_atmosphere && point_height > a.planet_atmosphere_radius)
        return float3(0.0);

    float view_angle_cos = dot_view_angle_cos / point_height;
    float sun_zenith_cos = dot(camera, sun_direction) / point_height;
    float sun_view_cos = dot(view_ray, sun_direction);
    bool intersects_ground = ray_intersects_ground(a, point_height, view_angle_cos);

    out_transmittance = intersects_ground
        ? float3(0.0)
        : get_transmittance_to_top(a, transmittance, s, point_height, view_angle_cos);

    float3 single_mie;
    float3 scattered = sample_4d(a, scattering, s, point_height, view_angle_cos,
                                 sun_zenith_cos, sun_view_cos, intersects_ground, single_mie);

    float3 result = scattered * rayleigh_phase(sun_view_cos) +
                    single_mie * mie_phase(a.mie_henyey_greenstein_g, sun_view_cos);

    result /= float3(a.star_irradiance) *
              (SUN_SPECTRAL_RADIANCE_TO_LUMINANCE / SKY_SPECTRAL_RADIANCE_TO_LUMINANCE);
    result *= SKY_IRRADIANCE_TO_RADIANCE;

    return result;
}

static float2 get_irradiance_uv(constant MTLAtmosphereParams &a,
                                float point_height, float sun_zenith_cos)
{
    float u_height = (point_height - a.planet_surface_radius) / a.atmosphere_height;
    float v_view = sun_zenith_cos * 0.5 + 0.5;

    return float2(ranged(v_view, IRRADIANCE_TEXTURE_WIDTH),
                  ranged(u_height, IRRADIANCE_TEXTURE_HEIGHT));
}

static float3 get_sky_irradiance(constant MTLAtmosphereParams &a,
                                 texture2d<float> irradiance,
                                 sampler s,
                                 float3 spoint, float3 sun_direction)
{
    float point_height = length(spoint);
    float sun_zenith_cos = dot(spoint, sun_direction) / point_height;

    float3 result = irradiance.sample(s, get_irradiance_uv(a, point_height, sun_zenith_cos)).rgb;

    result /= float3(a.star_irradiance) *
              (SUN_SPECTRAL_RADIANCE_TO_LUMINANCE / SKY_SPECTRAL_RADIANCE_TO_LUMINANCE);
    result *= SKY_IRRADIANCE_TO_RADIANCE;

    return result;
}

//
// Cube map rendering
//

// Inverse of the path tracer's sample_skybox() face mapping, so the array this
// kernel fills can be read back with exactly that function. Directions are in
// the engine's world space, which is +Z up - the orientation the atmosphere
// model already expects.
static float3 face_direction(uint face, float2 uv)
{
    float s = uv.x * 2.0 - 1.0;
    float t = 1.0 - uv.y * 2.0;

    switch (face) {
    case 0:  return normalize(float3( 1.0,  -s,    t));
    case 1:  return normalize(float3(-1.0,   s,    t));
    case 2:  return normalize(float3(  s,   1.0,   t));
    case 3:  return normalize(float3( -s,  -1.0,   t));
    case 4:  return normalize(float3( -t,   -s,  1.0));
    default: return normalize(float3(  t,   -s, -1.0));
    }
}

kernel void physical_sky_render(
    uint3 tid                                    [[thread_position_in_grid]],
    texture2d_array<float, access::write> out_sky [[texture(0)]],
    texture2d<float> transmittance                [[texture(1)]],
    texture3d<float> scattering                   [[texture(2)]],
    texture2d<float> irradiance                   [[texture(3)]],
    constant MTLAtmosphereParams &atmosphere      [[buffer(0)]],
    constant MTLPhysicalSkyUniforms &u            [[buffer(1)]],
    device MTLSkyAccumulator *accum               [[buffer(2)]])
{
    if (tid.x >= u.face_size || tid.y >= u.face_size)
        return;

    constexpr sampler lut_sampler(filter::linear, address::clamp_to_edge);

    float2 uv = (float2(tid.xy) + 0.5) / float(u.face_size);
    float3 eye = face_direction(tid.z, uv);

    float3 sun_direction = normalize(float3(u.sun_direction));

    // The viewer sits just above the planet surface, as in physical_sky.comp.
    float3 camera = float3(0.0, 0.0, atmosphere.planet_surface_radius + 0.1);

    float3 sun_transmittance = float3(0.0);
    float3 radiance = get_sky_radiance(atmosphere, transmittance, scattering, lut_sampler,
                                       camera, eye, sun_direction, sun_transmittance);

    // The solar disc itself, with a soft edge so the cube map can be filtered.
    float3 sun_direct = sun_transmittance / max(u.sun_solid_angle, 1e-6);
    sun_direct *= pow(saturate((dot(eye, sun_direction) - u.sun_cos_half_angle) * 1000.0 + 0.875), 10.0);

    radiance += sun_direct;

    // Below the horizon the atmosphere model has nothing to say, so fall back
    // to the preset's ground colour lit by the sky.
    if (eye.z < 0.0) {
        float3 ground_irradiance = get_sky_irradiance(atmosphere, irradiance, lut_sampler,
                                                      camera, sun_direction);
        float3 ground = float3(u.ground_radiance) * ground_irradiance / SM_PI;
        radiance = mix(radiance, ground, saturate(-eye.z * 8.0));
    }

    radiance *= float3(u.sun_color);

    out_sky.write(float4(radiance, 1.0), uint2(tid.x, tid.y), tid.z);

    // Sparse average of the sky, used for the sky area lights. Only samples
    // away from the solar disc so one very bright texel cannot dominate.
    if (dot(sun_direct, sun_direct) == 0.0 && (tid.x & 15u) == 7u && (tid.y & 15u) == 7u) {
        atomic_fetch_add_explicit(
            (device atomic_int *)&accum->color[0],
            int(radiance.r * MTL_SKY_ACCUM_SCALE), memory_order_relaxed);
        atomic_fetch_add_explicit(
            (device atomic_int *)&accum->color[1],
            int(radiance.g * MTL_SKY_ACCUM_SCALE), memory_order_relaxed);
        atomic_fetch_add_explicit(
            (device atomic_int *)&accum->color[2],
            int(radiance.b * MTL_SKY_ACCUM_SCALE), memory_order_relaxed);
        atomic_fetch_add_explicit(
            (device atomic_int *)&accum->count, 1, memory_order_relaxed);
    }
}
