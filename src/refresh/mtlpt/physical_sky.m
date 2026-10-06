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

// Physical sky, ported from refresh/vkpt/physical_sky.c and precomputed_sky.c.
// The Bruneton scattering tables are shipped as .dds files in the media pack,
// so this only has to load them and drive the cube map kernel.

#include "mtlpt_metal.h"
#include "physical_sky.h"

#include "common/files.h"
#include "common/math.h"

//
// DDS loading
//

#define DDS_MAGIC 0x20534444    // "DDS "

#define DDS_FOURCC_DX10 (('D') | ('X' << 8) | ('1' << 16) | ('0' << 24))

#define DDS_CUBEMAP          0x00000200
#define DDS_CUBEMAP_ALLFACES 0x0000fc00

#define DXGI_FORMAT_R32G32B32A32_FLOAT 2
#define DXGI_FORMAT_R32_FLOAT          41
#define DXGI_FORMAT_B8G8R8A8_UNORM     87

#define D3D11_RESOURCE_MISC_TEXTURECUBE 0x4

typedef struct {
    uint32_t size;
    uint32_t flags;
    uint32_t fourCC;
    uint32_t RGBBitCount;
    uint32_t RBitMask;
    uint32_t GBitMask;
    uint32_t BBitMask;
    uint32_t ABitMask;
} dds_pixelformat_t;

typedef struct {
    uint32_t          magic;
    uint32_t          size;
    uint32_t          flags;
    uint32_t          height;
    uint32_t          width;
    uint32_t          pitchOrLinearSize;
    uint32_t          depth;
    uint32_t          mipMapCount;
    uint32_t          reserved1[11];
    dds_pixelformat_t ddspf;
    uint32_t          caps;
    uint32_t          caps2;
    uint32_t          caps3;
    uint32_t          caps4;
    uint32_t          reserved2;
} dds_header_t;

typedef struct {
    uint32_t dxgiFormat;
    uint32_t resourceDimension;
    uint32_t miscFlag;
    uint32_t arraySize;
    uint32_t miscFlags2;
} dds_header_dxt10_t;

static bool bitmask_is(const dds_pixelformat_t *pf, uint32_t r, uint32_t g, uint32_t b, uint32_t a)
{
    return pf->RBitMask == r && pf->GBitMask == g && pf->BBitMask == b && pf->ABitMask == a;
}

// Loads one of the scattering tables. They are uncompressed 2D or 3D images
// with a single mip level, which keeps this much simpler than a general loader.
static id<MTLTexture> load_dds(const char *filename)
{
    byte *data = NULL;
    int len = FS_LoadFile(filename, (void **)&data);

    if (!data || len < (int)sizeof(dds_header_t)) {
        Com_EPrintf("Metal: couldn't read %s\n", filename);
        if (data)
            FS_FreeFile(data);
        return nil;
    }

    id<MTLTexture> texture = nil;

    const dds_header_t *dds = (const dds_header_t *)data;
    const dds_header_dxt10_t *dxt10 = (const dds_header_dxt10_t *)(data + sizeof(dds_header_t));

    if (dds->magic != DDS_MAGIC || dds->size != sizeof(dds_header_t) - 4) {
        Com_EPrintf("Metal: %s is not a DDS file\n", filename);
        goto done;
    }

    MTLPixelFormat format = MTLPixelFormatInvalid;
    size_t header_size = sizeof(dds_header_t);
    size_t bytes_per_pixel = 0;
    bool swizzle_bgra = false;

    if (dds->ddspf.fourCC == DDS_FOURCC_DX10) {
        header_size += sizeof(dds_header_dxt10_t);

        switch (dxt10->dxgiFormat) {
        case DXGI_FORMAT_B8G8R8A8_UNORM:
            format = MTLPixelFormatBGRA8Unorm;
            bytes_per_pixel = 4;
            break;
        case DXGI_FORMAT_R32_FLOAT:
            format = MTLPixelFormatR32Float;
            bytes_per_pixel = 4;
            break;
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
            format = MTLPixelFormatRGBA32Float;
            bytes_per_pixel = 16;
            break;
        default:
            Com_EPrintf("Metal: %s uses unsupported DXGI format %u\n", filename, dxt10->dxgiFormat);
            goto done;
        }
    } else {
        if (bitmask_is(&dds->ddspf, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000)) {
            format = MTLPixelFormatBGRA8Unorm;
            bytes_per_pixel = 4;
        } else if (bitmask_is(&dds->ddspf, 0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000)) {
            format = MTLPixelFormatRGBA8Unorm;
            bytes_per_pixel = 4;
        } else if (bitmask_is(&dds->ddspf, 0xffffffff, 0, 0, 0)) {
            format = MTLPixelFormatR32Float;
            bytes_per_pixel = 4;
        } else {
            Com_EPrintf("Metal: %s uses an unsupported pixel format\n", filename);
            goto done;
        }
    }
    (void)swizzle_bgra;

    uint32_t width = dds->width;
    uint32_t height = max(dds->height, 1u);
    uint32_t depth = max(dds->depth, 1u);

    MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
    desc.textureType = (depth > 1) ? MTLTextureType3D : MTLTextureType2D;
    desc.pixelFormat = format;
    desc.width = width;
    desc.height = height;
    desc.depth = depth;
    desc.mipmapLevelCount = 1;
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;

    texture = [mtl.device newTextureWithDescriptor:desc];
    [desc release];

    if (!texture) {
        Com_EPrintf("Metal: could not create a texture for %s\n", filename);
        goto done;
    }

    size_t bytes_per_row = (size_t)width * bytes_per_pixel;
    size_t bytes_per_image = bytes_per_row * height;
    size_t needed = bytes_per_image * depth;

    if (header_size + needed > (size_t)len) {
        Com_EPrintf("Metal: %s is truncated\n", filename);
        [texture release];
        texture = nil;
        goto done;
    }

    [texture replaceRegion:MTLRegionMake3D(0, 0, 0, width, height, depth)
               mipmapLevel:0
                     slice:0
                 withBytes:data + header_size
               bytesPerRow:bytes_per_row
             bytesPerImage:(depth > 1) ? bytes_per_image : 0];

    texture.label = [NSString stringWithUTF8String:filename];

done:
    FS_FreeFile(data);
    return texture;
}

//
// Atmosphere parameters, from precomputed_sky.c
//

#define EARTH_SURFACE_RADIUS      6360.0f
#define EARTH_ATMOSPHERE_RADIUS   6420.0f
#define STROGGOS_SURFACE_RADIUS   6360.0f
#define STROGGOS_ATMOSPHERE_RADIUS 6520.0f

#define DIST_TO_HORIZON(LOW, HIGH) ((HIGH) * (HIGH) - (LOW) * (LOW))

static const MTLAtmosphereParams params_earth = {
    .star_irradiance = { 1.47399998f, 1.85039997f, 1.91198003f },
    .star_angular_radius = 0.00467499997f,
    .rayleigh_scattering = { 0.00580233941f, 0.0135577619f, 0.0331000052f },
    .planet_surface_radius = EARTH_SURFACE_RADIUS,
    .mie_scattering = { 0.0014985f, 0.0014985f, 0.0014985f },
    .planet_atmosphere_radius = EARTH_ATMOSPHERE_RADIUS,
    .mie_henyey_greenstein_g = 0.8f,
    .sq_distance_to_horizontal_boundary =
        DIST_TO_HORIZON(EARTH_SURFACE_RADIUS, EARTH_ATMOSPHERE_RADIUS),
    .atmosphere_height = EARTH_ATMOSPHERE_RADIUS - EARTH_SURFACE_RADIUS,
};

static const MTLAtmosphereParams params_stroggos = {
    .star_irradiance = { 2.47399998f, 1.85039997f, 1.01198006f },
    .star_angular_radius = 0.00934999995f,
    .rayleigh_scattering = { 0.0270983186f, 0.0414223559f, 0.0647224262f },
    .planet_surface_radius = STROGGOS_SURFACE_RADIUS,
    .mie_scattering = { 0.00342514296f, 0.00342514296f, 0.00342514296f },
    .planet_atmosphere_radius = STROGGOS_ATMOSPHERE_RADIUS,
    .mie_henyey_greenstein_g = 0.9f,
    .sq_distance_to_horizontal_boundary =
        DIST_TO_HORIZON(STROGGOS_SURFACE_RADIUS, STROGGOS_ATMOSPHERE_RADIUS),
    .atmosphere_height = STROGGOS_ATMOSPHERE_RADIUS - STROGGOS_SURFACE_RADIUS,
};

//
// Sky presets, from physical_sky.c
//

typedef enum {
    SKY_NONE,
    SKY_EARTH,
    SKY_STROGGOS,
} sky_preset_id_t;

typedef struct {
    vec3_t          sun_color;
    float           sun_angular_diameter;
    vec3_t          ground_albedo;
    uint32_t        flags;
    sky_preset_id_t preset;
} sky_preset_t;

static const sky_preset_t sky_presets[3] = {
    {
        .flags = PHYSICAL_SKY_FLAG_USE_SKYBOX,
        .preset = SKY_NONE,
    },
    {
        .sun_color = { 1.45f, 1.29f, 1.27f },
        .sun_angular_diameter = 1.0f,
        .ground_albedo = { 0.3f, 0.15f, 0.14f },
        .flags = PHYSICAL_SKY_FLAG_DRAW_MOUNTAINS,
        .preset = SKY_EARTH,
    },
    {
        .sun_color = { 0.315f, 0.137f, 0.033f },
        .sun_angular_diameter = 5.0f,
        .ground_albedo = { 0.133f, 0.101f, 0.047f },
        .flags = PHYSICAL_SKY_FLAG_DRAW_MOUNTAINS,
        .preset = SKY_STROGGOS,
    },
};

static const sky_preset_t *get_sky_preset(int index)
{
    if (index >= 0 && index < (int)q_countof(sky_presets))
        return &sky_presets[index];
    return &sky_presets[0];
}

// Sun presets, from physical_sky.c's active_sun_preset()
enum {
    SUN_PRESET_NONE = 0,
    SUN_PRESET_CURRENT_TIME = 1,
    SUN_PRESET_FAST_TIME = 2,
    SUN_PRESET_NIGHT = 3,
    SUN_PRESET_DAWN,
    SUN_PRESET_MORNING,
    SUN_PRESET_NOON,
    SUN_PRESET_EVENING,
    SUN_PRESET_DUSK,
};

//
// State
//

static cvar_t *cvar_physical_sky;
static cvar_t *cvar_physical_sky_brightness;
static cvar_t *cvar_sun_preset;
static cvar_t *cvar_sun_azimuth;
static cvar_t *cvar_sun_elevation;
static cvar_t *cvar_sun_angle;
static cvar_t *cvar_sun_brightness;
static cvar_t *cvar_sun_color[3];

static id<MTLTexture> tex_transmittance;
static id<MTLTexture> tex_scattering;
static id<MTLTexture> tex_irradiance;
static id<MTLTexture> tex_sky_cube;

static id<MTLComputePipelineState> pipeline_sky;
static id<MTLBuffer>               atmosphere_buffer;
static id<MTLBuffer>               sky_accum_buffer;

static int   loaded_preset = -1;
static int   applied_sky_index = -1;
static bool  sky_needs_update = true;

static mtl_sun_light_t sun_light;
static vec3_t sky_average = { 0.f, 0.f, 0.f };

#define SKY_CUBE_SIZE 128

static void update_preset_cvars(void)
{
    const sky_preset_t *sky = get_sky_preset(cvar_physical_sky->integer);

    for (int i = 0; i < 3; i++)
        Cvar_SetValue(cvar_sun_color[i], sky->sun_color[i], FROM_CODE);

    Cvar_SetValue(cvar_sun_angle, sky->sun_angular_diameter, FROM_CODE);

    sky_needs_update = true;
}

static bool load_scatter_parameters(sky_preset_id_t preset)
{
    const char *planet = NULL;
    const MTLAtmosphereParams *constants = NULL;

    if (preset == SKY_EARTH) {
        planet = "earth";
        constants = &params_earth;
    } else if (preset == SKY_STROGGOS) {
        planet = "stroggos";
        constants = &params_stroggos;
    } else {
        return false;
    }

    [tex_transmittance release];
    [tex_scattering release];
    [tex_irradiance release];

    char path[MAX_QPATH];
    Q_snprintf(path, sizeof(path), "env/transmittance_%s.dds", planet);
    tex_transmittance = load_dds(path);
    Q_snprintf(path, sizeof(path), "env/inscatter_%s.dds", planet);
    tex_scattering = load_dds(path);
    Q_snprintf(path, sizeof(path), "env/irradiance_%s.dds", planet);
    tex_irradiance = load_dds(path);

    if (!tex_transmittance || !tex_scattering || !tex_irradiance) {
        Com_EPrintf("Metal: physical sky tables for '%s' are missing\n", planet);
        return false;
    }

    memcpy(atmosphere_buffer.contents, constants, sizeof(*constants));

    Com_Printf("Metal: loaded physical sky tables for '%s'\n", planet);
    return true;
}

bool mtl_physical_sky_init(void)
{
    cvar_physical_sky = Cvar_Get("physical_sky", "2", 0);
    cvar_physical_sky_brightness = Cvar_Get("physical_sky_brightness", "0", 0);
    cvar_sun_preset = Cvar_Get("sun_preset", va("%d", SUN_PRESET_MORNING), CVAR_ARCHIVE);
    cvar_sun_azimuth = Cvar_Get("sun_azimuth", "345", 0);
    cvar_sun_elevation = Cvar_Get("sun_elevation", "45", 0);
    cvar_sun_angle = Cvar_Get("sun_angle", "1.0", 0);
    cvar_sun_brightness = Cvar_Get("sun_brightness", "10", 0);

    static const char rgb[3] = { 'r', 'g', 'b' };
    for (int i = 0; i < 3; i++) {
        char name[32];
        Q_snprintf(name, sizeof(name), "sun_color_%c", rgb[i]);
        cvar_sun_color[i] = Cvar_Get(name, "1.0", 0);
    }

    NSError *err = nil;
    id<MTLFunction> fn = mtl_new_function(@"physical_sky_render");
    if (!fn) {
        Com_EPrintf("Metal: physical_sky_render kernel missing from the library\n");
        return false;
    }

    pipeline_sky = [mtl.device newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    if (!pipeline_sky)
        return mtl_log_error("physical_sky_render", err);

    atmosphere_buffer = [mtl.device newBufferWithLength:sizeof(MTLAtmosphereParams)
                                               options:MTLResourceStorageModeShared];
    atmosphere_buffer.label = @"atmosphere params";

    sky_accum_buffer = [mtl.device newBufferWithLength:sizeof(MTLSkyAccumulator)
                                              options:MTLResourceStorageModeShared];
    sky_accum_buffer.label = @"sky accumulator";

    MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
    desc.textureType = MTLTextureType2DArray;
    desc.pixelFormat = MTLPixelFormatRGBA16Float;
    desc.width = SKY_CUBE_SIZE;
    desc.height = SKY_CUBE_SIZE;
    desc.arrayLength = 6;
    desc.mipmapLevelCount = 1;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModePrivate;

    tex_sky_cube = [mtl.device newTextureWithDescriptor:desc];
    [desc release];
    tex_sky_cube.label = @"physical sky";

    return tex_sky_cube != nil;
}

void mtl_physical_sky_shutdown(void)
{
    [tex_transmittance release];
    tex_transmittance = nil;
    [tex_scattering release];
    tex_scattering = nil;
    [tex_irradiance release];
    tex_irradiance = nil;
    [tex_sky_cube release];
    tex_sky_cube = nil;
    [pipeline_sky release];
    pipeline_sky = nil;
    [atmosphere_buffer release];
    atmosphere_buffer = nil;
    [sky_accum_buffer release];
    sky_accum_buffer = nil;

    loaded_preset = -1;
    applied_sky_index = -1;
}

bool mtl_physical_sky_active(void)
{
    const sky_preset_t *sky = get_sky_preset(cvar_physical_sky->integer);
    return (sky->flags & PHYSICAL_SKY_FLAG_USE_SKYBOX) == 0 && tex_sky_cube != nil;
}

const mtl_sun_light_t *mtl_physical_sky_sun(void)
{
    return &sun_light;
}

void mtl_physical_sky_average_color(vec3_t out)
{
    VectorCopy(sky_average, out);
}

id<MTLTexture> mtl_physical_sky_texture(void)
{
    return tex_sky_cube;
}

// Port of vkpt_evaluate_sun_light(). The time based presets are left out; they
// only matter for sun_animate, which this backend does not drive yet.
static void evaluate_sun_light(void)
{
    const sky_preset_t *sky = get_sky_preset(cvar_physical_sky->integer);

    if (sky->flags & PHYSICAL_SKY_FLAG_USE_SKYBOX) {
        memset(&sun_light, 0, sizeof(sun_light));
        return;
    }

    float azimuth, elevation;

    switch (cvar_sun_preset->integer) {
    case SUN_PRESET_NIGHT:   elevation = -90.0f; azimuth = 0.0f;   break;
    case SUN_PRESET_DAWN:    elevation = -3.0f;  azimuth = 0.0f;   break;
    case SUN_PRESET_MORNING: elevation = 25.0f;  azimuth = -15.0f; break;
    case SUN_PRESET_NOON:    elevation = 80.0f;  azimuth = -75.0f; break;
    case SUN_PRESET_EVENING: elevation = 15.0f;  azimuth = 190.0f; break;
    case SUN_PRESET_DUSK:    elevation = -6.0f;  azimuth = 205.0f; break;
    default:
        azimuth = cvar_sun_azimuth->value;
        elevation = cvar_sun_elevation->value;
        break;
    }

    float elevation_rad = DEG2RAD(elevation);
    float azimuth_rad = DEG2RAD(azimuth);

    sun_light.direction[0] = cosf(azimuth_rad) * cosf(elevation_rad);
    sun_light.direction[1] = sinf(azimuth_rad) * cosf(elevation_rad);
    sun_light.direction[2] = sinf(elevation_rad);

    sun_light.angular_size_rad = DEG2RAD(Q_clipf(cvar_sun_angle->value, 1.0f, 10.0f));
    sun_light.use_physical_sky = true;

    for (int i = 0; i < 3; i++)
        sun_light.color[i] = cvar_sun_color[i]->value * cvar_sun_brightness->value;

    sun_light.visible = sun_light.direction[2] >= -sinf(sun_light.angular_size_rad * 0.5f);
}

void mtl_physical_sky_update(float time)
{
    if (!pipeline_sky)
        return;

    if (cvar_physical_sky->integer != applied_sky_index) {
        update_preset_cvars();
        applied_sky_index = cvar_physical_sky->integer;
    }

    evaluate_sun_light();

    if (!mtl_physical_sky_active())
        return;

    const sky_preset_t *sky = get_sky_preset(cvar_physical_sky->integer);

    if (sky->preset != loaded_preset) {
        if (!load_scatter_parameters(sky->preset))
            return;
        loaded_preset = sky->preset;
        sky_needs_update = true;
    }

    // The sky only has to be re-rendered when the sun moves or the preset
    // changes; it is otherwise constant for the whole map.
    static vec3_t applied_direction;
    static float applied_brightness = -1.0f;

    if (!VectorCompare(applied_direction, sun_light.direction) ||
        applied_brightness != cvar_sun_brightness->value)
        sky_needs_update = true;

    if (!sky_needs_update)
        return;

    VectorCopy(sun_light.direction, applied_direction);
    applied_brightness = cvar_sun_brightness->value;
    sky_needs_update = false;

    MTLSkyAccumulator *accum = (MTLSkyAccumulator *)sky_accum_buffer.contents;
    memset(accum, 0, sizeof(*accum));

    MTLPhysicalSkyUniforms u = { 0 };
    u.sun_direction.x = sun_light.direction[0];
    u.sun_direction.y = sun_light.direction[1];
    u.sun_direction.z = sun_light.direction[2];
    for (int i = 0; i < 3; i++)
        ((float *)&u.sun_color)[i] = cvar_sun_color[i]->value;
    u.sun_cos_half_angle = cosf(sun_light.angular_size_rad * 0.5f);
    u.sun_solid_angle = 2.0f * M_PI * (1.0f - cosf(sun_light.angular_size_rad * 0.5f));
    u.face_size = SKY_CUBE_SIZE;
    u.flags = sky->flags;
    u.ground_radiance.x = sky->ground_albedo[0];
    u.ground_radiance.y = sky->ground_albedo[1];
    u.ground_radiance.z = sky->ground_albedo[2];

    id<MTLCommandBuffer> cmd = [mtl.queue commandBuffer];
    cmd.label = @"physical sky";

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline_sky];
    [enc setTexture:tex_sky_cube atIndex:0];
    [enc setTexture:tex_transmittance atIndex:1];
    [enc setTexture:tex_scattering atIndex:2];
    [enc setTexture:tex_irradiance atIndex:3];
    [enc setBuffer:atmosphere_buffer offset:0 atIndex:0];
    [enc setBytes:&u length:sizeof(u) atIndex:1];
    [enc setBuffer:sky_accum_buffer offset:0 atIndex:2];

    MTLSize threads = MTLSizeMake(8, 8, 1);
    MTLSize groups = MTLSizeMake((SKY_CUBE_SIZE + 7) / 8, (SKY_CUBE_SIZE + 7) / 8, 6);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];

    // Average sky radiance, used to drive the sky area lights. Mirrors the
    // accumulator vkpt fills in physical_sky.comp.
    if (accum->count > 0) {
        float inv = 1.0f / (float)accum->count / MTL_SKY_ACCUM_SCALE;
        sky_average[0] = (float)accum->color[0] * inv;
        sky_average[1] = (float)accum->color[1] * inv;
        sky_average[2] = (float)accum->color[2] * inv;
    }

    Com_Printf("Metal: physical sky rendered, average (%.4f %.4f %.4f)\n",
               sky_average[0], sky_average[1], sky_average[2]);
}
