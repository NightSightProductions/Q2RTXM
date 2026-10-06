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

#include "mtlpt_metal.h"
#import <AppKit/NSScreen.h>

mtl_state_t mtl;

// Provided by src/unix/video/sdl.c
extern void *get_metal_layer(void);

cvar_t *mtl_vsync;
cvar_t *mtl_hdr;

bool mtl_log_error(const char *what, NSError *error)
{
    Com_EPrintf("Metal: %s failed: %s\n", what,
                error ? [[error localizedDescription] UTF8String] : "unknown error");
    return false;
}

// Development fallback for machines without Xcode (no offline `metal`
// compiler): with Q2RTX_MTL_SHADER_DIR pointing at src/refresh/mtlpt/shader,
// every .metal file there is compiled at startup by the runtime compiler. Each
// file becomes its own library, since the files share helper names that only
// work as separate translation units; mtl_new_function() searches them all.
static NSMutableArray *source_libraries;

static NSString *inline_shader_includes(NSString *path, NSMutableSet *seen)
{
    NSString *full = [path stringByStandardizingPath];
    if ([seen containsObject:full])
        return @"";
    [seen addObject:full];

    NSString *src = [NSString stringWithContentsOfFile:full encoding:NSUTF8StringEncoding error:NULL];
    if (!src) {
        Com_EPrintf("Metal: cannot read shader source %s\n", full.UTF8String);
        return @"";
    }

    // Source libraries have no include path, so local includes are inlined.
    NSString *dir = [full stringByDeletingLastPathComponent];
    NSMutableString *out = [NSMutableString string];
    NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:@"^\\s*#\\s*include\\s+\"([^\"]+)\""
                                                                        options:0 error:NULL];
    for (NSString *line in [src componentsSeparatedByString:@"\n"]) {
        NSTextCheckingResult *m = [re firstMatchInString:line options:0 range:NSMakeRange(0, line.length)];
        if (m) {
            NSString *inc = [line substringWithRange:[m rangeAtIndex:1]];
            [out appendString:inline_shader_includes([dir stringByAppendingPathComponent:inc], seen)];
            [out appendString:@"\n"];
        } else {
            [out appendString:line];
            [out appendString:@"\n"];
        }
    }
    return out;
}

static id<MTLLibrary> compile_shader_sources(id<MTLDevice> device, const char *shader_dir)
{
    NSString *dir = [NSString stringWithUTF8String:shader_dir];
    NSArray *files = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:dir error:NULL];
    if (!files.count) {
        Com_EPrintf("Metal: no shader sources in %s\n", shader_dir);
        return nil;
    }

    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    options.languageVersion = MTLLanguageVersion3_0;

    source_libraries = [[NSMutableArray alloc] init];
    for (NSString *file in [files sortedArrayUsingSelector:@selector(compare:)]) {
        if (![file.pathExtension isEqualToString:@"metal"])
            continue;
        NSString *src = inline_shader_includes([dir stringByAppendingPathComponent:file], [NSMutableSet set]);
        NSError *err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:options error:&err];
        if (!lib) {
            Com_EPrintf("Metal: compiling %s failed: %s\n", file.UTF8String, err.localizedDescription.UTF8String);
            [options release];
            [source_libraries release];
            source_libraries = nil;
            return nil;
        }
        [source_libraries addObject:lib];
        [lib release];
    }
    [options release];

    Com_Printf("Metal: compiled %d shader sources from %s\n", (int)source_libraries.count, shader_dir);
    return source_libraries.count ? [source_libraries[0] retain] : nil;
}

id<MTLFunction> mtl_new_function(NSString *name)
{
    id<MTLFunction> fn = [mtl.library newFunctionWithName:name];
    for (id<MTLLibrary> lib in source_libraries) {
        if (fn)
            break;
        fn = [lib newFunctionWithName:name];
    }
    return fn;
}

// Newest modification time of the shader sources (.metal and headers).
static NSDate *newest_source_date(NSString *shader_dir)
{
    NSFileManager *fm = [NSFileManager defaultManager];
    NSDate *newest = nil;
    NSArray *dirs = @[shader_dir, [shader_dir stringByDeletingLastPathComponent]];
    for (NSString *dir in dirs) {
        for (NSString *file in [fm contentsOfDirectoryAtPath:dir error:NULL]) {
            if (![file.pathExtension isEqualToString:@"metal"] && ![file.pathExtension isEqualToString:@"h"])
                continue;
            NSDate *d = [fm attributesOfItemAtPath:[dir stringByAppendingPathComponent:file] error:NULL].fileModificationDate;
            if (d && (!newest || [d compare:newest] == NSOrderedDescending))
                newest = d;
        }
    }
    return newest;
}

// Shader sources for a build run from the source tree: the explicit
// Q2RTX_MTL_SHADER_DIR, else src/refresh/mtlpt/shader next to the executable.
static NSString *find_shader_sources(bool *forced)
{
    const char *env = getenv("Q2RTX_MTL_SHADER_DIR");
    *forced = env && *env;
    if (*forced)
        return [NSString stringWithUTF8String:env];

    NSString *dir = [[[NSBundle mainBundle] bundlePath] stringByAppendingPathComponent:@"src/refresh/mtlpt/shader"];
    BOOL is_dir = NO;
    if ([[NSFileManager defaultManager] fileExistsAtPath:dir isDirectory:&is_dir] && is_dir)
        return dir;
    return nil;
}

// A metallib older than the sources next to it no longer matches the engine
// (missing kernels, changed uniform layouts), so the sources win.
static bool metallib_is_stale(NSString *shader_dir)
{
    if (!shader_dir)
        return false;
    NSString *lib = [[[NSBundle mainBundle] bundlePath] stringByAppendingPathComponent:@"baseq2/shader_mtlpt/q2rtx.metallib"];
    NSDate *lib_date = [[NSFileManager defaultManager] attributesOfItemAtPath:lib error:NULL].fileModificationDate;
    NSDate *src_date = newest_source_date(shader_dir);
    if (!lib_date || !src_date)
        return false;
    return [src_date compare:lib_date] == NSOrderedDescending;
}

static id<MTLLibrary> load_shader_library(id<MTLDevice> device)
{
    bool forced = false;
    NSString *shader_dir = find_shader_sources(&forced);
    if (forced || metallib_is_stale(shader_dir)) {
        if (!forced)
            Com_Printf("Metal: q2rtx.metallib is older than the shader sources, compiling them instead\n");
        id<MTLLibrary> lib = compile_shader_sources(device, shader_dir.UTF8String);
        if (lib)
            return lib;
    }

    void *data = NULL;
    int size = FS_LoadFile("shader_mtlpt/q2rtx.metallib", &data);

    if (size > 0 && data) {
        // DISPATCH_DATA_DESTRUCTOR_DEFAULT copies the bytes, so the engine's
        // copy can be released immediately afterwards.
        dispatch_data_t blob = dispatch_data_create(data, (size_t)size,
                                                    dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
                                                    DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        NSError *err = nil;
        id<MTLLibrary> lib = [device newLibraryWithData:blob error:&err];
        dispatch_release(blob);
        FS_FreeFile(data);

        if (lib)
            return lib;

        mtl_log_error("newLibraryWithData", err);
    } else if (data) {
        FS_FreeFile(data);
    }

    // Fall back to a metallib sitting next to the executable, which is how an
    // uninstalled development build is laid out.
    NSString *dir = [[NSBundle mainBundle] bundlePath];
    NSString *path = [dir stringByAppendingPathComponent:@"q2rtx.metallib"];
    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
    if (lib)
        return lib;

    mtl_log_error("newLibraryWithURL", err);
    if (shader_dir && !forced)
        return compile_shader_sources(device, shader_dir.UTF8String);
    return nil;
}

bool mtl_device_init(void)
{
    // vkpt's cvars, which the video menu sets. vid_vsync applies on the fly
    // (R_BeginFrame); a vid_hdr change restarts the renderer (CVAR_REFRESH)
    // since the drawable format and every pipeline drawing to it change.
    mtl_vsync = Cvar_Get("vid_vsync", "0", CVAR_ARCHIVE);
    mtl_hdr   = Cvar_Get("vid_hdr", "0", CVAR_ARCHIVE | CVAR_REFRESH);

    mtl.device = MTLCreateSystemDefaultDevice();
    if (!mtl.device) {
        Com_EPrintf("Metal: no compatible device found\n");
        return false;
    }

    mtl.queue = [mtl.device newCommandQueue];
    if (!mtl.queue) {
        Com_EPrintf("Metal: could not create a command queue\n");
        return false;
    }
    mtl.queue.label = @"q2rtx";

    CAMetalLayer *layer = (CAMetalLayer *)get_metal_layer();
    if (!layer) {
        Com_EPrintf("Metal: SDL did not provide a CAMetalLayer\n");
        return false;
    }

    mtl.layer = [layer retain];
    mtl.layer.device = mtl.device;
    mtl.layer.framebufferOnly = NO;     // the tone mapping pass writes it directly
    mtl.layer.displaySyncEnabled = mtl_vsync->integer != 0;
    // Two drawables like vkpt's two image swapchain; three queue another
    // frame for display and add ~16-30 ms of input lag.
    mtl.layer.maximumDrawableCount = 2;

    // HDR output through EDR, when asked for. Like vkpt, which turns vid_hdr
    // off when the surface has no HDR format, it needs a display with
    // headroom above SDR white (an HDR monitor with HDR enabled in System
    // Settings, or an XDR display).
    mtl.is_hdr = mtl_hdr->integer != 0;
    if (mtl.is_hdr) {
        CGFloat headroom = [NSScreen mainScreen].maximumPotentialExtendedDynamicRangeColorComponentValue;
        if (headroom <= 1.0) {
            Com_WPrintf("Metal: the display does not support HDR (EDR), disabling vid_hdr\n");
            Cvar_SetByVar(mtl_hdr, "0", FROM_CODE);
            mtl.is_hdr = false;
        } else {
            Com_Printf("Metal: HDR output, display headroom %.1fx SDR white\n", headroom);
        }
    }
    if (mtl.is_hdr) {
        mtl.drawable_format = MTLPixelFormatRGBA16Float;
        mtl.layer.wantsExtendedDynamicRangeContent = YES;
        mtl.layer.colorspace = CGColorSpaceCreateWithName(kCGColorSpaceExtendedLinearSRGB);
    } else {
        mtl.drawable_format = MTLPixelFormatBGRA8Unorm_sRGB;
        mtl.layer.wantsExtendedDynamicRangeContent = NO;
    }
    mtl.layer.pixelFormat = mtl.drawable_format;

    mtl.supports_raytracing = [mtl.device supportsRaytracing];

    mtl.library = load_shader_library(mtl.device);
    if (!mtl.library)
        return false;
    mtl.library.label = @"q2rtx";

    mtl.frame_sem = dispatch_semaphore_create(MTL_FRAMES_IN_FLIGHT);
    mtl.frame_index = 0;
    mtl.frame_counter = 0;

    CGSize size = mtl.layer.drawableSize;
    mtl.width = (int)size.width;
    mtl.height = (int)size.height;

    Com_Printf("Metal device: %s\n", mtl_device_name());
    Com_Printf("...unified memory: %s\n", mtl.device.hasUnifiedMemory ? "yes" : "no");
    Com_Printf("...ray tracing: %s\n", mtl.supports_raytracing ? "supported" : "NOT SUPPORTED");

    mtl.initialized = true;
    return true;
}

void mtl_device_shutdown(void)
{
    if (mtl.frame_sem) {
        // Drain in-flight frames so nothing references objects we release.
        for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++)
            dispatch_semaphore_wait(mtl.frame_sem, DISPATCH_TIME_FOREVER);
        // libdispatch traps (Trace/BPT trap on quit) when a semaphore is
        // released with a lower count than it was created with.
        for (int i = 0; i < MTL_FRAMES_IN_FLIGHT; i++)
            dispatch_semaphore_signal(mtl.frame_sem);
        dispatch_release(mtl.frame_sem);
        mtl.frame_sem = NULL;
    }

    [mtl.library release];
    [source_libraries release];
    source_libraries = nil;
    [mtl.queue release];
    [mtl.layer release];

    // MTLCreateSystemDefaultDevice returns a +1 reference.
    [mtl.device release];

    memset(&mtl, 0, sizeof(mtl));
}

void mtl_device_mode_changed(int width, int height)
{
    if (width <= 0 || height <= 0)
        return;

    mtl.width = width;
    mtl.height = height;

    if (mtl.layer)
        mtl.layer.drawableSize = CGSizeMake(width, height);
}

bool mtl_device_supports_raytracing(void)
{
    return mtl.supports_raytracing;
}

const char *mtl_device_name(void)
{
    if (!mtl.device)
        return "none";
    return [mtl.device.name UTF8String];
}
