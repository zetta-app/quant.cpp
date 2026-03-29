/**
 * TurboQuant — Metal backend dispatch (Objective-C host code)
 *
 * Loads the .metallib shader library, creates compute pipelines,
 * and provides the dispatch interface for Metal GPU kernels.
 */
#ifdef __APPLE__

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

/* Pipeline cache */
static id<MTLDevice>       tq_mtl_device    = nil;
static id<MTLCommandQueue> tq_mtl_queue     = nil;
static id<MTLLibrary>      tq_mtl_library   = nil;

/* Cached pipelines */
static id<MTLComputePipelineState> tq_pipe_polar_quantize  = nil;
static id<MTLComputePipelineState> tq_pipe_polar_attention  = nil;
static id<MTLComputePipelineState> tq_pipe_qjl_quantize    = nil;
static id<MTLComputePipelineState> tq_pipe_qjl_attention   = nil;
static id<MTLComputePipelineState> tq_pipe_value_quantize  = nil;

/**
 * Initialize Metal backend.
 * Returns 0 on success, -1 on failure.
 */
int tq_init_metal_backend(void) {
    @autoreleasepool {
        /* Get default Metal device */
        tq_mtl_device = MTLCreateSystemDefaultDevice();
        if (!tq_mtl_device) {
            NSLog(@"TurboQuant: No Metal device found");
            return -1;
        }

        /* Create command queue */
        tq_mtl_queue = [tq_mtl_device newCommandQueue];
        if (!tq_mtl_queue) {
            NSLog(@"TurboQuant: Failed to create command queue");
            return -1;
        }

        /* Load shader library from default.metallib or bundled path */
        NSError *error = nil;
        NSString *libPath = [[NSBundle mainBundle] pathForResource:@"turboquant"
                                                           ofType:@"metallib"];
        if (libPath) {
            tq_mtl_library = [tq_mtl_device newLibraryWithFile:libPath error:&error];
        } else {
            /* Try default library */
            tq_mtl_library = [tq_mtl_device newDefaultLibrary];
        }

        if (!tq_mtl_library) {
            NSLog(@"TurboQuant: Failed to load Metal library: %@", error);
            return -1;
        }

        /* Create compute pipelines */
        NSArray *kernelNames = @[
            @"tq_polar_quantize",
            @"tq_polar_attention",
            @"tq_qjl_quantize",
            @"tq_qjl_attention",
            @"tq_value_quantize_4b"
        ];

        id<MTLComputePipelineState> *pipes[] = {
            &tq_pipe_polar_quantize,
            &tq_pipe_polar_attention,
            &tq_pipe_qjl_quantize,
            &tq_pipe_qjl_attention,
            &tq_pipe_value_quantize
        };

        for (NSUInteger i = 0; i < kernelNames.count; i++) {
            id<MTLFunction> func = [tq_mtl_library newFunctionWithName:kernelNames[i]];
            if (func) {
                *pipes[i] = [tq_mtl_device newComputePipelineStateWithFunction:func
                                                                         error:&error];
                if (error) {
                    NSLog(@"TurboQuant: Pipeline error for %@: %@", kernelNames[i], error);
                }
            }
        }

        NSLog(@"TurboQuant: Metal backend initialized on %@", tq_mtl_device.name);
        return 0;
    }
}

/**
 * Free Metal resources.
 */
void tq_free_metal_backend(void) {
    tq_pipe_polar_quantize = nil;
    tq_pipe_polar_attention = nil;
    tq_pipe_qjl_quantize = nil;
    tq_pipe_qjl_attention = nil;
    tq_pipe_value_quantize = nil;
    tq_mtl_library = nil;
    tq_mtl_queue = nil;
    tq_mtl_device = nil;
}

/**
 * Get Metal device name.
 */
const char* tq_metal_device_name(void) {
    if (!tq_mtl_device) return "not initialized";
    return [[tq_mtl_device name] UTF8String];
}

#endif /* __APPLE__ */
