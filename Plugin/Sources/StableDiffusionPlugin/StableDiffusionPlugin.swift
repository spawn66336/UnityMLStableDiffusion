import CoreML
import Accelerate
import StableDiffusion

final class Plugin {

    var pipeline: StableDiffusionPipeline

    var config: StableDiffusionPipeline.Configuration

    var generated: CGImage?

    // Create a pipeline and load resources into it
    init(resourcePath: String, computeUnits: MLComputeUnits) throws {
        // Compute unit selection (all)
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits

        // Pipeline initialization
        let resourceURL = URL(filePath: resourcePath)
        pipeline = try StableDiffusionPipeline(
            resourcesAt: resourceURL,
            configuration: mlConfig,
            disableSafety: true,
            reduceMemory: false)
        try pipeline.loadResources()

        // Initial configuration
        config = StableDiffusionPipeline.Configuration(prompt: "")
        config.schedulerType = StableDiffusionScheduler.dpmSolverMultistepScheduler
    }

    // Run the pipeline and generate an image
    func runGenerator() throws {
        let images = try pipeline.generateImages(configuration: config)
        generated = images[0]!
    }
}

@_cdecl("SDCreate")
public func SDCreate(resourcePath: OpaquePointer, units: CInt) -> OpaquePointer! {
    let resourcePath = String(cString: UnsafePointer<CChar>(resourcePath))
    let units = MLComputeUnits(rawValue: Int(units))!
    //创建plugin对象
    if let plugin = try? Plugin(resourcePath: resourcePath, computeUnits: units) {
        //调用passRetained手动延长plugin对象引用计数，并返回透明指针
        return OpaquePointer(Unmanaged.passRetained(plugin).toOpaque())
    }
    return nil;
}

@_cdecl("SDDestroy")
public func SDDestroy(_ plugin: OpaquePointer) {
    //调用takeRetainedValue手动降低plugin对象引用技术，试图释放对象。与SDCreate中的调用相反
    _ = Unmanaged<Plugin>.fromOpaque(UnsafeRawPointer(plugin)).takeRetainedValue()
}


@_cdecl("SDSetConfig")
public func SDSetConfig(
    _ plugin: OpaquePointer,
    prompt: OpaquePointer,
    stepCount: CInt,
    seed: CInt,
    guidanceScale: CFloat
) {
    //在过程中使用plugin对象时，使用takeUnretainedValue来获取指针值，无需对对象引用计数进行修改
    let plugin = Unmanaged<Plugin>.fromOpaque(UnsafeRawPointer(plugin)).takeUnretainedValue()
    plugin.config.prompt = String(cString: UnsafePointer<CChar>(prompt))
    plugin.config.stepCount = Int(stepCount)
    plugin.config.seed = UInt32(seed)
    plugin.config.guidanceScale = guidanceScale
}

@_cdecl("SDGenerate")
public func SDGenerate(_ plugin: OpaquePointer) {
    //在过程中使用plugin对象时，使用takeUnretainedValue来获取指针值，无需对对象引用计数进行修改
    let plugin = Unmanaged<Plugin>.fromOpaque(UnsafeRawPointer(plugin)).takeUnretainedValue()
    try? plugin.runGenerator()
}

@_cdecl("SDGenerateFromImage")
public func SDGenerateFromImage(
    _ plugin: OpaquePointer,
    image: OpaquePointer,
    strength: CFloat
) {
    //在过程中使用plugin对象时，使用takeUnretainedValue来获取指针值，无需对对象引用计数进行修改
    let plugin = Unmanaged<Plugin>.fromOpaque(UnsafeRawPointer(plugin)).takeUnretainedValue()
    let pointer = UnsafeMutableRawPointer(image)

    // Raw pointer to CGImage object
    let buffer = vImage.PixelBuffer<vImage.Interleaved8x3>(
        data: pointer,
        width: 512,
        height: 512,
        byteCountPerRow: 512 * 3)
    let format = vImage_CGImageFormat(
        bitsPerComponent: 8,
        bitsPerPixel: 3 * 8,
        colorSpace: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue))!
    let cgImage = buffer.makeCGImage(cgImageFormat: format)!

    // img2img run
    plugin.config.startingImage = cgImage
    plugin.config.strength = strength
    try? plugin.runGenerator()
}

@_cdecl("SDGetImage")
public func SDGetImage(_ plugin: OpaquePointer) -> OpaquePointer! {
    //在过程中使用plugin对象时，使用takeUnretainedValue来获取指针值，无需对对象引用计数进行修改
    let plugin = Unmanaged<Plugin>.fromOpaque(UnsafeRawPointer(plugin)).takeUnretainedValue()
    return OpaquePointer(CFDataGetBytePtr(plugin.generated!.dataProvider!.data))
}

