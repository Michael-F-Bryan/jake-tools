// swift-tools-version: 6.2
//
// The native transcription service: a long-running macOS HTTP service that
// keeps FluidAudio's ASR and diarisation models warm and turns uploaded
// recordings into one structured transcript. The Python MCP tool in
// `src/jake_tools/transcription.py` is its only intended client.
//
// Every dependency is pinned exactly; bump them deliberately and re-run the
// real-service checks in ../tests/test_transcription_live.py.
import PackageDescription

let package = Package(
    name: "transcription-service",
    // macOS 15: FluidAudio's offline diarizer documents a BNNS crash on
    // macOS 14 (FluidAudio #878), and `Synchronization.Mutex` needs 15.
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "transcription-service", targets: ["transcription-service"]),
        .library(name: "TranscriptionService", targets: ["TranscriptionService"]),
    ],
    dependencies: [
        // The inference engine. At 0.17.7 the only trait is the NeMo
        // text-normalisation binary (TTS/ITN only), which `traits: []` drops;
        // the `Diarizer`/`TTS` traits arrived after this release, so switch to
        // `traits: ["Diarizer"]` when bumping past it.
        .package(
            url: "https://github.com/FluidInference/FluidAudio.git", exact: "0.17.7",
            traits: []),
        // HTTP server. `FullFoundation` only; the swift-configuration reader is
        // not used (flags and environment configure the service).
        .package(
            url: "https://github.com/hummingbird-project/hummingbird.git", exact: "2.28.0",
            traits: ["FullFoundation"]),
        // Streaming multipart parsing (parts are written to disk as they
        // arrive). 4.x has no streaming parser; 5.0.0-beta.2 is the pinned
        // pre-release that does.
        .package(url: "https://github.com/vapor/multipart-kit.git", exact: "5.0.0-beta.2"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", exact: "1.8.2"),
        .package(url: "https://github.com/apple/swift-log.git", exact: "1.16.2"),
        .package(url: "https://github.com/apple/swift-nio-ssl.git", exact: "2.37.5"),
    ],
    targets: [
        // Turns AVFoundation's NSExceptions into Swift errors (see its header).
        .target(name: "ObjCExceptionGuard"),
        .target(
            name: "TranscriptionService",
            dependencies: [
                "ObjCExceptionGuard",
                .product(name: "FluidAudio", package: "FluidAudio"),
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "HummingbirdCore", package: "hummingbird"),
                .product(name: "HummingbirdTLS", package: "hummingbird"),
                .product(name: "MultipartKit", package: "multipart-kit"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "NIOSSL", package: "swift-nio-ssl"),
            ]
        ),
        .executableTarget(
            name: "transcription-service",
            dependencies: [
                "TranscriptionService",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Logging", package: "swift-log"),
            ]
        ),
        .testTarget(
            name: "TranscriptionServiceTests",
            dependencies: ["TranscriptionService"]
        ),
    ]
)
