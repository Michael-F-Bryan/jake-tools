// The inference engine behind the service: FluidAudio's Parakeet TDT v3 for
// speech recognition and its offline pyannote community-1 pipeline (powerset
// segmentation, WeSpeaker embeddings, VBx clustering) for diarisation.
//
// Models are loaded once, in offline mode, and shared by every request for
// the life of the process. `provision` is the only code path that downloads.

import CoreML
import FluidAudio
import Foundation
import Logging

/// What the engine produces for one recording, before reconciliation.
public struct EngineOutput: Sendable {
    public var text: String
    public var words: [TimedWord]
    public var turns: [SpeakerTurn]
    public var asrSeconds: Double
    public var diarizationSeconds: Double

    public init(text: String, words: [TimedWord], turns: [SpeakerTurn], asrSeconds: Double, diarizationSeconds: Double) {
        self.text = text
        self.words = words
        self.turns = turns
        self.asrSeconds = asrSeconds
        self.diarizationSeconds = diarizationSeconds
    }
}

/// The seam the server depends on. `FluidAudioEngine` is the one production
/// implementation; the tests substitute a stub to exercise the HTTP layer.
public protocol TranscriptionEngine: Sendable {
    var info: EngineInfo { get }
    /// Called by at most one task at a time (see `JobGate`). Must honour
    /// task cancellation where the engine supports it.
    func transcribe(_ recording: DecodedRecording) async throws -> EngineOutput
}

public enum ModelPins {
    public static let fluidAudioVersion = "0.17.7"

    public static let asrVersion = AsrModelVersion.v3
    public static let asrRepo = Repo.parakeetV3
    /// Commit of `FluidInference/parakeet-tdt-0.6b-v3-coreml` on 2026-08-19.
    /// FluidAudio itself follows `main` for this repo; the service pins it so
    /// a provisioned host is reproducible.
    public static let asrRevision = "7dd20fe6b1797d35f5e3307e8b1732d9a178edfe"

    public static let diarizerRepo = Repo.diarizer
    /// FluidAudio pins the diarizer repo itself (`Repo.diarizer.revision`).
    public static var diarizerRevision: String { diarizerRepo.revision }

    public static var engineInfo: EngineInfo {
        EngineInfo(
            name: "FluidAudio",
            version: fluidAudioVersion,
            asrModel: "\(asrRepo.rawValue)@\(asrRevision)",
            diarizationModel: "\(diarizerRepo.rawValue)@\(diarizerRevision)")
    }

    /// Where the Parakeet bundle lives under a models root. The diarizer is
    /// addressed by the root itself (FluidAudio appends its own folder).
    public static func asrDirectory(modelsRoot: URL) -> URL {
        modelsRoot.appendingPathComponent(asrRepo.folderName, isDirectory: true)
    }

    /// FluidAudio's own cache: `~/Library/Application Support/FluidAudio/Models`.
    public static var defaultModelsRoot: URL { OfflineDiarizerModels.defaultModelsDirectory() }

    static func applyRevisionPins() {
        var overrides = ModelRegistry.revisionOverrides
        overrides[asrRepo.remotePath] = asrRevision
        ModelRegistry.revisionOverrides = overrides
    }
}

public let engineLimitations: [String] = [
    "Word timings are quantised to the engine's 80 ms encoder frames; sub-frame precision is not available.",
    "Overlapping speech is diarised as separate turns, but each recognised word is attributed to one speaker (the turn overlapping it most) and flagged `overlap`; simultaneous-speaker accuracy has not been measured.",
    "Speaker labels are anonymous and request-scoped: speaker_1 in one request is unrelated to speaker_1 in another.",
    "`text` is the recogniser's full transcript; segment text is rebuilt from timed words and can differ in spacing or punctuation placement.",
    "Recognition is Parakeet TDT v3 (25 European languages, no language hint); diarisation is pyannote community-1 with VBx clustering. Neither was tuned for this deployment.",
]

/// The production engine. Marked `@unchecked Sendable` because
/// `OfflineDiarizerManager` is not `Sendable`; `JobGate` guarantees one
/// `transcribe` at a time, and nothing mutates the managers after `load`.
public final class FluidAudioEngine: TranscriptionEngine, @unchecked Sendable {
    public let info: EngineInfo
    private let asr: AsrManager
    private let asrModels: AsrModels
    private let diarizer: OfflineDiarizerManager
    private let logger: Logger

    private init(asr: AsrManager, asrModels: AsrModels, diarizer: OfflineDiarizerManager, logger: Logger) {
        self.info = ModelPins.engineInfo
        self.asr = asr
        self.asrModels = asrModels
        self.diarizer = diarizer
        self.logger = logger
    }

    /// Download and compile the pinned models under `modelsRoot`, then load
    /// them offline to prove the directory is complete. Needs the network.
    public static func provision(modelsRoot: URL, logger: Logger) async throws {
        ModelHub.offlineMode = false
        ModelPins.applyRevisionPins()
        try FileManager.default.createDirectory(at: modelsRoot, withIntermediateDirectories: true)

        logger.info("downloading ASR models", metadata: ["repo": "\(ModelPins.asrRepo.rawValue)"])
        _ = try await AsrModels.download(
            to: ModelPins.asrDirectory(modelsRoot: modelsRoot), version: ModelPins.asrVersion)

        logger.info("downloading diarization models", metadata: ["repo": "\(ModelPins.diarizerRepo.rawValue)"])
        try await ModelHub.download(ModelPins.diarizerRepo, to: modelsRoot, variant: "offline")

        logger.info("verifying provisioned models load offline")
        _ = try await load(modelsRoot: modelsRoot, logger: logger, warmUp: false)
        logger.info("models provisioned", metadata: ["models_root": "\(modelsRoot.path)"])
    }

    /// Load the pinned models from `modelsRoot` without touching the network,
    /// then run a short warm-up so the first real request does not pay
    /// Core ML's first-prediction cost.
    public static func load(modelsRoot: URL, logger: Logger, warmUp: Bool = true) async throws -> FluidAudioEngine {
        ModelHub.offlineMode = true
        ModelPins.applyRevisionPins()
        let clock = ContinuousClock()

        let asrStart = clock.now
        let asrModels = try await AsrModels.load(
            from: ModelPins.asrDirectory(modelsRoot: modelsRoot), version: ModelPins.asrVersion)
        let asr = AsrManager(config: .default)
        try await asr.loadModels(asrModels)
        logger.info("ASR models loaded", metadata: ["seconds": "\(seconds(since: asrStart, clock: clock))"])

        let diarizerStart = clock.now
        let diarizerModels = try await OfflineDiarizerModels.load(from: modelsRoot)
        let diarizer = OfflineDiarizerManager(config: .default)
        diarizer.initialize(models: diarizerModels)
        logger.info(
            "diarization models loaded",
            metadata: ["seconds": "\(seconds(since: diarizerStart, clock: clock))"])

        let engine = FluidAudioEngine(asr: asr, asrModels: asrModels, diarizer: diarizer, logger: logger)
        if warmUp {
            let warmStart = clock.now
            try await engine.warmUp()
            logger.info("engine warm", metadata: ["seconds": "\(seconds(since: warmStart, clock: clock))"])
        }
        return engine
    }

    public func transcribe(_ recording: DecodedRecording) async throws -> EngineOutput {
        let clock = ContinuousClock()

        let asrStart = clock.now
        var decoderState = try TdtDecoderState(decoderLayers: asrModels.version.decoderLayers)
        let asrResult = try await asr.transcribe(recording.samples, decoderState: &decoderState)
        let asrSeconds = Self.seconds(since: asrStart, clock: clock)
        try Task.checkCancellation()

        let diarizationStart = clock.now
        let turns: [SpeakerTurn]
        do {
            let diarization = try await diarizer.process(audio: recording.samples)
            turns = diarization.segments.map {
                SpeakerTurn(
                    speaker: $0.speakerId, start: Double($0.startTimeSeconds), end: Double($0.endTimeSeconds))
            }
        } catch OfflineDiarizationError.noSpeechDetected {
            // Not a failure: a recording without detectable speech (or the
            // warm-up signal) has no speaker turns. Any recognised words
            // then stay unassigned, which the transcript reports honestly.
            turns = []
        }
        let diarizationSeconds = Self.seconds(since: diarizationStart, clock: clock)

        let words = Reconcile.words(fromTokens: asrResult.tokenTimings ?? [])
        return EngineOutput(
            text: asrResult.text,
            words: words,
            turns: turns,
            asrSeconds: asrSeconds,
            diarizationSeconds: diarizationSeconds)
    }

    /// Ten seconds of faint noise through both pipelines: Core ML's first
    /// prediction per model is paid here, not by the first caller. The
    /// diarizer finds no speech in it, which `transcribe` treats as no turns,
    /// so only a model that cannot run on this host makes this throw, and
    /// then the service must not report ready.
    private func warmUp() async throws {
        var generator = SystemRandomNumberGenerator()
        let samples = (0..<(DecodedRecording.sampleRate * 10)).map { _ in
            Float.random(in: -0.001...0.001, using: &generator)
        }
        _ = try await transcribe(DecodedRecording(samples: samples, files: []))
    }

    private static func seconds(since start: ContinuousClock.Instant, clock: ContinuousClock) -> Double {
        let duration = clock.now - start
        let (seconds, attoseconds) = duration.components
        return Double(seconds) + Double(attoseconds) / 1e18
    }
}
