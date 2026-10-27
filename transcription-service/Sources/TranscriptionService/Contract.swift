// The wire contract: what `POST /v1/transcriptions` returns, what every error
// looks like, and what `GET /health` reports. `src/jake_tools/transcription.py`
// mirrors these types; change both together and bump `schemaVersion` when a
// change is not backwards compatible.
//
// Timestamps are seconds (Double, three decimals) measured from the first
// sample of the concatenated recording. The engine's timing resolution is one
// encoder frame, 80 ms, so no word timing is finer than that.

import Foundation

public let contractSchemaVersion = 1
public let serviceVersion = "0.1.0"

/// One encoder frame of Parakeet TDT: the finest timing the engine produces.
public let timingResolutionSeconds = 0.08

public let timestampOriginDescription =
    "first sample of the concatenated recording (files joined in request order)"

/// Which engine and models produced a transcript.
public struct EngineInfo: Codable, Sendable, Equatable {
    public var name: String
    public var version: String
    public var asrModel: String
    public var diarizationModel: String

    public init(name: String, version: String, asrModel: String, diarizationModel: String) {
        self.name = name
        self.version = version
        self.asrModel = asrModel
        self.diarizationModel = diarizationModel
    }
}

/// Where one uploaded file sits on the concatenated timeline.
public struct FileSpan: Codable, Sendable, Equatable {
    public var index: Int
    public var name: String
    public var startSeconds: Double
    public var endSeconds: Double
    public var durationSeconds: Double

    public init(index: Int, name: String, startSeconds: Double, endSeconds: Double) {
        self.index = index
        self.name = name
        self.startSeconds = startSeconds.rounded3
        self.endSeconds = endSeconds.rounded3
        self.durationSeconds = (endSeconds - startSeconds).rounded3
    }
}

/// A recognised word with its timing and the speaker it was reconciled to.
///
/// `speaker` is `nil` when no diarised speaker turn covers or is near the
/// word. `overlap` is `true` when more than one speaker turn covers it: the
/// word is attributed to the turn with the largest overlap, which may be wrong.
public struct Word: Codable, Sendable, Equatable {
    public var text: String
    public var startSeconds: Double
    public var endSeconds: Double
    public var confidence: Double
    public var speaker: String?
    public var overlap: Bool

    public init(
        text: String, startSeconds: Double, endSeconds: Double, confidence: Double,
        speaker: String?, overlap: Bool
    ) {
        self.text = text
        self.startSeconds = startSeconds.rounded3
        self.endSeconds = endSeconds.rounded3
        self.confidence = confidence.rounded3
        self.speaker = speaker
        self.overlap = overlap
    }

    enum CodingKeys: String, CodingKey {
        case text, startSeconds, endSeconds, confidence, speaker, overlap
    }

    // Synthesised encoding omits a nil optional; the contract says `speaker`
    // is always present, null when unassigned.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(text, forKey: .text)
        try container.encode(startSeconds, forKey: .startSeconds)
        try container.encode(endSeconds, forKey: .endSeconds)
        try container.encode(confidence, forKey: .confidence)
        try container.encode(speaker, forKey: .speaker)
        try container.encode(overlap, forKey: .overlap)
    }
}

/// A run of consecutive words attributed to one speaker.
public struct Segment: Codable, Sendable, Equatable {
    public var speaker: String?
    public var startSeconds: Double
    public var endSeconds: Double
    public var text: String
    public var words: [Word]

    public init(speaker: String?, startSeconds: Double, endSeconds: Double, text: String, words: [Word]) {
        self.speaker = speaker
        self.startSeconds = startSeconds.rounded3
        self.endSeconds = endSeconds.rounded3
        self.text = text
        self.words = words
    }

    enum CodingKeys: String, CodingKey {
        case speaker, startSeconds, endSeconds, text, words
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(speaker, forKey: .speaker)
        try container.encode(startSeconds, forKey: .startSeconds)
        try container.encode(endSeconds, forKey: .endSeconds)
        try container.encode(text, forKey: .text)
        try container.encode(words, forKey: .words)
    }
}

/// One anonymous speaker identity within a request.
public struct SpeakerSummary: Codable, Sendable, Equatable {
    public var id: String
    public var speakingSeconds: Double
    public var segmentCount: Int

    public init(id: String, speakingSeconds: Double, segmentCount: Int) {
        self.id = id
        self.speakingSeconds = speakingSeconds.rounded3
        self.segmentCount = segmentCount
    }
}

public struct ProcessingTimings: Codable, Sendable, Equatable {
    public var decodeSeconds: Double
    public var asrSeconds: Double
    public var diarizationSeconds: Double
    public var totalSeconds: Double

    public init(decodeSeconds: Double, asrSeconds: Double, diarizationSeconds: Double, totalSeconds: Double) {
        self.decodeSeconds = decodeSeconds.rounded3
        self.asrSeconds = asrSeconds.rounded3
        self.diarizationSeconds = diarizationSeconds.rounded3
        self.totalSeconds = totalSeconds.rounded3
    }
}

/// The complete result of one request. There is no partial form: a request
/// that cannot be fully transcribed fails instead.
public struct Transcript: Codable, Sendable, Equatable {
    public var schemaVersion: Int
    public var engine: EngineInfo
    public var timestampUnit: String
    public var timestampOrigin: String
    public var timingResolutionSeconds: Double
    public var durationSeconds: Double
    public var files: [FileSpan]
    public var speakers: [SpeakerSummary]
    public var segments: [Segment]
    public var text: String
    public var unassignedWordCount: Int
    public var limitations: [String]
    public var processing: ProcessingTimings

    public init(
        engine: EngineInfo,
        durationSeconds: Double,
        files: [FileSpan],
        speakers: [SpeakerSummary],
        segments: [Segment],
        text: String,
        unassignedWordCount: Int,
        limitations: [String],
        processing: ProcessingTimings
    ) {
        self.schemaVersion = contractSchemaVersion
        self.engine = engine
        self.timestampUnit = "seconds"
        self.timestampOrigin = timestampOriginDescription
        self.timingResolutionSeconds = TranscriptionService.timingResolutionSeconds
        self.durationSeconds = durationSeconds.rounded3
        self.files = files
        self.speakers = speakers
        self.segments = segments
        self.text = text
        self.unassignedWordCount = unassignedWordCount
        self.limitations = limitations
        self.processing = processing
    }
}

/// The body of every non-2xx response.
public struct ErrorEnvelope: Codable, Sendable, Equatable {
    public struct Body: Codable, Sendable, Equatable {
        public var code: String
        public var message: String

        public init(code: String, message: String) {
            self.code = code
            self.message = message
        }
    }

    public var error: Body

    public init(code: String, message: String) {
        self.error = Body(code: code, message: message)
    }
}

/// `GET /health`. `ready` is the only field a supervisor needs.
public struct HealthReport: Codable, Sendable, Equatable {
    public struct Queue: Codable, Sendable, Equatable {
        public var running: Int
        public var queued: Int
        public var maxQueued: Int

        public init(running: Int, queued: Int, maxQueued: Int) {
            self.running = running
            self.queued = queued
            self.maxQueued = maxQueued
        }
    }

    /// `starting`, `ready` or `failed`.
    public var status: String
    public var ready: Bool
    public var version: String
    public var engine: EngineInfo?
    public var detail: String?
    public var queue: Queue

    public init(status: String, ready: Bool, version: String, engine: EngineInfo?, detail: String?, queue: Queue) {
        self.status = status
        self.ready = ready
        self.version = version
        self.engine = engine
        self.detail = detail
        self.queue = queue
    }
}

/// JSON settings shared by the server and the tests: snake_case keys, stable
/// key order.
public enum Wire {
    public static func encoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }

    public static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }
}

extension Double {
    /// Millisecond precision is already coarser than the engine's 80 ms frames.
    var rounded3: Double { (self * 1000).rounded() / 1000 }
}
