// Decoding uploaded files into one 16 kHz mono recording.
//
// Each file goes through FluidAudio's `AudioConverter` (AVFoundation
// underneath), so anything `AVAudioFile` reads is accepted: WAV, AIFF, CAF,
// M4A/AAC, MP3, FLAC. The outputs are concatenated in request order with no
// gap, which is what gives the transcript its single timeline.

import FluidAudio
import Foundation

public struct DecodedRecording: Sendable {
    public static let sampleRate = 16_000

    /// 16 kHz mono Float32 samples of every file, back to back.
    public var samples: [Float]
    public var files: [FileSpan]

    public var durationSeconds: Double { Double(samples.count) / Double(Self.sampleRate) }

    public init(samples: [Float], files: [FileSpan]) {
        self.samples = samples
        self.files = files
    }
}

public enum AudioDecoder {
    /// Decode and concatenate `uploads`. Rejects a file that is not decodable
    /// audio or decodes to nothing, and a total longer than
    /// `maxDurationSeconds`, as `ServiceError`s.
    public static func decode(_ uploads: [UploadedFile], maxDurationSeconds: Double) throws -> DecodedRecording {
        let converter = AudioConverter()
        var samples: [Float] = []
        var spans: [FileSpan] = []
        let maxSamples = Int(maxDurationSeconds * Double(DecodedRecording.sampleRate))
        for upload in uploads {
            try Task.checkCancellation()
            let decoded: [Float]
            do {
                decoded = try converter.resampleAudioFile(upload.url)
            } catch {
                throw ServiceError(
                    .invalidInput,
                    "file \(upload.index + 1) (\(upload.name)) could not be decoded as audio")
            }
            guard !decoded.isEmpty else {
                throw ServiceError(
                    .invalidInput, "file \(upload.index + 1) (\(upload.name)) contains no audio")
            }
            guard samples.count + decoded.count <= maxSamples else {
                throw ServiceError(
                    .payloadTooLarge,
                    "the recording exceeds the limit of \(Int(maxDurationSeconds)) seconds of audio")
            }
            let start = Double(samples.count) / Double(DecodedRecording.sampleRate)
            samples.append(contentsOf: decoded)
            let end = Double(samples.count) / Double(DecodedRecording.sampleRate)
            spans.append(FileSpan(index: upload.index, name: upload.name, startSeconds: start, endSeconds: end))
        }
        return DecodedRecording(samples: samples, files: spans)
    }
}
