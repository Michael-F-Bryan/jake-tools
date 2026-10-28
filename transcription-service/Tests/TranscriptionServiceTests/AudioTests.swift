// The decoder against real files AVFoundation writes: formats it must accept
// (Opus in M4A crashed the old decoder), and damaged or truncated files it
// must refuse rather than return as a shorter recording.

import AVFoundation
import Foundation
import Testing

@testable import TranscriptionService

/// `seconds` of a 440 Hz tone at 48 kHz mono.
func tone(seconds: Double, sampleRate: Double = 48_000) -> AVAudioPCMBuffer {
    let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
    let frames = AVAudioFrameCount(seconds * sampleRate)
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)!
    buffer.frameLength = frames
    for frame in 0..<Int(frames) {
        buffer.floatChannelData![0][frame] = Float(sin(2 * Double.pi * 440 * Double(frame) / sampleRate) * 0.3)
    }
    return buffer
}

/// Encode `seconds` of tone with AVFoundation's own encoder, as a client's
/// recorder would.
func makeEncoded(seconds: Double, format: AudioFormatID, fileExtension: String) throws -> URL {
    let url = try temporaryDirectory().appendingPathComponent("tone.\(fileExtension)")
    let buffer = tone(seconds: seconds)
    let file = try AVAudioFile(
        forWriting: url,
        settings: [AVFormatIDKey: format, AVSampleRateKey: 48_000, AVNumberOfChannelsKey: 1])
    try file.write(from: buffer)
    return url
}

/// Opus in an MPEG-4 container, the format of most of Michael's recordings.
/// `AVAudioFile` cannot write it; `AVAssetWriter` can, as `.mp4`.
func makeOpusMP4(seconds: Double) async throws -> URL {
    let url = try temporaryDirectory().appendingPathComponent("opus.m4a")
    let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let input = AVAssetWriterInput(
        mediaType: .audio,
        outputSettings: [
            AVFormatIDKey: kAudioFormatOpus, AVSampleRateKey: 48_000, AVNumberOfChannelsKey: 1,
            AVEncoderBitRateKey: 32_000,
        ])
    input.expectsMediaDataInRealTime = false
    writer.add(input)
    #expect(writer.startWriting())
    writer.startSession(atSourceTime: .zero)

    let buffer = tone(seconds: seconds)
    var description: CMAudioFormatDescription?
    CMAudioFormatDescriptionCreate(
        allocator: nil, asbd: buffer.format.streamDescription, layoutSize: 0, layout: nil,
        magicCookieSize: 0, magicCookie: nil, extensions: nil, formatDescriptionOut: &description)
    var sample: CMSampleBuffer?
    var timing = CMSampleTimingInfo(
        duration: CMTime(value: 1, timescale: 48_000), presentationTimeStamp: .zero, decodeTimeStamp: .invalid)
    CMSampleBufferCreate(
        allocator: nil, dataBuffer: nil, dataReady: false, makeDataReadyCallback: nil, refcon: nil,
        formatDescription: description, sampleCount: CMItemCount(buffer.frameLength),
        sampleTimingEntryCount: 1, sampleTimingArray: &timing, sampleSizeEntryCount: 0, sampleSizeArray: nil,
        sampleBufferOut: &sample)
    CMSampleBufferSetDataBufferFromAudioBufferList(
        sample!, blockBufferAllocator: nil, blockBufferMemoryAllocator: nil, flags: 0,
        bufferList: buffer.audioBufferList)
    while !input.isReadyForMoreMediaData {
        try await Task.sleep(for: .milliseconds(10))
    }
    #expect(input.append(sample!))
    input.markAsFinished()
    await writer.finishWriting()
    #expect(writer.status == .completed)
    return url
}

func upload(_ url: URL, index: Int = 0) throws -> UploadedFile {
    let bytes = try FileManager.default.attributesOfItem(atPath: url.path)[.size] as! Int
    return UploadedFile(index: index, name: url.lastPathComponent, url: url, bytes: bytes)
}

func writeTemporary(_ data: Data, name: String) throws -> URL {
    let url = try temporaryDirectory().appendingPathComponent(name)
    try data.write(to: url)
    return url
}

@Suite struct AudioTests {
    @Test func opusInMP4DecodesToItsFullLength() async throws {
        let url = try await makeOpusMP4(seconds: 3)
        let recording = try await AudioDecoder.decode([upload(url)], maxDurationSeconds: 60)
        #expect(abs(recording.durationSeconds - 3) < 0.1)
    }

    @Test(arguments: [(kAudioFormatMPEG4AAC, "m4a"), (kAudioFormatAppleLossless, "caf"), (kAudioFormatFLAC, "flac")])
    func compressedRecordingsDecodeToTheirFullLength(format: AudioFormatID, fileExtension: String) async throws {
        let url = try makeEncoded(seconds: 3, format: format, fileExtension: fileExtension)
        let recording = try await AudioDecoder.decode([upload(url)], maxDurationSeconds: 60)
        #expect(abs(recording.durationSeconds - 3) < 0.1)
        #expect(recording.files.count == 1)
    }

    @Test func aWAVCutShortOfItsHeaderIsRefusedAsTruncated() async throws {
        let wav = makeWAV(seconds: 2)
        let url = try writeTemporary(wav.prefix(wav.count / 2), name: "cut.wav")
        #expect(PCMContainer.truncation(of: url) != nil)
        await #expect {
            try await AudioDecoder.decode([upload(url)], maxDurationSeconds: 60)
        } throws: { error in
            let error = error as? ServiceError
            return error?.code == .invalidInput && error?.message.contains("(cut.wav) is truncated") == true
        }
    }

    @Test func intactAndStreamingPCMHeadersAreNotTruncations() throws {
        #expect(PCMContainer.truncation(of: try writeTemporary(makeWAV(seconds: 1), name: "ok.wav")) == nil)
        // Recorders that never finalise write 0xFFFFFFFF as the data size.
        var streaming = makeWAV(seconds: 1)
        streaming.replaceSubrange(40..<44, with: [0xFF, 0xFF, 0xFF, 0xFF])
        #expect(PCMContainer.truncation(of: try writeTemporary(streaming, name: "streaming.wav")) == nil)
        #expect(PCMContainer.truncation(of: try writeTemporary(Data("not audio".utf8), name: "x.wav")) == nil)
    }

    @Test func aDeclaredDurationOverTheLimitIsRefusedBeforeDecoding() async throws {
        let url = try makeEncoded(seconds: 5, format: kAudioFormatMPEG4AAC, fileExtension: "m4a")
        await #expect(throws: ServiceError.self) {
            try await AudioDecoder.decode([upload(url)], maxDurationSeconds: 2)
        }
    }

    @Test func webMIsRefusedWithAHintToConvert() async throws {
        // An EBML header: enough for the magic-byte check, and nothing
        // AVFoundation can open.
        let url = try writeTemporary(Data([0x1A, 0x45, 0xDF, 0xA3, 0x9F, 0x42, 0x86, 0x81, 0x01]), name: "r.webm")
        await #expect {
            try await AudioDecoder.decode([upload(url)], maxDurationSeconds: 60)
        } throws: { error in
            let error = error as? ServiceError
            return error?.code == .invalidInput && error?.message.contains("WebM/Matroska") == true
        }
    }
}
