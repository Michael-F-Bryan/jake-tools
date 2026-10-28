// Decoding uploaded files into one 16 kHz mono recording.
//
// Each file is read with `AVAssetReader`, which demuxes and decodes in one
// step and resamples straight to 16 kHz mono Float32, so anything
// AVFoundation opens is accepted: WAV, AIFF, CAF, M4A/MP4 (AAC, ALAC, Opus),
// MP3, FLAC. WebM and Ogg are not: AVFoundation has no demuxer for them. The
// outputs are concatenated in request order with no gap, which is what gives
// the transcript its single timeline.
//
// A file is accepted only if it decodes completely. Every file is opened
// before any is decoded, so an unreadable file, a PCM header promising more
// audio than the file holds, or a declared total over the duration limit is
// refused before the expensive work. A decode that stops early (damaged
// packets) or ends well short of the container's declared duration is
// refused too: the transcript is never silently partial. For fragmented
// MP4, whose declared duration is not trustworthy (see `AudioSource.open`),
// the comparison is against the summed durations of its packets.
//
// Decoding is not FluidAudio's `AudioConverter.resampleAudioFile`. That reads
// through `AVAudioFile`, which raises an uncaught NSException on Opus in MP4
// (aborting the service), holds the whole file at its source rate in memory,
// and treats a read error after the first chunk as end of file.

import AVFoundation
import CoreMedia
import Foundation
import ObjCExceptionGuard

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
    /// How far a decoded file may fall short of its container's declared
    /// duration. Encoder delay and padding account for tens of milliseconds;
    /// anything beyond this is lost audio.
    public static let durationToleranceSeconds = 0.5

    /// Decode and concatenate `uploads`. Rejects a file that is not decodable
    /// audio, decodes to nothing, or decodes incompletely, and a total longer
    /// than `maxDurationSeconds`, as `ServiceError`s. Checks for cancellation
    /// between decoded buffers.
    public static func decode(_ uploads: [UploadedFile], maxDurationSeconds: Double) async throws
        -> DecodedRecording
    {
        let tooLong = ServiceError(
            .payloadTooLarge, "the recording exceeds the limit of \(Int(maxDurationSeconds)) seconds of audio")

        var sources: [AudioSource] = []
        var declaredTotal = 0.0
        for upload in uploads {
            try Task.checkCancellation()
            let source = try await AudioSource.open(upload)
            declaredTotal += source.declaredSeconds ?? 0
            guard declaredTotal <= maxDurationSeconds + durationToleranceSeconds else { throw tooLong }
            sources.append(source)
        }

        let maxSamples = Int(maxDurationSeconds * Double(DecodedRecording.sampleRate))
        var samples: [Float] = []
        samples.reserveCapacity(min(maxSamples, Int((declaredTotal + 1) * Double(DecodedRecording.sampleRate))))
        var spans: [FileSpan] = []
        for source in sources {
            let first = samples.count
            try source.decode(appendingTo: &samples, maxSamples: maxSamples, tooLong: tooLong)
            let decodedSeconds = Double(samples.count - first) / Double(DecodedRecording.sampleRate)
            guard samples.count > first else {
                throw ServiceError(.invalidInput, "\(source.label) contains no audio")
            }
            if let declared = source.declaredSeconds, decodedSeconds < declared - durationToleranceSeconds {
                throw ServiceError(
                    .invalidInput,
                    "\(source.label) is truncated or damaged: it decoded to \(formatSeconds(decodedSeconds)) s "
                        + "of its declared \(formatSeconds(declared)) s")
            }
            spans.append(
                FileSpan(
                    index: source.upload.index, name: source.upload.name,
                    startSeconds: Double(first) / Double(DecodedRecording.sampleRate),
                    endSeconds: Double(samples.count) / Double(DecodedRecording.sampleRate)))
        }
        return DecodedRecording(samples: samples, files: spans)
    }
}

/// One opened upload: its asset, first audio track and declared duration.
struct AudioSource {
    let upload: UploadedFile
    let asset: AVURLAsset
    let track: AVAssetTrack
    /// The container's duration, when it states a trustworthy one.
    let declaredSeconds: Double?

    var label: String { "file \(upload.index + 1) (\(upload.name))" }

    static func open(_ upload: UploadedFile) async throws -> AudioSource {
        let label = "file \(upload.index + 1) (\(upload.name))"
        if let shortfall = PCMContainer.truncation(of: upload.url) {
            throw ServiceError(
                .invalidInput,
                "\(label) is truncated: its header declares \(shortfall.declaredBytes) bytes of audio "
                    + "but the file holds \(shortfall.presentBytes)")
        }
        let asset = AVURLAsset(url: upload.url, options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
        let tracks: [AVAssetTrack]
        let duration: CMTime
        let fragmented: Bool
        do {
            tracks = try await asset.loadTracks(withMediaType: .audio)
            duration = try await asset.load(.duration)
            fragmented = try await asset.load(.canContainFragments)
        } catch {
            throw ServiceError(.invalidInput, "\(label) could not be decoded as audio\(unsupportedHint(upload.url))")
        }
        guard let track = tracks.first else {
            throw ServiceError(.invalidInput, "\(label) has no audio track\(unsupportedHint(upload.url))")
        }
        // A fragmented MP4's own duration is not trustworthy. Most of
        // Michael's recordings are fragmented Opus whose `mdhd` duration is
        // written in milliseconds against a 48 kHz timescale; AVFoundation
        // adds that to the fragments' durations and declares about 2% more
        // audio than the file holds. The packets' own durations are right.
        let seconds = fragmented ? packetSeconds(asset: asset, track: track) : duration.seconds
        return AudioSource(
            upload: upload, asset: asset, track: track,
            declaredSeconds: seconds.flatMap { $0.isFinite && $0 > 0 ? $0 : nil })
    }

    /// The summed durations of the track's compressed packets, read without
    /// decoding (about 0.3 s for 90 minutes of Opus). `nil` if the read
    /// fails; decoding then reports the damage.
    private static func packetSeconds(asset: AVURLAsset, track: AVAssetTrack) -> Double? {
        guard let reader = try? AVAssetReader(asset: asset) else { return nil }
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        var total = CMTime.zero
        do {
            try ObjCExceptionGuard.run {
                guard reader.canAdd(output) else { return }
                reader.add(output)
                guard reader.startReading() else { return }
                while let buffer = output.copyNextSampleBuffer() {
                    total = CMTimeAdd(total, CMSampleBufferGetDuration(buffer))
                }
            }
        } catch {
            return nil
        }
        return reader.status == .completed ? total.seconds : nil
    }

    /// Append this file's audio, as 16 kHz mono Float32, to `samples`.
    func decode(appendingTo samples: inout [Float], maxSamples: Int, tooLong: ServiceError) throws {
        let reader: AVAssetReader
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            throw ServiceError(.invalidInput, "\(label) could not be decoded as audio")
        }
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: DecodedRecording.sampleRate,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 32,
                AVLinearPCMIsFloatKey: true,
                AVLinearPCMIsNonInterleaved: false,
                AVLinearPCMIsBigEndianKey: false,
            ])
        output.alwaysCopiesSampleData = false
        var started = false
        do {
            try ObjCExceptionGuard.run {
                if reader.canAdd(output) {
                    reader.add(output)
                    started = reader.startReading()
                }
            }
        } catch {
            throw ServiceError(.invalidInput, "\(label) could not be decoded as audio")
        }
        guard started else {
            throw ServiceError(.invalidInput, "\(label) could not be decoded as audio")
        }

        let first = samples.count
        while true {
            if Task.isCancelled {
                reader.cancelReading()
                throw CancellationError()
            }
            var next: CMSampleBuffer?
            do {
                try ObjCExceptionGuard.run { next = output.copyNextSampleBuffer() }
            } catch {
                reader.cancelReading()
                throw damaged(decodedSamples: samples.count - first)
            }
            guard let buffer = next else { break }
            guard let block = CMSampleBufferGetDataBuffer(buffer) else { continue }
            let count = CMBlockBufferGetDataLength(block) / MemoryLayout<Float>.size
            guard samples.count + count <= maxSamples else {
                reader.cancelReading()
                throw tooLong
            }
            let offset = samples.count
            samples.append(contentsOf: repeatElement(0, count: count))
            let status = samples.withUnsafeMutableBytes { bytes in
                CMBlockBufferCopyDataBytes(
                    block, atOffset: 0, dataLength: count * MemoryLayout<Float>.size,
                    destination: bytes.baseAddress! + offset * MemoryLayout<Float>.size)
            }
            guard status == kCMBlockBufferNoErr else {
                reader.cancelReading()
                throw ServiceError(.internalError, "could not copy decoded audio for \(label)")
            }
        }
        switch reader.status {
        case .completed:
            return
        case .cancelled:
            throw CancellationError()
        default:
            throw damaged(decodedSamples: samples.count - first)
        }
    }

    private func damaged(decodedSamples: Int) -> ServiceError {
        let decoded = Double(decodedSamples) / Double(DecodedRecording.sampleRate)
        let of = declaredSeconds.map { " of its declared \(formatSeconds($0)) s" } ?? ""
        return ServiceError(
            .invalidInput,
            "\(label) is damaged: decoding failed after \(formatSeconds(decoded)) s\(of)")
    }
}

/// A suffix naming a container AVFoundation cannot read, from its magic
/// bytes, so the caller learns to convert rather than retry.
private func unsupportedHint(_ url: URL) -> String {
    guard let handle = try? FileHandle(forReadingFrom: url) else { return "" }
    defer { try? handle.close() }
    let magic = (try? handle.read(upToCount: 4)).map { [UInt8]($0) } ?? []
    let container: String
    switch magic {
    case [0x1A, 0x45, 0xDF, 0xA3]: container = "WebM/Matroska"
    case Array("OggS".utf8): container = "Ogg"
    default: return ""
    }
    return "; it is \(container), which AVFoundation cannot read. Convert it to M4A, WAV or FLAC first"
}

private func formatSeconds(_ seconds: Double) -> String {
    String(format: "%.1f", seconds)
}

/// Header checks for uncompressed containers. AVFoundation sizes WAV, AIFF
/// and CAF audio by the bytes actually present, so a file cut short decodes
/// "completely" to less than its header declares; only the header shows the
/// loss.
enum PCMContainer {
    struct Truncation: Equatable {
        var declaredBytes: UInt64
        var presentBytes: UInt64
    }

    /// The shortfall if `url` is a RIFF/WAVE, AIFF/AIFC or CAF file whose
    /// audio data chunk declares more bytes than the file holds; `nil` for
    /// any other file, an intact one, or a streaming placeholder size.
    static func truncation(of url: URL) -> Truncation? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let fileSize = try? handle.seekToEnd(), let header = read(handle, at: 0, count: 12) else {
            return nil
        }
        let kind = String(decoding: header[0..<4], as: UTF8.self)
        let form = String(decoding: header[8..<12], as: UTF8.self)
        switch (kind, form) {
        case ("RIFF", "WAVE"):
            return walk(
                handle, fileSize: fileSize, from: 12, headerSize: 8, dataID: "data", padded: true,
                size: { UInt64(littleEndian32($0, 4)) },
                placeholder: { $0 == 0 || $0 == 0xFFFF_FFFF })
        case ("FORM", "AIFF"), ("FORM", "AIFC"):
            return walk(
                handle, fileSize: fileSize, from: 12, headerSize: 8, dataID: "SSND", padded: true,
                size: { UInt64(bigEndian32($0, 4)) },
                placeholder: { $0 == 0 })
        case ("caff", _):
            return walk(
                handle, fileSize: fileSize, from: 8, headerSize: 12, dataID: "data", padded: false,
                size: { bigEndian64($0, 4) },
                placeholder: { $0 == UInt64.max })
        default:
            return nil
        }
    }

    private static func walk(
        _ handle: FileHandle, fileSize: UInt64, from start: UInt64, headerSize: Int, dataID: String,
        padded: Bool, size: ([UInt8]) -> UInt64, placeholder: (UInt64) -> Bool
    ) -> Truncation? {
        var offset = start
        // A real file has a handful of chunks; the bound stops a crafted one
        // from making this loop long.
        for _ in 0..<64 {
            guard let chunk = read(handle, at: offset, count: headerSize) else { return nil }
            let id = String(decoding: chunk[0..<4], as: UTF8.self)
            let declared = size(chunk)
            let body = offset + UInt64(headerSize)
            if id == dataID {
                guard !placeholder(declared), body <= fileSize else { return nil }
                let present = fileSize - body
                return declared > present ? Truncation(declaredBytes: declared, presentBytes: present) : nil
            }
            let (next, overflow) = body.addingReportingOverflow(declared + (padded ? declared & 1 : 0))
            guard !overflow, next > offset else { return nil }
            offset = next
        }
        return nil
    }

    private static func read(_ handle: FileHandle, at offset: UInt64, count: Int) -> [UInt8]? {
        guard (try? handle.seek(toOffset: offset)) != nil,
            let data = try? handle.read(upToCount: count), data.count == count
        else { return nil }
        return [UInt8](data)
    }

    private static func littleEndian32(_ bytes: [UInt8], _ at: Int) -> UInt32 {
        (0..<4).reduce(UInt32(0)) { $0 | UInt32(bytes[at + $1]) << (8 * $1) }
    }

    private static func bigEndian32(_ bytes: [UInt8], _ at: Int) -> UInt32 {
        (0..<4).reduce(UInt32(0)) { $0 << 8 | UInt32(bytes[at + $1]) }
    }

    private static func bigEndian64(_ bytes: [UInt8], _ at: Int) -> UInt64 {
        (0..<8).reduce(UInt64(0)) { $0 << 8 | UInt64(bytes[at + $1]) }
    }
}
