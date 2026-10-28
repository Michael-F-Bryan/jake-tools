// A fragmented Opus MP4 in the shape of the vault recorder's files: an empty
// sample table in `moov`, every packet in a `moof`/`mdat` fragment, and an
// `mdhd` duration written in milliseconds against the 48 kHz timescale.
// AVFoundation adds that bogus `mdhd` duration to the fragments', so it
// declares about 2% more audio than the file holds. AVAssetWriter cannot
// produce this layout, so the boxes are written by hand around Opus packets
// AVFoundation encoded.

import AVFoundation
import Foundation
import Testing

private struct Box {
    var bytes = Data()

    init(_ type: String, _ build: (inout Box) -> Void = { _ in }) {
        bytes.append(contentsOf: [0, 0, 0, 0])
        bytes.append(contentsOf: type.utf8)
        build(&self)
        let size = UInt32(bytes.count).bigEndian
        withUnsafeBytes(of: size) { bytes.replaceSubrange(0..<4, with: $0) }
    }

    mutating func u8(_ value: UInt8) { bytes.append(value) }
    mutating func u16(_ value: UInt16) { withUnsafeBytes(of: value.bigEndian) { bytes.append(contentsOf: $0) } }
    mutating func u32(_ value: UInt32) { withUnsafeBytes(of: value.bigEndian) { bytes.append(contentsOf: $0) } }
    mutating func u64(_ value: UInt64) { withUnsafeBytes(of: value.bigEndian) { bytes.append(contentsOf: $0) } }
    mutating func zeros(_ count: Int) { bytes.append(Data(count: count)) }
    mutating func ascii(_ text: String) { bytes.append(contentsOf: text.utf8) }
    mutating func child(_ box: Box) { bytes.append(box.bytes) }
    /// Version and flags of a full box.
    mutating func full(_ version: UInt8, _ flags: UInt32 = 0) { u32(UInt32(version) << 24 | flags) }

    mutating func matrix() {
        for value: UInt32 in [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000] { u32(value) }
    }
}

/// Opus packets for `seconds` of tone, as AVFoundation's encoder writes them,
/// and the frames (at 48 kHz) each one covers.
private func opusPackets(seconds: Double) async throws -> (packets: [Data], frames: UInt32) {
    let source = try await makeOpusMP4(seconds: seconds)
    let asset = AVURLAsset(url: source)
    let track = try #require(try await asset.loadTracks(withMediaType: .audio).first)
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
    reader.add(output)
    reader.startReading()
    var packets: [Data] = []
    var frames: UInt32 = 0
    while let buffer = output.copyNextSampleBuffer() {
        guard let block = CMSampleBufferGetDataBuffer(buffer) else { continue }
        let count = CMSampleBufferGetNumSamples(buffer)
        if count > 0 {
            frames = UInt32(CMSampleBufferGetDuration(buffer).convertScale(48_000, method: .default).value) / UInt32(count)
        }
        var offset = 0
        for index in 0..<CMSampleBufferGetNumSamples(buffer) {
            let size = CMSampleBufferGetSampleSize(buffer, at: index)
            var packet = Data(count: size)
            _ = packet.withUnsafeMutableBytes {
                CMBlockBufferCopyDataBytes(block, atOffset: offset, dataLength: size, destination: $0.baseAddress!)
            }
            packets.append(packet)
            offset += size
        }
    }
    return (packets, frames)
}

/// Write the recorder-shaped file; its true duration is `seconds`. With
/// `gapSeconds`, the middle packet's duration also spans that much time in
/// which nothing was recorded, as when the recorder pauses or the machine
/// sleeps: the timeline jumps, the audio does not.
func makeRecorderFragmentedOpus(seconds: Double, gapSeconds: Double = 0) async throws -> URL {
    let (packets, frames) = try await opusPackets(seconds: seconds)
    let gapIndex = packets.count / 2
    let gapFrames = UInt32(gapSeconds * 48_000)
    let milliseconds = UInt32((Double(packets.count) * Double(frames) + Double(gapFrames)) / 48.0)

    let ftyp = Box("ftyp") {
        $0.ascii("isom")
        $0.u32(0x200)
        $0.ascii("isomiso6mp41")
    }
    let moov = Box("moov") {
        $0.child(
            Box("mvhd") {
                $0.full(0)
                $0.zeros(8)
                $0.u32(1000)
                $0.u32(milliseconds)
                $0.u32(0x0001_0000)
                $0.u16(0x0100)
                $0.zeros(10)
                $0.matrix()
                $0.zeros(24)
                $0.u32(2)
            })
        $0.child(
            Box("trak") {
                $0.child(
                    Box("tkhd") {
                        $0.full(0, 3)
                        $0.zeros(8)
                        $0.u32(1)
                        $0.zeros(4)
                        $0.u32(milliseconds)
                        $0.zeros(12)
                        $0.u16(0x0100)
                        $0.zeros(2)
                        $0.matrix()
                        $0.zeros(8)
                    })
                $0.child(
                    Box("mdia") {
                        $0.child(
                            Box("mdhd") {
                                $0.full(0)
                                $0.zeros(8)
                                $0.u32(48_000)
                                $0.u32(milliseconds)  // the recorder's bug
                                $0.u16(0x55C4)
                                $0.zeros(2)
                            })
                        $0.child(
                            Box("hdlr") {
                                $0.full(0)
                                $0.zeros(4)
                                $0.ascii("soun")
                                $0.zeros(12)
                                $0.ascii("SoundHandler\0")
                            })
                        $0.child(
                            Box("minf") {
                                $0.child(Box("smhd") { $0.full(0); $0.zeros(4) })
                                $0.child(
                                    Box("dinf") {
                                        $0.child(
                                            Box("dref") {
                                                $0.full(0)
                                                $0.u32(1)
                                                $0.child(Box("url ") { $0.full(0, 1) })
                                            })
                                    })
                                $0.child(
                                    Box("stbl") {
                                        $0.child(
                                            Box("stsd") {
                                                $0.full(0)
                                                $0.u32(1)
                                                $0.child(
                                                    Box("Opus") {
                                                        $0.zeros(6)
                                                        $0.u16(1)
                                                        $0.zeros(8)
                                                        $0.u16(1)
                                                        $0.u16(16)
                                                        $0.zeros(4)
                                                        $0.u32(48_000 << 16)
                                                        $0.child(
                                                            Box("dOps") {
                                                                $0.u8(0)
                                                                $0.u8(1)
                                                                $0.u16(312)
                                                                $0.u32(48_000)
                                                                $0.u16(0)
                                                                $0.u8(0)
                                                            })
                                                    })
                                            })
                                        $0.child(Box("stts") { $0.full(0); $0.u32(0) })
                                        $0.child(Box("stsc") { $0.full(0); $0.u32(0) })
                                        $0.child(Box("stsz") { $0.full(0); $0.u32(0); $0.u32(0) })
                                        $0.child(Box("stco") { $0.full(0); $0.u32(0) })
                                    })
                            })
                    })
            })
        $0.child(
            Box("mvex") {
                $0.child(
                    Box("trex") {
                        $0.full(0)
                        $0.u32(1)
                        $0.u32(1)
                        $0.u32(frames)
                        $0.zeros(8)
                    })
            })
    }

    func moof(dataOffset: UInt32) -> Box {
        Box("moof") {
            $0.child(Box("mfhd") { $0.full(0); $0.u32(1) })
            $0.child(
                Box("traf") {
                    $0.child(Box("tfhd") { $0.full(0, 0x02_0000); $0.u32(1) })
                    $0.child(Box("tfdt") { $0.full(1); $0.u64(0) })
                    $0.child(
                        Box("trun") {
                            // data offset, sample duration, sample size
                            $0.full(0, 0x0301)
                            $0.u32(UInt32(packets.count))
                            $0.u32(dataOffset)
                            for (index, packet) in packets.enumerated() {
                                $0.u32(index == gapIndex ? frames + gapFrames : frames)
                                $0.u32(UInt32(packet.count))
                            }
                        })
                })
        }
    }
    let fragment = moof(dataOffset: UInt32(moof(dataOffset: 0).bytes.count + 8))
    let mdat = Box("mdat") { box in packets.forEach { box.bytes.append($0) } }

    let url = try temporaryDirectory().appendingPathComponent("recorder.m4a")
    try (ftyp.bytes + moov.bytes + fragment.bytes + mdat.bytes).write(to: url)
    return url
}
