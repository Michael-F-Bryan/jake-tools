import Foundation
import NIOCore

/// A 16 kHz mono 16-bit PCM WAV of a sine tone: enough for AVFoundation to
/// decode and for the service's decoder to measure.
func makeWAV(seconds: Double, frequency: Double = 440, sampleRate: Int = 16_000, amplitude: Double = 0.3) -> Data {
    let frames = Int(seconds * Double(sampleRate))
    var pcm = Data(capacity: frames * 2)
    for frame in 0..<frames {
        let value = sin(2 * Double.pi * frequency * Double(frame) / Double(sampleRate)) * amplitude
        var sample = Int16(value * 32767).littleEndian
        withUnsafeBytes(of: &sample) { pcm.append(contentsOf: $0) }
    }
    var wav = Data()
    func append<T: FixedWidthInteger>(_ value: T) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { wav.append(contentsOf: $0) }
    }
    wav.append(contentsOf: "RIFF".utf8)
    append(UInt32(36 + pcm.count))
    wav.append(contentsOf: "WAVE".utf8)
    wav.append(contentsOf: "fmt ".utf8)
    append(UInt32(16))
    append(UInt16(1))
    append(UInt16(1))
    append(UInt32(sampleRate))
    append(UInt32(sampleRate * 2))
    append(UInt16(2))
    append(UInt16(16))
    wav.append(contentsOf: "data".utf8)
    append(UInt32(pcm.count))
    wav.append(pcm)
    return wav
}

struct MultipartFile {
    var name: String?
    var filename: String?
    var contentType: String = "application/octet-stream"
    var data: Data
}

/// A `multipart/form-data` body in the shape the Python client sends.
func multipartBody(boundary: String, files: [MultipartFile]) -> Data {
    var body = Data()
    for file in files {
        body.append(contentsOf: "--\(boundary)\r\n".utf8)
        var disposition = "Content-Disposition: form-data"
        if let name = file.name { disposition += "; name=\"\(name)\"" }
        if let filename = file.filename { disposition += "; filename=\"\(filename)\"" }
        body.append(contentsOf: "\(disposition)\r\n".utf8)
        body.append(contentsOf: "Content-Type: \(file.contentType)\r\n\r\n".utf8)
        body.append(file.data)
        body.append(contentsOf: "\r\n".utf8)
    }
    body.append(contentsOf: "--\(boundary)--\r\n".utf8)
    return body
}

/// Split `data` into an async stream of buffers of `chunkSize` bytes, so
/// parsers see boundaries straddling chunks as they would on a socket.
func chunkedStream(_ data: Data, chunkSize: Int) -> AsyncStream<ByteBuffer> {
    AsyncStream { continuation in
        var offset = 0
        while offset < data.count {
            let end = min(offset + chunkSize, data.count)
            continuation.yield(ByteBuffer(bytes: data[offset..<end]))
            offset = end
        }
        continuation.finish()
    }
}

func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("transcription-service-tests", isDirectory: true)
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
