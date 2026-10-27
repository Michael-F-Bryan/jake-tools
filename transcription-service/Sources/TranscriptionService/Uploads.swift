// Receiving the multipart upload. Parts are streamed straight to files in a
// per-request directory, so memory use does not grow with the recording. The
// request is one `multipart/form-data` body whose parts are all named `file`,
// each with a `filename`; part order is recording order.

import Foundation
import HTTPTypes
import MultipartKit
import NIOCore

public struct UploadLimits: Sendable, Equatable {
    public var maxUploadBytes: Int
    public var maxFiles: Int

    public init(maxUploadBytes: Int, maxFiles: Int) {
        self.maxUploadBytes = maxUploadBytes
        self.maxFiles = maxFiles
    }
}

public struct UploadedFile: Sendable, Equatable {
    /// Position in the request, from 0.
    public var index: Int
    /// The client's file name, sanitised to a single path component.
    public var name: String
    public var url: URL
    public var bytes: Int
}

/// The `boundary` parameter of a `multipart/form-data` content type.
public func multipartBoundary(contentType: String?) throws -> String {
    guard let contentType else {
        throw ServiceError(.invalidInput, "Content-Type must be multipart/form-data with a boundary")
    }
    let parts = contentType.split(separator: ";").map { $0.trimmingCharacters(in: .whitespaces) }
    guard let first = parts.first, first.lowercased() == "multipart/form-data" else {
        throw ServiceError(.invalidInput, "Content-Type must be multipart/form-data with a boundary")
    }
    for parameter in parts.dropFirst() {
        guard let equals = parameter.firstIndex(of: "=") else { continue }
        let key = parameter[..<equals].trimmingCharacters(in: .whitespaces).lowercased()
        guard key == "boundary" else { continue }
        let value = unquote(parameter[parameter.index(after: equals)...])
        if !value.isEmpty, value.count <= 70 {
            return value
        }
    }
    throw ServiceError(.invalidInput, "Content-Type must be multipart/form-data with a boundary")
}

/// Stream a multipart body into `directory`, one file per part.
///
/// Enforces `limits` as bytes arrive: the total of all part bodies may not
/// exceed `maxUploadBytes`, and there may be at most `maxFiles` parts. Any
/// violation, or a malformed body, throws a `ServiceError` and leaves the
/// caller to delete `directory`.
public func receiveUploads<Body: AsyncSequence & Sendable>(
    body: Body,
    boundary: String,
    into directory: URL,
    limits: UploadLimits
) async throws -> [UploadedFile] where Body.Element == ByteBuffer {
    let chunks = body.map { (buffer: ByteBuffer) -> [UInt8] in Array(buffer: buffer) }
    let sections = StreamingMultipartParserAsyncSequence(boundary: boundary, buffer: chunks)

    var files: [UploadedFile] = []
    var current: (file: UploadedFile, handle: FileHandle)?
    var totalBytes = 0

    func finishCurrent() throws {
        guard let open = current else { return }
        try open.handle.close()
        guard open.file.bytes > 0 else {
            throw ServiceError(.invalidInput, "file \(open.file.index + 1) (\(open.file.name)) is empty")
        }
        files.append(open.file)
        current = nil
    }

    do {
        for try await section in sections {
            switch section {
            case .headerFields(let fields):
                try finishCurrent()
                let disposition = parseContentDisposition(fields[.contentDisposition] ?? "")
                guard disposition.name == "file" else {
                    throw ServiceError(
                        .invalidInput,
                        "every multipart part must be named \"file\"; got \"\(disposition.name ?? "")\"")
                }
                guard files.count < limits.maxFiles else {
                    throw ServiceError(
                        .payloadTooLarge, "a request may carry at most \(limits.maxFiles) files")
                }
                let index = files.count
                let name = sanitisedFileName(disposition.filename, index: index)
                let url = directory.appendingPathComponent("\(index)-\(name)", isDirectory: false)
                guard
                    FileManager.default.createFile(
                        atPath: url.path, contents: nil, attributes: [.posixPermissions: 0o600])
                else {
                    throw ServiceError(.internalError, "could not create an upload file")
                }
                let handle = try FileHandle(forWritingTo: url)
                current = (UploadedFile(index: index, name: name, url: url, bytes: 0), handle)
            case .bodyChunk(let chunk):
                guard var open = current else {
                    throw ServiceError(.invalidInput, "multipart body data arrived before any part header")
                }
                totalBytes += chunk.count
                guard totalBytes <= limits.maxUploadBytes else {
                    throw ServiceError(
                        .payloadTooLarge,
                        "the upload exceeds the limit of \(limits.maxUploadBytes) bytes")
                }
                try open.handle.write(contentsOf: Data(chunk))
                open.file.bytes += chunk.count
                current = open
            case .boundary(let end):
                try finishCurrent()
                if end { break }
            }
        }
    } catch let error as ServiceError {
        try? current?.handle.close()
        throw error
    } catch is CancellationError {
        try? current?.handle.close()
        throw CancellationError()
    } catch {
        try? current?.handle.close()
        throw ServiceError(.invalidInput, "malformed multipart body")
    }
    try finishCurrent()
    guard !files.isEmpty else {
        throw ServiceError(.invalidInput, "the request carried no files")
    }
    return files
}

/// `name` and `filename` from a `Content-Disposition` header value.
func parseContentDisposition(_ value: String) -> (name: String?, filename: String?) {
    var name: String?
    var filename: String?
    for parameter in value.split(separator: ";").dropFirst() {
        guard let equals = parameter.firstIndex(of: "=") else { continue }
        let key = parameter[..<equals].trimmingCharacters(in: .whitespaces).lowercased()
        let raw = unquote(parameter[parameter.index(after: equals)...])
        switch key {
        case "name": name = raw
        case "filename": filename = raw
        case "filename*":
            // RFC 5987: charset'lang'percent-encoded. Only decode UTF-8.
            let pieces = raw.split(separator: "'", maxSplits: 2, omittingEmptySubsequences: false)
            if pieces.count == 3, pieces[0].lowercased() == "utf-8",
                let decoded = String(pieces[2]).removingPercentEncoding
            {
                filename = decoded
            }
        default: break
        }
    }
    return (name, filename)
}

/// The client's file name reduced to one safe path component.
func sanitisedFileName(_ raw: String?, index: Int) -> String {
    let fallback = "recording-\(index + 1)"
    guard let raw else { return fallback }
    let lastComponent = raw.split(whereSeparator: { $0 == "/" || $0 == "\\" }).last.map(String.init) ?? ""
    let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._- ()[]"))
    var cleaned = String(
        lastComponent.unicodeScalars.filter { allowed.contains($0) }
    ).trimmingCharacters(in: .whitespaces)
    if cleaned.isEmpty || cleaned == "." || cleaned == ".." {
        return fallback
    }
    if cleaned.hasPrefix(".") {
        cleaned = "_" + cleaned.dropFirst()
    }
    if cleaned.count > 120 {
        cleaned = String(cleaned.prefix(120))
    }
    return cleaned
}

private func unquote(_ value: Substring) -> String {
    var trimmed = value.trimmingCharacters(in: .whitespaces)
    if trimmed.hasPrefix("\""), trimmed.hasSuffix("\""), trimmed.count >= 2 {
        trimmed = String(trimmed.dropFirst().dropLast())
    }
    return trimmed.replacingOccurrences(of: "\\\"", with: "\"")
}
