import Foundation
import Testing

@testable import TranscriptionService

@Suite struct UploadsTests {
    let limits = UploadLimits(maxUploadBytes: 10_000, maxFiles: 3)

    @Test func boundaryIsParsedFromTheContentType() throws {
        #expect(try multipartBoundary(contentType: "multipart/form-data; boundary=abc123") == "abc123")
        #expect(try multipartBoundary(contentType: "Multipart/Form-Data; charset=utf-8; boundary=\"x y\"") == "x y")
        #expect(throws: ServiceError.self) { try multipartBoundary(contentType: nil) }
        #expect(throws: ServiceError.self) { try multipartBoundary(contentType: "application/json") }
        #expect(throws: ServiceError.self) { try multipartBoundary(contentType: "multipart/form-data") }
    }

    @Test func contentDispositionYieldsNameAndFilename() {
        let parsed = parseContentDisposition("form-data; name=\"file\"; filename=\"a b.m4a\"")
        #expect(parsed.name == "file")
        #expect(parsed.filename == "a b.m4a")
        let extended = parseContentDisposition("form-data; name=file; filename*=UTF-8''caf%C3%A9.wav")
        #expect(extended.filename == "café.wav")
    }

    @Test func fileNamesAreReducedToOneSafeComponent() {
        #expect(sanitisedFileName("../../etc/passwd", index: 0) == "passwd")
        #expect(sanitisedFileName("C:\\Users\\jake\\meeting (1).m4a", index: 0) == "meeting (1).m4a")
        #expect(sanitisedFileName(".hidden", index: 0) == "_hidden")
        #expect(sanitisedFileName("", index: 2) == "recording-3")
        #expect(sanitisedFileName(nil, index: 0) == "recording-1")
        #expect(sanitisedFileName("..", index: 4) == "recording-5")
        #expect(sanitisedFileName(String(repeating: "a", count: 300) + ".wav", index: 0).count == 120)
    }

    @Test(arguments: [1, 7, 64, 4096])
    func filesAreWrittenInOrderWhateverTheChunking(chunkSize: Int) async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let first = Data("first recording bytes".utf8)
        let second = Data((0..<1000).map { UInt8($0 % 251) })
        let body = multipartBody(
            boundary: "B",
            files: [
                MultipartFile(name: "file", filename: "one.wav", data: first),
                MultipartFile(name: "file", filename: "sub/dir/two.m4a", data: second),
            ])

        let files = try await receiveUploads(
            body: chunkedStream(body, chunkSize: chunkSize), boundary: "B", into: directory, limits: limits)

        #expect(files.map(\.index) == [0, 1])
        #expect(files.map(\.name) == ["one.wav", "two.m4a"])
        #expect(files.map(\.bytes) == [first.count, second.count])
        #expect(try Data(contentsOf: files[0].url) == first)
        #expect(try Data(contentsOf: files[1].url) == second)
        #expect(files.allSatisfy { $0.url.deletingLastPathComponent() == directory })
    }

    @Test func anUploadOverTheByteLimitIsRefusedMidStream() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let body = multipartBody(
            boundary: "B",
            files: [MultipartFile(name: "file", filename: "big.wav", data: Data(count: 20_000))])

        await #expect(throws: ServiceError(.payloadTooLarge, "the upload exceeds the limit of 10000 bytes")) {
            try await receiveUploads(
                body: chunkedStream(body, chunkSize: 1024), boundary: "B", into: directory, limits: limits)
        }
    }

    @Test func tooManyFilesIsRefusedAtTheExtraPartHeader() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let body = multipartBody(
            boundary: "B",
            files: (0..<4).map { MultipartFile(name: "file", filename: "\($0).wav", data: Data("x".utf8)) })

        do {
            _ = try await receiveUploads(
                body: chunkedStream(body, chunkSize: 32), boundary: "B", into: directory, limits: limits)
            Issue.record("expected payloadTooLarge")
        } catch let error as ServiceError {
            #expect(error.code == .payloadTooLarge)
            #expect(error.message.contains("at most 3 files"))
        }
    }

    @Test func partsNotNamedFileEmptyFilesAndMalformedBodiesAreInvalidInput() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let wrongName = multipartBody(
            boundary: "B", files: [MultipartFile(name: "audio", filename: "a.wav", data: Data("x".utf8))])
        await expectInvalidInput(wrongName, boundary: "B", directory: directory, containing: "named \"file\"")

        let empty = multipartBody(boundary: "B", files: [MultipartFile(name: "file", filename: "a.wav", data: Data())])
        await expectInvalidInput(empty, boundary: "B", directory: directory, containing: "is empty")

        let noFiles = multipartBody(boundary: "B", files: [])
        await expectInvalidInput(noFiles, boundary: "B", directory: directory, containing: "no files")

        let garbage = Data("this is not multipart at all".utf8)
        await expectInvalidInput(garbage, boundary: "B", directory: directory, containing: "malformed")
    }

    private func expectInvalidInput(_ body: Data, boundary: String, directory: URL, containing text: String) async {
        do {
            _ = try await receiveUploads(
                body: chunkedStream(body, chunkSize: 16), boundary: boundary, into: directory, limits: limits)
            Issue.record("expected invalidInput for \(text)")
        } catch let error as ServiceError {
            #expect(error.code == .invalidInput)
            #expect(error.message.contains(text), "\(error.message)")
        } catch {
            Issue.record("unexpected error \(error)")
        }
    }
}
