// The HTTP layer against the real Hummingbird server on an ephemeral port,
// with a stub engine standing in for the models. These prove routing, auth,
// upload handling, decoding, limits, queueing and cancellation mechanics;
// they say nothing about transcription quality, which only the real engine
// (tests/test_transcription_live.py) can show.

import Foundation
import Logging
import Synchronization
import Testing

@testable import TranscriptionService

final class StubEngine: TranscriptionEngine, @unchecked Sendable {
    let info = EngineInfo(name: "stub", version: "0", asrModel: "stub-asr", diarizationModel: "stub-diar")
    let delay: Duration
    private let received = Mutex<[DecodedRecording]>([])
    private let cancelled = Mutex(0)
    private let started = Mutex(0)

    init(delay: Duration = .zero) {
        self.delay = delay
    }

    var recordings: [DecodedRecording] { received.withLock { $0 } }
    var cancellations: Int { cancelled.withLock { $0 } }
    var starts: Int { started.withLock { $0 } }

    func transcribe(_ recording: DecodedRecording) async throws -> EngineOutput {
        started.withLock { $0 += 1 }
        received.withLock { $0.append(recording) }
        do {
            try await Task.sleep(for: delay)
        } catch is CancellationError {
            cancelled.withLock { $0 += 1 }
            throw CancellationError()
        }
        let duration = recording.durationSeconds
        return EngineOutput(
            text: "hello world",
            words: [
                TimedWord(text: "hello", start: 0.1, end: 0.5, confidence: 0.9),
                TimedWord(text: "world", start: min(0.6, duration), end: min(1.0, duration), confidence: 0.8),
            ],
            turns: [SpeakerTurn(speaker: "engine-A", start: 0, end: duration)],
            asrSeconds: 0.01,
            diarizationSeconds: 0.02)
    }
}

struct RunningServer {
    let baseURL: URL
    let state: ServiceState
    let task: Task<Void, any Error>
    let workDirectory: URL

    func stop() async {
        task.cancel()
        _ = await task.result
        try? FileManager.default.removeItem(at: workDirectory)
    }
}

let testToken = "test-token-123"

func startServer(
    engine: (any TranscriptionEngine)?,
    maxQueued: Int = 2,
    maxUploadBytes: Int = 5_000_000,
    maxDurationSeconds: Double = 30
) async throws -> RunningServer {
    let workDirectory = try temporaryDirectory()
    let configuration = ServiceConfiguration(
        host: "127.0.0.1",
        port: 0,
        token: testToken,
        uploadLimits: UploadLimits(maxUploadBytes: maxUploadBytes, maxFiles: 4),
        maxDurationSeconds: maxDurationSeconds,
        maxQueued: maxQueued,
        workDirectory: workDirectory)
    var logger = Logger(label: "test-server")
    logger.logLevel = .warning
    let state = ServiceState(configuration: configuration, logger: logger)
    if let engine {
        await state.engine.set(.ready(engine))
    }
    let (ports, continuation) = AsyncStream<Int>.makeStream()
    let application = try makeApplication(state: state) { channel in
        continuation.yield(channel.localAddress?.port ?? 0)
        continuation.finish()
    }
    let task = Task { try await application.runService() }
    guard let port = await ports.first(where: { _ in true }), port > 0 else {
        task.cancel()
        throw ServiceError(.internalError, "server did not report a port")
    }
    return RunningServer(
        baseURL: URL(string: "http://127.0.0.1:\(port)")!, state: state, task: task, workDirectory: workDirectory)
}

struct HTTPReply {
    var status: Int
    var headers: [String: String]
    var body: Data

    func json<T: Decodable>(_ type: T.Type) throws -> T {
        try Wire.decoder().decode(type, from: body)
    }
}

func send(
    _ method: String, _ path: String, to server: RunningServer, token: String? = testToken,
    contentType: String? = nil, body: Data? = nil
) async throws -> HTTPReply {
    var request = URLRequest(url: server.baseURL.appendingPathComponent(path))
    request.httpMethod = method
    if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
    if let contentType { request.setValue(contentType, forHTTPHeaderField: "Content-Type") }
    request.httpBody = body
    request.timeoutInterval = 30
    let (data, response) = try await URLSession.shared.data(for: request)
    let http = response as! HTTPURLResponse
    var headers: [String: String] = [:]
    for (key, value) in http.allHeaderFields {
        headers[String(describing: key).lowercased()] = String(describing: value)
    }
    return HTTPReply(status: http.statusCode, headers: headers, body: data)
}

func transcribeRequest(_ server: RunningServer, files: [MultipartFile], token: String? = testToken) async throws
    -> HTTPReply
{
    let boundary = "----jake-tests-\(UUID().uuidString)"
    return try await send(
        "POST", "/v1/transcriptions", to: server, token: token,
        contentType: "multipart/form-data; boundary=\(boundary)",
        body: multipartBody(boundary: boundary, files: files))
}

@Suite(.serialized) struct ServerTests {
    @Test func healthReportsStartingThenReady() async throws {
        let server = try await startServer(engine: nil)
        defer { Task { await server.stop() } }

        let starting = try await send("GET", "/health", to: server, token: nil)
        #expect(starting.status == 503)
        let startingReport = try starting.json(HealthReport.self)
        #expect(startingReport.status == "starting")
        #expect(startingReport.ready == false)
        #expect(startingReport.queue == HealthReport.Queue(running: 0, queued: 0, maxQueued: 2))

        let post = try await transcribeRequest(server, files: [MultipartFile(name: "file", filename: "a.wav", data: makeWAV(seconds: 1))])
        #expect(post.status == 503)
        #expect(try post.json(ErrorEnvelope.self).error.code == "not_ready")
        #expect(post.headers["retry-after"] != nil)

        await server.state.engine.set(.ready(StubEngine()))
        let ready = try await send("GET", "/health", to: server, token: nil)
        #expect(ready.status == 200)
        let readyReport = try ready.json(HealthReport.self)
        #expect(readyReport.ready)
        #expect(readyReport.engine?.name == "stub")

        await server.state.engine.set(.failed("no such directory"))
        let failed = try await send("GET", "/health", to: server, token: nil)
        #expect(failed.status == 503)
        #expect(try failed.json(HealthReport.self).detail == "no such directory")
        await server.stop()
    }

    @Test func transcriptionsRequireTheExactBearerToken() async throws {
        let server = try await startServer(engine: StubEngine())
        defer { Task { await server.stop() } }
        let files = [MultipartFile(name: "file", filename: "a.wav", data: makeWAV(seconds: 1))]

        for token in [nil, "", "wrong", testToken + "x", String(testToken.dropLast())] {
            let reply = try await transcribeRequest(server, files: files, token: token)
            #expect(reply.status == 401, "token \(String(describing: token))")
            #expect(try reply.json(ErrorEnvelope.self).error.code == "unauthorized")
        }
        #expect(try await transcribeRequest(server, files: files).status == 200)
        await server.stop()
    }

    @Test func orderedFilesBecomeOneTimelineAndAStructuredTranscript() async throws {
        let engine = StubEngine()
        let server = try await startServer(engine: engine)
        defer { Task { await server.stop() } }

        let reply = try await transcribeRequest(
            server,
            files: [
                MultipartFile(name: "file", filename: "part one.wav", data: makeWAV(seconds: 1.5)),
                MultipartFile(name: "file", filename: "nested/part-two.wav", data: makeWAV(seconds: 2.0, frequency: 220)),
            ])

        #expect(reply.status == 200)
        #expect(reply.headers["content-type"]?.hasPrefix("application/json") == true)
        let transcript = try reply.json(Transcript.self)
        #expect(transcript.schemaVersion == contractSchemaVersion)
        #expect(transcript.engine.name == "stub")
        #expect(transcript.timestampUnit == "seconds")
        #expect(transcript.timingResolutionSeconds == 0.08)
        #expect(transcript.durationSeconds == 3.5)
        #expect(transcript.files.map(\.name) == ["part one.wav", "part-two.wav"])
        #expect(transcript.files.map(\.startSeconds) == [0.0, 1.5])
        #expect(transcript.files.map(\.endSeconds) == [1.5, 3.5])
        #expect(transcript.files.map(\.durationSeconds) == [1.5, 2.0])
        #expect(transcript.speakers.map(\.id) == ["speaker_1"])
        #expect(transcript.speakers[0].speakingSeconds == 3.5)
        #expect(transcript.segments.count == 1)
        #expect(transcript.segments[0].speaker == "speaker_1")
        #expect(transcript.segments[0].text == "hello world")
        #expect(transcript.segments[0].words.map(\.text) == ["hello", "world"])
        #expect(transcript.text == "hello world")
        #expect(transcript.unassignedWordCount == 0)
        #expect(!transcript.limitations.isEmpty)
        // The stub reports fixed engine timings; the total is measured.
        #expect(transcript.processing.asrSeconds == 0.01)
        #expect(transcript.processing.diarizationSeconds == 0.02)
        #expect(transcript.processing.totalSeconds >= 0)

        // The engine saw the concatenation, not two separate recordings.
        #expect(engine.recordings.count == 1)
        #expect(engine.recordings[0].samples.count == Int(3.5 * 16_000))
        #expect(engine.recordings[0].files.map(\.index) == [0, 1])

        // Nothing is left in the work directory once the request is done.
        let leftovers = try FileManager.default.contentsOfDirectory(atPath: server.workDirectory.path)
        #expect(leftovers.isEmpty)

        // A null speaker is encoded as null, not omitted.
        let raw = try #require(String(data: reply.body, encoding: .utf8))
        #expect(raw.contains("\"speaker\":\"speaker_1\""))
        #expect(raw.contains("\"unassigned_word_count\":0"))
        await server.stop()
    }

    @Test func undecodableAndOverlongRecordingsAreRejectedWithoutInference() async throws {
        let engine = StubEngine()
        let server = try await startServer(engine: engine, maxDurationSeconds: 3)
        defer { Task { await server.stop() } }

        let garbage = try await transcribeRequest(
            server, files: [MultipartFile(name: "file", filename: "notes.txt", data: Data("not audio".utf8))])
        #expect(garbage.status == 400)
        let garbageError = try garbage.json(ErrorEnvelope.self).error
        #expect(garbageError.code == "invalid_input")
        #expect(garbageError.message.contains("notes.txt"))

        let long = try await transcribeRequest(
            server,
            files: [
                MultipartFile(name: "file", filename: "a.wav", data: makeWAV(seconds: 2)),
                MultipartFile(name: "file", filename: "b.wav", data: makeWAV(seconds: 2)),
            ])
        #expect(long.status == 413)
        #expect(try long.json(ErrorEnvelope.self).error.code == "payload_too_large")

        let wrongType = try await send(
            "POST", "/v1/transcriptions", to: server, contentType: "application/json", body: Data("{}".utf8))
        #expect(wrongType.status == 400)

        let declaredTooLarge = try await send(
            "POST", "/v1/transcriptions", to: server, contentType: "multipart/form-data; boundary=x",
            body: Data(count: 5_000_001))
        #expect(declaredTooLarge.status == 413)

        #expect(engine.starts == 0)
        let leftovers = try FileManager.default.contentsOfDirectory(atPath: server.workDirectory.path)
        #expect(leftovers.isEmpty)
        await server.stop()
    }

    @Test func concurrentRequestsQueueBehindOneRunningJobAndTheExcessIsRefused() async throws {
        let engine = StubEngine(delay: .milliseconds(600))
        let server = try await startServer(engine: engine, maxQueued: 1)
        defer { Task { await server.stop() } }
        let files = [MultipartFile(name: "file", filename: "a.wav", data: makeWAV(seconds: 1))]

        let statuses = try await withThrowingTaskGroup(of: Int.self, returning: [Int].self) { group in
            for _ in 0..<3 {
                group.addTask { try await transcribeRequest(server, files: files).status }
                try await Task.sleep(for: .milliseconds(100))
            }
            var statuses: [Int] = []
            for try await status in group { statuses.append(status) }
            return statuses.sorted()
        }

        #expect(statuses == [200, 200, 429])
        #expect(engine.starts == 2)
        let health = try await send("GET", "/health", to: server, token: nil).json(HealthReport.self)
        #expect(health.queue == HealthReport.Queue(running: 0, queued: 0, maxQueued: 1))
        await server.stop()
    }

    @Test func aClientThatDisconnectsCancelsItsRunningJob() async throws {
        let engine = StubEngine(delay: .seconds(10))
        let server = try await startServer(engine: engine)
        defer { Task { await server.stop() } }

        let boundary = "cancel-boundary"
        var request = URLRequest(url: server.baseURL.appendingPathComponent("/v1/transcriptions"))
        request.httpMethod = "POST"
        request.setValue("Bearer \(testToken)", forHTTPHeaderField: "Authorization")
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = multipartBody(
            boundary: boundary, files: [MultipartFile(name: "file", filename: "a.wav", data: makeWAV(seconds: 1))])
        let session = URLSession(configuration: .ephemeral)
        let client = Task { try await session.data(for: request) }

        // Wait until the engine is inside its (cancellable) work, then drop
        // the connection.
        for _ in 0..<100 where engine.starts == 0 {
            try await Task.sleep(for: .milliseconds(50))
        }
        #expect(engine.starts == 1)
        client.cancel()
        session.invalidateAndCancel()

        for _ in 0..<100 where engine.cancellations == 0 {
            try await Task.sleep(for: .milliseconds(50))
        }
        #expect(engine.cancellations == 1)

        // The slot is free again for the next caller.
        for _ in 0..<100 where await server.state.gate.report.running != 0 {
            try await Task.sleep(for: .milliseconds(50))
        }
        #expect(await server.state.gate.report == HealthReport.Queue(running: 0, queued: 0, maxQueued: 2))
        await server.stop()
    }
}
