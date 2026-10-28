// The HTTP service: two routes on a Hummingbird application.
//
// - `GET /health` (no auth): readiness for supervision. 200 once the models
//   are loaded and warm, 503 while starting or after a load failure.
// - `POST /v1/transcriptions` (bearer token): a multipart upload of ordered
//   recordings; returns the complete `Transcript`, or a `ServiceError` body.
//
// Client disconnect: Hummingbird does not cancel a handler when the socket
// closes, so the upload is consumed through `consumeWithInboundCloseHandler`
// and the request's work runs in a task that the close handler cancels. A
// queued job then leaves the queue; a running job stops at its next
// cancellation check (the decoder between buffers, ASR per chunk and decode
// step, the diarizer during PLDA scoring), which is not immediate.

import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdCore
import HummingbirdTLS
import Logging
import NIOCore
import NIOSSL
import Synchronization

public struct ServiceConfiguration: Sendable {
    public var host: String
    public var port: Int
    /// The bearer token every `POST` must carry.
    public var token: String
    public var uploadLimits: UploadLimits
    public var maxDurationSeconds: Double
    public var maxQueued: Int
    /// Per-request upload directories are created under here and removed
    /// when the request finishes, however it finishes.
    public var workDirectory: URL
    public var tls: TLS?

    public struct TLS: Sendable {
        public var certificateChainPath: String
        public var privateKeyPath: String

        public init(certificateChainPath: String, privateKeyPath: String) {
            self.certificateChainPath = certificateChainPath
            self.privateKeyPath = privateKeyPath
        }
    }

    public init(
        host: String,
        port: Int,
        token: String,
        uploadLimits: UploadLimits,
        maxDurationSeconds: Double,
        maxQueued: Int,
        workDirectory: URL,
        tls: TLS? = nil
    ) {
        self.host = host
        self.port = port
        self.token = token
        self.uploadLimits = uploadLimits
        self.maxDurationSeconds = maxDurationSeconds
        self.maxQueued = maxQueued
        self.workDirectory = workDirectory
        self.tls = tls
    }
}

/// The engine's lifecycle as the routes see it.
public actor EngineSlot {
    public enum Phase: Sendable {
        case starting
        case ready(any TranscriptionEngine)
        case failed(String)
    }

    public private(set) var phase: Phase = .starting

    public init() {}

    public func set(_ phase: Phase) {
        self.phase = phase
    }
}

/// Everything the routes share. One per process.
public final class ServiceState: Sendable {
    public let configuration: ServiceConfiguration
    public let gate: JobGate
    public let engine: EngineSlot
    public let logger: Logger

    public init(configuration: ServiceConfiguration, logger: Logger) {
        self.configuration = configuration
        self.gate = JobGate(maxQueued: configuration.maxQueued)
        self.engine = EngineSlot()
        self.logger = logger
    }
}

/// Build the application. `onServerRunning` receives the bound channel, which
/// is how the tests learn an ephemeral port.
public func makeApplication(
    state: ServiceState,
    onServerRunning: @escaping @Sendable (any Channel) async -> Void = { _ in }
) throws -> some ApplicationProtocol {
    let router = Router()
    router.add(middleware: LogRequestsMiddleware(.info))
    registerRoutes(router, state: state)

    let server: HTTPServerBuilder
    if let tls = state.configuration.tls {
        let certificates = try NIOSSLCertificate.fromPEMFile(tls.certificateChainPath)
        let key = try NIOSSLPrivateKey(file: tls.privateKeyPath, format: .pem)
        var tlsConfiguration = TLSConfiguration.makeServerConfiguration(
            certificateChain: certificates.map { .certificate($0) }, privateKey: .privateKey(key))
        tlsConfiguration.minimumTLSVersion = .tlsv12
        server = try .tls(.http1(), tlsConfiguration: tlsConfiguration)
    } else {
        server = .http1()
    }

    return Application(
        router: router,
        server: server,
        configuration: .init(
            address: .hostname(state.configuration.host, port: state.configuration.port),
            serverName: "jake-transcription-service"),
        onServerRunning: onServerRunning,
        logger: state.logger)
}

func registerRoutes(_ router: Router<BasicRequestContext>, state: ServiceState) {
    router.get("/health") { _, _ -> Response in
        let queue = await state.gate.report
        let report: HealthReport
        switch await state.engine.phase {
        case .starting:
            report = HealthReport(
                status: "starting", ready: false, version: serviceVersion, engine: nil, detail: nil,
                queue: queue)
        case .ready(let engine):
            report = HealthReport(
                status: "ready", ready: true, version: serviceVersion, engine: engine.info, detail: nil,
                queue: queue)
        case .failed(let detail):
            report = HealthReport(
                status: "failed", ready: false, version: serviceVersion, engine: nil, detail: detail,
                queue: queue)
        }
        return try jsonResponse(status: report.ready ? .ok : .serviceUnavailable, report)
    }

    router.post("/v1/transcriptions") { request, context -> Response in
        try authenticate(request, token: state.configuration.token)
        let engine: any TranscriptionEngine
        switch await state.engine.phase {
        case .starting:
            throw ServiceError(.notReady, "the transcription models are still loading", retryAfterSeconds: 15)
        case .failed(let detail):
            throw ServiceError(.notReady, "the transcription models failed to load: \(detail)")
        case .ready(let ready):
            engine = ready
        }
        let boundary = try multipartBoundary(contentType: request.headers[.contentType])
        if let declared = request.headers[.contentLength].flatMap(Int.init),
            declared > state.configuration.uploadLimits.maxUploadBytes
        {
            throw ServiceError(
                .payloadTooLarge,
                "the upload exceeds the limit of \(state.configuration.uploadLimits.maxUploadBytes) bytes")
        }

        let requestID = context.id
        let requestBody = request.body
        // Admission first: an overloaded service refuses before reading a
        // byte of the body.
        return try await state.gate.withAdmission {
            let job = CancellableJob()
            return try await requestBody.consumeWithInboundCloseHandler { body in
                let task = Task {
                    try await process(
                        body: body, boundary: boundary, engine: engine, state: state, requestID: requestID)
                }
                job.attach(task)
                defer { job.finish() }
                return try await withTaskCancellationHandler {
                    try await task.value
                } onCancel: {
                    task.cancel()
                }
            } onInboundClosed: {
                // Also fires when a client closes after its response; only
                // unfinished work is worth a log line.
                if job.cancel() {
                    state.logger.info(
                        "client disconnected; cancelling request", metadata: ["request": "\(requestID)"])
                }
            }
        }
    }
}

/// Constant-time bearer token check.
func authenticate(_ request: Request, token: String) throws {
    guard let header = request.headers[.authorization] else {
        throw ServiceError(.unauthorized, "missing Authorization: Bearer token")
    }
    let prefix = "Bearer "
    guard header.hasPrefix(prefix) else {
        throw ServiceError(.unauthorized, "Authorization must be a Bearer token")
    }
    let presented = Array(header.dropFirst(prefix.count).utf8)
    let expected = Array(token.utf8)
    var difference = presented.count ^ expected.count
    for index in 0..<max(presented.count, expected.count) {
        let a = index < presented.count ? presented[index] : 0
        let b = index < expected.count ? expected[index] : 0
        difference |= Int(a ^ b)
    }
    guard difference == 0 else {
        throw ServiceError(.unauthorized, "the bearer token was rejected")
    }
}

/// Upload, queue, decode, infer, reconcile. Owns the request's temporary
/// directory for exactly as long as it runs.
private func process(
    body: RequestBody,
    boundary: String,
    engine: any TranscriptionEngine,
    state: ServiceState,
    requestID: String
) async throws -> Response {
    let clock = ContinuousClock()
    let started = clock.now
    let logger = state.logger
    let configuration = state.configuration

    let directory = WorkDirectory.requestDirectory(in: configuration.workDirectory)
    try FileManager.default.createDirectory(
        at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
    defer { try? FileManager.default.removeItem(at: directory) }

    let uploads = try await receiveUploads(
        body: body, boundary: boundary, into: directory, limits: configuration.uploadLimits)
    let uploadedBytes = uploads.reduce(0) { $0 + $1.bytes }
    logger.info(
        "upload received",
        metadata: ["request": "\(requestID)", "files": "\(uploads.count)", "bytes": "\(uploadedBytes)"])
    try Task.checkCancellation()

    return try await state.gate.withSlot {
        let decodeStart = clock.now
        let recording = try await AudioDecoder.decode(
            uploads, maxDurationSeconds: configuration.maxDurationSeconds)
        let decodeSeconds = elapsed(since: decodeStart, clock: clock)
        logger.info(
            "recording decoded",
            metadata: [
                "request": "\(requestID)", "duration_seconds": "\(recording.durationSeconds)",
                "decode_seconds": "\(decodeSeconds)",
            ])

        let output: EngineOutput
        do {
            output = try await engine.transcribe(recording)
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as ServiceError {
            throw error
        } catch {
            logger.error(
                "inference failed", metadata: ["request": "\(requestID)", "error": "\(String(describing: error))"])
            throw ServiceError(.inferenceFailed, "the transcription engine failed on this recording")
        }

        let reconciled = Reconcile.reconcile(words: output.words, turns: output.turns)
        let transcript = Transcript(
            engine: engine.info,
            durationSeconds: recording.durationSeconds,
            files: recording.files,
            speakers: reconciled.speakers,
            segments: reconciled.segments,
            text: output.text,
            unassignedWordCount: reconciled.unassignedWordCount,
            limitations: engineLimitations,
            processing: ProcessingTimings(
                decodeSeconds: decodeSeconds,
                asrSeconds: output.asrSeconds,
                diarizationSeconds: output.diarizationSeconds,
                totalSeconds: elapsed(since: started, clock: clock)))
        logger.info(
            "transcript ready",
            metadata: [
                "request": "\(requestID)", "speakers": "\(reconciled.speakers.count)",
                "segments": "\(reconciled.segments.count)", "words": "\(output.words.count)",
                "asr_seconds": "\(output.asrSeconds)", "diarization_seconds": "\(output.diarizationSeconds)",
                "total_seconds": "\(transcript.processing.totalSeconds)",
            ])
        return try jsonResponse(status: .ok, transcript)
    }
}

/// Holds the request's task so the inbound-close handler can cancel it even
/// if the close is noticed before the task exists.
final class CancellableJob: Sendable {
    private struct State {
        var task: Task<Response, any Error>?
        var cancelled = false
        var finished = false
    }

    private let state = Mutex(State())

    func attach(_ task: Task<Response, any Error>) {
        let cancelNow = state.withLock { state in
            state.task = task
            return state.cancelled
        }
        if cancelNow { task.cancel() }
    }

    /// The request has its response (or its error); a later cancel is moot.
    func finish() {
        state.withLock { $0.finished = true }
    }

    /// Cancel the work unless it already finished. Returns whether there was
    /// unfinished work to cancel.
    @discardableResult
    func cancel() -> Bool {
        let (task, unfinished) = state.withLock { state in
            state.cancelled = true
            return (state.task, !state.finished)
        }
        guard unfinished else { return false }
        task?.cancel()
        return true
    }
}

func elapsed(since start: ContinuousClock.Instant, clock: ContinuousClock) -> Double {
    let (seconds, attoseconds) = (clock.now - start).components
    return Double(seconds) + Double(attoseconds) / 1e18
}
