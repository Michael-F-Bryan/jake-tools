// `transcription-service serve` runs the HTTP service; `provision` downloads
// the pinned models. Logs go to stderr and never include audio or transcript
// content.

import ArgumentParser
import Foundation
import Logging
import TranscriptionService

@main
struct TranscriptionServiceCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "transcription-service",
        abstract: "Local speech recognition and speaker diarisation over HTTP, with warm models.",
        version: serviceVersion,
        subcommands: [Serve.self, Provision.self]
    )
}

struct ModelOptions: ParsableArguments {
    @Option(
        name: .customLong("models-dir"),
        help: ArgumentHelp(
            "Directory holding the provisioned models.",
            discussion: "Defaults to $TRANSCRIPTION_MODELS_DIR, then FluidAudio's cache under ~/Library/Application Support/FluidAudio/Models."
        )
    )
    var modelsDir: String?

    var modelsRoot: URL {
        if let modelsDir {
            return URL(fileURLWithPath: modelsDir, isDirectory: true)
        }
        if let fromEnvironment = ProcessInfo.processInfo.environment["TRANSCRIPTION_MODELS_DIR"],
            !fromEnvironment.isEmpty
        {
            return URL(fileURLWithPath: fromEnvironment, isDirectory: true)
        }
        return ModelPins.defaultModelsRoot
    }
}

struct LoggingOptions: ParsableArguments {
    @Option(name: .customLong("log-level"), help: "trace, debug, info, notice, warning, error or critical.")
    var logLevel: String = "info"

    func bootstrap() throws -> Logger {
        guard let level = Logger.Level(rawValue: logLevel) else {
            throw ValidationError("unknown log level \(logLevel)")
        }
        LoggingSystem.bootstrap { label in
            var handler = StreamLogHandler.standardError(label: label)
            handler.logLevel = level
            return handler
        }
        return Logger(label: "transcription-service")
    }
}

struct Serve: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Run the HTTP service.")

    @OptionGroup var models: ModelOptions
    @OptionGroup var logging: LoggingOptions

    @Option(help: "Interface to bind.")
    var host: String = "127.0.0.1"

    @Option(help: "Port to bind.")
    var port: Int = 8787

    @Option(
        name: .customLong("token-file"),
        help: "File holding the bearer token clients must present. Alternatively set TRANSCRIPTION_API_TOKEN."
    )
    var tokenFile: String?

    @Option(name: .customLong("max-upload-bytes"), help: "Largest accepted request body.")
    var maxUploadBytes: Int = 2 * 1024 * 1024 * 1024

    @Option(name: .customLong("max-files"), help: "Most files in one request.")
    var maxFiles: Int = 64

    @Option(name: .customLong("max-duration-seconds"), help: "Longest decoded recording per request.")
    var maxDurationSeconds: Double = 3 * 60 * 60

    @Option(name: .customLong("max-queued"), help: "Requests allowed to wait behind the running one.")
    var maxQueued: Int = 4

    @Option(name: .customLong("work-dir"), help: "Where uploads are staged while a request runs.")
    var workDir: String?

    @Option(name: .customLong("tls-certificate"), help: "PEM certificate chain; enables HTTPS with --tls-private-key.")
    var tlsCertificate: String?

    @Option(name: .customLong("tls-private-key"), help: "PEM private key for --tls-certificate.")
    var tlsPrivateKey: String?

    func validate() throws {
        if (tlsCertificate == nil) != (tlsPrivateKey == nil) {
            throw ValidationError("--tls-certificate and --tls-private-key must be given together")
        }
        guard maxQueued >= 0, maxFiles >= 1, maxUploadBytes >= 1, maxDurationSeconds > 0 else {
            throw ValidationError("limits must be positive")
        }
    }

    func run() async throws {
        let logger = try logging.bootstrap()
        let token = try resolveToken()
        let workDirectory =
            workDir.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? FileManager.default.temporaryDirectory.appendingPathComponent(
                "jake-transcription-service", isDirectory: true)
        do {
            try WorkDirectory.prepare(workDirectory, logger: logger)
        } catch let error as ServiceError {
            throw ValidationError(error.message)
        }

        let configuration = ServiceConfiguration(
            host: host,
            port: port,
            token: token,
            uploadLimits: UploadLimits(maxUploadBytes: maxUploadBytes, maxFiles: maxFiles),
            maxDurationSeconds: maxDurationSeconds,
            maxQueued: maxQueued,
            workDirectory: workDirectory,
            tls: tlsCertificate.flatMap { certificate in
                tlsPrivateKey.map { ServiceConfiguration.TLS(certificateChainPath: certificate, privateKeyPath: $0) }
            })
        let state = ServiceState(configuration: configuration, logger: logger)
        let application = try makeApplication(state: state) { channel in
            logger.info(
                "listening",
                metadata: [
                    "address": "\(channel.localAddress.map(String.init(describing:)) ?? "?")",
                    "tls": "\(configuration.tls != nil)",
                ])
        }
        let modelsRoot = models.modelsRoot

        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { try await application.runService() }
            group.addTask {
                do {
                    let engine = try await FluidAudioEngine.load(modelsRoot: modelsRoot, logger: logger)
                    await state.engine.set(.ready(engine))
                    logger.info("ready", metadata: ["models_root": "\(modelsRoot.path)"])
                } catch {
                    let detail = String(describing: error)
                    await state.engine.set(.failed(detail))
                    logger.critical(
                        "models failed to load or warm up; if files are missing run `transcription-service provision` on this host",
                        metadata: ["models_root": "\(modelsRoot.path)", "error": "\(detail)"])
                    throw ExitCode.failure
                }
            }
            // The first task to finish is either the loader (ready, or failed
            // and throwing) or the server (signalled). A load failure must
            // take the server down with it, explicitly: the process exits
            // non-zero so a supervisor can act, instead of serving `failed`
            // forever.
            do {
                try await group.next()
            } catch {
                group.cancelAll()
                throw error
            }
            try await group.next()
        }
    }

    private func resolveToken() throws -> String {
        if let tokenFile {
            let raw: String
            do {
                raw = try String(contentsOfFile: tokenFile, encoding: .utf8)
            } catch {
                throw ValidationError("cannot read --token-file")
            }
            let token = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !token.isEmpty else { throw ValidationError("--token-file is empty") }
            return token
        }
        if let token = ProcessInfo.processInfo.environment["TRANSCRIPTION_API_TOKEN"]?
            .trimmingCharacters(in: .whitespacesAndNewlines), !token.isEmpty
        {
            return token
        }
        throw ValidationError("a bearer token is required: pass --token-file or set TRANSCRIPTION_API_TOKEN")
    }
}

struct Provision: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Download the pinned models so `serve` can run offline.")

    @OptionGroup var models: ModelOptions
    @OptionGroup var logging: LoggingOptions

    func run() async throws {
        let logger = try logging.bootstrap()
        try await FluidAudioEngine.provision(modelsRoot: models.modelsRoot, logger: logger)
    }
}
