// Every failure the service reports. The HTTP status says which class of
// failure it is; `code` is the stable string the Python client maps onto its
// own error codes. Messages name files by their uploaded name and limits by
// their value, never audio or transcript content.

import HTTPTypes
import Hummingbird
import NIOCore

public struct ServiceError: Error, Sendable, Equatable {
    public enum Code: String, Sendable {
        /// Missing or wrong bearer token.
        case unauthorized
        /// Malformed multipart body, no files, an undecodable file, or a file
        /// that is not audio.
        case invalidInput = "invalid_input"
        /// The upload, file count or decoded duration exceeds a configured
        /// limit.
        case payloadTooLarge = "payload_too_large"
        /// Models are still loading, or failed to load.
        case notReady = "not_ready"
        /// The bounded queue is full. Nothing was started.
        case overloaded
        /// The engine failed on an accepted recording.
        case inferenceFailed = "inference_failed"
        case internalError = "internal_error"

        public var status: HTTPResponse.Status {
            switch self {
            case .unauthorized: .unauthorized
            case .invalidInput: .badRequest
            case .payloadTooLarge: .contentTooLarge
            case .notReady: .serviceUnavailable
            case .overloaded: .tooManyRequests
            case .inferenceFailed: .internalServerError
            case .internalError: .internalServerError
            }
        }
    }

    public var code: Code
    public var message: String
    public var retryAfterSeconds: Int?

    public init(_ code: Code, _ message: String, retryAfterSeconds: Int? = nil) {
        self.code = code
        self.message = message
        self.retryAfterSeconds = retryAfterSeconds
    }

    public var envelope: ErrorEnvelope { ErrorEnvelope(code: code.rawValue, message: message) }
}

extension ServiceError: HTTPResponseError {
    public var status: HTTPResponse.Status { code.status }

    public func response(from request: Request, context: some RequestContext) throws -> Response {
        try jsonResponse(status: status, envelope, retryAfterSeconds: retryAfterSeconds)
    }
}

/// Encode a contract value as the body of a JSON response.
public func jsonResponse<Body: Encodable>(
    status: HTTPResponse.Status, _ body: Body, retryAfterSeconds: Int? = nil
) throws -> Response {
    let data = try Wire.encoder().encode(body)
    var headers: HTTPFields = [.contentType: "application/json; charset=utf-8"]
    if let retryAfterSeconds {
        headers[.retryAfter] = String(retryAfterSeconds)
    }
    return Response(status: status, headers: headers, body: .init(byteBuffer: ByteBuffer(data: data)))
}
