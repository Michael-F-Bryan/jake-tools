// Bounded admission: one inference job runs at a time and at most `maxQueued`
// wait behind it, first come first served. A waiter whose task is cancelled
// (its client went away) leaves the queue without taking a turn.
//
// Admission is decided before a request's body is read: `withAdmission`
// counts every request from the start of its upload to its response, and
// refuses one beyond `maxQueued + 1` immediately with `overloaded`. So a
// rejected client never uploads its recording first, and staged uploads
// never exceed `maxQueued + 1` requests' worth. Concurrent HTTP connections
// never mean concurrent model execution.

import Foundation

public actor JobGate {
    public let maxQueued: Int

    private var running = false
    private var admitted = 0
    private var waiters: [(id: UUID, continuation: CheckedContinuation<Void, any Error>)] = []

    public init(maxQueued: Int) {
        self.maxQueued = max(0, maxQueued)
    }

    public var report: HealthReport.Queue {
        HealthReport.Queue(running: running ? 1 : 0, queued: waiters.count, maxQueued: maxQueued)
    }

    /// Wait for the single running slot. Throws `ServiceError(.overloaded)` when
    /// the queue is full and `CancellationError` if the caller is cancelled
    /// while waiting. Every successful return must be paired with `release()`.
    public func acquire() async throws {
        if !running, waiters.isEmpty {
            running = true
            return
        }
        guard waiters.count < maxQueued else { throw overloaded() }
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                // The cancellation handler may already have fired, before this
                // waiter existed; honour it instead of waiting for a turn that
                // nobody wants.
                if Task.isCancelled {
                    continuation.resume(throwing: CancellationError())
                    return
                }
                waiters.append((id, continuation))
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
    }

    /// Hand the slot to the next waiter, or mark it free.
    public func release() {
        guard running else { return }
        if waiters.isEmpty {
            running = false
        } else {
            // `running` stays true: the slot moves straight to the next waiter.
            let next = waiters.removeFirst()
            next.continuation.resume()
        }
    }

    /// Run `body`, which uploads and then takes the slot with `withSlot`, as
    /// one of at most `maxQueued + 1` admitted requests. Throws
    /// `ServiceError(.overloaded)` without running `body` when that many are
    /// already admitted.
    public func withAdmission<T: Sendable>(_ body: @Sendable () async throws -> T) async throws -> T {
        guard admitted <= maxQueued else { throw overloaded() }
        admitted += 1
        defer { admitted -= 1 }
        return try await body()
    }

    /// Run `body` holding the slot.
    public func withSlot<T: Sendable>(_ body: @Sendable () async throws -> T) async throws -> T {
        try await acquire()
        defer { release() }
        return try await body()
    }

    private func overloaded() -> ServiceError {
        ServiceError(
            .overloaded, "the transcription queue is full (\(maxQueued) waiting); retry later",
            retryAfterSeconds: 30)
    }

    private func cancelWaiter(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(throwing: CancellationError())
    }
}
