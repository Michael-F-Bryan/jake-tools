// Bounded admission: one inference job runs at a time and at most `maxQueued`
// wait behind it, first come first served. A waiter whose task is cancelled
// (its client went away) leaves the queue without taking a turn.
//
// Concurrent HTTP connections therefore never mean concurrent model
// execution, and a burst beyond the queue bound is refused immediately with
// `overloaded` rather than accepted and starved.

import Foundation

public actor JobGate {
    public let maxQueued: Int

    private var running = false
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
        guard waiters.count < maxQueued else {
            throw ServiceError(
                .overloaded,
                "the transcription queue is full (\(maxQueued) waiting); retry later",
                retryAfterSeconds: 30)
        }
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

    /// Run `body` holding the slot.
    public func withSlot<T: Sendable>(_ body: @Sendable () async throws -> T) async throws -> T {
        try await acquire()
        defer { release() }
        return try await body()
    }

    private func cancelWaiter(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(throwing: CancellationError())
    }
}
