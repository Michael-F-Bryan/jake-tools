import Synchronization
import Testing

@testable import TranscriptionService

@Suite struct JobGateTests {
    @Test func onlyOneJobRunsAtATimeAndWaitersRunInOrder() async throws {
        let gate = JobGate(maxQueued: 4)
        let log = Mutex<[String]>([])
        let running = Mutex(0)

        try await withThrowingTaskGroup(of: Void.self) { group in
            for index in 0..<4 {
                group.addTask {
                    try await gate.withSlot {
                        let concurrent = running.withLock { $0 += 1; return $0 }
                        #expect(concurrent == 1)
                        log.withLock { $0.append("start \(index)") }
                        try await Task.sleep(for: .milliseconds(20))
                        running.withLock { $0 -= 1 }
                        log.withLock { $0.append("end \(index)") }
                    }
                }
                // Stagger admission so the order of arrival is defined.
                try await Task.sleep(for: .milliseconds(5))
            }
            try await group.waitForAll()
        }

        let entries = log.withLock { $0 }
        #expect(entries.count == 8)
        for pair in stride(from: 0, to: 8, by: 2) {
            #expect(entries[pair].hasPrefix("start "))
            #expect(entries[pair + 1].hasPrefix("end "))
        }
        #expect(await gate.report == HealthReport.Queue(running: 0, queued: 0, maxQueued: 4))
    }

    @Test func theQueueBoundRefusesTheNextArrivalWithoutStartingIt() async throws {
        let gate = JobGate(maxQueued: 1)
        try await gate.acquire()
        let waiter = Task { try await gate.withSlot { "waited" } }
        try await Task.sleep(for: .milliseconds(50))
        #expect(await gate.report == HealthReport.Queue(running: 1, queued: 1, maxQueued: 1))

        await #expect(throws: ServiceError.self) {
            try await gate.acquire()
        }
        do {
            try await gate.acquire()
        } catch let error as ServiceError {
            #expect(error.code == .overloaded)
            #expect(error.retryAfterSeconds != nil)
        }

        await gate.release()
        #expect(try await waiter.value == "waited")
        #expect(await gate.report == HealthReport.Queue(running: 0, queued: 0, maxQueued: 1))
    }

    @Test func aCancelledWaiterLeavesTheQueueAndNeverRuns() async throws {
        let gate = JobGate(maxQueued: 2)
        try await gate.acquire()
        let ran = Mutex(false)
        let waiter = Task {
            try await gate.withSlot { ran.withLock { $0 = true } }
        }
        try await Task.sleep(for: .milliseconds(50))
        #expect(await gate.report.queued == 1)

        waiter.cancel()
        await #expect(throws: CancellationError.self) { try await waiter.value }
        #expect(await gate.report.queued == 0)

        await gate.release()
        try await Task.sleep(for: .milliseconds(20))
        #expect(ran.withLock { $0 } == false)
        #expect(await gate.report.running == 0)
    }
}
