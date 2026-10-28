import Foundation
import Logging
import Testing

@testable import TranscriptionService

@Suite struct WorkDirectoryTests {
    let logger = Logger(label: "work-directory-tests")

    @Test func aForeignNonEmptyDirectoryIsRefusedAndLeftUntouched() throws {
        let directory = try temporaryDirectory()
        let notes = directory.appendingPathComponent("notes", isDirectory: true)
        try FileManager.default.createDirectory(at: notes, withIntermediateDirectories: true)
        try Data("keep".utf8).write(to: directory.appendingPathComponent("important.txt"))
        try Data("keep".utf8).write(to: notes.appendingPathComponent("a.md"))
        let stale = directory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: stale, withIntermediateDirectories: true)
        let before = try FileManager.default.attributesOfItem(atPath: directory.path)[.posixPermissions] as? Int

        #expect(throws: ServiceError.self) { try WorkDirectory.prepare(directory, logger: logger) }

        let entries = try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
        #expect(entries == [stale.lastPathComponent, "important.txt", "notes"].sorted())
        #expect(FileManager.default.fileExists(atPath: notes.appendingPathComponent("a.md").path))
        #expect(try FileManager.default.attributesOfItem(atPath: directory.path)[.posixPermissions] as? Int == before)
    }

    @Test func aMarkedDirectoryLosesOnlyItsRequestDirectories() throws {
        let directory = try temporaryDirectory().appendingPathComponent("work", isDirectory: true)
        #expect(try WorkDirectory.prepare(directory, logger: logger) == 0)
        #expect(FileManager.default.fileExists(atPath: directory.appendingPathComponent(WorkDirectory.markerName).path))
        #expect(try FileManager.default.attributesOfItem(atPath: directory.path)[.posixPermissions] as? Int == 0o700)

        let stale = WorkDirectory.requestDirectory(in: directory)
        try FileManager.default.createDirectory(at: stale, withIntermediateDirectories: true)
        try Data("audio".utf8).write(to: stale.appendingPathComponent("0-a.m4a"))
        let other = directory.appendingPathComponent("someone-elses.txt")
        try Data("keep".utf8).write(to: other)
        let uuidNamedFile = directory.appendingPathComponent(UUID().uuidString)
        try Data("keep".utf8).write(to: uuidNamedFile)

        #expect(try WorkDirectory.prepare(directory, logger: logger) == 1)
        #expect(!FileManager.default.fileExists(atPath: stale.path))
        #expect(FileManager.default.fileExists(atPath: other.path))
        #expect(FileManager.default.fileExists(atPath: uuidNamedFile.path))
    }

    @Test func anUnmarkedDirectoryHoldingOnlyRequestDirectoriesIsAdopted() throws {
        // What an older build left in the default work directory.
        let directory = try temporaryDirectory()
        let stale = WorkDirectory.requestDirectory(in: directory)
        try FileManager.default.createDirectory(at: stale, withIntermediateDirectories: true)

        #expect(try WorkDirectory.prepare(directory, logger: logger) == 1)
        #expect(FileManager.default.fileExists(atPath: directory.appendingPathComponent(WorkDirectory.markerName).path))
    }

    @Test func aRegularFileIsRefused() throws {
        let file = try temporaryDirectory().appendingPathComponent("file")
        try Data("x".utf8).write(to: file)
        #expect(throws: ServiceError.self) { try WorkDirectory.prepare(file, logger: logger) }
        #expect(try String(contentsOf: file, encoding: .utf8) == "x")
    }
}

@Suite struct ModelRevisionTests {
    @Test func directoriesFromAnotherRevisionAreReportedAndMissingOnesAreNot() throws {
        let root = try temporaryDirectory()
        #expect(ModelPins.revisionMismatches(modelsRoot: root).isEmpty)

        let asr = ModelPins.asrDirectory(modelsRoot: root)
        try FileManager.default.createDirectory(at: asr, withIntermediateDirectories: true)
        let marker = asr.appendingPathComponent(ModelPins.revisionMarkerName)
        try Data("0123abcd\n".utf8).write(to: marker)
        let mismatches = ModelPins.revisionMismatches(modelsRoot: root)
        #expect(mismatches.map(\.directory) == [asr])
        #expect(mismatches.first?.found == "0123abcd")
        #expect(String(describing: ModelRevisionError(mismatches: mismatches)).contains("provision"))

        try Data("\(ModelPins.asrRevision)\n".utf8).write(to: marker)
        #expect(ModelPins.revisionMismatches(modelsRoot: root).isEmpty)
    }
}
