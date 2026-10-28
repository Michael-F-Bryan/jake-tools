// The staging directory (`--work-dir`). Each request uploads into its own
// UUID-named subdirectory and removes it when it ends; one only outlives its
// request if the process dies, so startup clears those.
//
// Startup deletes nothing it did not create. A directory the service creates
// or adopts gets a marker file; only directories carrying the marker are
// cleaned, and only their UUID-named subdirectories. A non-empty directory
// without the marker is refused, so a mistaken `--work-dir ~` fails instead
// of wiping the home directory.

import Foundation
import Logging

public enum WorkDirectory {
    public static let markerName = ".jake-transcription-service"

    /// Create or adopt `directory` (owner-only), then remove stale request
    /// directories. Returns how many were removed.
    @discardableResult
    public static func prepare(_ directory: URL, logger: Logger) throws -> Int {
        let fileManager = FileManager.default
        let marker = directory.appendingPathComponent(markerName, isDirectory: false)
        var isDirectory: ObjCBool = false
        if fileManager.fileExists(atPath: directory.path, isDirectory: &isDirectory) {
            guard isDirectory.boolValue else {
                throw ServiceError(.internalError, "--work-dir \(directory.path) exists and is not a directory")
            }
            if !fileManager.fileExists(atPath: marker.path) {
                let entries = try fileManager.contentsOfDirectory(atPath: directory.path)
                // Only request directories may be adopted: an older build
                // made the default work directory without a marker.
                guard entries.allSatisfy(isRequestDirectoryName) else {
                    throw ServiceError(
                        .internalError,
                        "refusing --work-dir \(directory.path): it is not empty and was not created by "
                            + "this service (no \(markerName)); pass a new or empty directory")
                }
            }
        } else {
            try fileManager.createDirectory(
                at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        }
        try fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        if !fileManager.fileExists(atPath: marker.path) {
            try Data(
                "Staging directory of transcription-service. Its UUID-named subdirectories are removed at startup.\n"
                    .utf8
            ).write(to: marker)
        }

        var removed = 0
        for name in try fileManager.contentsOfDirectory(atPath: directory.path) where isRequestDirectoryName(name) {
            let item = directory.appendingPathComponent(name, isDirectory: true)
            var itemIsDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: item.path, isDirectory: &itemIsDirectory), itemIsDirectory.boolValue
            else { continue }
            if (try? fileManager.removeItem(at: item)) != nil {
                removed += 1
            }
        }
        if removed > 0 {
            logger.warning("removed stale upload directories", metadata: ["count": "\(removed)"])
        }
        return removed
    }

    /// Whether `name` is one the server gives a request's directory.
    static func isRequestDirectoryName(_ name: String) -> Bool {
        UUID(uuidString: name)?.uuidString == name
    }

    /// A new directory for one request, under `root`.
    static func requestDirectory(in root: URL) -> URL {
        root.appendingPathComponent(UUID().uuidString, isDirectory: true)
    }
}
