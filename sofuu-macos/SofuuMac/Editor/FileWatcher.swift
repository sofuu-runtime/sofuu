import Foundation
import CoreServices

/// Workspace file watching (FSEvents, file-level granularity). The agent
/// edits files outside the process constantly; without this the tree and
/// open tabs only caught up on manual refresh.
///
/// Callback contract, the part that crashed once: WITHOUT the
/// UseCFTypes flag FSEvents passes `eventPaths` as a plain C array of
/// char* — treating that as a CFArray faults in the ObjC bridge
/// (observed EXC_BAD_ACCESS with count > 0). The flag makes it a real
/// CFArray of CFStrings, decoded here by hand (no ObjC dynamic cast).
/// Events arrive on a private serial queue and hop to main for the store.
final class FileWatcher {
    private var stream: FSEventStreamRef?
    private let queue = DispatchQueue(label: "sofuu.fswatch", qos: .utility)
    private let onEvent: ([String]) -> Void

    init?(root: String, onEvent: @escaping ([String]) -> Void) {
        self.onEvent = onEvent
        var context = FSEventStreamContext(
            version: 0,
            info: Unmanaged.passRetained(self).toOpaque(),
            retain: nil, release: nil, copyDescription: nil)
        let callback: FSEventStreamCallback = { _, info, count, eventPaths, _, _ in
            guard let info, count > 0 else { return }
            let watcher = Unmanaged<FileWatcher>.fromOpaque(info).takeUnretainedValue()
            let cfList = unsafeBitCast(eventPaths, to: CFArray.self)
            var paths: [String] = []
            paths.reserveCapacity(count)
            for index in 0..<count {
                let value = unsafeBitCast(CFArrayGetValueAtIndex(cfList, index), to: CFString.self)
                paths.append(value as String)
            }
            // The dispatch queue below is NOT main — the store is.
            DispatchQueue.main.async {
                watcher.onEvent(paths)
            }
        }
        guard let stream = FSEventStreamCreate(
            kCFAllocatorDefault,
            callback,
            &context,
            [root] as CFArray,
            FSEventStreamEventId(kFSEventStreamEventIdSinceNow),
            0.3,
            UInt32(kFSEventStreamCreateFlagFileEvents
                   | kFSEventStreamCreateFlagNoDefer
                   | kFSEventStreamCreateFlagUseCFTypes)) else {
            // Balance the passRetained above — nothing owns the stream.
            Unmanaged.passUnretained(self).release()
            return nil
        }
        self.stream = stream
        // The dispatch-queue form replaces the deprecated run-loop
        // scheduling; the queue must be detached BEFORE the stream's
        // final release (done in stop()).
        FSEventStreamSetDispatchQueue(stream, queue)
        FSEventStreamStart(stream)
    }

    deinit { stop() }

    func stop() {
        guard let stream else { return }
        self.stream = nil
        FSEventStreamStop(stream)
        FSEventStreamInvalidate(stream)
        FSEventStreamSetDispatchQueue(stream, nil)
        FSEventStreamRelease(stream)
        // Balance init's passRetained.
        Unmanaged.passUnretained(self).release()
    }
}
