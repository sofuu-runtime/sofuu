import Foundation
import CoreServices

/// Workspace file watching (FSEvents, file-level granularity). The agent
/// edits files outside the process constantly; without this the tree and
/// open tabs only caught up on manual refresh. Events are delivered on
/// the main run loop; the store debounces downstream.
final class FileWatcher {
    private var stream: FSEventStreamRef?
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
            guard let list = unsafeBitCast(eventPaths, to: CFArray.self) as? [String] else { return }
            watcher.onEvent(list)
        }
        guard let stream = FSEventStreamCreate(
            kCFAllocatorDefault,
            callback,
            &context,
            [root] as CFArray,
            FSEventStreamEventId(kFSEventStreamEventIdSinceNow),
            0.3,
            UInt32(kFSEventStreamCreateFlagFileEvents | kFSEventStreamCreateFlagNoDefer)) else {
            // Balance the passRetained above — nothing owns the stream.
            Unmanaged.passUnretained(self).release()
            return nil
        }
        self.stream = stream
        FSEventStreamScheduleWithRunLoop(stream, CFRunLoopGetMain(), CFRunLoopMode.commonModes.rawValue)
        FSEventStreamStart(stream)
    }

    deinit { stop() }

    func stop() {
        guard let stream else { return }
        self.stream = nil
        FSEventStreamStop(stream)
        FSEventStreamInvalidate(stream)
        FSEventStreamRelease(stream)
        // Balance init's passRetained.
        Unmanaged.passUnretained(self).release()
    }
}
