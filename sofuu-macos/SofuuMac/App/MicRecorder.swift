import AVFoundation
import Foundation

/// Voice-note capture for the composer's mic: input tap → 16-bit PCM WAV
/// in a temp file. Runtime data is handed to the engine's
/// `sofuu.ai.transcribe` (OpenAI-compatible audio funnel) by the store —
/// this class only records. Mic permission is the AVCaptureDevice global
/// prompt (AVAudioSession record Permission does not exist on macOS).
final class MicRecorder: NSObject {
    enum MicError: LocalizedError {
        case noInput
        case busy

        var errorDescription: String? {
            switch self {
            case .noInput: return "No microphone input is available."
            case .busy: return "A recording is already running."
            }
        }
    }

    private let engine = AVAudioEngine()
    private var file: AVAudioFile?
    private let url = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("sofuu-voice-\(UUID().uuidString).wav")

    static func requestAccess(_ done: @escaping (Bool) -> Void) {
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            DispatchQueue.main.async { done(granted) }
        }
    }

    func start() throws {
        guard !engine.isRunning else { throw MicError.busy }
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else { throw MicError.noInput }
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: format.sampleRate,
            AVNumberOfChannelsKey: format.channelCount,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        file = try AVAudioFile(forWriting: url, settings: settings)
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
            guard let self, let file = self.file else { return }
            try? file.write(from: buffer)
        }
        engine.prepare()
        try engine.start()
    }

    func stop() -> Data? {
        if engine.isRunning {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }
        file = nil
        let data = try? Data(contentsOf: url)
        try? FileManager.default.removeItem(at: url)
        return data
    }

    deinit {
        if engine.isRunning {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }
    }
}
