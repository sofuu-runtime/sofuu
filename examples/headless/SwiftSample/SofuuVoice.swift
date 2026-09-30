// examples/headless/SwiftSample/SofuuVoice.swift
//
// M2: on-device voice for Sofuu iOS apps — OS speech APIs, zero servers,
// zero per-minute metering, works in airplane mode. Orchestrate with the
// agent loop: mic → STT → SofuuBridge.call("agent.run") → TTS.
//
// App capabilities (Info.plist):
//   NSSpeechRecognitionUsageDescription = "..."
//   NSMicrophoneUsageDescription = "..."
// Background audio (if speaking while backgrounded): UIBackgroundModes=audio.

import AVFoundation
import Foundation
import Speech

/// On-device speech I/O. All recognition can run on-device
/// (`onDevice: true` — no audio leaves the phone); synthesis is always
/// local. Provider STT/TTS via libsofuu remains the fallback for
/// languages the on-device models lack (see SofuuBridge voice funnel).
public enum SofuuVoice {

    // MARK: - Permissions

    /// Requests Speech + microphone authorization. Returns true when both
    /// are granted. Call once, early (e.g. onboarding).
    public static func requestPermissions() async -> Bool {
        let speech = await withCheckedContinuation { cont in
            SFSpeechRecognizer.requestAuthorization { cont.resume(returning: $0) }
        }
        guard speech == .authorized else { return false }
        return await withCheckedContinuation { cont in
            AVAudioApplication.requestRecordPermission { cont.resume(returning: $0) }
        }
    }

    // MARK: - File transcription (SFSpeechURLRecognitionRequest)

    /// Transcribe an audio file. On-device when the locale model is
    /// downloaded (Settings → General → Language); else falls back to
    /// server recognition unless `requiresOnDevice` pins local-only.
    public static func transcribeFile(
        _ url: URL,
        locale: Locale = .current,
        requiresOnDevice: Bool = true
    ) async throws -> String {
        guard let recognizer = SFSpeechRecognizer(locale: locale),
              recognizer.isAvailable else {
            throw VoiceError.unavailable
        }
        let req = SFSpeechURLRecognitionRequest(url: url)
        req.requiresOnDeviceRecognition = requiresOnDevice
        req.shouldReportPartialResults = false
        return try await withCheckedThrowingContinuation { cont in
            var done = false
            recognizer.recognitionTask(with: req) { result, error in
                guard !done else { return }
                if let result, result.isFinal {
                    done = true
                    cont.resume(returning: result.bestTranscription.formattedString)
                } else if let error {
                    done = true
                    cont.resume(throwing: error)
                } else if result == nil, error == nil {
                    // No result and no error: empty audio.
                    done = true
                    cont.resume(returning: "")
                }
            }
        }
    }

    // MARK: - Live mic transcription

    /// Streams mic audio into recognition; `onPartial` fires per hypothesis.
    /// Returns a stopper — call it to end capture and receive the final text
    /// via `onFinal`. One session at a time (audio engine is exclusive).
    @discardableResult
    public static func startLiveTranscription(
        locale: Locale = .current,
        requiresOnDevice: Bool = true,
        onPartial: @escaping (String) -> Void,
        onFinal: @escaping (String) -> Void
    ) throws -> () -> Void {
        guard let recognizer = SFSpeechRecognizer(locale: locale),
              recognizer.isAvailable else {
            throw VoiceError.unavailable
        }
        let engine = AVAudioEngine()
        let req = SFSpeechAudioBufferRecognitionRequest()
        req.shouldReportPartialResults = true
        req.requiresOnDeviceRecognition = requiresOnDevice
        var task: SFSpeechRecognitionTask?
        task = recognizer.recognitionTask(with: req) { result, error in
            guard let result else {
                if error != nil { onFinal("") }
                return
            }
            if result.isFinal { onFinal(result.bestTranscription.formattedString) }
            else { onPartial(result.bestTranscription.formattedString) }
        }
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
            req.append(buffer)
        }
        engine.prepare()
        try engine.start()
        return {
            engine.stop()
            input.removeTap(onBus: 0)
            req.endAudio()
            task?.cancel()
        }
    }

    // MARK: - Speech synthesis (always local)

    private static let synthesizer = AVSpeechSynthesizer()

    /// Speak text immediately (queues behind in-flight speech).
    /// `voice`: BCP-47 (e.g. "en-US"); nil = system default.
    public static func speak(_ text: String, voice: String? = nil, rate: Float = AVSpeechUtteranceDefaultSpeechRate) {
        let utterance = AVSpeechUtterance(string: text)
        if let voice {
            utterance.voice = AVSpeechSynthesisVoice(language: voice)
        }
        utterance.rate = rate
        synthesizer.speak(utterance)
    }

    /// Stop all queued speech.
    public static func stopSpeaking() {
        synthesizer.stopSpeaking(at: .immediate)
    }

    /// True while the synthesizer is producing audio.
    public static var isSpeaking: Bool { synthesizer.isSpeaking }
}

/// Voice-layer errors (OS capability / authorization, not model errors).
public enum VoiceError: Error {
    /// No recognizer for the locale, or recognition unavailable
    /// (e.g. on-device model missing and server fallback disabled).
    case unavailable
}
