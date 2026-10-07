# Sofuu.podspec — Sofuu as a CocoaPod (iOS + macOS).
#
# The other install path alongside SwiftPM, for the many apps that are still
# not on SPM (or use both during a migration). It ships the same
# libsofuu.xcframework from the GitHub release, so an app developer adds one
# line and has offline embeddings, an encrypted local brain, agents, MCP and
# speech — with no cloud account, no API key to ship, and no JIT (QuickJS is
# a pure interpreter, so the App Store executable-memory policy is satisfied
# by design).
#
# Install:
#   pod 'Sofuu', '~> 0.2'
#
# Then `import Sofuu` (the binary module) or `import SofuuBridge` (the typed
# Swift layer) — the same two products the Swift package exposes.
#
# Pod::Spec is validated with `pod lib lint`; the podspec is intentionally
# dependency-free (no vendored curl/libuv — they are static-linked into the
# xcframework) so a consumer's tree stays small.

Pod::Spec.new do |s|
  s.name             = 'Sofuu'
  s.version          = '0.2.0'
  s.summary          = 'The private, offline AI runtime for apps — 2MB, no JIT, no key to ship.'

  s.description      = <<-DESC
    Sofuu embeds a complete agent runtime in your app: offline embeddings
    with no model file, an encrypted local brain (memory that belongs to the
    user, not a provider), LLM streaming against any model you choose, MCP
    tools, sub-agents, and on-device speech.

    It runs entirely on-device — no cloud account, no API key in your
    binary, no telemetry. The engine is a pure interpreter (no JIT), so it
    is App-Store-eligible by construction.
  DESC

  s.homepage         = 'https://sofuu.xyz'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'Prianshu Boruah (Haruhito)' => 'hello@sofuu.xyz' }
  s.source           = {
    :http => 'https://github.com/sofuu-runtime/sofuu/releases/download/v0.2.0/libsofuu.xcframework.zip'
  }

  s.ios.deployment_target  = '16.0'
  s.osx.deployment_target  = '13.0'

  s.vendored_frameworks = 'libsofuu.xcframework'
  s.static_framework    = true
  s.requires_arc        = true

  # No dependencies: libcurl/libuv/QTSQ are already inside the binary, and
  # QTSQ-free release builds are pure MIT (a QTSQ-linked build additionally
  # falls under LICENSES/QTSQ-FORMAT.txt — see docs/EMBEDDING-DIST.md).
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    # The C ABI is plain C; keep the module interface C-clean for Swift.
    'CLANG_CXX_LANGUAGE_STANDARD' => 'gnu++17'
  }

  s.source_files = 'bindings/swift/Sources/SofuuBridge/**/*.swift'

  s.test_spec 'Tests' do |t|
    t.source_files = 'bindings/swift/Tests/SofuuBridgeTests/**/*.swift'
  end
end
