// swift-tools-version:5.9
//
// Package.swift — Sofuu as a Swift Package (iOS + macOS).
//
// This is the Phase 1 install path: an app developer adds one dependency
// line and gets the whole runtime — offline embeddings, an encrypted local
// brain, agents, MCP, speech — as a static binary. No cloud account, no API
// key to ship, and QuickJS is a pure interpreter, so the App Store's
// executable-memory policy is satisfied by design (no JIT).
//
// Usage:
//   .package(url: "https://github.com/sofuu-runtime/sofuu.git", from: "0.2.0")
//   // then depend on the "Sofuu" product
//
// The binary target resolves the `libsofuu.xcframework` that the matching
// GitHub release attaches. Release automation (scripts/dist/ios.sh +
// .github/workflows/release.yml) keeps the URL and checksum below in sync;
// `scripts/stamp-swiftpm.sh` re-stamps the checksum after a release.
//
// The C surface lives in include/sofuu_embed.h. A thin Swift wrapper with
// typed embedding/memory helpers is provided as a source target (SwiftBridge)
// so you can `import Sofuu` and get Swift ergonomics over the C ABI.

import PackageDescription

// The xcframework zip published on the GitHub release for this version.
let sofuuXCFrameworkURL =
    "https://github.com/sofuu-runtime/sofuu/releases/download/v0.2.0/libsofuu.xcframework.zip"

// SHA-256 of that zip. `make stamp-swiftpm` (scripts/stamp-swiftpm.sh)
// recomputes and rewrites this from the built artifact, then verifies the
// file exists — it is checked in on purpose so a corrupted or swapped
// download is caught by SwiftPM at resolve time, not at runtime.
let sofuuXCFrameworkChecksum = "0000000000000000000000000000000000000000000000000000000000000000"

let package = Package(
    name: "Sofuu",
    platforms: [
        // iOS 16 / macOS 13: the floor for the Swift Language Model
        // protocol and the on-device speech wrappers (see SofuuVoice).
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        // The umbrella most people want: the binary runtime.
        .library(name: "Sofuu", targets: ["SofuuBinary", "SofuuBridge"]),
        // Just the C ABI, for teams that want the raw header surface.
        .library(name: "SofuuC", targets: ["SofuuBinary"]),
    ],
    targets: [
        .binaryTarget(
            name: "SofuuBinary",
            url: sofuuXCFrameworkURL,
            checksum: sofuuXCFrameworkChecksum
        ),
        // Ergonomic Swift layer over the C ABI (embed / brain / stream).
        .target(
            name: "SofuuBridge",
            dependencies: ["SofuuBinary"],
            path: "bindings/swift/Sources/SofuuBridge"
        ),
    ]
)
