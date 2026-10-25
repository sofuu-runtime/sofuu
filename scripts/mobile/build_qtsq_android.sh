#!/usr/bin/env bash
# scripts/mobile/build_qtsq_android.sh — the proprietary QTSQ codec,
# cross-compiled for Android with the NDK clang (aarch64 / x86_64).
#
# The QTSQ checkout (SOFUU_QTSQ_SRC, default ~/projects/black-hole-disk)
# already builds under the NDK — the checkout ships an Android Gradle app
# whose CMakeLists compiles the same core + crypto + qtc sources — so the
# source lists below mirror the POSIX cross build (build_qtsq_cross.sh):
# the Makefile's SRC_CORE minus the Darwin-only (Metal/CoreAudio) and
# Opus-optional files, plus the four crypto dirs and the pure-C qtc
# compressor. This script COMPILES the checkout from source; it never
# copies the proprietary tree.
#
# Archive layout is what crates/sofuu-ffi/build.rs expects:
#   dist/qtsq-android-<abi>/libqtsq.a        (core + crypto)
#   dist/qtsq-android-<abi>/compressor/libqtc.a
#   dist/qtsq-android-<abi>/libz.a           (vendored zlib 1.3.1)
#   dist/qtsq-android-<abi>/include/         (staged headers: layout guard)
#
# Usage: bash scripts/mobile/build_qtsq_android.sh [arm64|x86_64 …]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO_ROOT/scripts/mobile/ndk_env.sh" "$@"

QTSQ_DIR="${SOFUU_QTSQ_SRC:-$HOME/projects/black-hole-disk}"
ZLIB_SRC="$REPO_ROOT/dist/zlib-src"

# Same lists as scripts/cross/build_qtsq_cross.sh (keep in lockstep).
QTSQ_CORE="qtsq_utils qtsq_format qtsq_types qtsq_codec qtsq_pixel \
qtsq_chandrasekhar qtsq_singularity qtsq_horizon qtsq_lensing qtsq_accretion \
qtsq_triangle qtsq_radiation qtsq_redshift qtsq_wormhole qtsq_vault \
qtsq_sanitize qtsq_secure qtsq_waves qtsq_paradox qtsq_gravitational \
qtsq_splat2d qtsq_splat_only qtsq_triangle_only qtsq_video_encode \
qtsq_video_decode qtsq_audio_encode qtsq_audio_decode qtsq_tensor \
qtsq_container qtsq_photon_sonar qtsq_photon_tdoa qtsq_photon_wifi \
qtsq_photon_fusion qtsq_splat4d qtsq_splat_tsubu qtsq_4dgs qtsq_geocrypt \
qtsq_shamir qtsq_deniable qtsq_region_crypt qtsq_fingerprint qtsq_censorship \
qtsq_keystore qtsq_seal qtsq_surfel qtsq_tsdf qtsq_seal_freqdct qtsq_turbo \
qtsq_ionic qtsq_mf_merge qtsq_c2pa"
QTC_SRC="qtc_model qtc_ans qtc_order1 qtc_lz qtc_image qtc_audio qtc_detect \
qtc_mdct qtc_audio_mdct qtc_audio_mdct2 qtc_audio_float qtc_delaunay \
qtc_image_mesh qtc_accretion qtc_image_dct qtc_tensor qtc_video qtc_core \
qtc_text"
ZLIB_OBJS="adler32 compress crc32 deflate inflate infback inftrees \
inffast trees uncompr zutil"

[ -f "$QTSQ_DIR/src/qtsq_codec.c" ] || { echo "✗ QTSQ checkout not found at $QTSQ_DIR (set SOFUU_QTSQ_SRC)"; exit 1; }

build_one() {
    local abi="$1"
    mobile_abi_env "$abi"
    local out="$REPO_ROOT/dist/qtsq-android-$MOBILE_ABI_NAME"
    # Headers are staged on every run, cache or not: build.rs compiles
    # sofuu_qtsq_table.c with -I <dist>/include and -I
    # <dist>/compressor/include, and an archive hit must not skip them
    # (a cached archive with unstaged headers died on qtc.h).
    mkdir -p "$out/include" "$out/compressor/include"
    cp "$QTSQ_DIR/include/"*.h "$out/include/"
    cp "$QTSQ_DIR/compressor/include/"*.h "$out/compressor/include/"
    if [ -f "$out/libqtsq.a" ] && [ -f "$out/compressor/libqtc.a" ] && [ -f "$out/libz.a" ]; then
        echo "✓ qtsq $MOBILE_ABI_NAME already built"
        return 0
    fi
    echo "── qtsq → $MOBILE_ABI_NAME ($MOBILE_NDK_PREFIX)"
    mkdir -p "$out/obj" "$out/compressor"

    local inc=(-I"$QTSQ_DIR/include" -I"$QTSQ_DIR/compressor/include" \
               -I"$QTSQ_DIR/crypto/common" -I"$QTSQ_DIR/crypto/ml-kem-768" \
               -I"$QTSQ_DIR/crypto/ml-dsa-65" -I"$QTSQ_DIR/crypto/argon2" \
               -I"$ZLIB_SRC")
    local defines=(-D_DEFAULT_SOURCE)

    # zlib first (independent archive; core + qtc reference it).
    local zobjs=()
    for o in $ZLIB_OBJS; do
        local f="$out/obj/z_$o.o"
        [ -f "$f" ] || "$MOBILE_CC" -O2 -std=c11 -c \
            "${inc[@]}" -I"$ZLIB_SRC" -DNO_GZIP \
            "$ZLIB_SRC/$o.c" -o "$f"
        zobjs+=("$f")
    done
    "$MOBILE_AR" crs "$out/libz.a" "${zobjs[@]}"

    # qtc compressor objects (must be the FIRST members of libqtc.a so
    # sofuu-ffi's two-archive link order resolves qtc_* symbols).
    local qtcobjs=()
    for s in $QTC_SRC; do
        local f="$out/obj/qtc_$s.o"
        [ -f "$f" ] || "$MOBILE_CC" -O2 -std=c11 -c "${defines[@]}" "${inc[@]}" \
            "$QTSQ_DIR/compressor/src/$s.c" -o "$f"
        qtcobjs+=("$f")
    done
    "$MOBILE_AR" crs "$out/compressor/libqtc.a" "${qtcobjs[@]}"

    # QTSQ core.
    local objs=()
    for s in $QTSQ_CORE; do
        local f="$out/obj/$s.o"
        [ -f "$f" ] || "$MOBILE_CC" -O2 -std=c11 -c "${defines[@]}" "${inc[@]}" \
            "$QTSQ_DIR/src/$s.c" -o "$f"
        objs+=("$f")
    done
    # crypto (tag per directory — ml-kem-768 and ml-dsa-65 share basenames).
    for CRYPTO_SUBDIR in crypto/common crypto/ml-kem-768 crypto/ml-dsa-65 crypto/argon2; do
        for CRYPTO_SRC in "$QTSQ_DIR/$CRYPTO_SUBDIR"/*.c; do
            [ -e "$CRYPTO_SRC" ] || { echo "✗ no crypto sources in $CRYPTO_SUBDIR"; exit 1; }
            local tag="${CRYPTO_SUBDIR#crypto/}"; tag="${tag//[^A-Za-z0-9]/_}"
            local f="$out/obj/crypto_${tag}_$(basename "$CRYPTO_SRC" .c).o"
            [ -f "$f" ] || "$MOBILE_CC" -O2 -std=c11 -c "${defines[@]}" "${inc[@]}" \
                "$CRYPTO_SRC" -o "$f"
            objs+=("$f")
        done
    done
    "$MOBILE_AR" crs "$out/libqtsq.a" "${objs[@]}"
    echo "✓ qtsq $MOBILE_ABI_NAME → $out (libqtsq.a $(du -h "$out/libqtsq.a" | cut -f1))"
}

# "all" (the Makefile's spelling) and an empty list both mean every ABI.
ABIS=()
for a in "$@"; do [ "$a" = "all" ] || ABIS+=("$a"); done
if [ ${#ABIS[@]} -eq 0 ]; then
    build_one arm64
    build_one x86_64
else
    for abi in "${ABIS[@]}"; do build_one "$abi"; done
fi
