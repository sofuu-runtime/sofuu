#!/usr/bin/env bash
# build_qtsq_cross.sh — cross-compile the QTSQ static libs for the CLI
# targets (Linux x86_64/arm64 musl, Windows x64 mingw) with zig cc.
#
# QTSQ is the proprietary checkout at ~/projects/black-hole-disk (see
# AUTHORSHIP_SEAL.md). This script compiles it from source — it never
# copies the checkout — producing:
#   dist/qtsq-linux-x86_64/libqtsq.a     (+ libqtc.a + libz.a)
#   dist/qtsq-linux-arm64/libqtsq.a      (+ libqtc.a + libz.a)
#   dist/qtsq-windows-x86_64/qtsq.lib    (+ libz.a; qtc baked in, shim
#                                         force-included, mirroring
#                                         windows/CMakeLists.txt)
#
# The CLI cross scripts point SOFUU_QTSQ_DIR / SOFUU_QTSQ_LIB /
# SOFUU_ZLIB_DIR at these dirs so sofuu-ffi's build.rs links QTSQ
# persistence (session store + brain/KV memory) into the cross builds.
#
# Excludes (mirroring the Makefile/CMake non-Apple builds): Metal/CoreAudio
# (.mm/.m — Darwin only) and the Opus audio path (optional dep).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ZIG="$REPO_ROOT/tools/zig/zig"
QTSQ_DIR="${SOFUU_QTSQ_SRC:-$HOME/projects/black-hole-disk}"
ZLIB_SRC="$REPO_ROOT/dist/zlib-src"

for dep in "$QTSQ_DIR/src/qtsq_codec.c" "$QTSQ_DIR/compressor/Makefile" \
           "$QTSQ_DIR/windows/shim/qtsq_win_compat.h"; do
    [ -e "$dep" ] || { echo "missing: $dep" >&2; exit 1; }
done
"$ZIG" version >/dev/null

# ── zlib source (one-time download, vendored into dist/) ────────────
# P1-17 rider (AUDIT-2026-09-07): the fresh-download branch extracted
# WITHOUT any hash check — same supply-chain class as the curl pin (P2-24).
# Both paths now verify against the pinned zlib 1.3.1 hash before extract.
ZLIB_SHA256="9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23"
if [ ! -f "$ZLIB_SRC/zlib.h" ]; then
    mkdir -p "$ZLIB_SRC"
    if [ ! -f /tmp/zlib-1.3.1.tar.gz ]; then
        curl -fsSL --max-time 60 -o /tmp/zlib-1.3.1.tar.gz \
            https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
    fi
    GOT_SHA="$(shasum -a 256 /tmp/zlib-1.3.1.tar.gz | cut -d' ' -f1)"
    if [ "$GOT_SHA" != "$ZLIB_SHA256" ]; then
        echo "✗ zlib tarball checksum mismatch:"
        echo "    expected $ZLIB_SHA256"
        echo "    got      $GOT_SHA"
        echo "  Refusing to build from an unverified download."
        exit 1
    fi
    tar xzf /tmp/zlib-1.3.1.tar.gz -C /tmp
    cp -R /tmp/zlib-1.3.1/. "$ZLIB_SRC/"
fi

# QTSQ module sets.
#
# POSIX links the FULL upstream source set. The old reduced subset (utils,
# format, types, codec, container, tensor, vault, secure, sanitize,
# keystore, horizon, shamir, singularity) produces archives with dangling
# references — qtsq_c2pa/qtsq_seal call qtsq_decompress (defined in
# qtsq_radiation.c) and qtc_core.o dispatches to qtc_audio_*/
# qtc_accretion_* (defined in qtc_audio.c/qtc_accretion.c). A static BINARY
# link tolerates those because the referencing members are never pulled, but
# a SHARED library must resolve every relocation it contains: libsofuu.so
# refused to dlopen with "undefined symbol: qtsq_decompress". The musl
# archives shipped since 2026-10 were already full builds (all 51 qtsq .c
# compiled for musl); these lists match them. qtsq_audio_opus,
# qtsq_image_file, qtsq_model and qtsq_video_stubs stay out — they need
# external codecs/models and were never in the shipped archives.
QTSQ_CORE_POSIX="qtsq_utils qtsq_format qtsq_types qtsq_codec qtsq_pixel \
qtsq_chandrasekhar qtsq_singularity qtsq_horizon qtsq_lensing \
qtsq_accretion qtsq_triangle qtsq_radiation qtsq_redshift qtsq_wormhole \
qtsq_vault qtsq_sanitize qtsq_secure qtsq_waves qtsq_paradox \
qtsq_gravitational qtsq_splat2d qtsq_splat_only qtsq_triangle_only \
qtsq_video_encode qtsq_video_decode qtsq_audio_encode qtsq_audio_decode \
qtsq_tensor qtsq_container qtsq_photon_sonar qtsq_photon_tdoa \
qtsq_photon_wifi qtsq_photon_fusion qtsq_splat4d qtsq_splat_tsubu \
qtsq_4dgs qtsq_geocrypt qtsq_shamir qtsq_deniable qtsq_region_crypt \
qtsq_fingerprint qtsq_censorship qtsq_keystore qtsq_seal qtsq_surfel \
qtsq_tsdf qtsq_seal_freqdct qtsq_turbo qtsq_ionic qtsq_mf_merge qtsq_c2pa"

# Windows keeps the reduced set: it mirrors windows/CMakeLists.txt (the
# Opus/spatial modules are not part of that port) and bakes qtc INTO the
# single qtsq.lib, so nothing there links a shared object.
QTSQ_CORE_WIN="qtsq_utils qtsq_format qtsq_types qtsq_codec qtsq_container \
qtsq_tensor qtsq_vault qtsq_secure qtsq_sanitize qtsq_keystore \
qtsq_horizon qtsq_shamir qtsq_singularity"
# P3 (AUDIT-2026-09-07): the crypto list came from `ls crypto/*/*.c | sort` —
# a missing or renamed directory made ls emit the LITERAL glob pattern into
# the list, breaking the link far away from the cause. Enumerate the four
# dirs explicitly and refuse to build if any has no sources; the output set
# is identical (same relative crypto/<dir>/<file>.c entries, sorted).
QTSQ_CRYPTO=""
for CRYPTO_SUBDIR in crypto/common crypto/ml-kem-768 crypto/ml-dsa-65 crypto/argon2; do
    for CRYPTO_SRC in "$QTSQ_DIR/$CRYPTO_SUBDIR"/*.c; do
        if [ ! -e "$CRYPTO_SRC" ]; then
            echo "✗ No crypto sources found in $QTSQ_DIR/$CRYPTO_SUBDIR — refusing to build from an incomplete list." >&2
            exit 1
        fi
        QTSQ_CRYPTO="$QTSQ_CRYPTO $CRYPTO_SUBDIR/$(basename "$CRYPTO_SRC")"
    done
done
QTSQ_CRYPTO="$(printf '%s\n' $QTSQ_CRYPTO | sort)"
# QTC compression core. POSIX = the whole compressor/src set (qtc_core
# references the audio/accretion/image/video dispatchers); Windows = the
# minimal tensor/text core the CMake port builds.
QTC_SRC_POSIX="qtc_accretion qtc_ans qtc_audio qtc_audio_float qtc_audio_mdct \
qtc_audio_mdct2 qtc_core qtc_delaunay qtc_detect qtc_image qtc_image_dct \
qtc_image_mesh qtc_lz qtc_mdct qtc_model qtc_order1 qtc_table qtc_tensor \
qtc_text qtc_video"
QTC_SRC_WIN="qtc_core qtc_ans qtc_lz qtc_order1 qtc_model qtc_tensor qtc_text"

# zlib objects (vendored; no gz* — the QTSQ callers only use the raw
# compress/uncompress/deflate/inflate/crc32 surface).
ZLIB_OBJS="adler32 compress crc32 deflate inflate infback inftrees \
inffast trees uncompr zutil"

build_one() {
    local target="$1" outdir="$2" win="$3"
    local out="$REPO_ROOT/dist/$outdir"
    mkdir -p "$out/obj"
    # Stage the checkout's public headers: sofuu-ffi's build.rs layout
    # guard compiles qtsq_layout_guard.c against $SOFUU_QTSQ_DIR/include.
    mkdir -p "$out/include" "$out/compressor/include"
    cp "$QTSQ_DIR/include/"*.h "$out/include/"
    # sofuu-ffi's build.rs also puts <dist>/compressor/include on the include
    # path for sofuu_qtsq_table.c (#include "qtc.h"), but this script never
    # staged it there — only the Android script did — so every x86_64 QTSQ
    # build died in that TU ("'qtc.h' file not found"). Stage it with the
    # archive, exactly like scripts/mobile/build_qtsq_android.sh.
    cp "$QTSQ_DIR/compressor/include/"*.h "$out/compressor/include/"
    local linkargs=(-target "$target")
    local cc=("$ZIG" cc "${linkargs[@]}" -O2 -std=c11 -c)
    local inc=(-I"$QTSQ_DIR/include" -I"$QTSQ_DIR/compressor/include" \
               -I"$QTSQ_DIR/crypto/common" -I"$QTSQ_DIR/crypto/ml-kem-768" \
               -I"$QTSQ_DIR/crypto/ml-dsa-65" -I"$QTSQ_DIR/crypto/argon2" \
               -I"$ZLIB_SRC")
    local defines=(-D_DEFAULT_SOURCE)
    local qtsq_list="$QTSQ_CORE_POSIX" qtc_list="$QTC_SRC_POSIX"
    if [ "$win" = 1 ]; then
        qtsq_list="$QTSQ_CORE_WIN"
        qtc_list="$QTC_SRC_WIN"
        defines+=(-D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS \
                  -D_CRT_NONSTDC_NO_WARNINGS -DWIN32_LEAN_AND_MEAN \
                  -DNOMINMAX -DNOGDI -DQTSQ_MMAP_WIN)
        inc=(-I"$QTSQ_DIR/windows/shim" "${inc[@]}")
    fi

    echo "── $target → dist/$outdir"

    # zlib (independent archive; QTSQ core + qtc reference it).
    local zobjs=()
    for o in $ZLIB_OBJS; do
        local f="$out/obj/z_$o.o"
        [ -f "$f" ] || "${cc[@]}" -I"$ZLIB_SRC" -DNO_GZIP \
            "$ZLIB_SRC/$o.c" -o "$f"
        zobjs+=("$f")
    done
    # zig ar (llvm-ar), not the host ar: members are ELF/COFF for the
    # TARGET, and the Apple host ar can't index those.
    local AR=("$ZIG" ar)
    "${AR[@]}" crs "$out/libz.a" "${zobjs[@]}"

    # qtc compressor core (POSIX: separate libqtc.a archive; Windows: baked
    # into qtsq.lib, mirroring the CMake single-archive layout).
    local qtc_objs=()
    for s in $qtc_list; do
        local f="$out/obj/qtc_$s.o"
        [ -f "$f" ] || "${cc[@]}" "${defines[@]}" "${inc[@]}" \
            "$QTSQ_DIR/compressor/src/$s.c" -o "$f"
        qtc_objs+=("$f")
    done

    # QTSQ core + crypto. Windows: force-include the shim master header in
    # every TU (CMake /FIqtsq_win_compat.h) — ordering-proof for TUs that
    # include POSIX headers before the shim's shadowing ones.
    # bash 3.2 + set -u: "${fi[@]}" on an empty array is an unbound error,
    # hence the ${fi[@]+...} guard.
    local fi=()
    [ "$win" = 1 ] && fi=(-include "$QTSQ_DIR/windows/shim/qtsq_win_compat.h")
    local qtsq_objs=()
    for s in $qtsq_list; do
        local f="$out/obj/$s.o"
        [ -f "$f" ] || "${cc[@]}" "${defines[@]}" "${inc[@]}" \
            ${fi[@]+"${fi[@]}"} "$QTSQ_DIR/src/$s.c" -o "$f"
        qtsq_objs+=("$f")
    done
    for c in $QTSQ_CRYPTO; do
        # ml-kem-768 and ml-dsa-65 share basenames (poly.c, ntt.c, …) — a
        # flat per-basename object name lets the later dir clobber the
        # earlier one, silently swapping in the wrong PQCLEAN_* symbols
        # (the exact trap the upstream Makefile's crypto/build/ scheme
        # avoids). Tag every crypto object with its directory.
        local tag; tag="$(dirname "$c")"; tag="${tag#crypto/}"
        tag="${tag//[^A-Za-z0-9]/_}"
        local f="$out/obj/crypto_${tag}_$(basename "$c" .c).o"
        [ -f "$f" ] || "${cc[@]}" "${defines[@]}" "${inc[@]}" \
            ${fi[@]+"${fi[@]}"} "$QTSQ_DIR/$c" -o "$f"
        qtsq_objs+=("$f")
    done
    if [ "$win" = 1 ]; then
        local win_objs=("${qtc_objs[@]}" "${qtsq_objs[@]}")
        for s in qtsq_win_compat qtsq_mmap_win; do
            local f="$out/obj/shim_$s.o"
            [ -f "$f" ] || "${cc[@]}" "${defines[@]}" "${inc[@]}" \
                "$QTSQ_DIR/windows/shim/$s.c" -o "$f"
            win_objs+=("$f")
        done
        "${AR[@]}" crs "$out/qtsq.lib" "${win_objs[@]}"
        # rustc's -lstatic=qtsq/-lzlib go through the mingw search path,
        # which resolves `libqtsq.a`/`libzlib.a`; sofuu-core's has_qtsq
        # probe additionally requires `$SOFUU_QTSQ_DIR/libqtsq.a`, and the
        # build.rs zlib probe requires a literal `zlib.lib`. Same members,
        # alias filenames (cp is fine here, but ar-from-objects is exact).
        "${AR[@]}" crs "$out/libqtsq.a" "${win_objs[@]}"
        "${AR[@]}" crs "$out/zlib.lib" "${zobjs[@]}"
        "${AR[@]}" crs "$out/libzlib.a" "${zobjs[@]}"
        echo "   qtsq.lib ($(du -h "$out/qtsq.lib" | cut -f1)) + libqtsq.a + zlib.lib/libzlib.a"
    else
        # build.rs looks for the compressor archive at
        # $SOFUU_QTSQ_DIR/compressor/libqtc.a (the checkout's compressor
        # Makefile product) — stage it exactly there or -lqtc is never
        # emitted and the link dies on qtc_* symbols.
        mkdir -p "$out/compressor"
        "${AR[@]}" crs "$out/compressor/libqtc.a" "${qtc_objs[@]}"
        cp "$out/compressor/libqtc.a" "$out/libqtc.a"
        "${AR[@]}" crs "$out/libqtsq.a" "${qtsq_objs[@]}"
        echo "   libqtsq.a ($(du -h "$out/libqtsq.a" | cut -f1)) + compressor/libqtc.a + libz.a"
    fi
}

build_one x86_64-linux-musl qtsq-linux-x86_64 0
build_one aarch64-linux-musl qtsq-linux-arm64 0
build_one x86_64-windows-gnu qtsq-windows-x86_64 1
# glibc variant: rustc DROPS the cdylib crate type for *-linux-musl (musl has
# no dynamic libc in the Rust target spec), so the only buildable libsofuu.so
# is a glibc one — it needs glibc QTSQ objects. See
# scripts/cross/build_linux_capi_so.sh.
build_one x86_64-linux-gnu qtsq-linux-x86_64-gnu 0
echo "all QTSQ cross libs built."
