#!/usr/bin/env bash
# scripts/cross/zig-ld.sh — linker wrapper for rustc cross builds.
#
# Problems this solves:
# 1. rustc invokes the linker as a bare executable with its own leading
#    flags (-m64 on x86_64, `-m i386pep` on windows-gnu) BEFORE any
#    `-C link-arg` values, so `zig` itself can't be the linker. The
#    wrapper guarantees `zig cc` is argv[0..1] and handles rustc's own
#    flags.
# 2. rustc's gcc-flavor link line contains BARE ld-style options
#    (--as-needed, -z <opt>, -Bstatic, ...) that clang rejects as
#    "Unknown Clang option". They are rewritten to -Wl,<opt> form.
# 3. Target-specific arg fixes, mirroring cargo-zigbuild's filter:
#    - windows-gnu: drop rustc's MinGW CRT libs (zig links its own
#      msvcrt/unwind/compiler_rt), drop rust's libcompiler_builtins rlib
#      (its ___chkstk_ms duplicates zig's compiler_rt), -lgcc_eh → -lc++
#      (zig < 0.14), -Bdynamic → -search_paths_first.
#    - musl: skip GNU-ld-only -z/ld options lld rejects (unneeded for
#      static output).
#
# Used by scripts/cross/build_linux_*.sh and build_windows_x86_64.sh via
# RUSTFLAGS="-C linker=$REPO_ROOT/scripts/cross/zig-ld.sh ...".
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

is_windows=0
for a in "$@"; do
    case "$a" in
        *-windows-gnu*|*-windows-msvc*) is_windows=1 ;;
    esac
done

args=()
pending_z=0
pending_m=0
for a in "$@"; do
    if [ "$pending_z" = 1 ]; then
        pending_z=0
        case "$a" in
            text) ;;  # GNU-ld-only; irrelevant for static musl output
            *) args+=("-Wl,-z,$a") ;;
        esac
        continue
    fi
    if [ "$pending_m" = 1 ]; then
        pending_m=0
        # PE ld-emulation from rustc's mingw line; zig cc derives the PE
        # arch from -target alone (i386pep IS x86_64's PE emulation).
        case "$a" in
            i386pep) ;;
            *) echo "zig-ld.sh: unexpected -m emulation: $a" >&2; exit 1 ;;
        esac
        continue
    fi
    case "$a" in
        -z) pending_z=1 ;;
        -m) pending_m=1 ;;
        # GNU-ld-only flags lld doesn't implement (and doesn't need — lld
        # links correct static PIE output without --no-dynamic-linker).
        --no-dynamic-linker) ;;
        # MinGW PE-specific ld options rustc emits bare; lld enables the
        # same protections (DEP/ASLR/high-entropy VA) by default for PE.
        --dynamicbase|--disable-auto-image-base|--high-entropy-va|--nxcompat) ;;
        --*) args+=("-Wl,$a") ;;
        -Bstatic|-Bgroup|-Bsymbolic) args+=("-Wl,$a") ;;
        -Bdynamic)
            if [ "$is_windows" = 1 ]; then
                args+=("-Wl,-search_paths_first")
            else
                args+=("-Wl,-Bdynamic")
            fi ;;
        *)
            if [ "$is_windows" = 1 ]; then
                case "$a" in
                    # MinGW CRT libs rustc names explicitly; zig's
                    # windows-gnu driver links its own equivalents.
                    -l:libpthread.a|-lgcc|-lmsvcrt|-lwindows) ;;
                    -lgcc_eh) args+=("-lc++") ;;
                    # rust's compiler_builtins duplicates zig's compiler_rt
                    *libcompiler_builtins-*.rlib) ;;
                    *) args+=("$a") ;;
                esac
            else
                case "$a" in
                    -lgcc|-lgcc_eh|-lmsvcrt) ;;
                    *) args+=("$a") ;;
                esac
            fi ;;
    esac
done
if [ "$pending_z" = 1 ]; then
    echo "zig-ld.sh: dangling -z at end of linker args" >&2
    exit 1
fi

exec "$DIR/../../tools/zig/zig" cc "${args[@]}"
