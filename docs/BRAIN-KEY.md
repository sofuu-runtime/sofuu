# The Brain Key — machine-local encryption, migration, and recovery

> **Status:** B4 shipped 2026-10-14 (ROADMAP-BRAIN-CMA). Companion to
> `BRAIN-CARD.md` (the portable format) and `AUDIT-2026-10-13-BRAIN-CMA.md`
> (why this exists).

## What changed

Every brain file used to be encrypted with ONE password compiled into the
binary (`sofuu-kv-local-v1`), so any `.qtsq` file was readable by anyone
holding the source. The vault password is now a **machine-local key**.

## Where the key comes from

Resolution order (first hit wins):

1. **`SOFUU_BRAIN_KEY`** — 64 hex characters (32 bytes). An explicit
   override for CI, containers, or a deliberately shared key across a
   fleet. Set it and the key file is never read.
2. **`~/.sofuu/brain.key`** — 32 random bytes as hex, mode `0600`, created
   on first use from the OS entropy source (`/dev/urandom`). Written to a
   `0600` temp sibling and renamed, so a reader never sees a partial key
   and the file is never briefly world-readable.
3. **No usable key** (no `HOME`, read-only home): a process-local random
   key is used and the reason is printed. Brains written in that process
   will not reopen — the message says so rather than silently downgrading
   to the legacy password.

The key is read once per process and cached.

## Migration from a pre-B4 brain

A brain written before B4 is encrypted with the legacy password. Opening it
works: the machine key is tried first, the legacy password second (the
codec verifies the HMAC before touching the ciphertext, so a failed attempt
is free and harmless). When the legacy password is the one that opens it,
the brain is marked and its **first write** re-encrypts it under the
machine key:

- the manifest (or the single legacy file) is copied to
  `<path>.legacy-key.bak` first — a unique name, never clobbered;
- the rewrite rides the compaction path, so every segment is re-encrypted
  and the old generation stays on disk until the age-guarded orphan sweep
  (10 minutes) can prove no concurrent reader holds it;
- if the backup cannot be written, the write is **refused** rather than
  proceeding without one.

Migration is automatic and one-way. Nothing needs to be run by hand.

## Recovery: a new machine, or a lost key file

**The brain is machine-local by design.** The key file is the only thing
that can decrypt a brain written on this machine, and it is not backed up
anywhere. Losing it means the brain file is unreadable — permanently. That
is the point of the change: before it, the "key" was public.

What to do:

| Situation | What happens | What to do |
|---|---|---|
| New machine, brain copied over | The open **refuses** with a message naming the file (`is a container this key cannot open`). Nothing is written, nothing is corrupted. | Start a fresh brain there, or move the key file too if you trust the destination. |
| Key file deleted, brain intact | Same refusal. | Restore `~/.sofuu/brain.key` from your own backup, or set `SOFUU_BRAIN_KEY` to the original value. |
| Key file deleted, no backup | The brain is unreadable. | Delete the brain file and start fresh. The refusal is loud; there is no silent data loss. |
| Sharing a brain with someone | `/share` writes a **plain JSON card** (`BRAIN-CARD.md`) — deliberately not encrypted, so it travels. | Use cards to share memories; use the key file to keep a brain private. |

**Back up the key if the brain matters.** `~/.sofuu/brain.key` is one line
of hex; copy it wherever you keep secrets. The brain file alone is not
enough.

## What this does not protect against

- **A process running as you.** The key is readable by your user account;
  anything running as you can decrypt the brain. This is at-rest
  protection, not a sandbox.
- **A compromised machine.** Same reasoning.
- **The legacy password era.** Brains written before B4 stay decryptable
  with the public password until their first write migrates them. If a
  pre-B4 brain was ever copied off the machine, that copy is already
  exposed — migration cannot undo it.

## Why not the codec's device keystore

The codec ships `qtsq_keystore.h` (ML-KEM-768 + ML-DSA-65, Keychain-backed
on macOS). It was evaluated first, as the roadmap asks, and rejected for
two reasons: an unsigned binary triggers a **Keychain prompt on every
launch** — the exact friction the fixed password existed to avoid — and its
encrypt path is ML-KEM encrypt-to-self, a different mechanism from the
password vault every existing brain file uses, so adopting it would change
the on-disk scheme rather than just the key. The `0600` file keeps the
vault, keeps the format, and still makes the key machine-local.
