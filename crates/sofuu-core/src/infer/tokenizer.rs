// crates/sofuu-core/src/infer/tokenizer.rs — Fast BPE Tokenizer for GGUF models.
//
// Reads vocab tokens, scores, and merges directly from GGUF metadata. Zero external dependencies.

use std::collections::HashMap;

use super::gguf::GgufFile;

#[derive(Clone)]
pub struct IndexedStringTable {
    table: Vec<u32>,
    mask: usize,
}

impl IndexedStringTable {
    pub fn build(vocab: &[String]) -> Self {
        let min_cap = (vocab.len() as f64 / 0.6) as usize;
        let cap = min_cap.next_power_of_two().max(1024);
        let mask = cap - 1;
        let mut table = vec![u32::MAX; cap];

        for (id, token) in vocab.iter().enumerate() {
            let mut idx = (Self::hash(token) as usize) & mask;
            loop {
                if table[idx] == u32::MAX {
                    table[idx] = id as u32;
                    break;
                }
                idx = (idx + 1) & mask;
            }
        }

        Self { table, mask }
    }

    #[inline(always)]
    fn hash(s: &str) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for &b in s.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    #[inline(always)]
    pub fn get(&self, s: &str, vocab: &[String]) -> Option<u32> {
        let mut idx = (Self::hash(s) as usize) & self.mask;
        loop {
            let id = self.table[idx];
            if id == u32::MAX {
                return None;
            }
            if vocab[id as usize] == s {
                return Some(id);
            }
            idx = (idx + 1) & self.mask;
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpaceStyle {
    Gpt2,
    SentencePiece,
    None,
}

/// Pre-digested merge table for the fast BPE path.
///
/// The legacy encoder rebuilds a `"a b"` string for every adjacent pair on
/// every merge iteration — O(chars²) allocations, which on a Pi 5 costs
/// ~3.3 ms per prompt token (a 1210-token conversation turn burned 4 s in the
/// tokenizer alone). Here each merge is resolved once at load into a pair of
/// token ids, and the merge order is reproduced by a heap keyed on
/// (rank, left position) — the same "lowest rank, leftmost on ties" rule the
/// legacy scan implements, so token ids are identical.
pub struct BpeFastTables {
    pair_ranks: HashMap<(u32, u32), u32>,
}

pub struct BpeTokenizer {
    pub vocab: Vec<String>,
    pub token_to_id: IndexedStringTable,
    pub merges: HashMap<String, usize>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub special_tokens: Vec<(String, u32)>,
    pub space_style: SpaceStyle,
    /// `None` when the merge table cannot be represented exactly (a merge half
    /// that is not itself a vocab token, or `SpaceStyle::None` where symbols can
    /// contain the key separator); those tokenizers keep the legacy encoder.
    pub bpe_fast: Option<BpeFastTables>,
}

impl BpeTokenizer {
    pub fn from_gguf(gguf: &mut GgufFile) -> Option<Self> {
        let vocab = gguf.take_string_array("tokenizer.ggml.tokens")?;
        let token_to_id = IndexedStringTable::build(&vocab);

        let mut merges = HashMap::new();
        if let Some(merges_list) = gguf.take_string_array("tokenizer.ggml.merges") {
            merges.reserve(merges_list.len());
            for (rank, merge_str) in merges_list.into_iter().enumerate() {
                merges.insert(merge_str, rank);
            }
        }

        let bos_token_id = gguf.get_u32("tokenizer.ggml.bos_token_id");
        let eos_token_id = gguf.get_u32("tokenizer.ggml.eos_token_id");

        let space_style = if token_to_id.get("\u{2581}", &vocab).is_some() {
            SpaceStyle::SentencePiece
        } else if token_to_id.get("Ġ", &vocab).is_some() {
            SpaceStyle::Gpt2
        } else {
            SpaceStyle::None
        };
        let mut special_tokens: Vec<(String, u32)> = Vec::new();
        for (id, tok) in vocab.iter().enumerate() {
            if tok.starts_with("<|") && tok.ends_with("|>") {
                special_tokens.push((tok.clone(), id as u32));
            }
        }
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let mut tok = Self {
            vocab,
            token_to_id,
            merges,
            bos_token_id,
            eos_token_id,
            special_tokens,
            space_style,
            bpe_fast: None,
        };
        tok.bpe_fast = build_fast_tables(&tok);
        Some(tok)
    }

    pub fn from_hf_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, String> {
        let path_ref = path.as_ref();
        let content = std::fs::read_to_string(path_ref)
            .map_err(|e| format!("failed to read tokenizer.json at {}: {}", path_ref.display(), e))?;
        let mut tok = Self::from_hf_json(&content)?;

        // If tokenizer_config.json exists alongside tokenizer.json, refine bos_token_id / eos_token_id
        if let Some(parent) = path_ref.parent() {
            let config_path = parent.join("tokenizer_config.json");
            if let Ok(cfg_str) = std::fs::read_to_string(&config_path) {
                if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&cfg_str) {
                    if let Some(bos_val) = cfg.get("bos_token") {
                        let bos_name = if let Some(s) = bos_val.as_str() {
                            Some(s)
                        } else {
                            bos_val.get("content").and_then(|c| c.as_str())
                        };
                        if let Some(name) = bos_name {
                            if let Some(id) = tok.get_token_id(name) {
                                tok.bos_token_id = Some(id);
                            }
                        }
                    }
                    if let Some(eos_val) = cfg.get("eos_token") {
                        let eos_name = if let Some(s) = eos_val.as_str() {
                            Some(s)
                        } else {
                            eos_val.get("content").and_then(|c| c.as_str())
                        };
                        if let Some(name) = eos_name {
                            if let Some(id) = tok.get_token_id(name) {
                                tok.eos_token_id = Some(id);
                            }
                        }
                    }
                }
            }
        }

        Ok(tok)
    }

    pub fn from_hf_json(json_str: &str) -> Result<Self, String> {
        let root: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| format!("invalid tokenizer JSON: {}", e))?;

        let model = root.get("model").ok_or("tokenizer.json missing 'model' object")?;
        let vocab_obj = model.get("vocab").and_then(|v| v.as_object())
            .ok_or("tokenizer.json model missing 'vocab' map")?;

        let mut max_id = 0usize;
        for id_val in vocab_obj.values() {
            if let Some(id) = id_val.as_u64() {
                max_id = max_id.max(id as usize);
            }
        }

        if let Some(added) = root.get("added_tokens").and_then(|a| a.as_array()) {
            for tok in added {
                if let Some(id) = tok.get("id").and_then(|i| i.as_u64()) {
                    max_id = max_id.max(id as usize);
                }
            }
        }

        let mut vocab = vec![String::new(); max_id + 1];
        for (tok, id_val) in vocab_obj {
            if let Some(id) = id_val.as_u64() {
                vocab[id as usize] = tok.clone();
            }
        }

        let mut special_tokens: Vec<(String, u32)> = Vec::new();
        if let Some(added) = root.get("added_tokens").and_then(|a| a.as_array()) {
            for tok in added {
                let id = tok.get("id").and_then(|i| i.as_u64()).map(|i| i as u32);
                let content = tok.get("content").and_then(|c| c.as_str());
                let is_special = tok.get("special").and_then(|s| s.as_bool()).unwrap_or(false);
                if let (Some(id), Some(content)) = (id, content) {
                    vocab[id as usize] = content.to_string();
                    if is_special || (content.starts_with("<|") && content.ends_with("|>")) {
                        special_tokens.push((content.to_string(), id));
                    }
                }
            }
        }

        let mut merges = HashMap::new();
        if let Some(merges_arr) = model.get("merges").and_then(|m| m.as_array()) {
            merges.reserve(merges_arr.len());
            for (rank, m_val) in merges_arr.iter().enumerate() {
                if let Some(pair) = m_val.as_array() {
                    if pair.len() == 2 {
                        let a = pair[0].as_str().unwrap_or("");
                        let b = pair[1].as_str().unwrap_or("");
                        merges.insert(format!("{} {}", a, b), rank);
                    }
                } else if let Some(s) = m_val.as_str() {
                    merges.insert(s.to_string(), rank);
                }
            }
        }

        let token_to_id = IndexedStringTable::build(&vocab);

        // 1. Detect BOS and EOS tokens from special added_tokens first
        let mut bos_token_id = None;
        let mut eos_token_id = None;

        if let Some(added) = root.get("added_tokens").and_then(|a| a.as_array()) {
            for tok in added {
                let id = tok.get("id").and_then(|i| i.as_u64()).map(|i| i as u32);
                let content = tok.get("content").and_then(|c| c.as_str());
                let is_special = tok.get("special").and_then(|s| s.as_bool()).unwrap_or(false);
                if is_special {
                    if let (Some(id), Some(content)) = (id, content) {
                        if content == "<bos>" || content == "<|startoftext|>" || content == "<|im_start|>" {
                            bos_token_id = Some(id);
                        } else if content == "<s>" && bos_token_id.is_none() {
                            bos_token_id = Some(id);
                        }

                        if content == "<eos>" || content == "<|endoftext|>" || content == "<|im_end|>" {
                            eos_token_id = Some(id);
                        } else if content == "</s>" && eos_token_id.is_none() {
                            eos_token_id = Some(id);
                        }
                    }
                }
            }
        }

        // 2. Check post_processor if not found
        if bos_token_id.is_none() {
            if let Some(post) = root.get("post_processor") {
                if let Some(spec) = post.get("special_tokens").and_then(|s| s.as_object()) {
                    for key in ["<bos>", "<|startoftext|>", "<|im_start|>", "<s>"] {
                        if let Some(entry) = spec.get(key) {
                            if let Some(ids) = entry.get("ids").and_then(|i| i.as_array()) {
                                if let Some(id) = ids.first().and_then(|i| i.as_u64()) {
                                    bos_token_id = Some(id as u32);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // 3. Fallback to token_to_id lookup with strict priority (<bos> before <s>)
        if bos_token_id.is_none() {
            bos_token_id = token_to_id.get("<bos>", &vocab)
                .or_else(|| token_to_id.get("<|startoftext|>", &vocab))
                .or_else(|| token_to_id.get("<|im_start|>", &vocab))
                .or_else(|| token_to_id.get("<s>", &vocab));
        }

        if eos_token_id.is_none() {
            eos_token_id = token_to_id.get("<eos>", &vocab)
                .or_else(|| token_to_id.get("<|endoftext|>", &vocab))
                .or_else(|| token_to_id.get("<|im_end|>", &vocab))
                .or_else(|| token_to_id.get("</s>", &vocab));
        }

        // Determine space style:
        let norm_str = root.get("normalizer").map(|n| n.to_string()).unwrap_or_default();
        let dec_str = root.get("decoder").map(|d| d.to_string()).unwrap_or_default();
        let pre_str = root.get("pre_tokenizer").map(|p| p.to_string()).unwrap_or_default();

        let space_style = if norm_str.contains("\u{2581}") || dec_str.contains("\u{2581}") {
            SpaceStyle::SentencePiece
        } else if pre_str.contains("ByteLevel") || dec_str.contains("ByteLevel") {
            SpaceStyle::Gpt2
        } else if token_to_id.get("\u{2581}", &vocab).is_some() {
            SpaceStyle::SentencePiece
        } else if token_to_id.get("Ġ", &vocab).is_some() {
            SpaceStyle::Gpt2
        } else {
            SpaceStyle::None
        };
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let mut tok = Self {
            vocab,
            token_to_id,
            merges,
            bos_token_id,
            eos_token_id,
            special_tokens,
            space_style,
            bpe_fast: None,
        };
        tok.bpe_fast = build_fast_tables(&tok);
        Ok(tok)
    }

    #[inline(always)]
    pub fn get_token_id(&self, s: &str) -> Option<u32> {
        self.token_to_id.get(s, &self.vocab)
    }

    /// Encode input text into a list of token IDs with special token support.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut tokens = Vec::new();
        if let Some(bos) = self.bos_token_id {
            tokens.push(bos);
        }

        if self.special_tokens.is_empty() {
            tokens.extend(self.encode_segment(text));
            return tokens;
        }

        let mut cursor = 0;
        while cursor < text.len() {
            let mut earliest: Option<(usize, usize, u32)> = None;
            for (st, id) in &self.special_tokens {
                if let Some(pos) = text[cursor..].find(st) {
                    let abs_pos = cursor + pos;
                    match earliest {
                        None => earliest = Some((abs_pos, st.len(), *id)),
                        Some((earliest_pos, _, _)) if abs_pos < earliest_pos => {
                            earliest = Some((abs_pos, st.len(), *id));
                        }
                        _ => {}
                    }
                }
            }

            match earliest {
                Some((pos, len, id)) => {
                    if pos > cursor {
                        let seg = &text[cursor..pos];
                        tokens.extend(self.encode_segment(seg));
                    }
                    tokens.push(id);
                    cursor = pos + len;
                }
                None => {
                    let seg = &text[cursor..];
                    tokens.extend(self.encode_segment(seg));
                    break;
                }
            }
        }

        tokens
    }

    /// Byte-level (GPT-2) or SentencePiece normalization applied before BPE.
    fn normalize_text(&self, text: &str) -> String {
        match self.space_style {
            // GPT-2/Llama BPE is byte-level. Mapping only spaces corrupts
            // prompts containing newlines, tabs, quotes, and non-ASCII text.
            SpaceStyle::Gpt2 => text.bytes().map(gpt2_byte_to_char).collect(),
            SpaceStyle::SentencePiece => text.replace(' ', "\u{2581}"),
            SpaceStyle::None => text.to_string(),
        }
    }

    /// Encode a single regular text segment using BPE merges.
    fn encode_segment(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let normalized = self.normalize_text(text);
        // SOFUU_BPE_LEGACY=1 restores the O(n²) scanner (also the test oracle
        // for the fast path). Resolved once per process.
        static LEGACY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let legacy = *LEGACY.get_or_init(|| super::quant::env_flag("SOFUU_BPE_LEGACY"));
        match (&self.bpe_fast, legacy) {
            (Some(fast), false) => self.encode_segment_fast(fast, &normalized),
            _ => self.encode_segment_legacy(&normalized),
        }
    }

    /// Heap-driven BPE over interned token ids: same merge order as the legacy
    /// scanner (lowest rank first, leftmost on ties), so the ids are identical
    /// — only the pair keys and the repeated full rescans are gone.
    fn encode_segment_fast(&self, fast: &BpeFastTables, normalized: &str) -> Vec<u32> {
        let n = normalized.chars().count();
        if n == 0 {
            return Vec::new();
        }

        // Local symbol arena: a merged symbol is the concatenation of two
        // adjacent ones, so bytes are only ever appended.
        let mut arena: Vec<u8> = Vec::with_capacity(normalized.len() * 2);
        let mut sym_off: Vec<u32> = Vec::with_capacity(n * 2);
        let mut sym_len: Vec<u32> = Vec::with_capacity(n * 2);
        let mut sym_id: Vec<u32> = Vec::with_capacity(n * 2); // vocab id or u32::MAX
        // Distinct characters repeat heavily; one hash probe per character
        // would otherwise dominate short texts.
        let mut char_cache: HashMap<char, u32> = HashMap::new();
        for c in normalized.chars() {
            let mut buf = [0u8; 4];
            let s = c.encode_utf8(&mut buf);
            let id = *char_cache
                .entry(c)
                .or_insert_with(|| self.token_to_id.get(s, &self.vocab).unwrap_or(u32::MAX));
            sym_off.push(arena.len() as u32);
            sym_len.push(s.len() as u32);
            arena.extend_from_slice(s.as_bytes());
            sym_id.push(id);
        }

        // Slots: one per initial symbol, in text order. A merge folds the right
        // slot into the left one, so slot order stays left-to-right and slot
        // index is the position used for the leftmost tie-break.
        let mut slot_sym: Vec<u32> = (0..n as u32).collect();
        let mut prev: Vec<i32> = (0..n as i32).map(|i| i - 1).collect();
        let mut next: Vec<i32> = (0..n as i32).map(|i| i + 1).collect();
        next[n - 1] = -1;
        let mut alive = vec![true; n];

        let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(u32, u32, u32)>> =
            std::collections::BinaryHeap::new();
        for i in 0..n.saturating_sub(1) {
            push_pair(&mut heap, &fast.pair_ranks, &sym_id, &slot_sym, i, i + 1);
        }

        while let Some(std::cmp::Reverse((rank, l, r))) = heap.pop() {
            let (li, ri) = (l as usize, r as usize);
            if !alive[li] || !alive[ri] || next[li] != ri as i32 {
                continue; // stale: a neighbour merged since this was queued
            }
            // The stored rank can be stale when the left symbol was just
            // rewritten by another merge; only an unchanged pair may fire.
            let (la, ra) = (slot_sym[li] as usize, slot_sym[ri] as usize);
            match fast.pair_ranks.get(&(sym_id[la], sym_id[ra])) {
                Some(&cur) if cur == rank => {}
                _ => continue,
            }

            let (o1, l1) = (sym_off[la], sym_len[la]);
            let (o2, l2) = (sym_off[ra], sym_len[ra]);
            let start = arena.len() as u32;
            arena.extend_from_within(o1 as usize..(o1 + l1) as usize);
            arena.extend_from_within(o2 as usize..(o2 + l2) as usize);
            let total = l1 + l2;
            let merged_id = std::str::from_utf8(&arena[start as usize..(start + total) as usize])
                .ok()
                .and_then(|s| self.token_to_id.get(s, &self.vocab))
                .unwrap_or(u32::MAX);
            sym_off.push(start);
            sym_len.push(total);
            sym_id.push(merged_id);
            slot_sym[li] = (sym_id.len() - 1) as u32;
            alive[ri] = false;
            next[li] = next[ri];
            if next[li] >= 0 {
                prev[next[li] as usize] = li as i32;
            }
            if prev[li] >= 0 {
                push_pair(&mut heap, &fast.pair_ranks, &sym_id, &slot_sym, prev[li] as usize, li);
            }
            if next[li] >= 0 {
                push_pair(&mut heap, &fast.pair_ranks, &sym_id, &slot_sym, li, next[li] as usize);
            }
        }

        let mut out = Vec::with_capacity(n);
        let mut slot = 0i32;
        while slot >= 0 {
            let si = slot as usize;
            let sym = slot_sym[si] as usize;
            let id = sym_id[sym];
            if id != u32::MAX {
                out.push(id);
            } else {
                // Byte fallback: encode individual bytes as "<0xXX>" tokens.
                let o = sym_off[sym] as usize;
                let l = sym_len[sym] as usize;
                for &b in &arena[o..o + l] {
                    let byte_repr = format!("<0x{:02X}>", b);
                    if let Some(id) = self.get_token_id(&byte_repr) {
                        out.push(id);
                    }
                }
            }
            slot = next[si];
        }
        out
    }

    /// Reference encoder: rescans the whole segment for the best pair on every
    /// merge. Kept as the fallback for tokenizers the fast tables cannot
    /// represent exactly, and as a test oracle for the fast path.
    fn encode_segment_legacy(&self, normalized: &str) -> Vec<u32> {
        // Initial character tokens
        let mut word_parts: Vec<String> = normalized.chars().map(|c| c.to_string()).collect();

        // Iteratively apply BPE merges based on rank
        loop {
            if word_parts.len() < 2 {
                break;
            }

            let mut best_pair: Option<(usize, usize)> = None;
            for i in 0..word_parts.len() - 1 {
                let pair_key = format!("{} {}", word_parts[i], word_parts[i + 1]);
                if let Some(&rank) = self.merges.get(&pair_key) {
                    match best_pair {
                        None => best_pair = Some((i, rank)),
                        Some((_, best_rank)) if rank < best_rank => best_pair = Some((i, rank)),
                        _ => {}
                    }
                }
            }

            let Some((best_idx, _)) = best_pair else {
                break;
            };

            let merged = format!("{}{}", word_parts[best_idx], word_parts[best_idx + 1]);
            word_parts[best_idx] = merged;
            word_parts.remove(best_idx + 1);
        }

        let mut tokens = Vec::with_capacity(word_parts.len());
        // Map final merged substrings to token IDs
        for part in word_parts {
            if let Some(id) = self.get_token_id(&part) {
                tokens.push(id);
            } else {
                // Byte fallback: encode individual bytes as tokens
                for b in part.as_bytes() {
                    let byte_repr = format!("<0x{:02X}>", b);
                    if let Some(id) = self.get_token_id(&byte_repr) {
                        tokens.push(id);
                    }
                }
            }
        }

        tokens
    }

    /// Decode token ID back into human-readable string slice.
    pub fn decode_token(&self, token_id: u32) -> Option<&str> {
        let s = self.vocab.get(token_id as usize)?;
        Some(s.as_str())
    }

    /// Decode a single token ID into a clean text piece for streaming.
    pub fn decode_piece(&self, token_id: u32) -> String {
        if let Some(s) = self.decode_token(token_id) {
            if s.starts_with("<|") && s.ends_with("|>") {
                return String::new();
            }
            if s.starts_with('<') && s.ends_with('>') && !s.starts_with("<0x") {
                return String::new();
            }
            if s.starts_with("<0x") && s.ends_with('>') && s.len() == 6 {
                if let Ok(b) = u8::from_str_radix(&s[3..5], 16) {
                    return (b as char).to_string();
                }
            }
            match self.space_style {
                SpaceStyle::Gpt2 => {
                    let dec = decode_byte_level(s);
                    dec.replace('\u{2581}', " ")
                }
                SpaceStyle::SentencePiece => {
                    s.replace('\u{2581}', " ")
                }
                SpaceStyle::None => s.to_string(),
            }
        } else {
            String::new()
        }
    }

    /// Decode a sequence of token IDs to clean text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut out = String::new();
        for &t in tokens {
            out.push_str(&self.decode_piece(t));
        }
        out
    }
}

/// Queue the pair starting at `left`, unless either symbol is not a vocab
/// token (such a pair can never match a merge, see `build_fast_tables`).
fn push_pair(
    heap: &mut std::collections::BinaryHeap<std::cmp::Reverse<(u32, u32, u32)>>,
    pair_ranks: &HashMap<(u32, u32), u32>,
    sym_id: &[u32],
    slot_sym: &[u32],
    left: usize,
    right: usize,
) {
    let a = sym_id[slot_sym[left] as usize];
    let b = sym_id[slot_sym[right] as usize];
    if a != u32::MAX && b != u32::MAX {
        if let Some(&rank) = pair_ranks.get(&(a, b)) {
            heap.push(std::cmp::Reverse((rank, left as u32, right as u32)));
        }
    }
}

/// Resolve the merge table into `(left_id, right_id) -> rank` for the fast
/// encoder. Returns `None` (legacy encoder) when a merge half is not itself a
/// vocab token — the pair-key mapping would then not be exact — or when the
/// style's symbols can contain the `' '` key separator (`SpaceStyle::None`).
fn build_fast_tables(tok: &BpeTokenizer) -> Option<BpeFastTables> {
    if tok.space_style == SpaceStyle::None || tok.merges.is_empty() {
        return None;
    }
    let mut pair_ranks = HashMap::with_capacity(tok.merges.len());
    for (key, &rank) in &tok.merges {
        // A merge key is exactly "<left> <right>"; a key with more spaces
        // would be silently split into the wrong pair.
        if key.matches(' ').count() != 1 {
            return None;
        }
        let (a, b) = key.split_once(' ')?;
        let ia = tok.token_to_id.get(a, &tok.vocab)?;
        let ib = tok.token_to_id.get(b, &tok.vocab)?;
        pair_ranks.insert((ia, ib), rank as u32);
    }
    Some(BpeFastTables { pair_ranks })
}

fn gpt2_bytes_table() -> [Option<u8>; 512] {
    let mut table = [None; 512];
    let mut bs = Vec::with_capacity(256);
    for b in b'!'..=b'~' { bs.push(b); }
    for b in 0xa1..=0xac { bs.push(b); }
    for b in 0xae..=0xff { bs.push(b); }
    let mut cs = bs.iter().map(|&b| b as u32).collect::<Vec<_>>();
    let mut n = 0u32;
    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    for (b, c) in bs.into_iter().zip(cs.into_iter()) {
        if (c as usize) < 512 {
            table[c as usize] = Some(b);
        }
    }
    table
}

fn gpt2_byte_to_char(b: u8) -> char {
    static ENCODER: std::sync::OnceLock<[char; 256]> = std::sync::OnceLock::new();
    let table = ENCODER.get_or_init(|| {
        let mut bs = Vec::with_capacity(256);
        for byte in b'!'..=b'~' {
            bs.push(byte);
        }
        for byte in 0xa1..=0xac {
            bs.push(byte);
        }
        for byte in 0xae..=0xff {
            bs.push(byte);
        }
        let mut cs: Vec<u32> = bs.iter().map(|&byte| byte as u32).collect();
        let mut n = 0u32;
        for byte in 0..=255u8 {
            if !bs.contains(&byte) {
                bs.push(byte);
                cs.push(256 + n);
                n += 1;
            }
        }
        let mut encoder = ['\0'; 256];
        for (byte, code) in bs.into_iter().zip(cs.into_iter()) {
            encoder[byte as usize] = char::from_u32(code).unwrap_or('\0');
        }
        encoder
    });
    table[b as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_tokenizer() -> BpeTokenizer {
        // GPT-2 byte-level vocab over the characters the tests use, with the
        // merge table of a handful of words so chained merges are exercised.
        let mut vocab: Vec<String> = Vec::new();
        let mut push = |s: &str| {
            vocab.push(s.to_string());
        };
        for b in 0u8..=255 {
            push(&gpt2_byte_to_char(b).to_string());
        }
        for s in [
            "he", "ll", "hel", "hell", "hello", "Ġw", "Ġwo", "wo", "wor", "worl",
            "world", "ĠĠ", "!!", ",Ġ",
        ] {
            push(s);
        }
        for b in 0u8..=255 {
            push(&format!("<0x{:02X}>", b));
        }
        let mut merges = HashMap::new();
        for (rank, key) in [
            "h e", "l l", "he l", "hell o", "Ġ w", "Ġw o", "wo r", "wor l",
            "worl d", "Ġ Ġ", "! !", ", Ġ",
        ]
        .iter()
        .enumerate()
        {
            merges.insert(key.to_string(), rank);
        }
        let token_to_id = IndexedStringTable::build(&vocab);
        let space_style = SpaceStyle::Gpt2;
        let mut tok = BpeTokenizer {
            vocab,
            token_to_id,
            merges,
            bos_token_id: None,
            eos_token_id: None,
            special_tokens: Vec::new(),
            space_style,
            bpe_fast: None,
        };
        tok.bpe_fast = build_fast_tables(&tok);
        tok
    }

    fn assert_same(tok: &BpeTokenizer, text: &str) {
        let fast = tok.bpe_fast.as_ref().expect("toy tokenizer must use the fast path");
        let normalized = tok.normalize_text(text);
        let a = tok.encode_segment_fast(fast, &normalized);
        let b = tok.encode_segment_legacy(&normalized);
        assert_eq!(a, b, "fast vs legacy diverged for {text:?}");
    }

    #[test]
    fn fast_bpe_matches_legacy_on_edge_cases() {
        let tok = toy_tokenizer();
        for text in [
            "",
            "h",
            "hello",
            "hello world",
            "hello  world",
            "helloworld",
            "hello, world!!",
            "\n\thello\n",
            "hello hello hello",
            "wörld",           // chars outside the toy vocab -> byte fallback
            "héllo wörld",
            "....",
            "  ",
            "a b c",
            "hello\nworld",
        ] {
            assert_same(&tok, text);
        }
    }

    /// Random text over the toy alphabet, including the separator and bytes
    /// that only exist as <0xXX> tokens: 400 strings, compared id-for-id.
    #[test]
    fn fast_bpe_matches_legacy_fuzz() {
        let tok = toy_tokenizer();
        let alphabet: Vec<char> = "helowrd,!? \n\tĠ".chars().collect();
        let mut rng: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for _ in 0..400 {
            let len = (next() % 24) as usize;
            let s: String = (0..len)
                .map(|_| alphabet[(next() as usize) % alphabet.len()])
                .collect();
            assert_same(&tok, &s);
        }
    }

    /// A merge that duplicates an earlier key must keep the later rank (the
    /// string map overwrites); the pair map has to reproduce that.
    #[test]
    fn fast_bpe_duplicate_merge_rank() {
        let mut tok = toy_tokenizer();
        tok.merges.insert("l l".to_string(), 99);
        tok.merges.insert("h e".to_string(), 100);
        tok.bpe_fast = build_fast_tables(&tok);
        assert_same(&tok, "hello");
    }

    /// End-to-end equivalence on the real production prompt with the real
    /// Qwen tokenizer, when those files exist on this machine.
    #[test]
    fn fast_bpe_matches_legacy_real_qwen_prompt() {
        let candidates = [
            "/Users/priyanshuboruah/Downloads/Qwen3-0.6B-Q4_K_M.gguf",
            "/Users/priyanshuboruah/Downloads/Qwen-3-0.6B-Claude-4.7-Opus-Distilled.i1-Q4_K_M.gguf",
            "/Users/priyanshuboruah/Downloads/Llama-3.2-1B-Instruct-Q8_0.gguf",
        ];
        let model = match candidates.iter().find(|p| std::path::Path::new(p).exists()) {
            Some(p) => std::path::Path::new(p),
            None => return,
        };
        let prompt = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../scripts/bench_prompt.txt");
        if !prompt.exists() {
            return;
        }
        let mut gguf = GgufFile::open(model).expect("open model");
        let tok = BpeTokenizer::from_gguf(&mut gguf).expect("tokenizer");
        assert!(tok.bpe_fast.is_some(), "Qwen merge table must be representable");
        let text = std::fs::read_to_string(prompt).expect("read prompt");
        assert_same(&tok, &text);
        // A multi-turn-ish concatenation, with special-token-like text.
        let convo = format!(
            "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\nSure — here\tis\n\n a reply.<|im_end|>\n"
        );
        assert_same(&tok, &convo);
        let ids = tok.encode(&convo);
        assert!(!ids.is_empty());
    }
}

pub fn decode_byte_level(s: &str) -> String {
    static TABLE: std::sync::OnceLock<[Option<u8>; 512]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(gpt2_bytes_table);
    let mut bytes = Vec::with_capacity(s.len());
    let mut has_mapped = false;
    for c in s.chars() {
        let cp = c as usize;
        if cp < 512 {
            if let Some(b) = table[cp] {
                bytes.push(b);
                has_mapped = true;
                continue;
            }
        }
        let mut buf = [0u8; 4];
        let enc = c.encode_utf8(&mut buf);
        bytes.extend_from_slice(enc.as_bytes());
    }
    if has_mapped {
        String::from_utf8_lossy(&bytes).into_owned()
    } else {
        s.to_string()
    }
}

