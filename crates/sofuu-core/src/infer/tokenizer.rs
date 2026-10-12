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

pub struct BpeTokenizer {
    pub vocab: Vec<String>,
    pub token_to_id: IndexedStringTable,
    pub merges: HashMap<String, usize>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub special_tokens: Vec<(String, u32)>,
    pub use_gpt2_space: bool,
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

        let use_gpt2_space = token_to_id.get("Ġ", &vocab).is_some();
        let mut special_tokens: Vec<(String, u32)> = Vec::new();
        for (id, tok) in vocab.iter().enumerate() {
            if tok.starts_with("<|") && tok.ends_with("|>") {
                special_tokens.push((tok.clone(), id as u32));
            }
        }
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        Some(Self {
            vocab,
            token_to_id,
            merges,
            bos_token_id,
            eos_token_id,
            special_tokens,
            use_gpt2_space,
        })
    }

    pub fn from_hf_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, String> {
        let path_ref = path.as_ref();
        let content = std::fs::read_to_string(path_ref)
            .map_err(|e| format!("failed to read tokenizer.json at {}: {}", path_ref.display(), e))?;
        Self::from_hf_json(&content)
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
        let bos_token_id = token_to_id.get("<|startoftext|>", &vocab)
            .or_else(|| token_to_id.get("<s>", &vocab))
            .or_else(|| token_to_id.get("<|im_start|>", &vocab));
        let eos_token_id = token_to_id.get("<|endoftext|>", &vocab)
            .or_else(|| token_to_id.get("</s>", &vocab))
            .or_else(|| token_to_id.get("<|im_end|>", &vocab));

        let use_gpt2_space = token_to_id.get("Ġ", &vocab).is_some();
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        Ok(Self {
            vocab,
            token_to_id,
            merges,
            bos_token_id,
            eos_token_id,
            special_tokens,
            use_gpt2_space,
        })
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

    /// Encode a single regular text segment using BPE merges.
    fn encode_segment(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let normalized = if self.use_gpt2_space {
            text.replace(' ', "Ġ")
        } else {
            text.replace(' ', " ")
        };

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
            let dec = decode_byte_level(s);
            dec.replace(' ', " ")
        } else {
            String::new()
        }
    }

    /// Decode a sequence of token IDs to clean text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut raw = String::new();
        for &t in tokens {
            if let Some(s) = self.decode_token(t) {
                // Skip special tokens in standard text rendering
                if s.starts_with("<|") && s.ends_with("|>") {
                    continue;
                }
                raw.push_str(s);
            }
        }
        let dec = decode_byte_level(&raw);
        dec.replace(' ', " ")
    }
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

