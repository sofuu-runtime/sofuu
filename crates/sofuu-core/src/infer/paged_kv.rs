// crates/sofuu-core/src/infer/paged_kv.rs — Paged KV Cache Memory Architecture.
//
// Eliminates contiguous preallocation of KV caches (which wastes 250MB–1.2GB per connection).
// Allocates fixed 16-token physical memory blocks on demand as sequences grow,
// with immediate recycling back to a shared block pool upon stream completion.

pub const KV_BLOCK_SIZE: usize = 16;

/// A physical memory block holding up to 16 tokens of Key and Value tensors.
#[derive(Clone, Debug)]
pub struct KvBlock {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub latent: Option<Vec<f32>>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub latent_dim: usize,
}

impl KvBlock {
    pub fn new(n_kv_heads: usize, head_dim: usize) -> Self {
        let cap = KV_BLOCK_SIZE * n_kv_heads * head_dim;
        Self {
            k: vec![0.0f32; cap],
            v: vec![0.0f32; cap],
            latent: None,
            n_kv_heads,
            head_dim,
            latent_dim: 0,
        }
    }

    pub fn new_latent(latent_dim: usize) -> Self {
        let cap = KV_BLOCK_SIZE * latent_dim;
        Self {
            k: Vec::new(),
            v: Vec::new(),
            latent: Some(vec![0.0f32; cap]),
            n_kv_heads: 0,
            head_dim: 0,
            latent_dim,
        }
    }

    #[inline]
    pub fn store(&mut self, offset_in_block: usize, kv_head: usize, k_vec: &[f32], v_vec: &[f32]) {
        let idx = (offset_in_block * self.n_kv_heads + kv_head) * self.head_dim;
        self.k[idx..idx + self.head_dim].copy_from_slice(k_vec);
        self.v[idx..idx + self.head_dim].copy_from_slice(v_vec);
    }

    #[inline]
    pub fn get_k(&self, offset_in_block: usize, kv_head: usize) -> &[f32] {
        let idx = (offset_in_block * self.n_kv_heads + kv_head) * self.head_dim;
        &self.k[idx..idx + self.head_dim]
    }

    #[inline]
    pub fn get_v(&self, offset_in_block: usize, kv_head: usize) -> &[f32] {
        let idx = (offset_in_block * self.n_kv_heads + kv_head) * self.head_dim;
        &self.v[idx..idx + self.head_dim]
    }

    #[inline]
    pub fn store_latent(&mut self, offset_in_block: usize, latent_vec: &[f32]) {
        if let Some(ref mut l) = self.latent {
            let idx = offset_in_block * self.latent_dim;
            let len = latent_vec.len().min(self.latent_dim);
            l[idx..idx + len].copy_from_slice(&latent_vec[..len]);
        }
    }

    #[inline]
    pub fn get_latent(&self, offset_in_block: usize) -> &[f32] {
        if let Some(ref l) = self.latent {
            let idx = offset_in_block * self.latent_dim;
            &l[idx..idx + self.latent_dim]
        } else {
            &[]
        }
    }
}

/// A physical memory pool of KvBlocks for a specific layer.
pub struct BlockPool {
    blocks: Vec<KvBlock>,
    free_list: Vec<usize>,
    n_kv_heads: usize,
    head_dim: usize,
    latent_dim: usize,
    is_latent: bool,
    max_blocks: usize,
}

impl BlockPool {
    pub fn new(n_kv_heads: usize, head_dim: usize, initial_blocks: usize, max_blocks: usize) -> Self {
        let mut blocks = Vec::with_capacity(initial_blocks);
        let mut free_list = Vec::with_capacity(initial_blocks);
        for i in 0..initial_blocks {
            blocks.push(KvBlock::new(n_kv_heads, head_dim));
            free_list.push(i);
        }
        Self {
            blocks,
            free_list,
            n_kv_heads,
            head_dim,
            latent_dim: 0,
            is_latent: false,
            max_blocks: max_blocks.max(initial_blocks),
        }
    }

    pub fn new_latent(latent_dim: usize, initial_blocks: usize, max_blocks: usize) -> Self {
        let mut blocks = Vec::with_capacity(initial_blocks);
        let mut free_list = Vec::with_capacity(initial_blocks);
        for i in 0..initial_blocks {
            blocks.push(KvBlock::new_latent(latent_dim));
            free_list.push(i);
        }
        Self {
            blocks,
            free_list,
            n_kv_heads: 0,
            head_dim: 0,
            latent_dim,
            is_latent: true,
            max_blocks: max_blocks.max(initial_blocks),
        }
    }

    /// Allocate a block ID from the pool, creating a new block if free list is empty and below cap.
    pub fn alloc_block(&mut self) -> Result<usize, String> {
        if let Some(id) = self.free_list.pop() {
            return Ok(id);
        }
        if self.blocks.len() < self.max_blocks {
            let id = self.blocks.len();
            if self.is_latent {
                self.blocks.push(KvBlock::new_latent(self.latent_dim));
            } else {
                self.blocks.push(KvBlock::new(self.n_kv_heads, self.head_dim));
            }
            return Ok(id);
        }
        Err(format!(
            "Paged KV pool exhausted (cap of {} blocks reached). Drop idle streams or expand pool.",
            self.max_blocks
        ))
    }

    /// Free a block back to the pool.
    pub fn free_block(&mut self, id: usize) {
        if id < self.blocks.len() && !self.free_list.contains(&id) {
            self.free_list.push(id);
        }
    }

    /// Free multiple blocks back to the pool.
    pub fn free_blocks(&mut self, ids: &[usize]) {
        for &id in ids {
            self.free_block(id);
        }
    }

    #[inline]
    pub fn get_block(&self, id: usize) -> &KvBlock {
        &self.blocks[id]
    }

    #[inline]
    pub fn get_block_mut(&mut self, id: usize) -> &mut KvBlock {
        &mut self.blocks[id]
    }

    pub fn total_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn free_blocks_count(&self) -> usize {
        self.free_list.len()
    }

    pub fn allocated_blocks_count(&self) -> usize {
        self.blocks.len().saturating_sub(self.free_list.len())
    }

    pub fn memory_bytes(&self) -> usize {
        let block_bytes = if self.is_latent {
            KV_BLOCK_SIZE * self.latent_dim * std::mem::size_of::<f32>()
        } else {
            KV_BLOCK_SIZE * self.n_kv_heads * self.head_dim * std::mem::size_of::<f32>() * 2
        };
        self.blocks.len() * block_bytes
    }
}

/// A logical block table mapping sequence token positions to physical blocks in a BlockPool.
#[derive(Debug, Clone, Default)]
pub struct SequenceBlockTable {
    pub block_ids: Vec<usize>,
    pub num_tokens: usize,
}

impl SequenceBlockTable {
    pub fn new() -> Self {
        Self {
            block_ids: Vec::new(),
            num_tokens: 0,
        }
    }

    /// Append a single token's KV across all heads. Allocates new block when crossing 16-token boundary.
    pub fn append_token(
        &mut self,
        pool: &mut BlockPool,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), String> {
        let pos = self.num_tokens;
        let block_idx = pos / KV_BLOCK_SIZE;
        let offset = pos % KV_BLOCK_SIZE;

        if block_idx >= self.block_ids.len() {
            let b_id = pool.alloc_block()?;
            self.block_ids.push(b_id);
        }

        let b_id = self.block_ids[block_idx];
        let n_kv_heads = pool.n_kv_heads;
        let head_dim = pool.head_dim;
        let block = pool.get_block_mut(b_id);

        for h in 0..n_kv_heads {
            let src_off = h * head_dim;
            block.store(offset, h, &k[src_off..src_off + head_dim], &v[src_off..src_off + head_dim]);
        }
        self.num_tokens += 1;
        Ok(())
    }

    /// Append a single token's compressed MLA latent vector.
    pub fn append_token_latent(
        &mut self,
        pool: &mut BlockPool,
        latent: &[f32],
    ) -> Result<(), String> {
        let pos = self.num_tokens;
        let block_idx = pos / KV_BLOCK_SIZE;
        let offset = pos % KV_BLOCK_SIZE;

        if block_idx >= self.block_ids.len() {
            let b_id = pool.alloc_block()?;
            self.block_ids.push(b_id);
        }

        let b_id = self.block_ids[block_idx];
        let block = pool.get_block_mut(b_id);
        block.store_latent(offset, latent);
        self.num_tokens += 1;
        Ok(())
    }

    /// Get Key vector for token `pos` and `kv_head`.
    #[inline]
    pub fn get_k<'a>(&self, pool: &'a BlockPool, kv_head: usize, pos: usize) -> Option<&'a [f32]> {
        if pos >= self.num_tokens {
            return None;
        }
        let block_idx = pos / KV_BLOCK_SIZE;
        let offset = pos % KV_BLOCK_SIZE;
        let b_id = self.block_ids.get(block_idx)?;
        Some(pool.get_block(*b_id).get_k(offset, kv_head))
    }

    /// Get Value vector for token `pos` and `kv_head`.
    #[inline]
    pub fn get_v<'a>(&self, pool: &'a BlockPool, kv_head: usize, pos: usize) -> Option<&'a [f32]> {
        if pos >= self.num_tokens {
            return None;
        }
        let block_idx = pos / KV_BLOCK_SIZE;
        let offset = pos % KV_BLOCK_SIZE;
        let b_id = self.block_ids.get(block_idx)?;
        Some(pool.get_block(*b_id).get_v(offset, kv_head))
    }

    /// Get compressed MLA latent vector for token `pos`.
    #[inline]
    pub fn get_latent<'a>(&self, pool: &'a BlockPool, pos: usize) -> Option<&'a [f32]> {
        if pos >= self.num_tokens {
            return None;
        }
        let block_idx = pos / KV_BLOCK_SIZE;
        let offset = pos % KV_BLOCK_SIZE;
        let b_id = self.block_ids.get(block_idx)?;
        Some(pool.get_block(*b_id).get_latent(offset))
    }

    /// Release all blocks back to pool and clear the table.
    pub fn clear(&mut self, pool: &mut BlockPool) {
        pool.free_blocks(&self.block_ids);
        self.block_ids.clear();
        self.num_tokens = 0;
    }

    /// Truncate to `target_len` tokens, freeing newly unreferenced tail blocks.
    pub fn truncate_to(&mut self, pool: &mut BlockPool, target_len: usize) {
        if target_len >= self.num_tokens {
            return;
        }
        self.num_tokens = target_len;
        let needed_blocks = if target_len == 0 {
            0
        } else {
            (target_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE
        };
        if needed_blocks < self.block_ids.len() {
            let to_free = &self.block_ids[needed_blocks..];
            pool.free_blocks(to_free);
            self.block_ids.truncate(needed_blocks);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paged_kv_block_allocation_and_boundary() {
        let mut pool = BlockPool::new(4, 64, 2, 10);
        assert_eq!(pool.total_blocks(), 2);
        assert_eq!(pool.free_blocks_count(), 2);

        let mut table = SequenceBlockTable::new();
        let k_sample = vec![1.5f32; 4 * 64];
        let v_sample = vec![2.5f32; 4 * 64];

        // Store 35 tokens (should cross 16, 32 boundaries -> 3 blocks allocated)
        for i in 0..35 {
            let mut k = k_sample.clone();
            k[0] = i as f32;
            table.append_token(&mut pool, &k, &v_sample).unwrap();
        }

        assert_eq!(table.num_tokens, 35);
        assert_eq!(table.block_ids.len(), 3);
        assert_eq!(pool.allocated_blocks_count(), 3);

        // Verify values at boundary points
        for i in [0, 15, 16, 31, 32, 34] {
            let k_read = table.get_k(&pool, 0, i).unwrap();
            assert_eq!(k_read[0], i as f32);
            assert_eq!(k_read[1], 1.5f32);
            let v_read = table.get_v(&pool, 0, i).unwrap();
            assert_eq!(v_read[0], 2.5f32);
        }

        // Truncate to 20 tokens (should free 1 block: needed_blocks = 2)
        table.truncate_to(&mut pool, 20);
        assert_eq!(table.num_tokens, 20);
        assert_eq!(table.block_ids.len(), 2);
        assert_eq!(pool.allocated_blocks_count(), 2);

        // Clear table (should free all blocks back to pool)
        table.clear(&mut pool);
        assert_eq!(table.num_tokens, 0);
        assert_eq!(table.block_ids.len(), 0);
        assert_eq!(pool.allocated_blocks_count(), 0);
    }

    #[test]
    fn test_paged_kv_latent_mla() {
        let mut pool = BlockPool::new_latent(576, 2, 8);
        let mut table = SequenceBlockTable::new();
        let latent_sample = vec![0.42f32; 576];

        for i in 0..25 {
            let mut l = latent_sample.clone();
            l[0] = (i * 10) as f32;
            table.append_token_latent(&mut pool, &l).unwrap();
        }

        assert_eq!(table.num_tokens, 25);
        assert_eq!(table.block_ids.len(), 2);

        let l_read = table.get_latent(&pool, 24).unwrap();
        assert_eq!(l_read[0], 240.0f32);
        assert_eq!(l_read[1], 0.42f32);

        table.clear(&mut pool);
        assert_eq!(pool.allocated_blocks_count(), 0);
    }
}
