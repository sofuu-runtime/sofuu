// ml/relevance — the pre-retrieval relevance gate (PLAN-ML-GATES §6).
//
// Phase 4. Scores a candidate set (files, memories, search results) against
// the current task and advises the retriever on which candidates deserve the
// tokens — before they are spent. Advise-only (Principle 1): the guidance
// rides the ephemeral context message; the LLM decides. features.rs is the
// 37-feature extractor (BM25 over the candidate set + semantic cosine +
// morphological stem match + duplication + shape + task×content
// interactions); model.rs the trained net + policy; eval.rs the
// committed-weights fixtures.

pub mod features;
pub mod model;

#[cfg(test)]
mod eval;
