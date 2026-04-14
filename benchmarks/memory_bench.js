// In our engine, Sofuu global is injected.
try {
    console.log("== Cognitive Memory Architecture Test ==");
    const dim = 16;
    
    // Create dummy vectors
    const vec1 = new Float32Array(dim);
    for (let i = 0; i < dim; i++) vec1[i] = 1.0;
    
    const vec2 = new Float32Array(dim);
    for (let i = 0; i < dim; i++) vec2[i] = 2.0;

    const vec3 = new Float32Array(dim);
    for (let i = 0; i < dim; i++) vec3[i] = 3.0;

    // 1. Open Memory
    const mindPath = "test_agent.mind.qtsq";
    console.log(`Opening memory at ${mindPath} (dim=${dim})...`);
    let mem = Sofuu.memory.open(mindPath, dim);
    
    // 2. Remember
    console.log("Remembering vector 1 (User)...");
    const id1 = mem.remember(vec1, "Hello, who are you?", "user", 101);
    console.log("Remembering vector 2 (Assistant)...");
    const id2 = mem.remember(vec2, "I am Sofuu CMA.", "assistant", 102);
    
    // 3. Recall
    console.log("Recalling vector closest to vector 1...");
    const results1 = mem.recall(vec1, 2);
    console.log("Recall results for query 1:", JSON.stringify(results1));

    // 4. KV Hints
    console.log("KV Hints for vector 1...");
    const hints = mem.kvHints(vec1, 2);
    console.log("KV Hints:", JSON.stringify(hints));

    // 5. Flush and Re-Open (Testing Serialization)
    console.log("Flushing to disk...");
    mem.flush();
    
    console.log("Re-opening memory from disk...");
    let mem2 = Sofuu.memory.open(mindPath, dim);
    
    console.log("Recalling vector closest to vector 2 after reload...");
    const results2 = mem2.recall(vec2, 2);
    console.log("Recall results from reloaded memory:", JSON.stringify(results2));

    console.log("== Test Complete ==");
} catch (e) {
    console.error("Test failed:", e);
}
