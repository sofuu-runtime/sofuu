try {
    console.log("== Agent Orchestration Test ==");
    
    const dim = 16;
    let mem = Sofuu.memory.open("agent_mind.qtsq", dim);
    
    let kv = Sofuu.kv.open("agent_kv", {
        modelId: "test", nLayers: 2, nHeads: 4, headDim: 32
    });
    
    // Create an agent tying CMA and KV together
    let agent = Sofuu.agent.create("Alpha", mem, kv);
    console.log("Agent 'Alpha' successfully created uniting CMA and KV Store.");
    
    // Populate memories and KV cache
    const kData = new Float32Array(2*4*10*32);
    const vData = new Float32Array(2*4*10*32);
    const pageId = kv.save(kData, vData, 10);
    
    const contextVec = new Float32Array(dim);
    contextVec[1] = 1.0;
    
    // Remember a fact and link it to the exact KV page
    mem.remember(contextVec, "User prefers dark mode.", "system", pageId);
    console.log(`Saved fact into Cognitive Memory linking to KV Page ID ${pageId}`);
    
    console.log("Agent orchestrating prefetch routine based on a similar inference query...");
    const query = new Float32Array(dim);
    query[1] = 0.99; // very similar
    
    // Instruct agent to prefetch
    const numPrefetched = agent.prefetch(query, 5);
    console.log("Agent Prefetch successful.");
    console.log(`Agent loaded ${numPrefetched} specific KV sequence blocks into ultra-low latency RAM from SSD based on Cognitive Similarity!`);
    
    console.log("== Test Complete ==");
} catch(e) { console.error(e); }
