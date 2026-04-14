try {
    console.log("== Cognitive SSD KV Cache Test ==");
    
    // Simulate layer config for Llama-3-8B like structure (minified)
    const nLayers = 2;
    const nHeads = 4;
    const headDim = 32;
    const nToks = 10;
    
    const count = nLayers * nHeads * nToks * headDim;
    
    console.log(`Generating dummy tensors: array of ${count} float32s`);
    const kData = new Float32Array(count);
    const vData = new Float32Array(count);
    
    // Fill with values (e.g., 0.123, 0.456, etc)
    for (let i = 0; i < count; i++) {
        kData[i] = (i % 100) / 100.0;
        vData[i] = (i % 50) / 50.0;
    }
    
    // 1. Open KV store
    const kvPath = "test_agent_kv";
    console.log(`Opening KV store at ${kvPath}...`);
    let kv = Sofuu.kv.open(kvPath, {
        modelId: "test-model-v1",
        nLayers: nLayers,
        nHeads: nHeads,
        headDim: headDim
    });
    
    // 2. Save KV Page
    console.log("Saving new KV Page (f8/f16 precision)...");
    const pageId = kv.save(kData, vData, nToks);
    console.log(`Saved successfully as Page ID: ${pageId}`);
    
    // 3. Search via Index
    const query = new Float32Array(64); // summary size
    for(let i=0; i<64; i++) query[i] = 0.5; // dummy summary
    
    console.log("Searching KV Index for page hints...");
    const hints = kv.search(query, 5);
    console.log(`Search Results (hints): ${JSON.stringify(hints)}`);
    
    // 4. Flush
    console.log("Flushing KV Store index to JSON...");
    kv.flush();
    
    console.log("== Test Complete == SSD directory size should now contain page files and index.json");
} catch (e) {
    console.error("Test failed:", e);
}
