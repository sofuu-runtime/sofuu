// examples/ai_complete_test.js
// Test ai.complete() with Ollama (no API key needed — runs locally)
// Start Ollama first: ollama serve

const { ai } = sofuu;

console.log("=== ai.complete() test (Ollama / llama3) ===\n");

ai.complete("What is 2 + 2? Answer in one word.", {
    provider: "ollama",
    model: "llama3"
}).then(result => {
    console.log("Response:", result.text);
    console.log("\n=== PASSED ===");
}).catch(err => {
    console.error("Error:", err);
});
