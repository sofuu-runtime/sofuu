// examples/ai_stream_test.js
// Test ai.stream() with Ollama streaming
// Start Ollama first: ollama serve

const { ai } = sofuu;

console.log("=== ai.stream() test (Ollama / llama3) ===\n");

async function run() {
    const stream = ai.stream("Count from 1 to 5, one number per line.", {
        provider: "ollama",
        model: "llama3"
    });

    let full = "";
    for await (const chunk of stream) {
        process.stdout.write(chunk.text);
        full += chunk.text;
    }

    console.log("\n\n=== Stream complete — " + full.length + " chars received ===");
}

run().catch(err => console.error("Stream error:", err));
