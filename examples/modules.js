// examples/modules.js — ESM import/export test
import { greet, VERSION } from "./greet.js";

greet("World");
greet("AI Era");
console.log("Imported version:", VERSION);

// Test object printing
const info = { runtime: "sofuu", version: VERSION, aiNative: true };
console.log("Runtime info:", info);

// Test array printing
const features = ["streaming", "mcp", "vectors", "agents"];
console.log("Features:", features);
