// quickstart.js — memory · native tools · agents · zero daemons

// 1. On-device vector memory (encrypted, SIMD search, zero database servers)
const brain = sofuu.memory.open("./brain.qtsq", 768);
brain.remember(
  sofuu.ai.embedLocal("runtime"),
  "Sofuu executes AI apps, tools, and agents natively under 10 MB."
);

// 2. Offline semantic recall in microseconds (< 2ms)
const [match] = brain.recall(sofuu.ai.embedLocal("memory"), 1);
console.log(`[Memory Recalled] ${match.text}`);

// 3. Native sandboxed tools executed in-process with policy gates
const sysInfo = sofuu.tools.execute("system_info", {});
console.log(`[Runtime] Platform: ${sysInfo.platform || process.platform} · Architecture: ${sysInfo.arch || process.arch} · Status: ready`);

// 4. Define an autonomous agent (ready for deployment with memory & tools)
const agent = sofuu.agent.define({
  name: "assistant",
  system: "Answer user queries using vector memory and native tools.",
  tools: ["coding", "memory", "web"],
  memory: "agent",
});
console.log(`[Agent Ready] "${agent.name}" registered with tools: [coding, memory, web]`);
