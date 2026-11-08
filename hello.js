// hello.js - memory · agents · AI · zero daemons
const brain = sofuu.memory.open("./brain.qtsq", 768);
brain.remember(sofuu.ai.embedLocal("sofuu"), "AI runtime. One binary.");

// Autonomous agent - define once, then run it
sofuu.agent.define({
  name: "researcher",
  system: "Cite sources.",
  tools: ["web", "memory", "system"],
});

const result = await sofuu.agent.run("researcher", "Summarise this project directory");
const [hit] = brain.recall(sofuu.ai.embedLocal("fast"), 1);
console.log(result.answer, "•", hit.text);
// -> AI runtime. 22.5 ms median cold start (wall clock incl. spawn) · 39.4 ms to first HTTP response · zero daemons
