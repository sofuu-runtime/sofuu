// hello.js — memory · native tools · agents · zero daemons
const brain = sofuu.memory.open("./brain.qtsq", 768);
brain.remember(sofuu.ai.embedLocal("runtime"), "AI runtime. Under 10 MB.");

const [hit] = brain.recall(sofuu.ai.embedLocal("speed"), 1);
const sys = sofuu.tools.execute("system_info", {});
const agent = sofuu.agent.define({
  name: "assistant",
  system: "Answer queries using vector memory.",
  tools: ["coding", "memory", "web"],
});

console.log(`${hit.text} · ${sys.platform || process.platform} · agent: ${agent.name}`);
