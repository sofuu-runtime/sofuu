// examples/process_test.js — process bindings test
console.log("argv:", process.argv);
console.log("version:", process.version);
console.log("platform:", process.platform);
console.log("HOME:", process.env.HOME);
console.log("PATH exists:", typeof process.env.PATH === "string");

// Test console methods
console.warn("This is a warning");
console.error("This is an error");
console.info("This is info");
