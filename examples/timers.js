// examples/timers.js — setTimeout, setInterval, sofuu.sleep test

console.log("1. Start");

// Basic setTimeout
setTimeout(() => {
  console.log("3. setTimeout fired after 50ms");
}, 50);

// sofuu.sleep - Promise-based delay
async function runAsync() {
  console.log("2. Before sleep");
  await sofuu.sleep(100);
  console.log("4. After 100ms sleep");

  await sofuu.sleep(50);
  console.log("5. After another 50ms sleep");
}

runAsync().then(() => {
  console.log("6. All done!");
});
