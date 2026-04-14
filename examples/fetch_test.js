const { fetch } = sofuu;

console.log("Starting fetch test...");
fetch("https://httpbin.org/get")
  .then(res => {
    console.log("Fetch succeeded! Length: " + res.length);
    console.log(res);
  })
  .catch(err => {
    console.error("Fetch failed:", err);
  });
