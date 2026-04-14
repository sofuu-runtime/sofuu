// examples/fs_test.js — Async file I/O test

async function main() {
  console.log("=== Sofuu Async File I/O Test ===\n");

  // Write a file
  console.log("Writing file...");
  await sofuu.fs.writeFile("/tmp/sofuu_test.txt", "Hello from Sofuu!\nLine 2\nLine 3");
  console.log("✓ writeFile done");

  // Read it back
  const content = await sofuu.fs.readFile("/tmp/sofuu_test.txt");
  console.log("✓ readFile:", content);

  // Append to it
  await sofuu.fs.appendFile("/tmp/sofuu_test.txt", "\nAppended line!");
  console.log("✓ appendFile done");

  // Read again
  const updated = await sofuu.fs.readFile("/tmp/sofuu_test.txt");
  console.log("✓ After append:\n" + updated);

  // Check exists
  const exists = await sofuu.fs.exists("/tmp/sofuu_test.txt");
  const noexist = await sofuu.fs.exists("/tmp/definitely_not_here_XYZ.txt");
  console.log("✓ exists('/tmp/sofuu_test.txt'):", exists);
  console.log("✓ exists('/tmp/not_here'):", noexist);

  // readdir
  const files = await sofuu.fs.readdir("/tmp");
  const hasSofuuFile = files.some(f => f === "sofuu_test.txt");
  console.log("✓ readdir /tmp contains sofuu_test.txt:", hasSofuuFile);

  console.log("\n=== All fs tests passed! ===");
}

main().catch(err => {
  console.error("Test failed:", err);
  process.exit(1);
});
