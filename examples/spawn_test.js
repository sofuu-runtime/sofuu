const { spawn } = sofuu;

console.log("Starting spawn test...");

const p = spawn({
    command: "sh",
    args: ["-c", "echo 'Hello from child process!'; sleep 0.1; echo 'And goodbye!'; >&2 echo 'Error output!'; exit 42"],
    onStdout: (data) => {
        console.log(">> STDOUT:", data.trim());
    },
    onStderr: (data) => {
        console.log(">> STDERR:", data.trim());
    },
    onExit: (code) => {
        console.log(">> EXIT:", code);
    }
});

console.log("Child process spawned. Waiting for events...");
