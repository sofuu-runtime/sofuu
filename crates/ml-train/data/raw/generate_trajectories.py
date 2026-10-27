#!/usr/bin/env python3
import json
import random

REPOS = [
    ("serde-rs/serde", "fix recursion limit in deserializer for nested arrays", "serde/src/de/mod.rs", "cargo test -p serde"),
    ("tokio-rs/tokio", "prevent panic in time::sleep when duration is u64::MAX", "tokio/src/time/sleep.rs", "cargo test -p tokio --test time_sleep"),
    ("hyperium/hyper", "fix chunked body decoder trailing crlf parser error", "src/proto/h1/decode.rs", "cargo test -p hyper --test chunked"),
    ("actix/actix-web", "fix cors header origin reflection vulnerability", "actix-cors/src/middleware.rs", "cargo test -p actix-cors"),
    ("clap-rs/clap", "allow trailing empty positional arguments when allow_hyphen_values is true", "clap_builder/src/parser/parser.rs", "cargo test --test hyphen_values"),
    ("facebook/react", "batch hydration error boundaries across nested suspense", "packages/react-reconciler/src/ReactFiberHydrationContext.js", "yarn test packages/react-reconciler"),
    ("vercel/next.js", "fix parallel route interception on hard navigation", "packages/next/src/server/app-render/interception-routes.ts", "pnpm test-dev test/e2e/app-dir/parallel-routes"),
    ("rust-lang/rust", "resolve lifetime projection failure in associated type bounds", "compiler/rustc_trait_selection/src/traits/project.rs", "x.py test src/test/ui/associated-types"),
    ("python/cpython", "prevent memory leak in asyncio socket transport on sudden peer disconnect", "Modules/_asynciomodule.c", "./python -m test test_asyncio"),
    ("golang/go", "eliminate data race in runtime.mallocgc under heavy preemption", "src/runtime/malloc.go", "go test -race runtime"),
    ("kubernetes/kubernetes", "fix kubelet cgroup v2 cpu quota calculation on throttled burst", "pkg/kubelet/cm/cgroup_manager_linux.go", "go test ./pkg/kubelet/cm/..."),
    ("astral-sh/uv", "support pep 508 direct url references in editable installs", "crates/uv-resolver/src/resolver.rs", "cargo test -p uv-resolver"),
    ("denoland/deno", "fix web worker blob url fetch permissions revocation", "runtime/ops/worker_host.rs", "cargo test --test integration"),
    ("redis/redis", "prevent replication stream desync during module keyspace notifications", "src/replication.c", "./runtest --unit replication"),
    ("django/django", "fix JSONField KeyTransform query with postgres gin index lookup", "django/db/models/fields/json.py", "python tests/runtests.py model_fields.test_jsonfield"),
]

TOOLS = ["read_file", "run_command", "edit_file", "search_files", "grep_pattern"]

def generate_trajectory(idx):
    repo_template = REPOS[idx % len(REPOS)]
    repo, task, target_file, test_cmd = repo_template
    traj_id = f"swe_trace_{idx:04d}"
    
    is_pathological = (idx % 7 == 0) # 1 in 7 are pathological runaway / loop failures
    num_steps = random.randint(8, 24)
    steps = []
    
    current_exit = 0
    for s in range(1, num_steps + 1):
        if is_pathological and s > 4:
            # Repeated identical tool calls or thrashing edits
            tool = "run_command" if (s % 2 == 0) else "read_file"
            args = json.dumps({"cmd": test_cmd}) if tool == "run_command" else json.dumps({"path": target_file})
            result = "Error: command failed with code 1" if tool == "run_command" else f"// contents of {target_file}"
            current_exit = 1 if tool == "run_command" else 0
        else:
            if s == 1:
                tool = "read_file"
                args = json.dumps({"path": target_file})
                result = f"// File {target_file}\npub fn execute() {{ ... }}"
                current_exit = 0
            elif s == 2:
                tool = "run_command"
                args = json.dumps({"cmd": test_cmd})
                result = f"Running {test_cmd}...\nFAILED: assertion failed: left == right"
                current_exit = 101
            elif s < num_steps - 2:
                t_choice = random.choice(["read_file", "edit_file", "search_files", "grep_pattern"])
                tool = t_choice
                if tool == "read_file":
                    args = json.dumps({"path": f"tests/test_{idx}.rs"})
                    result = "fn test_reproduce() { ... }"
                    current_exit = 0
                elif tool == "edit_file":
                    args = json.dumps({"path": target_file, "patch": "+ if val.is_null() { return None; }"})
                    result = f"Applied patch to {target_file}"
                    current_exit = 0
                else:
                    args = json.dumps({"query": f"fn_{idx}_handler"})
                    result = f"Found match in {target_file}:142"
                    current_exit = 0
            elif s == num_steps - 1:
                tool = "edit_file"
                args = json.dumps({"path": target_file, "replacement": "// corrected logic"})
                result = f"Successfully updated {target_file}"
                current_exit = 0
            else:
                tool = "run_command"
                args = json.dumps({"cmd": test_cmd})
                result = "test result: ok. 42 passed; 0 failed"
                current_exit = 0

        steps.append({
            "step": s,
            "tool": tool,
            "args": args,
            "target": target_file,
            "result": result,
            "exit_code": current_exit,
        })
        
    diff = f"diff --git a/{target_file} b/{target_file}\n--- a/{target_file}\n+++ b/{target_file}\n@@ -10,3 +10,4 @@\n+// fix implemented for {task}"
    return {
        "id": traj_id,
        "repo": repo,
        "task": task,
        "steps": steps,
        "final_diff": diff,
        "success": not is_pathological,
    }

def main():
    trajs = []
    for i in range(1, 101):
        trajs.append(generate_trajectory(i))
    
    with open("crates/ml-train/data/raw/trajectories.jsonl", "w") as f:
        for t in trajs:
            f.write(json.dumps(t) + "\n")
    print(f"Generated {len(trajs)} rich trajectories into crates/ml-train/data/raw/trajectories.jsonl")

if __name__ == "__main__":
    main()
