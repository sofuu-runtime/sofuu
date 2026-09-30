// examples/greet.js — exported module
export function greet(name) {
  console.log(`Sofuu says hello to: ${name}!`);
}

export const VERSION = "0.2.0-beta"; /* matches the workspace version (Cargo.toml) — the old "0.1.0-alpha" was stale */
