// examples/ts_test.ts — TypeScript stripper integration test

// Interfaces (stripped)
interface Point {
    x: number;
    y: number;
}

// Type aliases (stripped)
type ID = string | number;
type Callback<T> = (value: T) => void;

// Generic function with typed params
function identity<T>(value: T): T {
    return value;
}

// Class with access modifiers
class Vector {
    private x: number;
    private y: number;

    constructor(x: number, y: number) {
        this.x = x;
        this.y = y;
    }

    public magnitude(): number {
        return Math.sqrt(this.x * this.x + this.y * this.y);
    }

    public add(other: Vector): Vector {
        return new Vector(this.x + other.x, this.y + other.y);
    }
}

// as-type assertions
const raw: unknown = "hello sofuu";
const msg = raw as string;

// Optional params
function greet(name: string, prefix?: string): string {
    return `${prefix ?? "Hello"}, ${name}!`;
}

// satisfies
const config = { host: "localhost", port: 8080 } satisfies object;

// import type (this line would be stripped — tested via no-error execution)

// ── Assertions ────────────────────────────────────────────────
const results = { passed: 0, failed: 0 };
function assert(label: string, cond: boolean): void {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else       { console.error("  ❌ FAIL: " + label); results.failed++; }
}

console.log("\n=== TypeScript Stripper Test ===\n");

// identity works with any type
assert("identity<number>(42) === 42",    identity(42) === 42);
assert("identity<string>('hi') === 'hi'", identity("hi") === "hi");

// class methods work
const v1 = new Vector(3, 4);
assert("Vector.magnitude() === 5", Math.abs(v1.magnitude() - 5) < 0.001);
const v2 = new Vector(1, 2);
const v3 = v1.add(v2);
assert("Vector.add() x===4", (v3 as any).x === 4);

// type assertion stripped cleanly
assert("as-cast string works", msg.toUpperCase() === "HELLO SOFUU");

// optional param
assert("greet with prefix", greet("World", "Hey") === "Hey, World!");
assert("greet without prefix", greet("World") === "Hello, World!");

// satisfies doesn't change runtime value
assert("satisfies keeps value", (config as any).port === 8080);

// Generics with union types
function first<T>(arr: T[]): T | undefined {
    return arr[0];
}
assert("generic first()", first([10, 20, 30]) === 10);

const length: ID = 42;
assert("union type alias works", typeof length === "number");

console.log(`\n=== RESULTS ===`);
console.log(`Passed: ${results.passed} | Failed: ${results.failed}`);
if (results.failed === 0) {
    console.log("\n✅ TypeScript stripper — all tests PASSED\n");
} else {
    console.error(`\n❌ ${results.failed} test(s) failed`);
    process.exit(1);
}
