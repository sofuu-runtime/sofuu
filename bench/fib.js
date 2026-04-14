// bench/fib.js — CPU benchmark: recursive fibonacci
function fib(n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}
const result = fib(40);
console.log(result); // 102334155
