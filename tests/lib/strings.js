// examples/lib/strings.js — tests transitive import (imports math.js itself)
import { PI } from './math.js';

export function greet(name) {
    return `Hello, ${name}! Welcome to Sofuu (π ≈ ${PI.toFixed(4)})`;
}
export function shout(s) { return s.toUpperCase() + '!'; }
