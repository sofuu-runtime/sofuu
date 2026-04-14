#pragma once

/*
 * repl.h — Sofuu interactive REPL
 *
 * Launches a readline-backed REPL loop that shares one persistent
 * QuickJS runtime across all entered lines (variables survive).
 *
 * Returns when the user types .exit or sends Ctrl-D.
 */
int sofuu_repl(void);
