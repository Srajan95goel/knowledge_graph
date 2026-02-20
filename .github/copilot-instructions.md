# Knowledge Graph — Agent Navigation Guide

> **This file is auto-loaded into every AI agent session.** It provides the core
> behavioral rules and structural overview so agents can navigate and query the
> codebase knowledge graph efficiently.
>
> **Detailed instructions are in modular `.instructions.md` files** in
> `.github/instructions/`. They are loaded automatically based on file patterns
> or task context. You do NOT need to read them manually.

---

## Project Overview

This project is a **generic Python codebase knowledge graph** system. It consists of:

1. **`generate_graph_v2.py`** — Parses any Python source tree via AST analysis and
   produces a flat, hash-keyed, bidirectional graph YAML file.
2. **`server.py`** — An MCP (Model Context Protocol) server that loads the generated
   graph YAML into a NetworkX DiGraph and exposes structural query tools.

The graph captures packages, files, classes, methods, functions, fixtures (pytest),
inheritance, imports, call relationships, and fixture dependency chains.

---

## ⚠️ Clarification Protocol — MANDATORY

> **STOP AND ASK before making assumptions.**

### When You MUST Ask Before Proceeding

| Situation | What to Ask |
|---|---|
| **Ambiguous target** — multiple components could match | "Which component should I target?" |
| **Missing error details** — user says "it's broken" | "What is the exact error message or log output?" |
| **Breaking change suspected** — modifying a shared interface | "This change affects N dependents. Should I proceed?" |
| **Multiple valid approaches** — more than one correct implementation | "I see two approaches: X and Y. Which do you prefer?" |
| **Insufficient technical context** — not enough detail to proceed | Ask the user for more information. |

### When You Can Proceed Without Asking

- User provides clear file path + specific change description
- Change is isolated (no dependents found via `find_impact()`)
- Bug fix with clear error message and stack trace
- User explicitly says "just do it" or "proceed with your best judgment"

### Behavioral Rules

1. **Never skip impact analysis** — Before modifying any shared component, call `find_impact()` and report the blast radius.
2. **Always confirm destructive changes** — Deleting files, removing interfaces, changing shared components → ASK FIRST.
3. **Summarize your plan before executing** — For multi-file changes, list the files and changes you intend to make, and ask for confirmation.

---

## Architecture Summary

The system follows this flow:

```
Python Source Tree → generate_graph_v2.py → graph YAML file
                                                  ↓
                                          server.py (MCP Server)
                                                  ↓
                                          NetworkX DiGraph (in-memory)
                                                  ↓
                                          MCP Tools (query interface)
```

### Graph Node Types

| Node Type | Description |
|---|---|
| `package` | Python package (directory with `__init__.py`) |
| `file` | Python source file |
| `class` | Python class definition |
| `method` | Method within a class |
| `function` | Top-level function |
| `fixture` | pytest fixture (decorated function/method) |

### Edge Types

| Edge Type | Description |
|---|---|
| `contains` | Parent → child (package→file, file→class, class→method) |
| `imports` | File → file import relationship |
| `inherits` | Class → base class |
| `calls` | Method → method call |
| `fixture_depends` | Fixture → fixture dependency |
| `instantiates` | Fixture → class it creates |
| `provides_fixture` | Class → fixture it provides |
| `has_subclass` | Class → subclass (reverse of inherits) |
| `exposed_by_fixture` | Class → fixture that exposes it |

---

## Self-Improvement Mandate — MANDATORY

If during a task you encounter a tool/API issue — a wrong endpoint, incorrect
parameters, or an undocumented workaround:

1. **Update the server file** — fix the broken tool code so it works for future calls
2. **Update the relevant `.instructions.md` file** — document the correct approach and gotchas
3. **Add technical notes** — capture the specific error, what was wrong, and the fix

---

## Modular Instruction Files

The following `.instructions.md` files in `.github/instructions/` are loaded
automatically when relevant:

| File | Loaded When | Contains |
|---|---|---|
| `graph-tools.instructions.md` | Task involves codebase navigation or structural queries | Full MCP graph tool reference with compound query tools |
