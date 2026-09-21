# AGENTS.md

## Core Rules

- Communicate with the user in Chinese by default.
- Before taking action, clarify the expected outcome, authorized scope, completion criteria, and required validation.
- By default, access is authorized for this repository, not for the entire computer.
- Respect and preserve existing user changes, changes made by other agents, and content already staged in Git.
- Use the clearest and most direct language possible in the final response.
- Any command containing `rm -rf` is prohibited. This system runs macOS; use the `trash` command to move files to the Trash when deleting them.

---

## Python Environment Rules

- The Python environment is managed with `uv` at the repository root: the interpreter and tools are located in `.venv/`, and declared dependencies are listed in `pyproject.toml`.
- Only the Python environment at `.venv/` in the repository root may be used.
- Before using a dependency, you may inspect the direct dependencies in `pyproject.toml` and confirm that the required command exists in `.venv/`.
- Do not use `pip`, `python -m pip`, `uv pip`, or any other installation method.
- After validation, install only dependencies that have been confirmed to be safe and widely adopted by the community (not niche or unmaintained), and do so exclusively with `uv add` or by explicitly editing `pyproject.toml`. **Important: any dependency installation must be clearly reported to the user.**

---

## Other Code Environment Rules

Examples of other code environments that may be used include:

- Rust environments that use Cargo for package management
- Node.js and its package managers

Package management remains subject to the same rules as the Python environment above. If a particular environment is not present on the current **device**, the rules in this section do not apply to that environment, and installing it is **strictly prohibited**.

---

## Version Control Rules

Create an audit-style Git commit for each completed, self-contained task that modifies files. By default, one user request constitutes one task and one commit. Split a request into multiple task-level commits only when it contains multiple deliverables with clear boundaries that can be completed and validated independently.

### Safe Commit Procedure

1. Before committing, record the existing working-tree state and any content already staged, solely to protect task boundaries. If the task paths contain existing user changes or staged content, distinguish those changes from the changes made for the current task.
2. Review the current task's changes with path-scoped commands such as `git diff -- <task-owned-paths>` and `git diff --cached -- <task-owned-paths>`. Review the full contents of untracked files because ordinary `git diff` output does not include them.
3. When all changes within the task paths belong exclusively to the current task, stage them only with `git add -- <task-owned-path-1> <task-owned-path-2> ...`. If a file contains mixed changes, stage only the hunks verified as belonging to the current task. Do not use `git add -A`, `git add .`, `git commit -a`, or any other command that could include out-of-scope changes.
4. After staging, inspect `git diff --cached --name-status` and `git diff --cached -- <task-owned-paths>`. If the index contains any pre-existing staged content that does not belong to the current task, do not unstage, overwrite, reorder, or include it in the commit. Do not use a path-scoped commit to circumvent the shared index. Stop the commit process and report the situation so that the user can handle the existing staged content first. Proceed only after verifying that all content in the index belongs to the current task.
5. After committing, verify the paths included in the commit. If a boundary violation is found, do not rewrite history without authorization; report it immediately.

Do not create empty commits. Do not amend commits, rewrite history, force-push, or perform destructive Git operations without explicit authorization.

### Commit Message Format

```text
<type>(<scope>): <concise summary>

Background:
- task context and motivation

Changes:
- specific changes and affected components

Notes:
- implementation details, limitations, or trade-offs

Validation:
- checks and tests actually performed
- unverified items, if any
```

---

## Temporary File Rules

- Create a task-specific directory at `./tmp/<timestamp>-<short-id>/` under the project root's `./tmp/` directory.
- Immediately after creating it, add a `.codex-tmp` marker inside the directory and record that the directory was created by the current task.
- A temporary directory does not automatically become committable content. Do not stage it unless the user explicitly requests this.
- When the user explicitly declares the task finished, clean up only the dedicated directories that the current task created and recorded.

---

## Subagent Strategy

The main agent's context is a scarce resource. It should contain only conclusions and key evidence, not large volumes of raw command output. As the main agent, delegate tasks to subagent processes and focus on receiving, reviewing, and synthesizing their results so that the main agent's context remains high quality.

The user has granted standing authorization to use subagents, delegation, and parallel agent work within this repository.

### General Principles for Subagent Delegation

- The main agent determines which commands to use, how many subagents to create, and how to divide the work according to the task.
  - Delegate by required capability and natural boundaries, such as code exploration, test analysis, documentation review, or visual inspection. Do not divide work mechanically by file count alone.
  - Prefer delegation when the expected raw output is much larger than the final conclusions, the entry point is unclear, cross-module investigation is costly, or different capabilities can proceed independently.
  - Handle the task directly when the entry point is clear, the scope is small, or coordination would cost more than it saves.
  - Delegated agents inherit the same instruction priority, authorization boundaries, and prohibitions. The main agent remains responsible for scope, results, and validation.
- The main agent must protect its context while waiting for subagents:
  - Use a long timeout with `wait_agent` to wait for results. Waiting does not consume the main thread's context, and subagents continue working in parallel, so progress during the elapsed time is unaffected.
  - While waiting, the main thread must not search the repository or read files. Do not conduct additional exploration under the pretext of performing a "lightweight index," checking potential issues in advance, or avoiding idle time.
  - The main agent must not duplicate work similar to a subagent's assignment. Once work has been delegated, the main agent must wait for the result and must not perform overlapping or related work.
  - The only exception is when the user explicitly assigns a new task while the main agent is waiting.
- A delegation prompt must, at minimum, state the objective and motivation, in-scope and out-of-scope work, known clues, permitted read and write operations, completion criteria, and expected evidence. Agents should return only conclusions, key evidence, uncertainties, and recommendations, not large raw logs. Evidence should be concise. The main agent is responsible for synthesizing the results and should perform only limited spot checks of critical uncertainties rather than repeating the entire task.

---

## Avoid Excessive Defensive Programming

1. We are not writing a cybersecurity research paper. Validation is permitted, but excessive defensive programming is strictly, strictly, strictly prohibited.
2. Do not generate hashes or SHA-256 checksums.
3. Do not implement defensive logic for duplicate events that are virtually impossible.

---

## Project Status Snapshot (as of v0.11.0, 2026-09-21)

This repo is a dual-language data toolkit built from an empty scaffold:

- **Python package `kimi_space`** (src layout, installable via `uv sync`):
  `stats` / `io` / `plotting` / `outliers` / `transform` modules and a
  `kimi-space` CLI with subcommands `summary` / `hist` / `series` / `heatmap` /
  `box` / `corr` / `outliers` / `transform` / `autocorr` / `percentile` /
  `mad` / `report`.
- **Rust crate** (same repo): numeric kernel `src/lib.rs` mirroring the Python
  statistics semantics, plus the `kimi-space-rs` stdin binary (`src/bin/`)
  and criterion benchmarks (`benches/kernels.rs`).
- Quality gates that must stay green: `.venv/bin/pytest` (coverage 100%,
  CI floor 95%), `.venv/bin/ruff check .`, `.venv/bin/ruff format --check .`,
  `.venv/bin/mypy src/kimi_space` (strict), `cargo test`,
  `cargo clippy --all-targets -- -D warnings`, `cargo fmt --check`,
  `pre-commit run --all-files`.
- Cross-language numerical parity is enforced by `tests/test_parity.py`
  (skipped when cargo is unavailable); doc/CLI drift is guarded by
  `tests/test_meta.py`.
- Version numbers live in four places kept in lockstep by
  `tests/test_meta.py`: `pyproject.toml`, `Cargo.toml`,
  `src/kimi_space/__init__.py`, `CHANGELOG.md` (top heading).
- Releases: run `.venv/bin/python scripts/bump_version.py X.Y.Z`, fill the
  CHANGELOG TODO, run the full gate suite, commit, then `git tag -a vX.Y.Z`.
  Every feature lands with tests, docs (README / docs/api.md), and a
  CHANGELOG entry.
