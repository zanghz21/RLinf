---
name: review-pr
description: Review or explain an RLinf pull request, trace behavior changes against its baseline, assess algorithm and architecture choices, and draft evidence-backed review comments. Use for a PR URL or a PR number with repository context.
---

# Review PR

Help the reviewer understand what changes, verify the behavior, and identify actionable issues. Prioritize correctness and responsibility boundaries; check contribution requirements without letting metadata dominate. Apply [AGENTS.md](../../../AGENTS.md), [CONTRIBUTING.md](../../../CONTRIBUTING.md), and the voice rules in [docs/STYLE_GUIDE.md](../../../docs/STYLE_GUIDE.md).

## Match the request

- For a full review, explain the change briefly, then report findings by severity.
- For a question about one component, trace that component and its relevant callers. Keep the answer within the question's scope.
- For a before/after comparison, distinguish behavior changes from moves, renames, and extracted abstractions.
- For comment drafting, preserve the user's concern and requested tone, verify factual claims, and label suggestions that can be deferred. Drafting does not authorize posting.

Infer the repository and PR number from the conversation when unambiguous. Ask only for missing information that prevents the requested work.

## Establish the versions

Use supplied, complete evidence when it identifies the reviewed revisions. Otherwise fetch PR metadata, the current diff, and relevant review discussions through the available GitHub API, browser, or CLI; no `gh` dependency is required. A PR description may describe an earlier revision. Record the head SHA and base repository/branch, and check that downloaded files and diffs correspond to that head. Refresh this information when reviewing subsequent updates or when the user requests the latest state.

Keep these three views separate:

| View | Purpose |
|---|---|
| PR merge-base | The before state for attributing changes to this PR |
| Pinned PR head | The after state, including unchanged callers and dependencies |
| Current upstream target branch | Current conventions, reusable helpers, and integration conflicts |

Resolve the base remote from PR metadata; `origin` may be a personal fork, and the target branch may not be `main`. When working from Git, fetch missing refs and compute `git merge-base <base-ref> <head-sha>`. Use `git diff <merge-base> <head-sha>` for the changes and `git show <sha>:<path>` for complete files. If the target branch has advanced, label its differences separately rather than attributing them to the PR.

Use an isolated checkout or revision-qualified reads so the user's working branch stays intact. Search the pinned head for consumers of new code and the baseline for old behavior. Do not treat an unrelated local checkout or a concatenation of main and the diff as the PR's runtime tree.

If access is incomplete, identify which evidence is unavailable and which conclusions remain possible. Continue with available evidence; request an exported diff or missing source only when needed. Do not claim a complete review from a truncated diff.

## Reconstruct behavior before judging it

Map the affected path from configuration and construction through the caller, implementation, and downstream consumer. For a large PR, organize by behavior or responsibility rather than file order.

- Read the old and new implementations together with their call sites. Resolve inherited methods, overrides, omitted arguments, default values, configuration overrides, and train/eval branches. A method name, `.eval()`, or `no_grad()` does not establish whether custom sampling, mutation, or updates occur.
- Follow values to their effect: tensor shapes and masks, action selection and execution, worker messages, replay storage, loss inputs, checkpoint state, or cleanup. An interface existing in one file does not prove it is reached in a supported run.
- Check relevant modes and alternate implementations: training/evaluation, sync/async, supported environments/backends, warmup, resets, termination, and resume. Focus on paths affected by the change, with concrete inputs that expose a difference.
- Distinguish a default preset from an optional capability. For routing or takeover logic, establish precedence, fallbacks, disabled/shadow modes, and which action is executed versus recorded for learning.
- Before calling code unused, inspect factories, registries, inherited hooks, serialization, configuration, and external extension contracts. Zero direct text matches is a lead, not proof of dead code.

When the PR changes a learning objective, sampling rule, or update schedule, or the user asks whether an algorithm is standard, read [Algorithm semantics](reference.md#algorithm-semantics). For conditional docs, installation, and CI checks, use the relevant sections of [reference.md](reference.md); do not expand a focused question into an unrelated audit.

## Evaluate correctness and architecture

Review correctness through the affected contract: masks and numerical behavior, rank participation and synchronization, message ordering, state ownership, resource lifecycle, and error handling. Describe a reachable trigger and its consequence. A minimal reproducer or calculation is useful when the source trace alone leaves uncertainty.

For design and reuse, compare with a relevant current upstream sibling and verify compatibility at the PR head:

- Identify the responsibility being added: network structure, policy composition, learning objective, rollout coordination, replay management, or environment adaptation. Judge placement and dependencies by that responsibility, not the class name or directory alone.
- Before proposing reuse, compare input/output contracts, shapes, action distributions, parameter ownership, and lifecycle. Check which helpers are already reused. Prefer composition or a shared smaller helper when inheriting a whole implementation would import incompatible behavior.
- Test a claim of generality against an actual alternative component. Trace factory selection, type checks, callbacks, and required observation/action fields. Separate task-specific geometry or hardware handling from generic routing or learning decisions; explain the concrete coupling that prevents reuse.
- Trace inherited behavior before criticizing a worker's parent class. Identify algorithm-specific assumptions that leak through it; distinguish a runtime problem from an architectural cleanup that can wait.

Classify each concern before reporting it:

| Classification | Evidence needed |
|---|---|
| PR-introduced bug/regression | Changed behavior, reachable trigger, violated contract, and impact |
| Pre-existing issue | Evidence at the merge-base; explain if this PR broadens its impact |
| Deliberate behavior change | Source/history or author statement establishing intent; evaluate correctness separately |
| Design suggestion | Concrete ownership, reuse, or extension cost; state whether it blocks the PR |
| Open question | Specific missing evidence and what would resolve it |

Code establishes implemented behavior. Comments, docs, commits, and author replies provide evidence of intent. Tests establish only the properties they exercise. A test written to assert a new behavior, or a successful training smoke test, does not by itself establish algorithmic correctness. Do not invent rationale from names such as "ablation" or from the fact that code already exists. Correct earlier conclusions explicitly if a deeper trace changes them.

## Validate and present the result

Run checks proportionate to the findings and available environment. Prefer focused existing tests or a small reproducer; do not install a full GPU/robot stack just to answer a source question. Report what ran, what it established, and any material untested path. Missing hardware is a validation limit, not evidence of a bug.

For a full review, identify the PR and reviewed revision, explain its purpose and main changes, then list actionable findings in severity order. Keep pre-existing debt, non-blocking suggestions, and unresolved questions distinguishable from PR-introduced bugs. Do not manufacture findings to fill categories.

Each finding should explain the expected and actual behavior in plain language, the triggering conditions, and the practical impact. Include a verified file/line link and enough of the code trace to make the conclusion checkable. Show short inline snippets when defaults or multiple calls are central to the issue; scale the walkthrough to its complexity. Suggest a concrete fix direction without prescribing an unnecessary redesign.

For an unfamiliar subsystem or a requested comparison, use a table with **area, before, after, and behavioral impact**, followed by the relevant call flow. Explain the role before introducing a class name. Mark moves and extractions as such instead of presenting old capabilities as new features.

For algorithm comparisons, keep **reference algorithm, pre-PR implementation, and PR implementation** separate. For an English review comment, put the concrete concern first, provide the evidence, and ask for a specific change or explanation. Preserve uncertainty and non-blocking status; do not turn a question into an asserted bug.

Use links to the pinned revision for source evidence or verified lines in the current PR diff. Cite the merge-base for historical claims, current upstream for convention checks, and primary sources for algorithm claims. End with relevant validation limits rather than an exhaustive checklist.
