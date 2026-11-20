# Conditional PR review checks

Use the sections relevant to the requested review. The version and evidence rules in [SKILL.md](SKILL.md) apply throughout.

## Algorithm semantics

Use this section when the PR changes a learning objective, action distribution, target, or update schedule, or when the user asks how an implementation compares with a named algorithm.

1. When a named algorithm or claimed standard is relevant, identify the reference variant from the paper, authors' implementation, or official documentation. Verify disputed details against primary sources; do not reconstruct a standard from an algorithm's name or a neighboring implementation. A question about behavior preservation can be answered from before/after code without inventing a reference algorithm.
2. Translate the effective code into the relevant equations or operations, accounting for callers, defaults, inherited methods, and resolved configuration. Inspect both the merge-base and pinned head.
3. Compare the parts that matter to the question:

   | Part | Questions to resolve |
   |---|---|
   | Policy and exploration | Deterministic or stochastic output? Which sampling/noise occurs during rollout, actor training, evaluation, and target construction? Where are clipping or squashing applied? |
   | Critic target | Which target actor/critics are used? How are Q values reduced? Are entropy terms, smoothing, discounts, terminal masks, and action chunks handled as intended? |
   | Actor objective | Which critic or reduction supplies gradients? Are entropy, behavior cloning, demonstrations, or auxiliary losses included? Which parameters receive gradients? |
   | Update schedule | What advances the counter? Which condition gates critic, actor, and target updates? Are updates nested under the same condition? How does accumulation or distributed execution affect the cadence? |
   | Training data | Which action was proposed, executed, and stored? How are intervention flags, next observations, resets, and replay sampling used? |

4. Separate three conclusions: what differs from the reference, which differences this PR introduces, and what justifies or validates them. An intentional variant can be reasonable, but naming it does not establish its behavior or effectiveness.
5. If intent is disputed, inspect relevant commit history and review discussion. Attribute an explanation to its author; keep an inferred rationale explicitly tentative. A new comment and a test asserting the same behavior demonstrate an implementation choice, not independent validation of that choice.

A useful comparison has columns for **reference algorithm / pre-PR / PR / consequence**. Report only relevant rows. For a suspected bug, identify a violated contract or an observable consequence; for an undocumented variant, ask for the intended objective and supporting explanation or experiment.

## Documentation and examples

For changed docs or public behavior, check the affected examples and paired EN/ZH pages against the pinned head. Use [docs-check](../docs-check/SKILL.md) for a documentation review; keep a code review focused on the affected contract.

- Verify commands, config keys, paths, public symbols, supported combinations, and capability claims. For additions, removals, or renames, identify the exact corresponding EN/ZH pages that need updating. Follow [CONTRIBUTING.md](../../../CONTRIBUTING.md) for user-facing test and documentation requirements.
- Check EN/ZH agreement on behavior, setup, keys, claims, numbers, and relevant structure, including when only one language changed. Give a concrete correction for a reported mismatch.
- For changed model-weight paths in example configs, match the environment, task variant, model family, and full-model versus adapter role to the recipe. Check the documented local-directory convention and any linked model repository. Use current recipes rather than copying a fixed list of model names into this skill.
- Verify external artifacts through the provider when needed. A canonical-name redirect can identify a stale link; an authorization error, gated repository, network failure, or unavailable API does not by itself prove the artifact is missing. Report the observation and its effect on reproducibility accurately.

## Configuration, installation, and CI

- Follow configuration requirements in [CONTRIBUTING.md](../../../CONTRIBUTING.md): static YAML values, current sibling templates, and user-settable fields treated as read-only. Resolve Hydra defaults before alleging a missing value or unnecessary override; an explicit override may select a different variant.
- Check registrations, lazy imports, optional dependencies, and process-local initialization through the actual builder/worker path. Do not require every subclass to redefine a valid inherited hook.
- For install or Docker changes, use [install-check](../install-check/SKILL.md). For a new model or environment, use [add-install-docker-ci-e2e](../add-install-docker-ci-e2e/SKILL.md) for the integration points, checking applicability against current upstream siblings. Some hardware environments do not ship Docker images; do not demand a stage solely because an environment was added.
- Check that moved/new code reaches the intended CI jobs and that e2e configurations are actually referenced. Flag duplicated path filters or jobs only after checking their coverage and execution conditions.
- Assess tests through caller-visible behavior and repository component ownership. Follow the test layout rules in [AGENTS.md](../../../AGENTS.md#tests). A new component can justify a new test file; a regression-specific file or mock-only assertion does not establish a new component.
- Identify hardware, model assets, or dependency requirements that limit validation. Note when maintainer assistance is needed, without sending messages on the user's behalf.

## Style and contribution metadata

Apply current repository rules to relevant files. Keep these findings secondary to correctness and design:

- Public API documentation and types, meaningful errors, project logging, and accelerator portability.
- Applicable license headers on newly added Python/source files, preserving third-party attribution. Check the repository's header/year rule and the file's provenance before alleging a mismatch; the review date alone does not prove an older contribution has an invalid year.
- Conventional Commit and sign-off requirements, PR title constraints, the template, and reported testing. Verify the title with the current rules in [CONTRIBUTING.md](../../../CONTRIBUTING.md), which are stricter than commit subjects.
- For changes that can affect training performance or stability, check whether the supplied evidence addresses that claim. Distinguish a successful launch or finite-loss smoke test from comparative learning results.

Do not include positive checklist filler, label unavailable checks as passed, or raise a preference as a blocking defect without a concrete cost or repository requirement.
