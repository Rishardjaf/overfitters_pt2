# Agent knowledge base

This folder is the shared, durable memory for humans and LLM agents working on the project. It contains only reusable conclusions that can save another contributor time or prevent a repeated mistake. It is not a chat log, scratchpad, task tracker, or substitute for source code and tests.

## Start here

1. Read [INDEX.md](INDEX.md).
2. Open only the files relevant to the current task.
3. Check [conflicts/OPEN.md](conflicts/OPEN.md) before relying on a disputed claim.
4. Verify important claims against their linked source, code, command, dataset slice, or experiment.

## Where information belongs

| Information | Location |
| --- | --- |
| Reusable result from an investigation | `findings/<topic>.md` |
| Candidate or validated model feature | `features/<feature_name>.md` |
| Accepted technical or analytical choice | `decisions/<decision_name>.md` |
| Two credible but incompatible findings | `conflicts/<topic>.md` and `conflicts/OPEN.md` |
| Temporary notes, guesses, or raw transcripts | Do not add them here |

Use lowercase `snake_case` filenames. Keep one topic per file so future agents can load narrowly scoped context.

## Required update workflow

1. Search this folder for the topic before adding a file.
2. Copy the matching file from `templates/`.
3. Record the conclusion, scope, evidence, date, contributor, and confidence. Link repository paths with relative Markdown links.
4. Separate observed facts from interpretations and recommendations. Never present an untested hypothesis as established fact.
5. Add or update one row in [INDEX.md](INDEX.md).
6. If new evidence disagrees with an existing claim, preserve both claims. Create a conflict record, add it to `conflicts/OPEN.md`, and define the test needed to resolve it.
7. Update affected entries when code, data, assumptions, or experiments make them stale. Prefer marking a conclusion superseded with a link to its replacement over silently deleting history.
8. Commit the knowledge update with the code or analysis that produced the evidence when practical.

## Quality bar

A useful entry lets another contributor answer: What was learned? Under what conditions is it true? Where is the evidence? How confident are we? What should change because of it? What could invalidate it?

Do not store secrets, credentials, personal data, large generated outputs, duplicated code documentation, or claims supported only by an agent's memory.


## Current dataset and migration

This workflow and its original templates were copied from [overfitters at 97525ff](https://github.com/Rishardjaf/overfitters/tree/97525ff208185a706aa69128536ac18b8432ea3c/agent_knowledge). All 11 source Markdown files were checked against the GitHub blob hashes before adapting them. That source knowledge index had no recorded findings or feature/decision entries.

Read [the current release](findings/dataset_release.md) and [the old-knowledge review](findings/previous_knowledge_review.md) before adopting historical assumptions. Every new entry must identify its dataset version or SHA-256 fingerprints. Feature benefit and model performance require fresh validation on the revised schema.
