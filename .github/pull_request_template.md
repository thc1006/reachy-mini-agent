<!-- Thanks for contributing! Please fill this in so reviewers can move fast. -->

## What does this PR do?

<!-- One-paragraph summary. Link the issue it resolves with "Closes #123". -->

## Why?

<!-- User-facing or maintainer-facing motivation. What did it feel like before this PR and why is that bad? -->

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change (behavior / env-var / API)
- [ ] Documentation only
- [ ] Internal / refactor (no user-visible change)

## Testing

<!-- Describe exactly what you ran. "Tested against hardware" vs. "syntax + ruff only" are both valid, just be honest. -->

- [ ] `python -m py_compile src/*.py` passes
- [ ] `ruff check src/` is clean
- [ ] Tested end-to-end against a Reachy Mini
- [ ] Tested with `TTS_ENGINE=edge` / `TTS_ENGINE=kokoro`
- [ ] Docs (README / .env.example / docs/) updated if behavior changed

## Checklist

- [ ] Commit messages follow the project's style (imperative, descriptive, **no AI / bot signatures**).
- [ ] New env vars documented in `.env.example` and `README.md`.
- [ ] No secrets, hostnames, or personal paths committed.
- [ ] `CHANGELOG.md` updated under `## [Unreleased]` if user-visible.
