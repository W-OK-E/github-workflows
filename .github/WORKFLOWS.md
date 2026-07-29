# GitHub Automation

This repo uses only free, built-in GitHub Actions/bots — no paid add-ons, no
external services. Everything below runs automatically once merged to `main`;
no setup required beyond what's noted per-workflow.

## What's included

| Workflow | File | Trigger | What it does |
|---|---|---|---|
| PR path labeler | `workflows/pr-labeler.yml` + `pr-labeler.yml` | PR opened/updated | Labels PRs (`algo`, `env/robot`, `rewards`, `configs`, `scripts`, `docs`, `dependencies`, `ci`, ...) based on which files changed. |
| Community text labeler | `workflows/community.yml` + `labeler.yml` | Issue/PR opened | Labels issues/PRs (`bug`, `help wanted`, `enhancement`) based on keywords in the title/description, and posts a welcome comment explaining the labels. |
| Dependabot | `dependabot.yml` | Weekly (Mon) | Opens PRs to bump outdated Python deps (`requirements.txt`) and GitHub Action versions. |
| Dependency review | `workflows/dependency-review.yml` | PR opened (touches deps) | Blocks/flags PRs that introduce dependencies with known vulnerabilities. |
| CodeQL | `workflows/codeql.yml` | Push/PR to `main`, weekly | Static security analysis of the Python code, results show up under the repo's **Security → Code scanning** tab. |
| Stale bot | `workflows/stale.yml` | Daily | Labels issues/PRs `stale` after 60 days of inactivity, closes them 14 days later unless labeled `pinned` or `security`. |
| Greeting bot | `workflows/greetings.yml` | First issue/PR from a user | Posts a one-time welcome comment on someone's first issue or PR. |

## One-time setup (do this once)

1. **Push to GitHub.** All workflows live in `.github/` and activate as soon as
   this is on the default branch (`main`) — nothing to install.
2. **Enable code scanning alerts** (for CodeQL): repo → *Settings → Code
   security* → make sure "Code scanning" is on. On a **public** repo this is
   free automatically. On a **private** repo, CodeQL and dependency review
   need GitHub Advanced Security enabled (also free for private repos owned
   by orgs/users, but must be turned on under *Settings → Code security*).
3. **Labels**: `actions/labeler` (path labeler) auto-creates any label it
   needs the first time it's used, so nothing to pre-create. Dependabot's
   `dependencies`/`ci` labels are the same — GitHub creates them on first use.
4. Nothing needs secrets — everything uses the automatically-provided
   `secrets.GITHUB_TOKEN`.

## Day-to-day usage

- **Opening a PR**: labels get applied automatically within ~30s based on
  changed files and title/description keywords. Just review them — remove or
  add labels manually if the automation guessed wrong.
- **Opening an issue**: same text-based labeling + a welcome comment if it's
  your first one.
- **Dependency bumps**: Dependabot will open its own PRs weekly (e.g. "Bump
  torch from 2.1.0 to 2.2.0"). Review the changelog, let CI run, merge like
  any other PR. To change frequency (`daily`/`weekly`/`monthly`) edit
  `.github/dependabot.yml`.
- **Security alerts**: check the **Security** tab on GitHub for CodeQL
  findings and Dependabot vulnerability alerts. Dependency-review will also
  comment directly on a PR if it adds a vulnerable package.
- **Stale issues/PRs**: if something goes quiet for 60 days it gets an
  automatic `stale` label + comment; another 14 days of silence and it's
  closed. Comment on it (or add a `pinned`/`security` label) to keep it open.
- **Manually trigger stale sweep**: *Actions* tab → "Stale" workflow → *Run
  workflow* (it also has `workflow_dispatch` enabled for on-demand runs).

## Customizing

- **Path→label mapping**: edit `.github/pr-labeler.yml` (glob patterns per
  label).
- **Keyword→label mapping**: edit `.github/labeler.yml` (regex on title +
  description).
- **Stale timing/messages**: edit `.github/workflows/stale.yml`
  (`days-before-stale`, `days-before-close`, messages, exempt labels).
- **Welcome messages**: edit `.github/workflows/greetings.yml`.
- **Dependency update cadence/ecosystems**: edit `.github/dependabot.yml`.
