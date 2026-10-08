# Publishing `verdict` to GitHub

Eve built the repo and committed it locally, but she can't touch your GitHub —
so here's the handoff. Three steps, about two minutes, on your work laptop.

## What you're getting

A file called `verdict.bundle` — a full git repo in a single file (all history,
all files). You'll clone it locally, create the GitHub repo, and push.

## Steps

**1. Download `verdict.bundle`** to your work laptop (it's attached / linked
in chat).

**2. Clone it and create the GitHub repo:**

```bash
git clone verdict.bundle verdict
cd verdict
```

Then on github.com: **New repository** → name it `verdict` → **do not**
initialize with a README (the repo already has one) → Create.

**3. Push:**

```bash
git remote set-url origin git@github.com:<your-username>/verdict.git
git push -u origin main
```

Done. Share the repo URL with whoever needs it.

## After it's up

- Teammates: `git clone git@github.com:<your-username>/verdict.git`,
  then `pip install -e ".[test]"` and `verdict backends`.
- The `pyproject.toml` still says `https://github.com/REPLACE_ME/verdict`
  under `[project.urls]` — update it to your real URL whenever (one line,
  no urgency).
- The `LICENSE` is Apache 2.0 with a bare copyright line — add the
  copyright holder if your team wants it named.
- The seed commit is authored as you (`rcamejo04@gmail.com`). If that's not
  the email on your GitHub account, run this once after cloning and the
  commits will link to you properly:
  `git commit --amend --author="Rebecca Camejo <your-github-email>"`
