# GitHub Publication Guide

## Before publishing

1. Review `LICENSE`; MIT is the selected default.
2. Replace `REPLACE_WITH_GITHUB_REPOSITORY_URL` in `CITATION.cff`.
3. Add real authors/maintainers if desired.
4. Review both manifests and the development report.
5. Confirm the third-party paper PDF is not staged.
6. Confirm no credentials or private datasets are present.

## Create the repository

Install Git if it is not already available, then extract the source archive:

```powershell
git init
git add .
git status
git commit -m "Initial open-source release"
git branch -M main
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

Review `git status` before every commit. Do not use `git add -f` to bypass the exclusions
without understanding why the file is ignored.

## Publish the weights and complete example

Create tag `v0.1.0` and a GitHub Release. Upload:

- `minslope-v0.1.0-release-assets.zip`
- the adjacent `.sha256` checksum file

The asset contains final weights and the full continuous demonstration. It is separate from
the source repository to keep clones lightweight and avoid permanent binary history.

## Suggested release notes

```markdown
## Minslope v0.1.0

Initial research release of the height-field soil, wheel-loader dynamics, SAC,
large-pile validation, and animation pipeline.

Highlights:
- 40 passing unit tests
- four-model data-scale study
- nine load-verified final policy weights
- 100/100 supervised large-pile validation successes at the 20% target
- 309-scoop demonstration reaching 9.9327% remaining volume
- continuous 5:09 MP4 replay

See `docs/DEVELOPMENT_REPORT.md` for limitations and reproducibility details.
```

## GitHub settings

- Enable Actions for the test workflow.
- Enable Issues and Discussions as appropriate.
- Add repository topics such as `reinforcement-learning`, `robotics`,
  `wheel-loader`, `soil-simulation`, `gymnasium`, and `stable-baselines3`.
- Protect `main` after the initial push if multiple contributors will work on the project.
