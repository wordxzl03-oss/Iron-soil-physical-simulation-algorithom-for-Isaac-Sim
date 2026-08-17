# Contributing

Contributions are welcome through GitHub issues and pull requests.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Expectations

- Add or update tests for behavior changes.
- Preserve deterministic seeds in examples and experiments.
- Report units and distinguish direct-policy results from supervised rollouts.
- Do not commit virtual environments, caches, third-party papers, credentials, or
  large generated datasets.
- Put reusable final weights and large demonstrations in GitHub Release assets.
- Record new experiments with exact commands, configurations, seeds, and evidence paths.
- Preserve authoritative physics semantics and distinguish accepted validation
  from open or not-yet-evaluated production behavior.

By contributing, you agree that your contribution is licensed under the MIT License.
