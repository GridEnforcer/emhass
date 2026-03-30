# CLAUDE.md

GridEnforcer fork of EMHASS (Energy Management for Home Assistant). Adds multi-battery CVXPY support.

**Branch**: `feature/multiple-batteries-v0.16.2` (based on upstream v0.16.2)

## Development Commands

```bash
# Install deps and run tests
uv sync --reinstall --extra test
pytest

# Lint
uvx ruff check .
uvx ruff format --check --diff
```

Python 3.10+, <3.13. Uses UV for dependency management. Build system: Hatchling.

## Multi-Battery Additions (this fork)

- Per-battery SOC init/final (`set_battery_soc_initial_list`, `set_battery_soc_final_list`)
- DC/AC bus routing (`battery_is_dc_coupled_list`)
- Battery availability windows (`batt_start_timestep`/`batt_end_timestep`)
- SOC-by-departure constraints
- Mixed-timezone fix in `publish_data` (`pd.to_datetime(utc=True)`)

## Key Files

| File | Purpose |
|------|---------|
| `src/emhass/optimization.py` | CVXPY optimization core (multi-battery logic here) |
| `src/emhass/command_line.py` | CLI orchestration and `publish_data` |
| `src/emhass/utils.py` | Validators, helpers |
| `src/emhass/forecast.py` | PV/load forecasting (pvlib) |
| `src/emhass/web_server.py` | REST API (Quart + Gunicorn/Uvicorn) |
| `tests/test_multi_battery.py` | Multi-battery feature tests |

## Conventions

- Solver: CVXPY with HiGHS backend (`highspy`)
- Web server: Gunicorn + Uvicorn workers on port 5000
- EMHASS crashes on `inf` values — always replace with finite bounds

## Definition of done

A feature is complete when:

1. The specified behavior works correctly across all described scenarios
2. Edge cases identified in the specification are handled
3. A corresponding unit test exists and passes
4. No linting errors are introduced
5. A oneline summary of the feature is added to CHANGELOG.md
6. README.md and PRD.md is updated to reflect the changes and user approved the updates
