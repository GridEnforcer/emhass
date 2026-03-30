# EMHASS — GridEnforcer Fork

GridEnforcer fork of [EMHASS](https://github.com/davidusb-geek/emhass) (Energy Management for Home Assistant) with multi-battery CVXPY support.

**Branch**: `feature/multiple-batteries-v0.16.2` (based on upstream v0.16.2)

## Fork Additions

- **Per-battery SOC**: Individual initial/final SOC for each battery (`set_battery_soc_initial_list`, `set_battery_soc_final_list`)
- **DC/AC coupling**: Per-battery bus routing (`battery_is_dc_coupled_list`) — DC-coupled batteries bypass the inverter
- **Battery availability windows**: `batt_start_timestep` / `batt_end_timestep` for time-limited batteries (e.g., EV connected 18:00-07:00)
- **SOC-by-departure constraints**: Ensure target SOC is reached before battery disconnects
- **Battery stress cost**: Per-battery cycling penalty in the objective function
- **Mixed-timezone fix**: `publish_data` uses `pd.to_datetime(utc=True)` to prevent timezone errors

## Development

```bash
# Install with UV
uv sync --reinstall --extra test

# Run tests
pytest

# Lint
uvx ruff check .
uvx ruff format --check --diff
```

Python 3.10+, <3.13. Uses UV for dependency management.

## Key Files

| File | Purpose |
|------|---------|
| `src/emhass/optimization.py` | CVXPY optimization core (multi-battery logic) |
| `src/emhass/command_line.py` | CLI orchestration and `publish_data` |
| `src/emhass/utils.py` | Validators, helpers |
| `src/emhass/forecast.py` | PV/load forecasting (pvlib) |
| `src/emhass/web_server.py` | REST API (Gunicorn + Uvicorn on port 5000) |
| `tests/test_multi_battery.py` | Multi-battery feature tests |

---

*Below is the upstream EMHASS README.*

---

<div align="center">
  <br>
  <img alt="EMHASS" src="https://raw.githubusercontent.com/davidusb-geek/emhass/master/docs/images/logo_docs.png" width="700px">
  <h1>Energy Management for Home Assistant</h1>
</div>

## Introduction

EMHASS (Energy Management for Home Assistant) is an optimization tool designed for residential households. The package uses a Linear Programming approach to optimize energy usage while considering factors such as electricity prices, power generation from solar panels, and energy storage from batteries.

The complete documentation is [available here](https://emhass.readthedocs.io/en/latest/).

## License

MIT License — Copyright (c) 2021-2026 David HERNANDEZ TORRES
