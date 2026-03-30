# PRD: EMHASS (GridEnforcer Fork)

Energy Management for Home Assistant — CVXPY-based LP/MPC optimization for residential energy systems. This fork adds multi-battery support on top of upstream v0.16.2.

## Behavioral Specification

### Multi-Battery Optimization (Fork Additions)

- **When `number_of_batteries > 1`**, the solver creates per-battery CVXPY variables: `Pbatt_chg[b]`, `Pbatt_dischg[b]`, `Ebatt[b]` for each battery b.
- **When `set_battery_soc_initial_list` is provided**, each battery starts at its own SOC (0-1 scale).
- **When `set_battery_soc_final_list` is provided**, each battery has its own target SOC at the end of the horizon.
- **When `battery_is_dc_coupled_list` is provided**, DC-coupled batteries use inverter input power constraints; AC-coupled batteries use inverter output.
- **When `batt_start_timestep` / `batt_end_timestep` are provided**, each battery's charge/discharge variables are forced to 0 outside its availability window.

### Energy Balance (Per Battery, Per Timestep)

- `E[t+1] = E[t] + P_chg[t] × eff_chg - P_dischg[t] / eff_dischg`
- SOC bounds: `E_min ≤ E[t] ≤ E_max` (per battery)
- Power limits: `0 ≤ P_chg[t] ≤ P_chg_max`, `0 ≤ P_dischg[t] ≤ P_dischg_max` (per battery)
- Initial/final SOC: `E[0] = E_init`, `E[T] = E_final` (per battery)

### Result Format

- **Single battery**: `P_batt` (alias for `P_batt0`), `SOC_opt` (alias for `SOC_opt0`).
- **Multiple batteries**: `P_batt0`, `P_batt1`, ..., `SOC_opt0`, `SOC_opt1`, ... (no scalar aliases).
- **Power sign convention**: Negative = charging, positive = discharging (EMHASS convention).

### MPC Optimization Endpoint

- **When `/action/naive-mpc-optim` is called**, the solver runs CVXPY with HiGHS backend.
- **When the problem is infeasible**, a text response indicates infeasibility (not JSON).
- **When the problem is unbounded**, a text response indicates unboundedness.
- **When successful**, results are returned as JSON (or HTML, requiring publish-data followup).

### Publish Data

- **When `/action/publish-data` is called**, optimization results are pushed to HA sensor entities.
- **When timestamps have mixed timezone offsets** (DST transitions), `pd.to_datetime(utc=True)` normalizes them.
- **Per-battery sensors created**: `p_batt0_forecast`, `soc_batt0_forecast`, `p_batt1_forecast`, etc.

### Web Server

- **Gunicorn + Uvicorn workers** on port 5000.
- **REST API endpoints**: `/action/naive-mpc-optim`, `/action/publish-data`, `/action/forecast-model-fit`, `/action/forecast-model-predict`, `/action/forecast-model-tune`, `/action/get-table-data`.

## Acceptance Criteria

1. Single-battery optimization produces `P_batt` and `P_batt0` aliases that are equal.
2. Two-battery optimization produces independent SOC trajectories per battery.
3. DC/AC coupling flag correctly routes battery power through appropriate bus constraints.
4. Availability windows force charge/discharge to 0 outside the specified timesteps.
5. Energy balance equation holds for all batteries at all timesteps.
6. SOC initial/final constraints are respected per battery.
7. Published results include per-battery indexed sensor entities.
8. Mixed-timezone timestamps are handled without errors during publish.
9. Infeasible problems are detected and reported (not silently returning bad results).

## Edge Cases

- `inf` values in input parameters crash the solver — always use finite bounds.
- Mixed timezone offsets during DST transitions in result CSV — fixed with `utc=True`.
- SOC list length mismatch (fewer SOC values than batteries) — validation rejects.
- Single battery with list-style parameters — backwards compatibility preserved.
- HiGHS solver timeout on large problems (96+ timesteps × multiple batteries) — may need horizon reduction.
- Deferrable load count mismatch between parameter arrays — solver may crash or produce garbage.
- Empty forecast arrays — solver returns infeasible.
