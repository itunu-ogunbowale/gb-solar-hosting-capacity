"""
Power system study of the reduced GB transmission network with pandapower:
how much solar can a weak bus take?

  1. Inspect the network
  1b. Dispatch: one AC optimal power flow (OPF) per demand level, without solar.
      Double circuits are limited to 50% per circuit so the dispatch is N-1 secure.
  2. Base case load flow
  3. N-1 line contingency screening of the base case
  4. Short-circuit (fault) analysis, IEC 60909
  5. Add a solar farm and compare (weakest bus, plus the weakest bus with no
     generators if the weakest bus has some)
  6. Fault levels with the solar farm
  7. Hosting capacity sweep (peak vs summer minimum; unity pf, 0.95 absorbing,
     or the solar farm controlling voltage)
  8. N-1 hosting capacity: the same sweep, but no line outage may be worse
     than the same outage without solar

After Section 1b, every study uses AC load flows from the frozen OPF dispatch.
Only line outages are tested in N-1, not transformer or generator outages.

Install:  py -3.12 -m venv .venv
          .venv/Scripts/python -m pip install -r requirements-lock.txt
Run:      .venv/Scripts/python gb_power_system_study.py   (or step through the "# %%" cells)
All tables and figures are saved in ./results
"""

# %% 0. Imports and assumptions ------------------------------------------------
import os
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.networks as pn
import pandapower.shortcircuit as sc

OUT = "results"
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 140)
# pandapower raises pandas deprecation warnings internally; they do not affect results
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Assumptions (state these in your report) ---
EXT_GRID_SSC_MAX_MVA = 20000   # strength of the slack/external grid, max case
EXT_GRID_SSC_MIN_MVA = 15000   # ... min case
EXT_GRID_RX = 0.1              # R/X ratio of the external grid
GEN_XDSS_PU = 0.20             # synchronous generator subtransient reactance
GEN_RX = 0.07                  # generator R/X ratio (typical large machine)
GEN_COS_PHI = 0.85             # generator rated power factor
LINE_END_TEMP_C = 80           # conductor temperature for the IEC "min" case
PV_K = 1.2                     # inverter fault current = 1.2 x rated current
SUMMER_MIN_LOAD_SCALE = 0.45   # summer minimum demand ~45% of the case's demand
PV_BUSES = None                # list of bus indices to choose the solar buses yourself
PV_SIZE_MW = 2000              # aggregated regional solar for the comparison
SWEEP_SIZES_MW = np.arange(0, 8001, 250)
PV_Q_RANGE_PF = 0.95           # ... reactive range: +/- MVAr of 0.95 pf at full output
N1_STEP_EVERY = 2             # N-1 sweep tests every 2nd size (500 MW steps) to save time,
                              # then fills in the skipped size below the first failure
OVERLOAD_TOL_PCT = 1.0         # a line already above 100% may rise this many % points
VOLTAGE_TOL_PU = 0.005         # a bus already outside its limits may move this much further out
DOUBLE_CIRCUIT_OPF_LIMIT_PCT = 50.0  # OPF loading limit per circuit of a double circuit, so
                                     # losing one circuit cannot overload the other (N-1 secure)


def voltage_limits(vn_kv):
    """Planning voltage limits in pu by voltage level (check the current NETS SQSS)."""
    return (0.95, 1.05) if vn_kv >= 300 else (0.90, 1.10)


# %% Helper functions -----------------------------------------------------------
def run_lf(net):
    """Run a Newton-Raphson load flow. Returns True if it converged.
    Generator reactive limits are enforced; otherwise generators hold their
    voltage setpoint with unlimited MVAr and voltage rise never shows."""
    try:
        pp.runpp(net, algorithm="nr", calculate_voltage_angles=True,
                 init="auto", max_iteration=50, enforce_q_lims=True)
        return True
    except Exception:
        return False


def summarise(net):
    """Key results of the last load flow as a dict."""
    vm = net.res_bus.vm_pu
    viol = (voltage_excursion(net, vm) > 0) & net.bus.in_service
    line_load = net.res_line.loading_percent.max() if len(net.line) else 0.0
    trafo_load = net.res_trafo.loading_percent.max() if len(net.trafo) else 0.0
    losses = net.res_line.pl_mw.sum() + (net.res_trafo.pl_mw.sum() if len(net.trafo) else 0.0)
    return {
        "v_min_pu": vm.min(),
        "v_max_pu": vm.max(),
        "n_voltage_violations": int(viol.sum()),
        "max_line_loading_pct": line_load,
        "max_trafo_loading_pct": trafo_load,
        "losses_mw": losses,
        "ext_grid_p_mw": net.res_ext_grid.p_mw.sum(),
    }


def snapshot(net):
    """Bus voltages and line and transformer loadings of the last load flow, for
    element-by-element comparison. Callers use None in place of a snapshot for a
    non-converged case."""
    return {"vm": net.res_bus.vm_pu.copy(), "line": net.res_line.loading_percent.copy(),
            "trafo": net.res_trafo.loading_percent.copy()}


def voltage_excursion(net, vm):
    """How far each bus is outside its voltage limits, in pu (0 if inside)."""
    lims = net.bus.vn_kv.apply(voltage_limits)
    vmin, vmax = lims.str[0], lims.str[1]
    return np.maximum(np.maximum(vmin - vm, vm - vmax), 0).fillna(0)


def describe_violations(net, snap):
    """Buses outside their voltage limits, and lines and transformers above 100%,
    as readable text."""
    if snap is None:
        return "did not converge", ""
    vm = snap["vm"]
    exc = voltage_excursion(net, vm)
    buses = "; ".join(f"bus {b}: {vm[b]:.3f} pu" for b in exc.index[exc > 0])
    branches = "; ".join(f"{kind} {i}: {snap[kind][i]:.1f}%" for kind in ("line", "trafo")
                         for i in snap[kind].index[snap[kind] > 100])
    return buses, branches


def worsening(net, snap, ref):
    """What is worse in `snap` than in the reference `ref` (e.g. the same case
    without solar), as readable text; "" means nothing is worse.
    A bus fails if it leaves its voltage limits, or if it was already outside
    them and moves more than VOLTAGE_TOL_PU further out. A line or transformer
    fails if it goes above 100%, or if it was already above 100% and rises more
    than OVERLOAD_TOL_PCT points."""
    if ref is None:
        return ""  # nothing to compare against: the reference did not converge either
    if snap is None:
        return "did not converge"
    vm, vm_ref = snap["vm"], ref["vm"]
    exc, exc_ref = voltage_excursion(net, vm), voltage_excursion(net, vm_ref)
    bad_bus = exc > exc_ref + np.where(exc_ref > 0, VOLTAGE_TOL_PU, 0.0)
    parts = [f"bus {b}: {vm_ref[b]:.4f} -> {vm[b]:.4f} pu" for b in exc.index[bad_bus]]
    for kind in ("line", "trafo"):
        ld, ld_ref = snap[kind], ref[kind]
        limit = ld_ref.where(ld_ref <= 100, ld_ref + OVERLOAD_TOL_PCT).clip(lower=100)
        parts += [f"{kind} {i}: {ld_ref[i]:.1f} -> {ld[i]:.1f}%" for i in ld.index[ld > limit]]
    return "; ".join(parts)


def n1_screen(net):
    """Trip each in-service line in turn, run a load flow, record the results,
    then restore the line. Returns (one row per outage indexed by line,
    {line: snapshot or None})."""
    rows, snaps = [], {}
    for idx in list(net.line.index[net.line.in_service]):
        net.line.at[idx, "in_service"] = False
        ok = run_lf(net)
        row = {"line_out": idx, "converged": ok}
        if ok:
            row.update(summarise(net))
        snaps[idx] = snapshot(net) if ok else None
        rows.append(row)
        net.line.at[idx, "in_service"] = True
    run_lf(net)  # leave the network solved in its intact state
    return pd.DataFrame(rows).set_index("line_out"), snaps


def n1_failures(net, snaps, snaps_ref):
    """Outages that are worse than the same outage in the reference.
    Returns {line: reason}; empty means the N-1 test passed."""
    reasons = {l: worsening(net, snaps[l], snaps_ref.get(l)) for l in snaps}
    return {l: r for l, r in reasons.items() if r}


def add_short_circuit_data(net):
    """MATPOWER-based cases have no short-circuit data, so add assumed values."""
    net.ext_grid["s_sc_max_mva"] = EXT_GRID_SSC_MAX_MVA
    net.ext_grid["s_sc_min_mva"] = EXT_GRID_SSC_MIN_MVA
    net.ext_grid["rx_max"] = EXT_GRID_RX
    net.ext_grid["rx_min"] = EXT_GRID_RX

    net.gen["vn_kv"] = net.bus.loc[net.gen.bus, "vn_kv"].values
    p_rating = net.gen["max_p_mw"] if "max_p_mw" in net.gen else net.gen["p_mw"]
    p_rating = p_rating.fillna(net.gen["p_mw"]).abs()
    net.gen["sn_mva"] = (p_rating / GEN_COS_PHI).clip(lower=10)
    net.gen["xdss_pu"] = GEN_XDSS_PU
    x_ohm = GEN_XDSS_PU * net.gen["vn_kv"] ** 2 / net.gen["sn_mva"]
    net.gen["rdss_ohm"] = GEN_RX * x_ohm
    net.gen["cos_phi"] = GEN_COS_PHI

    # The case already has a static generator with no rating; IEC 60909 needs one
    s_sgen = np.hypot(net.sgen.p_mw, net.sgen.q_mvar)
    if "max_p_mw" in net.sgen:
        s_sgen = np.maximum(s_sgen, net.sgen.max_p_mw.fillna(0).abs())
    net.sgen["sn_mva"] = net.sgen["sn_mva"].fillna(s_sgen).clip(lower=1)
    if "k" not in net.sgen:
        net.sgen["k"] = PV_K
    net.sgen["k"] = net.sgen["k"].fillna(PV_K)

    net.line["endtemp_degree"] = LINE_END_TEMP_C


def run_short_circuit(net, fault="3ph", case="max"):
    sc.calc_sc(net, fault=fault, case=case, ip=(fault == "3ph"))
    return net.res_bus_sc.copy()


def opf_dispatch(net, base, load_scale):
    """Cost-optimal dispatch for one demand level, without solar, from an AC
    optimal power flow. It respects generator P and Q limits, the slack's limits,
    bus voltage limits and line/transformer ratings (set in Section 1b).
    If the OPF fails the script stops: no fallback dispatch is used."""
    net.load["p_mw"] = base["load_p"] * load_scale
    net.load["q_mvar"] = base["load_q"] * load_scale
    try:
        pp.runopp(net, calculate_voltage_angles=True)
    except Exception as e:
        raise RuntimeError(
            f"OPF failed for demand scale {load_scale} ({type(e).__name__}). No fallback "
            "dispatch is used: adjust the OPF limits, costs or demand and run again.") from e
    return {"gen_p": net.res_gen.p_mw.copy(), "gen_vm": net.res_gen.vm_pu.copy(),
            "sgen_p": net.res_sgen.p_mw.copy(), "bus_vm": net.res_bus.vm_pu.copy(),
            "cost": net.res_cost}


def set_scenario(net, base, load_scale):
    """Set demand, and the OPF dispatch for that demand level (generator outputs
    and voltage setpoints, Section 1b). Generators added later (the solar farm in
    voltage-control mode) hold their bus at its OPF voltage, so they agree with
    any other generator on the same bus."""
    d = base["dispatch"][load_scale]
    net.load["p_mw"] = base["load_p"] * load_scale
    net.load["q_mvar"] = base["load_q"] * load_scale
    net.gen.loc[d["gen_p"].index, "p_mw"] = d["gen_p"]
    net.gen.loc[d["gen_vm"].index, "vm_pu"] = d["gen_vm"]
    added = net.gen.index.difference(d["gen_vm"].index)
    net.gen.loc[added, "vm_pu"] = d["bus_vm"][net.gen.bus[added]].values
    net.sgen.loc[d["sgen_p"].index, "p_mw"] = d["sgen_p"]


def set_case(net, base, load_scale, pv, p_mw, pf):
    """Set one study case: demand and OPF dispatch for load_scale (set_scenario),
    plus p_mw of solar. pv holds the solar farm's two models: a fixed power
    factor static generator ("sgen") and a voltage-controlling generator ("gen").
    pf < 1 means the inverter absorbs reactive power at that power factor.
    pf=None means voltage control (setpoint from set_scenario), with reactive power limited to
    +/- p_mw x tan(acos(PV_Q_RANGE_PF)).
    Conventional generators are turned down in proportion to make room
    (a simple stand-in for real market redispatch)."""
    set_scenario(net, base, load_scale)
    gen_p_ref = base["dispatch"][load_scale]["gen_p"]
    v_control = pf is None
    net.sgen.at[pv["sgen"], "in_service"] = not v_control
    net.gen.at[pv["gen"], "in_service"] = v_control
    if v_control:
        q_max = p_mw * math.tan(math.acos(PV_Q_RANGE_PF))
        net.gen.loc[pv["gen"], ["p_mw", "min_q_mvar", "max_q_mvar", "sn_mva"]] = [
            p_mw, -q_max, q_max, max(p_mw / PV_Q_RANGE_PF, 1.0)]
    else:
        net.sgen.at[pv["sgen"], "p_mw"] = p_mw
        net.sgen.at[pv["sgen"], "q_mvar"] = -p_mw * math.tan(math.acos(pf)) if pf < 1 else 0.0
        net.sgen.at[pv["sgen"], "sn_mva"] = max(p_mw / pf, 1.0)
    total = gen_p_ref.clip(lower=0).sum()
    factor = max(0.0, 1 - p_mw / total) if total > 0 else 1.0
    net.gen.loc[gen_p_ref.index, "p_mw"] = gen_p_ref * factor


def select_pv(net, pvs, bus):
    """Take the solar farms at every bus except `bus` out of service
    (set_case then chooses which model of the one at `bus` is used)."""
    for b, pv in pvs.items():
        if b != bus:
            net.sgen.at[pv["sgen"], "in_service"] = False
            net.gen.at[pv["gen"], "in_service"] = False


def n1_check(net, base, load_scale, pv, p_mw, pf, n1_ref):
    """N-1 test of one solar size against n1_ref, the N-1 snapshots without solar.
    Returns None if the intact load flow does not converge, else
    {line: reason} for outages worse than without solar (empty = pass)."""
    set_case(net, base, load_scale, pv, p_mw, pf)
    if not run_lf(net):
        return None
    return n1_failures(net, n1_screen(net)[1], n1_ref)


# %% 1. Load and inspect the network ---------------------------------------------
net = pn.GBreducednetwork()

print(net)  # element counts
print("\nBus voltage levels (kV):\n", net.bus.vn_kv.value_counts())
print(f"\nTotal load: {net.load.p_mw.sum():.0f} MW, {net.load.q_mvar.sum():.0f} MVAr")
print(f"Conventional generation stored in the case: {net.gen.p_mw.sum():.0f} MW "
      f"from {len(net.gen)} generators (not balanced with load; replaced in Section 1b)")
print("External grid (slack) at bus:", net.ext_grid.bus.tolist())
print("\nLine thermal ratings (kA):\n", net.line.max_i_ka.describe())

# Store the original demand so every scenario starts from the same place
base = {
    "load_p": net.load.p_mw.copy(),
    "load_q": net.load.q_mvar.copy(),
}

# %% 1b. Dispatch by optimal power flow --------------------------------------------
# The generator outputs stored in the case are starting values for an optimiser,
# not a balanced operating point (Section 1 prints the totals: about 73.7 GW of
# generation for 56.3 GW of load, with the surplus forced into the slack). So for each demand level we run one AC OPF
# without solar to get a balanced, cost-optimal dispatch, then freeze it. All
# later studies use plain AC load flows from that starting point.
# The case has no line/transformer limits and uses 0.90-1.10 pu at every bus, so
# set the limits this study uses.
lims = net.bus.vn_kv.apply(voltage_limits)
net.bus["min_vm_pu"], net.bus["max_vm_pu"] = lims.str[0], lims.str[1]
net.line["max_loading_percent"] = 100.0
net.trafo["max_loading_percent"] = 100.0
# pandapower's OPF only checks the intact network. To make the dispatch N-1
# secure, as the real system is operated, each circuit of a double circuit
# (parallel lines between the same two buses) is limited to
# DOUBLE_CIRCUIT_OPF_LIMIT_PCT, so it can take its partner's flow after an outage.
# This applies to the OPF only: load flow checks still use 100%.
circuit = net.line[["from_bus", "to_bus"]].apply(lambda r: tuple(sorted(r)), axis=1)
double_circuit = circuit.duplicated(keep=False)
net.line.loc[double_circuit, "max_loading_percent"] = DOUBLE_CIRCUIT_OPF_LIMIT_PCT
print(f"\n{double_circuit.sum()} of {len(net.line)} lines are in double circuits; "
      f"OPF limit {DOUBLE_CIRCUIT_OPF_LIMIT_PCT:.0f}% per circuit")

base["dispatch"] = {s: opf_dispatch(net, base, s) for s in (1.0, SUMMER_MIN_LOAD_SCALE)}
opf_table = pd.DataFrame({f"gen_p_mw_x{s}": d["gen_p"] for s, d in base["dispatch"].items()}
                         | {f"gen_vm_pu_x{s}": d["gen_vm"] for s, d in base["dispatch"].items()})
opf_table.insert(0, "bus", net.gen.bus)
opf_table.insert(1, "max_p_mw", net.gen.max_p_mw)
opf_table.to_csv(f"{OUT}/opf_dispatch.csv")
for s, d in base["dispatch"].items():
    print(f"\nOPF dispatch at {s:.0%} demand: {d['gen_p'].sum():.0f} MW from generators "
          f"({(d['gen_p'] > 0.1).sum()} running), {d['sgen_p'].sum():.0f} MW static gen, "
          f"cost {d['cost']:.0f}")
set_scenario(net, base, 1.0)

# %% 2. Base case load flow --------------------------------------------------------
assert run_lf(net), "Base case did not converge - check the network data."
base_summary = summarise(net)
base_snap = snapshot(net)
print("\nBase case:", pd.Series(base_summary).round(3), sep="\n")

net.res_bus.to_csv(f"{OUT}/base_bus_results.csv")
net.res_line.to_csv(f"{OUT}/base_line_results.csv")

# Sanity check: generation = load + losses (+ shunts)
gen_total = net.res_ext_grid.p_mw.sum() + net.res_gen.p_mw.sum() + net.res_sgen.p_mw.sum()
load_total = net.res_load.p_mw.sum()
print(f"\nPower balance: generation {gen_total:.1f} MW, load {load_total:.1f} MW, "
      f"difference {gen_total - load_total:.1f} MW (losses + shunts)")

# Each bus is drawn against its own limits (400 kV and 132/275 kV differ)
lims = net.bus.vn_kv.apply(voltage_limits)
outside = voltage_excursion(net, net.res_bus.vm_pu) > 0
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(net.bus))
ax.bar(x, net.res_bus.vm_pu, color=np.where(outside, "tab:red", "tab:blue"))
ax.hlines(lims.str[0], x - 0.4, x + 0.4, colors="k", linestyles="--", label="Limits for each bus")
ax.hlines(lims.str[1], x - 0.4, x + 0.4, colors="k", linestyles="--")
ax.set_xticks(x); ax.set_xticklabels(net.bus.index.astype(str))
ax.set_ylim(0.85, 1.12)
ax.set_xlabel("Bus"); ax.set_ylabel("Voltage (pu)")
ax.set_title("Base case bus voltages (red: outside limits)"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/1_base_voltages.png", dpi=150); plt.close(fig)

# %% 3. N-1 line contingency screening -------------------------------------------
# For each outage: every violation present, and what is worse than the intact case
n1, n1_snaps = n1_screen(net)
n1.insert(0, "from_bus", net.line.from_bus.reindex(n1.index))
n1.insert(1, "to_bus", net.line.to_bus.reindex(n1.index))
viol = {l: describe_violations(net, s) for l, s in n1_snaps.items()}
n1["buses_outside_limits"] = [viol[l][0] for l in n1.index]
n1["branches_over_100pct"] = [viol[l][1] for l in n1.index]
n1["worse_than_intact"] = [worsening(net, n1_snaps[l], base_snap) for l in n1.index]
n1.to_csv(f"{OUT}/n1_contingencies.csv")

intact_buses, intact_branches = describe_violations(net, base_snap)
print(f"\nIntact base case violations:\n  buses: {intact_buses or 'none'}"
      f"\n  lines/transformers: {intact_branches or 'none'}")
problems = n1[n1.worse_than_intact != ""]
print(f"\nN-1: {len(n1)} outages studied, {len(problems)} make things worse than intact:")
for l, r in problems.iterrows():
    print(f"  line {l} ({r.from_bus}-{r.to_bus}) out: {r.worse_than_intact}")

# %% 4. Fault analysis (IEC 60909) -------------------------------------------------
add_short_circuit_data(net)

sc_3ph_max = run_short_circuit(net, "3ph", "max")
sc_3ph_min = run_short_circuit(net, "3ph", "min")
sc_2ph_max = run_short_circuit(net, "2ph", "max")

faults = pd.DataFrame({
    "vn_kv": net.bus.vn_kv,
    "ikss_3ph_max_ka": sc_3ph_max.ikss_ka,
    "ip_3ph_max_ka": sc_3ph_max.ip_ka,
    "skss_3ph_max_mva": sc_3ph_max.skss_mw,
    "ikss_3ph_min_ka": sc_3ph_min.ikss_ka,
    "ikss_2ph_max_ka": sc_2ph_max.ikss_ka,
})
faults.to_csv(f"{OUT}/fault_levels_base.csv")
print("\nFault levels (base):\n", faults.round(2))

# %% 5. Choose the solar buses and add the solar farms --------------------------
# The weakest bus (lowest minimum fault level, excluding the slack) is where
# voltage rise from new generation is worst. But generators on a bus hold its
# voltage at their setpoint (within their reactive limits), which masks voltage
# rise and the effect of the solar power factor. So if the weakest bus has
# generators, we also study the weakest bus with none.
# Override with PV_BUSES if you prefer buses that represent southern England.
if PV_BUSES is None:
    candidates = sc_3ph_min.skss_mw.drop(index=net.ext_grid.bus.tolist(), errors="ignore")
    PV_BUSES = [int(candidates.idxmin())]
    if net.gen.bus.isin(PV_BUSES).any():
        no_gen = candidates.drop(index=net.gen.bus.unique().tolist(), errors="ignore")
        PV_BUSES.append(int(no_gen.idxmin()))

# Number of conventional generators at each solar bus (before solar models are added)
n_conv_gen = {bus: int(net.gen.bus.isin([bus]).sum()) for bus in PV_BUSES}

# Each solar farm gets two models; set_case puts the right one in service:
# a static generator for fixed power factor, a generator for voltage control
pvs = {}
for bus in PV_BUSES:
    sgen = pp.create_sgen(net, bus=bus, p_mw=0, q_mvar=0, sn_mva=1, type="PV",
                          name=f"Solar farm bus {bus}", in_service=False)
    net.sgen.at[sgen, "k"] = PV_K  # inverter fault current limit
    gen = pp.create_gen(net, bus=bus, p_mw=0, vm_pu=1.0, min_q_mvar=0, max_q_mvar=0,
                        name=f"Solar farm bus {bus} (voltage control)", in_service=False)
    pvs[bus] = {"sgen": sgen, "gen": gen}
    print(f"\nSolar farm studied at bus {bus} "
          f"(min fault level {sc_3ph_min.at[bus, 'skss_mw']:.0f} MVA, "
          f"{n_conv_gen[bus]} generators)")

comparison = {"base": base_summary}
for bus in PV_BUSES:
    select_pv(net, pvs, bus)
    set_case(net, base, 1.0, pvs[bus], PV_SIZE_MW, pf=1.0)
    # Non-convergence is a result here (voltage collapse), not a script error
    if run_lf(net):
        comparison[f"{PV_SIZE_MW} MW PV at bus {bus}"] = summarise(net)
    else:
        print(f"WARNING: load flow with {PV_SIZE_MW} MW PV at bus {bus} did not converge.")
        comparison[f"{PV_SIZE_MW} MW PV at bus {bus}"] = {k: np.nan for k in base_summary}

comparison = pd.DataFrame(comparison)
print("\nBase vs solar:\n", comparison.round(3))
comparison.to_csv(f"{OUT}/base_vs_solar.csv")

# %% 6. Fault levels with the solar farm --------------------------------------------
for bus in PV_BUSES:
    select_pv(net, pvs, bus)
    set_case(net, base, 1.0, pvs[bus], PV_SIZE_MW, pf=1.0)
    sc_pv = run_short_circuit(net, "3ph", "max")
    faults[f"ikss_3ph_max_pv_bus{bus}_ka"] = sc_pv.ikss_ka
    faults[f"change_pct_pv_bus{bus}"] = 100 * (sc_pv.ikss_ka / faults.ikss_3ph_max_ka - 1)
    print(f"\nFault current at bus {bus}: {faults.at[bus, 'ikss_3ph_max_ka']:.2f} kA -> "
          f"{sc_pv.ikss_ka[bus]:.2f} kA ({faults.at[bus, f'change_pct_pv_bus{bus}']:+.1f}%)")
faults.to_csv(f"{OUT}/fault_levels_with_pv.csv")

# Note: IEC 60909 counts every in-service generator at its rating, so turning
# generators down for the solar does not reduce their fault contribution here.
# To model plants actually closing, set net.gen.in_service = False for them.

x = np.arange(len(faults)); w = 0.8 / (len(PV_BUSES) + 1)
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(x - 0.4 + w/2, faults.ikss_3ph_max_ka, w, label="Base")
for i, bus in enumerate(PV_BUSES, start=1):
    ax.bar(x - 0.4 + w/2 + i*w, faults[f"ikss_3ph_max_pv_bus{bus}_ka"], w,
           label=f"With {PV_SIZE_MW} MW PV at bus {bus}")
ax.set_xticks(x); ax.set_xticklabels(faults.index.astype(str))
ax.set_xlabel("Bus"); ax.set_ylabel("Initial symmetrical fault current I''k (kA)")
ax.set_title("Three-phase fault current, max case"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/2_fault_levels.png", dpi=150); plt.close(fig)

# %% 7. Hosting capacity sweep -----------------------------------------------------
scenarios = {
    "Peak, unity pf": (1.0, 1.00),
    "Peak, 0.95 absorbing": (1.0, 0.95),
    "Summer min, unity pf": (SUMMER_MIN_LOAD_SCALE, 1.00),
    "Summer min, 0.95 absorbing": (SUMMER_MIN_LOAD_SCALE, 0.95),
    # pf=None: the solar farm controls voltage, as GB grid code plant must be able to
    "Peak, voltage control": (1.0, None),
    "Summer min, voltage control": (SUMMER_MIN_LOAD_SCALE, None),
}
# Colour = demand case, line style = reactive mode, so lines that coincide stay visible
styles = {
    "Peak, unity pf": dict(c="tab:blue", ls="-", marker="o"),
    "Peak, 0.95 absorbing": dict(c="tab:blue", ls="--", marker="x"),
    "Summer min, unity pf": dict(c="tab:orange", ls="-", marker="o"),
    "Summer min, 0.95 absorbing": dict(c="tab:orange", ls="--", marker="x"),
    "Peak, voltage control": dict(c="tab:purple", ls="-.", marker="s"),
    "Summer min, voltage control": dict(c="tab:green", ls="-.", marker="s"),
}

sweeps, hosting, intact_binding = {}, {}, {}
for bus in PV_BUSES:
    select_pv(net, pvs, bus)
    pv = pvs[bus]
    for name, (load_scale, pf) in scenarios.items():
        # Reference point: the scenario with no solar
        set_case(net, base, load_scale, pv, 0, 1.0)
        ref = snapshot(net) if run_lf(net) else None

        rows = []
        for size in SWEEP_SIZES_MW:
            set_case(net, base, load_scale, pv, float(size), pf)
            ok = run_lf(net)
            row = {"pv_mw": size, "converged": ok}
            if ok:
                row.update(summarise(net))
                row["v_pv_bus_pu"] = net.res_bus.vm_pu[bus]
            # A size is acceptable if nothing is worse than the no-solar case
            row["worse_than_no_solar"] = worsening(net, snapshot(net) if ok else None, ref)
            row["ok"] = ok and not row["worse_than_no_solar"]
            rows.append(row)
        df = pd.DataFrame(rows)

        # Hosting capacity: the last size before the first failure. If nothing
        # fails, it is at least the largest size swept.
        if df.ok.all():
            hc, why = SWEEP_SIZES_MW[-1], "no limit reached within sweep"
        else:
            first_fail = df.index[~df.ok][0]
            hc = df.pv_mw[first_fail - 1] if first_fail > 0 else 0
            why = f"fails at {df.pv_mw[first_fail]} MW: {df.worse_than_no_solar[first_fail]}"
        sweeps[bus, name], hosting[bus, name], intact_binding[bus, name] = df, hc, why
        df.to_csv(f"{OUT}/sweep_bus{bus}_{name.replace(' ', '_').replace(',', '')}.csv",
                  index=False)

    print("\nHosting capacity at bus", bus)
    for name in scenarios:
        print(f"  {name:30s} {hosting[bus, name]} MW  ({intact_binding[bus, name]})")

# One row of plots per solar bus: voltage at the solar bus on the left, highest
# line loading on the right. Lines end where the load flow stops converging.
fig, axes = plt.subplots(len(PV_BUSES), 2, figsize=(12, 4.5 * len(PV_BUSES)),
                         sharex=True, squeeze=False)
for (ax1, ax2), bus in zip(axes, PV_BUSES):
    for name in scenarios:
        df = sweeps[bus, name]
        label = f"{name} (HC {'>=' if df.ok.all() else ''}{hosting[bus, name]} MW)"
        ax1.plot(df.pv_mw, df.v_pv_bus_pu, label=label, markevery=4, **styles[name])
        branch_pct = df[["max_line_loading_pct", "max_trafo_loading_pct"]].max(axis=1, skipna=False)
        ax2.plot(df.pv_mw, branch_pct, label=label, markevery=4, **styles[name])
    vn = net.bus.vn_kv[bus]
    for lim in voltage_limits(vn):
        ax1.axhline(lim, ls=":", c="k", lw=1.5)
    ax1.plot([], [], ls=":", c="k", lw=1.5, label=f"Limits ({vn:.0f} kV bus)")
    ax2.axhline(100, ls=":", c="k", lw=1.5, label="Limit (100%)")
    gens = "with generators" if n_conv_gen[bus] else "no generators"
    ax1.set_title(f"Solar at bus {bus} ({gens}): voltage at bus {bus}")
    ax2.set_title(f"Solar at bus {bus} ({gens}): max line/transformer loading")
    ax1.set_ylabel("Voltage (pu)"); ax2.set_ylabel("Loading (%)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("Solar capacity (MW)")
fig.suptitle("Hosting capacity (HC) sweep, intact network\n"
             "HC can also be set by a voltage violation at another bus")
fig.tight_layout(); fig.savefig(f"{OUT}/3_hosting_capacity.png", dpi=150); plt.close(fig)

# %% 8. N-1 hosting capacity ----------------------------------------------------
# Same idea as Section 7, but each solar size must also pass an N-1 screen.
n1_rows, n1_fail_rows = [], []
for bus in PV_BUSES:
    select_pv(net, pvs, bus)
    pv = pvs[bus]
    for name, (load_scale, pf) in scenarios.items():
        # Reference: N-1 results for this scenario with no solar. Each outage is
        # compared with the same outage without solar, so problems the network
        # already has under that outage do not count against the solar.
        set_case(net, base, load_scale, pv, 0, 1.0)
        run_lf(net)
        _, n1_ref = n1_screen(net)
        unsolved = [l for l, s in n1_ref.items() if s is None]
        if unsolved:
            print(f"  Bus {bus}, {name}: outage(s) of line(s) {unsolved} do not converge even "
                  "without solar, so they are not tested")

        # N-1 hosting capacity cannot exceed the intact value, so only sizes up to
        # the intact limit (Section 7) are tested. To save time every
        # N1_STEP_EVERY-th size is tested, plus the intact limit itself.
        ok_intact = sweeps[bus, name].ok.values
        cap = len(ok_intact) if ok_intact.all() else int(np.argmax(~ok_intact))
        sizes = list(SWEEP_SIZES_MW[1:cap][::N1_STEP_EVERY])
        if cap > 1 and SWEEP_SIZES_MW[cap - 1] not in sizes:
            sizes.append(SWEEP_SIZES_MW[cap - 1])

        hc_n1, fail_size, failing = 0, None, None
        for size in sizes:
            failing = n1_check(net, base, load_scale, pv, float(size), pf, n1_ref)
            if failing is None or failing:
                fail_size = size
                break
            hc_n1 = size
        if fail_size is not None:
            # Test the sizes the coarse step skipped, between the last pass and the failure
            for fine in SWEEP_SIZES_MW[(SWEEP_SIZES_MW > hc_n1) & (SWEEP_SIZES_MW < fail_size)]:
                result = n1_check(net, base, load_scale, pv, float(fine), pf, n1_ref)
                if result is None or result:
                    fail_size, failing = fine, result
                    break
                hc_n1 = fine

        if fail_size is None:
            binding = ("no limit reached within sweep" if cap == len(ok_intact)
                       else f"intact limit (Section 7, {intact_binding[bus, name]})")
        elif failing is None:
            binding = f"intact load flow did not converge at {fail_size} MW"
        else:
            binding = f"{len(failing)} outage(s) fail at {fail_size} MW, e.g. line(s) " \
                      f"{list(failing)[:5]}"
            n1_fail_rows += [{"pv_bus": bus, "scenario": name, "pv_mw": fail_size,
                              "line_out": l, "from_bus": net.line.from_bus[l],
                              "to_bus": net.line.to_bus[l], "worse_than_no_solar": r}
                             for l, r in failing.items()]

        n1_rows.append({"pv_bus": bus, "scenario": name, "intact_hc_mw": hosting[bus, name],
                        "n1_hc_mw": hc_n1, "binding_constraint": binding})
    print(f"N-1 sweep done for bus {bus}", flush=True)

n1_hc = pd.DataFrame(n1_rows)
n1_hc.to_csv(f"{OUT}/hosting_capacity_n1.csv", index=False)
print("\nIntact vs N-1 hosting capacity:\n", n1_hc.to_string(index=False))

# Every failing outage at the first size that fails N-1, and why
n1_fail = pd.DataFrame(n1_fail_rows, columns=["pv_bus", "scenario", "pv_mw", "line_out",
                                               "from_bus", "to_bus", "worse_than_no_solar"])
n1_fail.to_csv(f"{OUT}/n1_failures_with_pv.csv", index=False)
print("\nN-1 failures with solar (first failing size; 'was -> now' vs no solar):")
for (bus, name), d in n1_fail.groupby(["pv_bus", "scenario"], sort=False):
    print(f"  Bus {bus}, {name}, {d.pv_mw.iloc[0]} MW:")
    for _, r in d.iterrows():
        print(f"    line {r.line_out} ({r.from_bus}-{r.to_bus}) out: {r.worse_than_no_solar}")

fig, axes = plt.subplots(1, len(PV_BUSES), figsize=(7 * len(PV_BUSES), 4.5),
                         sharey=True, squeeze=False)
for ax, bus in zip(axes[0], PV_BUSES):
    d = n1_hc[n1_hc.pv_bus == bus]
    x = np.arange(len(d)); w = 0.4
    ax.bar(x - w/2, d.intact_hc_mw, w, label="Intact network")
    ax.bar(x + w/2, d.n1_hc_mw, w, label="N-1 secure")
    ax.set_xticks(x); ax.set_xticklabels(d.scenario, rotation=15, ha="right")
    ax.set_title(f"Solar at bus {bus}: intact vs N-1"); ax.legend()
axes[0][0].set_ylabel("Hosting capacity (MW)")
fig.tight_layout(); fig.savefig(f"{OUT}/4_hosting_capacity_n1.png", dpi=150); plt.close(fig)

print(f"\nDone. Results and figures saved in ./{OUT}")
