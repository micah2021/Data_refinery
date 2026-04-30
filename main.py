"""
Main Experiment Runner
Run all experiments and generate all plots.

Usage:
    python main.py                    # full suite
    python main.py --quick            # fast test (500 episodes)
    python main.py --variant classic  # single variant
    python main.py --compare-nsteps   # n-step comparison only
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from core.environment import VARIANTS
from core.trainer import (
    ExperimentConfig, run_experiment,
    compare_nsteps, run_vs_baseline
)
from visualization.plots import (
    plot_cooperation_curve, plot_qtable_heatmap,
    plot_nstep_comparison, plot_variant_comparison,
    plot_action_heatmap
)

os.makedirs("plots",   exist_ok=True)
os.makedirs("results", exist_ok=True)


def run_full_suite(quick: bool = False):
    n_ep = 500 if quick else 5000
    print(f"\n{'='*60}")
    print(f"  2-STEP SARSA — PRISONER'S DILEMMA RESEARCH")
    print(f"  Episodes: {n_ep} | Mode: {'QUICK' if quick else 'FULL'}")
    print(f"{'='*60}")

    # ── Experiment 1: Classic PD with n=2 ─────────────────────
    print("\n[1/4] Classic PD — 2-Step SARSA")
    cfg = ExperimentConfig(
        variant="classic", n_step=2,
        n_episodes=n_ep, verbose=True, log_every=n_ep//5
    )
    res_classic = run_experiment(cfg)
    print(res_classic.summary())
    plot_cooperation_curve(res_classic)
    plot_qtable_heatmap(res_classic)

    # ── Experiment 2: All variants ─────────────────────────────
    print("\n[2/4] All Variants Comparison")
    variant_results = {}
    for v in ["classic", "noisy", "asymmetric", "harsh", "lenient"]:
        print(f"  Running {v}...")
        cfg_v = ExperimentConfig(
            variant=v, n_step=2,
            n_episodes=n_ep, verbose=False
        )
        variant_results[v] = run_experiment(cfg_v)
        coop = variant_results[v].final_coop_rate
        print(f"    → coop={coop*100:.1f}%  policy_A={variant_results[v].policy_a}")
    plot_variant_comparison(variant_results)

    # ── Experiment 3: N-step comparison ───────────────────────
    print("\n[3/4] N-Step Comparison (n=1,2,3,5)")
    n_seeds = 3 if quick else 5
    nstep_res = compare_nsteps(
        variant="classic",
        n_steps=[1, 2, 3, 5],
        n_episodes=n_ep,
        n_seeds=n_seeds
    )
    plot_nstep_comparison(nstep_res, variant="classic")

    # ── Experiment 4: vs Baselines ─────────────────────────────
    print("\n[4/4] SARSA vs Baselines (TFT, AllD, Random)")
    baseline_res = run_vs_baseline(sarsa_n=2, variant="classic", n_episodes=n_ep)

    # ── Action history plot ────────────────────────────────────
    from core.environment import PDEnvironment, Action
    from core.agent import NStepSARSA, AgentConfig, Transition
    env2 = PDEnvironment(VARIANTS["classic"])
    states2 = env2.get_all_states()
    a_cfg = AgentConfig(n_step=2, name="A")
    b_cfg = AgentConfig(n_step=2, name="B")
    ag_a = NStepSARSA(states2, a_cfg)
    ag_b = NStepSARSA(states2, b_cfg)
    env_states = env2.reset()
    sa = env_states[0]; sb = env_states[1]
    aa = ag_a.choose_action(sa); ab = ag_b.choose_action(sb)
    for _ in range(500):
        new_states, rewards, _ = env2.step([Action(aa), Action(ab)])
        sa1, sb1 = new_states
        ra, rb   = rewards
        aa1 = ag_a.choose_action(sa1); ab1 = ag_b.choose_action(sb1)
        ag_a.store_transition(Transition(sa, aa, ra, sa1, aa1))
        ag_b.store_transition(Transition(sb, ab, rb, sb1, ab1))
        ag_a.update(); ag_b.update()
        ag_a.record_step(aa, ra); ag_b.record_step(ab, rb)
        sa, sb = sa1, sb1; aa, ab = aa1, ab1
    plot_action_heatmap(ag_a.action_history, ag_b.action_history,
                        "Action History — Last 500 Steps (2-Step SARSA)")

    # ── Save summary JSON ──────────────────────────────────────
    summary = {
        "classic_coop_rate":    res_classic.final_coop_rate,
        "classic_converged":    res_classic.converged,
        "classic_convergence_ep": res_classic.convergence_ep,
        "classic_policy_a":     res_classic.policy_a,
        "classic_policy_b":     res_classic.policy_b,
        "nstep_comparison":     {str(k): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                                 for k, v in nstep_res.items()},
        "variant_coop_rates":   {v: float(r.final_coop_rate) for v, r in variant_results.items()},
        "baseline_results":     baseline_res,
    }
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[✓] Summary saved → results/summary.json")

    # ── Final report ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"  Classic PD cooperation rate : {res_classic.final_coop_rate*100:.1f}%")
    print(f"  Best variant for coop       : {max(variant_results, key=lambda v: variant_results[v].final_coop_rate)}")
    print(f"  Best n-step for coop        : n={max(nstep_res, key=lambda n: np.mean(nstep_res[n]))}")
    print(f"\n  Plots saved to  : plots/")
    print(f"  Results saved to: results/")
    print(f"{'='*60}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="2-Step SARSA Prisoner's Dilemma Research")
    parser.add_argument("--quick",         action="store_true", help="Run 500 episodes (fast test)")
    parser.add_argument("--variant",       default=None,        help="Run single variant only")
    parser.add_argument("--compare-nsteps",action="store_true", help="Run n-step comparison only")
    parser.add_argument("--n-episodes",    type=int, default=5000)
    parser.add_argument("--n-step",        type=int, default=2)
    parser.add_argument("--alpha",         type=float, default=0.1)
    parser.add_argument("--gamma",         type=float, default=0.9)
    parser.add_argument("--epsilon",       type=float, default=0.1)
    args = parser.parse_args()

    if args.compare_nsteps:
        res = compare_nsteps(n_episodes=args.n_episodes)
        plot_nstep_comparison(res)
        plt.show()
        return

    if args.variant:
        print(f"\nRunning single variant: {args.variant}")
        cfg = ExperimentConfig(
            variant=args.variant,
            n_step=args.n_step,
            n_episodes=args.n_episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            verbose=True,
            log_every=max(1, args.n_episodes // 10)
        )
        result = run_experiment(cfg)
        print(result.summary())
        plot_cooperation_curve(result)
        plot_qtable_heatmap(result)
        plt.show()
        return

    run_full_suite(quick=args.quick)


if __name__ == "__main__":
    main()
