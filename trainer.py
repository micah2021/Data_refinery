"""
Training Loop
Runs multi-episode experiments and collects results.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import time

from core.environment import PDEnvironment, PDConfig, VARIANTS, Action
from core.agent import NStepSARSA, AgentConfig, RandomAgent, TitForTatAgent, AlwaysDefectAgent


@dataclass
class ExperimentConfig:
    n_episodes:     int   = 5000
    steps_per_ep:   int   = 100
    variant:        str   = "classic"
    n_step:         int   = 2
    alpha:          float = 0.1
    gamma:          float = 0.9
    epsilon:        float = 0.1
    epsilon_decay:  float = 0.9995
    epsilon_min:    float = 0.01
    seed:           int   = 42
    verbose:        bool  = True
    log_every:      int   = 500


@dataclass
class ExperimentResult:
    config:           ExperimentConfig
    coop_history:     List[float] = field(default_factory=list)
    reward_a_history: List[float] = field(default_factory=list)
    reward_b_history: List[float] = field(default_factory=list)
    q_table_a:        Dict        = field(default_factory=dict)
    q_table_b:        Dict        = field(default_factory=dict)
    policy_a:         Dict        = field(default_factory=dict)
    policy_b:         Dict        = field(default_factory=dict)
    final_coop_rate:  float       = 0.0
    converged:        bool        = False
    convergence_ep:   int         = -1
    total_time:       float       = 0.0
    agent_a_name:     str         = "SARSA-A"
    agent_b_name:     str         = "SARSA-B"

    def summary(self) -> str:
        lines = [
            f"\n{'='*55}",
            f"  EXPERIMENT RESULTS",
            f"{'='*55}",
            f"  Variant       : {self.config.variant}",
            f"  N-step        : {self.config.n_step}",
            f"  Episodes      : {self.config.n_episodes}",
            f"  Final Coop %  : {self.final_coop_rate*100:.1f}%",
            f"  Converged     : {'Yes @ ep ' + str(self.convergence_ep) if self.converged else 'No'}",
            f"  Avg Reward A  : {np.mean(self.reward_a_history):.3f}",
            f"  Avg Reward B  : {np.mean(self.reward_b_history):.3f}",
            f"  Time          : {self.total_time:.2f}s",
            f"  Policy A      : {self.policy_a}",
            f"  Policy B      : {self.policy_b}",
            f"{'='*55}",
        ]
        return '\n'.join(lines)


def run_experiment(cfg: ExperimentConfig) -> ExperimentResult:
    """Run a full experiment and return results."""
    np.random.seed(cfg.seed)
    start_time = time.time()

    # Build environment
    env = PDEnvironment(VARIANTS[cfg.variant])
    states = env.get_all_states()

    # Build agents
    agent_cfg_a = AgentConfig(
        alpha=cfg.alpha, gamma=cfg.gamma,
        epsilon=cfg.epsilon, n_step=cfg.n_step,
        epsilon_decay=cfg.epsilon_decay,
        epsilon_min=cfg.epsilon_min,
        name="SARSA-A"
    )
    agent_cfg_b = AgentConfig(
        alpha=cfg.alpha, gamma=cfg.gamma,
        epsilon=cfg.epsilon, n_step=cfg.n_step,
        epsilon_decay=cfg.epsilon_decay,
        epsilon_min=cfg.epsilon_min,
        name="SARSA-B"
    )
    agent_a = NStepSARSA(states, agent_cfg_a)
    agent_b = NStepSARSA(states, agent_cfg_b)

    result = ExperimentResult(config=cfg)
    window = 100

    for ep in range(cfg.n_episodes):
        # Reset
        env_states = env.reset()
        agent_a.reset_episode()
        agent_b.reset_episode()

        sa = env_states[0]
        sb = env_states[1] if len(env_states) > 1 else env_states[0]
        aa = agent_a.choose_action(sa)
        ab = agent_b.choose_action(sb)

        ep_coop = 0

        for step in range(cfg.steps_per_ep):
            # Environment step
            new_states, rewards, _ = env.step([Action(aa), Action(ab)])
            sa1, sb1 = new_states[0], new_states[1] if len(new_states) > 1 else new_states[0]
            ra, rb   = rewards[0], rewards[1] if len(rewards) > 1 else rewards[0]

            # Choose next actions
            aa1 = agent_a.choose_action(sa1)
            ab1 = agent_b.choose_action(sb1)

            # Store transitions
            from core.agent import Transition
            agent_a.store_transition(Transition(sa, aa, ra, sa1, aa1))
            agent_b.store_transition(Transition(sb, ab, rb, sb1, ab1))

            # Update Q-values
            agent_a.update()
            agent_b.update()

            # Record
            agent_a.record_step(aa, ra)
            agent_b.record_step(ab, rb)

            if aa == 0: ep_coop += 1

            # Advance
            sa, sb = sa1, sb1
            aa, ab = aa1, ab1

        # End of episode
        agent_a.end_of_episode()
        agent_b.end_of_episode()

        # Record episode metrics
        coop_rate = ep_coop / cfg.steps_per_ep
        result.coop_history.append(coop_rate)
        result.reward_a_history.append(agent_a.episode_reward / cfg.steps_per_ep)
        result.reward_b_history.append(agent_b.episode_reward / cfg.steps_per_ep)

        # Check convergence (stable coop rate for 500 episodes)
        if not result.converged and ep >= window:
            recent = result.coop_history[-window:]
            if np.std(recent) < 0.05:
                result.converged     = True
                result.convergence_ep = ep

        # Log
        if cfg.verbose and (ep + 1) % cfg.log_every == 0:
            recent_coop = np.mean(result.coop_history[-window:])
            print(f"  ep {ep+1:>5}/{cfg.n_episodes} | "
                  f"coop={recent_coop*100:.1f}% | "
                  f"ε={agent_a.epsilon:.3f} | "
                  f"ΔQ={agent_a.q_delta:.4f}")

    # Store final results
    result.final_coop_rate = np.mean(result.coop_history[-window:])
    result.q_table_a       = {s: agent_a.Q[s].tolist() for s in states}
    result.q_table_b       = {s: agent_b.Q[s].tolist() for s in states}
    result.policy_a        = agent_a.greedy_policy()
    result.policy_b        = agent_b.greedy_policy()
    result.total_time      = time.time() - start_time

    return result


def compare_nsteps(variant: str = "classic",
                   n_steps: List[int] = [1, 2, 3, 5],
                   n_episodes: int = 3000,
                   n_seeds: int = 5) -> Dict[int, List[float]]:
    """
    Compare different n-step values.
    Returns dict: {n_step: [coop_rate per seed]}
    """
    print(f"\n{'='*55}")
    print(f"  COMPARING N-STEP VALUES ON '{variant}'")
    print(f"  Seeds: {n_seeds} | Episodes: {n_episodes}")
    print(f"{'='*55}")

    results = {}
    for n in n_steps:
        coop_rates = []
        for seed in range(n_seeds):
            cfg = ExperimentConfig(
                variant=variant, n_step=n,
                n_episodes=n_episodes, seed=seed,
                verbose=False
            )
            res = run_experiment(cfg)
            coop_rates.append(res.final_coop_rate)

        results[n] = coop_rates
        mean = np.mean(coop_rates)
        std  = np.std(coop_rates)
        print(f"  n={n}: coop={mean*100:.1f}% ± {std*100:.1f}%")

    return results


def run_vs_baseline(sarsa_n: int = 2,
                    variant:  str = "classic",
                    n_episodes: int = 3000) -> Dict:
    """
    Run SARSA against TFT and AllD baselines.
    """
    print(f"\n{'='*55}")
    print(f"  SARSA (n={sarsa_n}) vs BASELINES")
    print(f"{'='*55}")

    env = PDEnvironment(VARIANTS[variant])
    states = env.get_all_states()

    results = {}
    baselines = {
        "TFT":    TitForTatAgent("TFT"),
        "AllD":   AlwaysDefectAgent("AllD"),
        "Random": RandomAgent("Random"),
    }

    for name, baseline in baselines.items():
        sarsa_cfg = AgentConfig(n_step=sarsa_n, name=f"SARSA-{sarsa_n}")
        sarsa = NStepSARSA(states, sarsa_cfg)
        np.random.seed(42)
        env_states = env.reset()

        for ep in range(n_episodes):
            env_states = env.reset()
            sarsa.reset_episode()
            baseline.reset_episode()

            sa = env_states[0]
            sb = env_states[1] if len(env_states) > 1 else env_states[0]
            aa = sarsa.choose_action(sa)
            ab = baseline.choose_action(sb)

            for step in range(100):
                new_states, rewards, _ = env.step([Action(aa), Action(ab)])
                sa1 = new_states[0]
                sb1 = new_states[1] if len(new_states) > 1 else new_states[0]
                ra, rb = rewards

                aa1 = sarsa.choose_action(sa1)
                ab1 = baseline.choose_action(sb1)

                from core.agent import Transition
                sarsa.store_transition(Transition(sa, aa, ra, sa1, aa1))
                sarsa.update()
                sarsa.record_step(aa, ra)
                baseline.record_step(ab, rb)

                sa, sb = sa1, sb1
                aa, ab = aa1, ab1

            sarsa.end_of_episode()
            baseline.end_of_episode()

        sarsa_coop = sarsa.cooperation_rate(500)
        base_coop  = baseline.cooperation_rate(500)
        sarsa_rew  = np.mean(sarsa.reward_history[-500:])
        base_rew   = np.mean(baseline.reward_history[-500:])

        results[name] = {
            "sarsa_coop": sarsa_coop,
            "baseline_coop": base_coop,
            "sarsa_avg_reward": sarsa_rew,
            "baseline_avg_reward": base_rew,
        }
        print(f"  vs {name:8}: SARSA coop={sarsa_coop*100:.1f}%  "
              f"reward={sarsa_rew:.2f} | "
              f"{name} coop={base_coop*100:.1f}%  reward={base_rew:.2f}")

    return results


if __name__ == "__main__":
    print("Running quick test experiment...")
    cfg = ExperimentConfig(
        variant="classic", n_step=2,
        n_episodes=1000, verbose=True, log_every=200
    )
    result = run_experiment(cfg)
    print(result.summary())
    print(f"\nAgent A Q-table:")
    print(f"  CC: C={result.q_table_a['CC'][0]:.3f}  D={result.q_table_a['CC'][1]:.3f}")
    print(f"  DD: C={result.q_table_a['DD'][0]:.3f}  D={result.q_table_a['DD'][1]:.3f}")
