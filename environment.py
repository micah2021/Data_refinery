"""
Prisoner's Dilemma Environment
Supports: Classic, Noisy, Asymmetric, N-Person variants
"""

import numpy as np
from enum import IntEnum
from dataclasses import dataclass
from typing import Tuple, List, Optional


class Action(IntEnum):
    COOPERATE = 0
    DEFECT    = 1


@dataclass
class PDConfig:
    """Payoff matrix configuration. Must satisfy T > R > P > S."""
    T: float = 5.0   # Temptation (defect vs cooperator)
    R: float = 3.0   # Reward (mutual cooperation)
    P: float = 1.0   # Punishment (mutual defection)
    S: float = 0.0   # Sucker (cooperate vs defector)
    T_b: Optional[float] = None   # Asymmetric temptation for agent B
    noise: float = 0.0            # Prob of action being flipped
    n_agents: int = 2             # Number of agents (>2 = N-person)

    def __post_init__(self):
        assert self.T > self.R > self.P > self.S, \
            f"Payoff must satisfy T>R>P>S, got T={self.T} R={self.R} P={self.P} S={self.S}"
        if self.T_b is None:
            self.T_b = self.T


# ── Preset variants ────────────────────────────────────────────
VARIANTS = {
    "classic": PDConfig(T=5, R=3, P=1, S=0),
    "noisy":   PDConfig(T=5, R=3, P=1, S=0, noise=0.10),
    "asymmetric": PDConfig(T=5, R=3, P=1, S=0, T_b=7),
    "nperson": PDConfig(T=5, R=3, P=1, S=0, n_agents=3),
    "harsh":   PDConfig(T=9, R=3, P=1, S=0),   # High temptation
    "lenient": PDConfig(T=4, R=3, P=2, S=1),   # Weak dilemma
}


class PDEnvironment:
    """
    Iterated Prisoner's Dilemma environment for n agents.
    State = tuple of last actions of all agents (own + opponents).
    """

    def __init__(self, config: PDConfig):
        self.config = config
        self.n = config.n_agents
        self.step_count = 0
        self.history: List[Tuple] = []
        # Last actions (start: all defect)
        self.last_actions = [Action.DEFECT] * self.n

    def reset(self) -> List[str]:
        """Reset environment, return initial states for all agents."""
        self.step_count = 0
        self.history = []
        self.last_actions = [Action.DEFECT] * self.n
        return [self._get_state(i) for i in range(self.n)]

    def _get_state(self, agent_idx: int) -> str:
        """
        State for agent i = own last action + opponent's last action.
        For n>2: own action + count of cooperators among others.
        """
        if self.n == 2:
            own  = 'C' if self.last_actions[agent_idx] == Action.COOPERATE else 'D'
            opp  = 'C' if self.last_actions[1 - agent_idx] == Action.COOPERATE else 'D'
            return own + opp   # CC, CD, DC, DD
        else:
            own   = 'C' if self.last_actions[agent_idx] == Action.COOPERATE else 'D'
            coops = sum(1 for i, a in enumerate(self.last_actions)
                        if i != agent_idx and a == Action.COOPERATE)
            return f"{own}{coops}"   # e.g. C2, D1, C0

    def _apply_noise(self, action: Action) -> Action:
        if self.config.noise > 0 and np.random.random() < self.config.noise:
            return Action.COOPERATE if action == Action.DEFECT else Action.DEFECT
        return action

    def _compute_payoffs(self, actions: List[Action]) -> List[float]:
        """Compute payoffs for all agents given their actions."""
        cfg = self.config
        if self.n == 2:
            a, b = actions
            payoff_a = {
                (Action.COOPERATE, Action.COOPERATE): cfg.R,
                (Action.COOPERATE, Action.DEFECT):    cfg.S,
                (Action.DEFECT,    Action.COOPERATE): cfg.T,
                (Action.DEFECT,    Action.DEFECT):    cfg.P,
            }[(a, b)]
            payoff_b = {
                (Action.COOPERATE, Action.COOPERATE): cfg.R,
                (Action.COOPERATE, Action.DEFECT):    cfg.S,
                (Action.DEFECT,    Action.COOPERATE): cfg.T_b,
                (Action.DEFECT,    Action.DEFECT):    cfg.P,
            }[(b, a)]
            return [payoff_a, payoff_b]
        else:
            # N-person: reward based on number of cooperators
            n_coop = sum(1 for a in actions if a == Action.COOPERATE)
            payoffs = []
            for i, act in enumerate(actions):
                others_coop = n_coop - (1 if act == Action.COOPERATE else 0)
                if act == Action.COOPERATE:
                    r = cfg.S + (cfg.R - cfg.S) * others_coop / (self.n - 1)
                else:
                    r = cfg.P + (cfg.T - cfg.P) * others_coop / (self.n - 1)
                payoffs.append(r)
            return payoffs

    def step(self, actions: List[Action]) -> Tuple[List[str], List[float], List[str]]:
        """
        Execute one round.
        Returns: (new_states, rewards, next_states_for_sarsa)
        """
        # Apply noise
        noisy_actions = [self._apply_noise(a) for a in actions]
        # Compute payoffs
        rewards = self._compute_payoffs(noisy_actions)
        # Update history
        self.last_actions = noisy_actions
        self.history.append(tuple(noisy_actions))
        self.step_count += 1
        # New states
        new_states = [self._get_state(i) for i in range(self.n)]
        return new_states, rewards, new_states

    def get_all_states(self) -> List[str]:
        """Return all possible states for this config."""
        if self.n == 2:
            return ['CC', 'CD', 'DC', 'DD']
        else:
            return [f"{own}{k}" for own in ['C','D']
                    for k in range(self.n)]

    def cooperation_rate(self) -> float:
        if not self.history:
            return 0.0
        return np.mean([a == Action.COOPERATE
                        for round_ in self.history
                        for a in round_])

    def recent_cooperation_rate(self, window: int = 100) -> float:
        recent = self.history[-window:]
        if not recent:
            return 0.0
        return np.mean([a == Action.COOPERATE
                        for round_ in recent
                        for a in round_])


if __name__ == "__main__":
    env = PDEnvironment(VARIANTS["classic"])
    states = env.reset()
    print(f"Initial states: {states}")
    print(f"All possible states: {env.get_all_states()}")

    # Test one step
    actions = [Action.COOPERATE, Action.DEFECT]
    new_states, rewards, _ = env.step(actions)
    print(f"Actions: C, D → Rewards: {rewards} → New states: {new_states}")
    print("Environment OK.")
