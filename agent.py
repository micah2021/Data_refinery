"""
N-Step SARSA Agent
Implements n-step temporal difference learning (on-policy).
n=1: standard SARSA
n=2: your thesis (2-step SARSA)
n=inf: Monte Carlo
"""

import numpy as np
from collections import deque
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class Transition:
    """One step transition for the n-step buffer."""
    state:      str
    action:     int
    reward:     float
    next_state: str
    next_action: int


@dataclass
class AgentConfig:
    alpha:   float = 0.1    # Learning rate
    gamma:   float = 0.9    # Discount factor
    epsilon: float = 0.1    # Exploration rate
    n_step:  int   = 2      # Number of steps to look ahead
    epsilon_decay: float = 0.9995  # Decay per episode
    epsilon_min:   float = 0.01
    name: str = "Agent"


class NStepSARSA:
    """
    N-Step SARSA agent for the iterated Prisoner's Dilemma.
    
    Key equations:
      G(t:t+n) = sum_{k=0}^{n-1} gamma^k * R_{t+k+1}
                 + gamma^n * Q(S_{t+n}, A_{t+n})
      Q(S_t, A_t) <- Q(S_t, A_t) + alpha * [G(t:t+n) - Q(S_t, A_t)]
    """

    def __init__(self, states: List[str], config: AgentConfig):
        self.config   = config
        self.states   = states
        self.n_actions = 2   # cooperate=0, defect=1

        # Q-table: {state: [Q(C), Q(D)]}
        self.Q: Dict[str, np.ndarray] = {
            s: np.zeros(self.n_actions) for s in states
        }

        # N-step buffer
        self.buffer: deque = deque(maxlen=config.n_step + 1)

        # Tracking
        self.total_reward   = 0.0
        self.episode_reward = 0.0
        self.n_updates      = 0
        self.epsilon        = config.epsilon
        self.q_delta        = 0.0   # Last Q-value change magnitude

        # History for analysis
        self.action_history:  List[int]   = []
        self.reward_history:  List[float] = []
        self.coop_per_episode: List[float] = []

    def choose_action(self, state: str) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def store_transition(self, t: Transition):
        """Add transition to n-step buffer."""
        self.buffer.append(t)

    def update(self) -> float:
        """
        Perform n-step SARSA update.
        Returns the TD error magnitude.
        """
        n = self.config.n_step
        if len(self.buffer) < n:
            return 0.0

        # Get the oldest transition (what we're updating)
        t0 = self.buffer[0]

        # Compute n-step return G
        G = 0.0
        for k, trans in enumerate(self.buffer):
            G += (self.config.gamma ** k) * trans.reward

        # Bootstrap from the last transition in buffer
        last = self.buffer[-1]
        G += (self.config.gamma ** n) * self.Q[last.next_state][last.next_action]

        # Update Q(s0, a0)
        old_q = self.Q[t0.state][t0.action]
        td_error = G - old_q
        self.Q[t0.state][t0.action] += self.config.alpha * td_error
        self.q_delta = abs(td_error)
        self.n_updates += 1

        return abs(td_error)

    def end_of_episode(self):
        """Flush remaining buffer at episode end, decay epsilon."""
        # Drain buffer with no bootstrapping
        while len(self.buffer) > 0:
            t0 = self.buffer[0]
            G = sum((self.config.gamma ** k) * t.reward
                    for k, t in enumerate(self.buffer))
            old_q = self.Q[t0.state][t0.action]
            self.Q[t0.state][t0.action] += self.config.alpha * (G - old_q)
            self.q_delta = abs(G - old_q)
            self.buffer.popleft()

        # Epsilon decay
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay
        )

        # Track episode cooperation
        if self.action_history:
            ep_coop = self.action_history[-100:].count(0) / min(100, len(self.action_history))
            self.coop_per_episode.append(ep_coop)

    def record_step(self, action: int, reward: float):
        self.total_reward   += reward
        self.episode_reward += reward
        self.action_history.append(action)
        self.reward_history.append(reward)

    def cooperation_rate(self, window: int = 100) -> float:
        recent = self.action_history[-window:]
        if not recent:
            return 0.0
        return recent.count(0) / len(recent)

    def greedy_policy(self) -> Dict[str, str]:
        """Return current greedy policy (best action per state)."""
        return {
            s: 'C' if self.Q[s][0] >= self.Q[s][1] else 'D'
            for s in self.states
        }

    def q_table_str(self) -> str:
        lines = [f"\n{'='*50}",
                 f"  Q-Table: {self.config.name}",
                 f"{'='*50}",
                 f"  {'State':<6} {'Q(C)':>8} {'Q(D)':>8} {'Policy':>8}"]
        for s in self.states:
            qc, qd = self.Q[s]
            pol = 'COOP' if qc >= qd else 'DEFECT'
            lines.append(f"  {s:<6} {qc:>8.3f} {qd:>8.3f} {pol:>8}")
        lines.append(f"  epsilon: {self.epsilon:.4f}")
        lines.append(f"  updates: {self.n_updates}")
        return '\n'.join(lines)

    def reset_episode(self):
        self.episode_reward = 0.0
        self.buffer.clear()


class RandomAgent:
    """Baseline: random 50/50 agent."""
    def __init__(self, name="Random"):
        self.config = AgentConfig(name=name)
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0.0

    def choose_action(self, state): return np.random.randint(2)
    def store_transition(self, t): pass
    def update(self): return 0.0
    def end_of_episode(self): pass
    def record_step(self, a, r):
        self.action_history.append(a)
        self.reward_history.append(r)
        self.total_reward += r
    def cooperation_rate(self, window=100):
        return sum(1 for a in self.action_history[-window:] if a==0) / max(1,min(window,len(self.action_history)))
    def reset_episode(self): pass


class TitForTatAgent:
    """Classic TFT: cooperate first, then copy opponent's last move."""
    def __init__(self, name="TFT"):
        self.config = AgentConfig(name=name)
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0.0
        self._last_opp = 0  # cooperate first

    def choose_action(self, state):
        # State is 'CC','CD','DC','DD' — second char is opponent's last
        opp_last = 0 if state[1] == 'C' else 1
        return opp_last

    def store_transition(self, t): pass
    def update(self): return 0.0
    def end_of_episode(self): self._last_opp = 0
    def record_step(self, a, r):
        self.action_history.append(a)
        self.reward_history.append(r)
        self.total_reward += r
    def cooperation_rate(self, window=100):
        return sum(1 for a in self.action_history[-window:] if a==0) / max(1,min(window,len(self.action_history)))
    def reset_episode(self): pass


class AlwaysDefectAgent:
    """Baseline: always defect."""
    def __init__(self, name="AllD"):
        self.config = AgentConfig(name=name)
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0.0

    def choose_action(self, state): return 1
    def store_transition(self, t): pass
    def update(self): return 0.0
    def end_of_episode(self): pass
    def record_step(self, a, r):
        self.action_history.append(a)
        self.reward_history.append(r)
        self.total_reward += r
    def cooperation_rate(self, window=100): return 0.0
    def reset_episode(self): pass


if __name__ == "__main__":
    cfg = AgentConfig(alpha=0.1, gamma=0.9, epsilon=0.1, n_step=2, name="TestAgent")
    agent = NStepSARSA(states=['CC','CD','DC','DD'], config=cfg)
    print(agent.q_table_str())
    print("Agent OK.")
