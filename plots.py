"""
Visualization Module
Generates all plots for the research project.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional
import os

# Style
plt.rcParams.update({
    'font.family':      'monospace',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   '--',
    'figure.facecolor': 'white',
    'axes.facecolor':   '#FAFAFA',
})

COLORS = {
    'coop':    '#1D9E75',
    'defect':  '#D85A30',
    'neutral': '#4A7BB7',
    'accent':  '#BA7517',
    'light':   '#E8E8E8',
}

os.makedirs("plots", exist_ok=True)


def smooth(data: List[float], window: int = 50) -> np.ndarray:
    """Moving average smoothing."""
    if len(data) < window:
        return np.array(data)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot_cooperation_curve(result, save: bool = True) -> plt.Figure:
    """Plot cooperation rate over episodes."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"2-Step SARSA — Prisoner's Dilemma\nVariant: {result.config.variant.upper()}  "
                 f"|  n={result.config.n_step}  |  α={result.config.alpha}  "
                 f"|  γ={result.config.gamma}  |  ε={result.config.epsilon}",
                 fontsize=13, fontweight='bold', y=0.98)

    # ── Panel 1: Cooperation Rate ──────────────────────────────
    ax = axes[0]
    coop = np.array(result.coop_history)
    eps  = np.arange(len(coop))

    ax.fill_between(eps, smooth(coop, 20), alpha=0.15, color=COLORS['coop'])
    ax.plot(eps, smooth(coop, 50), color=COLORS['coop'], lw=2, label='Cooperation Rate (smoothed)')
    ax.axhline(0.5, color='gray', lw=0.8, ls=':', label='50% baseline')
    ax.axhline(result.final_coop_rate, color=COLORS['accent'], lw=1.2, ls='--',
               label=f'Final: {result.final_coop_rate*100:.1f}%')

    if result.converged:
        ax.axvline(result.convergence_ep, color=COLORS['defect'], lw=1, ls=':',
                   label=f'Converged @ ep {result.convergence_ep}')

    ax.set_ylabel('Cooperation Rate', fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title('Cooperation Rate Over Training', fontsize=11)

    # ── Panel 2: Rewards ───────────────────────────────────────
    ax2 = axes[1]
    rew_a = np.array(result.reward_a_history)
    rew_b = np.array(result.reward_b_history)

    ax2.plot(smooth(rew_a, 50), color=COLORS['coop'],   lw=2, label='Agent A avg reward')
    ax2.plot(smooth(rew_b, 50), color=COLORS['neutral'], lw=2, label='Agent B avg reward', ls='--')
    ax2.axhline(result.config.__class__.__name__ and 1.0, color='gray', lw=0.8, ls=':')

    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Avg Reward / Step', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_title('Average Reward Per Step', fontsize=11)

    plt.tight_layout()
    if save:
        path = f"plots/coop_curve_{result.config.variant}_n{result.config.n_step}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [✓] Saved → {path}")
    return fig


def plot_qtable_heatmap(result, save: bool = True) -> plt.Figure:
    """Visualize Q-tables as heatmaps."""
    states  = list(result.q_table_a.keys())
    actions = ['Cooperate', 'Defect']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Q-Table Heatmap — {result.config.variant.upper()} PD  "
                 f"(n={result.config.n_step})", fontsize=13, fontweight='bold')

    cmap = LinearSegmentedColormap.from_list('pd', [COLORS['defect'], 'white', COLORS['coop']])

    for idx, (agent_name, q_data) in enumerate([
        ('Agent A', result.q_table_a),
        ('Agent B', result.q_table_b)
    ]):
        ax = axes[idx]
        matrix = np.array([q_data[s] for s in states])
        vmax = np.abs(matrix).max() + 0.1

        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)

        # Labels
        ax.set_xticks(range(len(actions)))
        ax.set_yticks(range(len(states)))
        ax.set_xticklabels(actions, fontsize=11)
        ax.set_yticklabels(states, fontsize=11)

        # Values in cells
        for i, s in enumerate(states):
            for j in range(2):
                val  = matrix[i, j]
                best = np.argmax(matrix[i])
                ax.text(j, i, f'{val:.2f}',
                        ha='center', va='center', fontsize=10,
                        fontweight='bold' if j == best else 'normal',
                        color='white' if abs(val) > vmax * 0.6 else 'black')

        # Policy arrow
        for i, s in enumerate(states):
            best = np.argmax(matrix[i])
            ax.add_patch(plt.Rectangle((best - 0.48, i - 0.48), 0.96, 0.96,
                                       fill=False, edgecolor=COLORS['accent'],
                                       linewidth=2.5))

        ax.set_title(f'{agent_name}  (policy: {result.policy_a if idx==0 else result.policy_b})',
                     fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Q-value')

    plt.tight_layout()
    if save:
        path = f"plots/qtable_{result.config.variant}_n{result.config.n_step}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [✓] Saved → {path}")
    return fig


def plot_nstep_comparison(comparison_results: Dict[int, List[float]],
                          variant: str = "classic",
                          save: bool = True) -> plt.Figure:
    """Bar chart comparing cooperation rates across n-step values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"N-Step SARSA Comparison — {variant.upper()} PD",
                 fontsize=13, fontweight='bold')

    n_steps = sorted(comparison_results.keys())
    means   = [np.mean(comparison_results[n]) * 100 for n in n_steps]
    stds    = [np.std(comparison_results[n]) * 100  for n in n_steps]
    colors  = [COLORS['coop'] if n == 2 else COLORS['neutral'] for n in n_steps]

    bars = ax.bar(range(len(n_steps)), means, yerr=stds, capsize=6,
                  color=colors, edgecolor='white', linewidth=1.2,
                  error_kw={'linewidth': 1.5})

    # Highlight n=2 (your thesis)
    ax.bar(n_steps.index(2), means[n_steps.index(2)],
           color=COLORS['coop'], edgecolor=COLORS['accent'],
           linewidth=2.5, label='n=2 (your thesis)')

    # Value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(n_steps)))
    ax.set_xticklabels([f'n={n}' for n in n_steps], fontsize=12)
    ax.set_ylabel('Final Cooperation Rate (%)', fontsize=12)
    ax.set_ylim(0, 110)
    ax.axhline(50, color='gray', lw=0.8, ls=':', label='50% baseline')
    ax.legend(fontsize=10)
    ax.set_title(f'Cooperation Rate by N-Step Value\n(mean ± std over {len(list(comparison_results.values())[0])} seeds)',
                 fontsize=11)

    plt.tight_layout()
    if save:
        path = f"plots/nstep_comparison_{variant}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [✓] Saved → {path}")
    return fig


def plot_variant_comparison(variant_results: Dict[str, object],
                            save: bool = True) -> plt.Figure:
    """Compare cooperation rates across PD variants."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("2-Step SARSA Across PD Variants", fontsize=13, fontweight='bold')

    variants = list(variant_results.keys())
    coop_rates = [variant_results[v].final_coop_rate * 100 for v in variants]
    colors_v   = [COLORS['coop'] if r > 50 else COLORS['defect'] for r in coop_rates]

    # Bar chart
    ax = axes[0]
    bars = ax.bar(range(len(variants)), coop_rates, color=colors_v, edgecolor='white', linewidth=1.2)
    for bar, rate in zip(bars, coop_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, fontsize=11)
    ax.set_ylabel('Final Cooperation Rate (%)', fontsize=11)
    ax.set_ylim(0, 110)
    ax.axhline(50, color='gray', lw=0.8, ls=':')
    ax.set_title('Cooperation Rate by Variant', fontsize=11)

    # Learning curves overlay
    ax2 = axes[1]
    cmap_list = plt.cm.tab10(np.linspace(0, 1, len(variants)))
    for (v, color) in zip(variants, cmap_list):
        res  = variant_results[v]
        data = smooth(res.coop_history, 50)
        ax2.plot(data, lw=2, label=v, color=color)
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Cooperation Rate', fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.axhline(0.5, color='gray', lw=0.8, ls=':')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.set_title('Learning Curves by Variant', fontsize=11)

    plt.tight_layout()
    if save:
        path = "plots/variant_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [✓] Saved → {path}")
    return fig


def plot_action_heatmap(action_history_a: List[int],
                        action_history_b: List[int],
                        title: str = "Action History",
                        save: bool = True) -> plt.Figure:
    """Show action sequences as colored heatmap strips."""
    n   = min(500, len(action_history_a))
    fig, axes = plt.subplots(2, 1, figsize=(14, 4))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    for ax, hist, name in zip(axes,
                               [action_history_a[-n:], action_history_b[-n:]],
                               ['Agent A', 'Agent B']):
        data = np.array(hist).reshape(1, -1)
        ax.imshow(data, cmap=LinearSegmentedColormap.from_list('ca', [COLORS['coop'], COLORS['defect']]),
                  aspect='auto', vmin=0, vmax=1)
        ax.set_yticks([0])
        ax.set_yticklabels([name], fontsize=10)
        ax.set_xlabel('Step (last 500)', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save:
        path = f"plots/action_history.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [✓] Saved → {path}")
    return fig


if __name__ == "__main__":
    print("Visualization module loaded OK.")
    print("Run main.py to generate all plots.")
