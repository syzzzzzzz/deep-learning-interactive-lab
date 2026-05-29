"""
Reinforcement learning introduction page.

Run:
    streamlit run part6_universal_framework/reinforcement_learning.py
or:
    python main.py part6/reinforcement_learning
"""

from __future__ import annotations

import html
import math
import time
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import FancyArrowPatch, Rectangle


try:
    import sitecustomize

    configure_matplotlib = getattr(sitecustomize, "_configure_matplotlib", None)
    if callable(configure_matplotlib):
        configure_matplotlib()
except Exception:
    pass


INK = "#172026"
MUTED = "#596772"
LINE = "#d8dee3"
PAPER = "#fbfaf6"
TEAL = "#0f8b8d"
ROSE = "#bf3f5b"
AMBER = "#c4871f"
BLUE = "#3268a8"
GREEN = "#3f7d58"
VIOLET = "#7353ba"

Action = Literal[0, 1, 2, 3]
ACTIONS: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_NAMES = ("上", "右", "下", "左")
ARROWS = ("↑", "→", "↓", "←")


st.set_page_config(
    page_title="强化学习入门",
    layout="wide",
    initial_sidebar_state="auto",
)


def css() -> str:
    return """
    <style>
    :root {
        --ink: #172026;
        --muted: #596772;
        --line: #d8dee3;
        --paper: #fbfaf6;
        --teal: #0f8b8d;
        --rose: #bf3f5b;
        --amber: #c4871f;
        --blue: #3268a8;
        --green: #3f7d58;
        --violet: #7353ba;
    }
    .stApp {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(239,246,243,0.96) 100%),
            #fbfaf6;
        color: var(--ink);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.2rem;
    }
    h1, h2, h3 { letter-spacing: 0; }
    section[data-testid="stSidebar"] {
        background: #eef4f2;
        border-right: 1px solid var(--line);
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.82);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.72rem;
    }
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #172026;
        background: #172026;
        color: white;
        min-height: 2.45rem;
        font-weight: 700;
    }
    .stButton > button:hover {
        border-color: #0f8b8d;
        background: #0f8b8d;
        color: white;
    }
    .hero {
        border-bottom: 1px solid var(--line);
        padding-bottom: 0.95rem;
        margin-bottom: 0.95rem;
    }
    .hero h1 {
        font-size: clamp(2.05rem, 3vw, 3.35rem);
        line-height: 1.08;
        margin: 0;
    }
    .hero p {
        color: var(--muted);
        max-width: 980px;
        line-height: 1.75;
        margin: 0.45rem 0 0 0;
        font-size: 1.02rem;
    }
    .lesson-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.72rem;
        margin: 0.65rem 0 1rem 0;
    }
    .lesson-card {
        background: rgba(255,255,255,0.78);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.78rem 0.9rem;
        min-height: 112px;
    }
    .lesson-card strong {
        display: block;
        color: #1f2d35;
        margin-bottom: 0.35rem;
    }
    .lesson-card p {
        color: var(--muted);
        margin: 0;
        line-height: 1.62;
        font-size: 0.92rem;
    }
    .note {
        border-left: 4px solid var(--teal);
        background: rgba(255,255,255,0.74);
        border-radius: 0 8px 8px 0;
        padding: 0.72rem 0.9rem;
        color: #26343b;
        line-height: 1.7;
        margin: 0.35rem 0 0.85rem 0;
    }
    .mini-table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.25rem 0 0.85rem 0;
        font-size: 0.93rem;
    }
    .mini-table td {
        border-bottom: 1px solid rgba(216,222,227,0.9);
        padding: 0.5rem 0.38rem;
        color: var(--muted);
        vertical-align: top;
    }
    .mini-table td:first-child {
        color: var(--ink);
        font-weight: 700;
        width: 28%;
    }
    .code-box {
        background: rgba(23,32,38,0.94);
        color: #f6fbfc;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        font-family: Consolas, "Courier New", monospace;
        font-size: 0.9rem;
        line-height: 1.55;
        white-space: pre-wrap;
    }
    @media (max-width: 1000px) {
        .lesson-grid { grid-template-columns: 1fr; }
        .lesson-card { min-height: auto; }
    }
    </style>
    """


def e(text: str) -> str:
    return html.escape(text, quote=True)


def render_cards(cards: list[tuple[str, str]]) -> None:
    body = "".join(
        '<div class="lesson-card">'
        f"<strong>{e(title)}</strong>"
        f"<p>{e(description)}</p>"
        "</div>"
        for title, description in cards
    )
    st.markdown(f'<div class="lesson-grid">{body}</div>', unsafe_allow_html=True)


def render_note(text: str) -> None:
    st.markdown(f'<div class="note">{e(text)}</div>', unsafe_allow_html=True)


def render_table(rows: list[tuple[str, str]]) -> None:
    body = "".join(f"<tr><td>{e(left)}</td><td>{e(right)}</td></tr>" for left, right in rows)
    st.markdown(f'<table class="mini-table">{body}</table>', unsafe_allow_html=True)


def new_fig(width: float = 7.2, height: float = 4.4) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), dpi=120)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor("#fffdf8")
    return fig, ax


def close_and_show(fig: plt.Figure) -> None:
    st.pyplot(fig, clear_figure=True, width="stretch")
    plt.close(fig)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    window = max(1, min(window, len(values)))
    kernel = np.ones(window) / window
    left_pad = np.full(window - 1, values[0])
    padded = np.r_[left_pad, values]
    return np.convolve(padded, kernel, mode="valid")


@dataclass(frozen=True)
class BanditResult:
    rewards: np.ndarray
    choices: np.ndarray
    estimates: np.ndarray
    counts: np.ndarray
    true_means: np.ndarray
    optimal_arm: int


@st.cache_data(show_spinner=False)
def simulate_bandit(
    epsilon: float,
    steps: int,
    arms: int,
    reward_noise: float,
    seed: int,
) -> BanditResult:
    rng = np.random.default_rng(seed)
    true_means = rng.normal(0.0, 1.0, size=arms)
    optimal_arm = int(np.argmax(true_means))
    estimates = np.zeros(arms)
    counts = np.zeros(arms, dtype=int)
    rewards = np.zeros(steps)
    choices = np.zeros(steps, dtype=int)

    for step in range(steps):
        if rng.random() < epsilon:
            action = int(rng.integers(arms))
        else:
            best_value = estimates.max()
            best_actions = np.flatnonzero(np.isclose(estimates, best_value))
            action = int(rng.choice(best_actions))

        reward = rng.normal(true_means[action], reward_noise)
        counts[action] += 1
        estimates[action] += (reward - estimates[action]) / counts[action]
        rewards[step] = reward
        choices[step] = action

    return BanditResult(rewards, choices, estimates, counts, true_means, optimal_arm)


@dataclass
class GridWorld:
    width: int = 6
    height: int = 5
    start: tuple[int, int] = (4, 0)
    goal: tuple[int, int] = (0, 5)
    traps: tuple[tuple[int, int], ...] = ((1, 2), (2, 4), (3, 1))
    walls: tuple[tuple[int, int], ...] = ((1, 1), (2, 1), (3, 3))
    step_cost: float = -0.04
    goal_reward: float = 1.0
    trap_reward: float = -1.0

    @property
    def n_states(self) -> int:
        return self.width * self.height

    @property
    def n_actions(self) -> int:
        return 4

    def state_index(self, pos: tuple[int, int]) -> int:
        row, col = pos
        return row * self.width + col

    def state_pos(self, state: int) -> tuple[int, int]:
        return divmod(state, self.width)

    def reset(self) -> tuple[int, int]:
        return self.start

    def terminal(self, pos: tuple[int, int]) -> bool:
        return pos == self.goal or pos in self.traps

    def step(self, pos: tuple[int, int], action: Action) -> tuple[tuple[int, int], float, bool]:
        if self.terminal(pos):
            return pos, 0.0, True

        dr, dc = ACTIONS[action]
        row = int(np.clip(pos[0] + dr, 0, self.height - 1))
        col = int(np.clip(pos[1] + dc, 0, self.width - 1))
        next_pos = (row, col)
        if next_pos in self.walls:
            next_pos = pos

        if next_pos == self.goal:
            return next_pos, self.goal_reward, True
        if next_pos in self.traps:
            return next_pos, self.trap_reward, True
        return next_pos, self.step_cost, False


@dataclass(frozen=True)
class QLearningTrace:
    q_tables: np.ndarray
    rewards: np.ndarray
    lengths: np.ndarray
    success: np.ndarray
    paths: tuple[tuple[tuple[int, int], ...], ...]
    snapshots: tuple[int, ...]


def greedy_action(q_values: np.ndarray, rng: np.random.Generator) -> int:
    best_value = q_values.max()
    best_actions = np.flatnonzero(np.isclose(q_values, best_value))
    return int(rng.choice(best_actions))


def run_greedy_episode(env: GridWorld, q_table: np.ndarray, max_steps: int = 80) -> tuple[tuple[tuple[int, int], ...], float]:
    rng = np.random.default_rng(123)
    pos = env.reset()
    path = [pos]
    total_reward = 0.0
    for _ in range(max_steps):
        state = env.state_index(pos)
        action = greedy_action(q_table[state], rng)
        pos, reward, done = env.step(pos, action)  # type: ignore[arg-type]
        total_reward += reward
        path.append(pos)
        if done:
            break
    return tuple(path), total_reward


@st.cache_data(show_spinner=False)
def train_q_learning(
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    max_steps: int,
    seed: int,
) -> QLearningTrace:
    env = GridWorld()
    rng = np.random.default_rng(seed)
    q_table = np.zeros((env.n_states, env.n_actions), dtype=float)
    rewards = np.zeros(episodes)
    lengths = np.zeros(episodes, dtype=int)
    success = np.zeros(episodes, dtype=bool)

    snapshot_set = set(np.unique(np.linspace(0, episodes - 1, min(36, episodes), dtype=int)))
    q_tables: list[np.ndarray] = []
    paths: list[tuple[tuple[int, int], ...]] = []
    snapshots: list[int] = []

    for episode in range(episodes):
        pos = env.reset()
        total_reward = 0.0
        done = False

        for step in range(max_steps):
            state = env.state_index(pos)
            if rng.random() < epsilon:
                action = int(rng.integers(env.n_actions))
            else:
                action = greedy_action(q_table[state], rng)

            next_pos, reward, done = env.step(pos, action)  # type: ignore[arg-type]
            next_state = env.state_index(next_pos)
            target = reward
            if not done:
                target += gamma * np.max(q_table[next_state])
            q_table[state, action] += alpha * (target - q_table[state, action])

            pos = next_pos
            total_reward += reward
            if done:
                lengths[episode] = step + 1
                break

        if lengths[episode] == 0:
            lengths[episode] = max_steps
        rewards[episode] = total_reward
        success[episode] = pos == env.goal

        if episode in snapshot_set:
            q_copy = q_table.copy()
            q_tables.append(q_copy)
            paths.append(run_greedy_episode(env, q_copy)[0])
            snapshots.append(episode + 1)

    if not q_tables:
        q_tables.append(q_table.copy())
        paths.append(run_greedy_episode(env, q_table)[0])
        snapshots.append(episodes)

    return QLearningTrace(
        q_tables=np.array(q_tables),
        rewards=rewards,
        lengths=lengths,
        success=success,
        paths=tuple(paths),
        snapshots=tuple(snapshots),
    )


def draw_concept_graph() -> plt.Figure:
    fig, ax = new_fig(9.6, 5.4)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    nodes = {
        "智能体\nAgent": (1.7, 3.1, TEAL),
        "环境\nEnvironment": (8.0, 3.1, BLUE),
        "状态\nState": (4.9, 4.8, VIOLET),
        "动作\nAction": (4.9, 3.0, AMBER),
        "奖励\nReward": (4.9, 1.2, ROSE),
        "策略\nPolicy": (1.7, 1.1, GREEN),
        "价值函数\nValue / Q": (8.0, 1.1, "#6b5b95"),
    }
    for label, (x, y, color) in nodes.items():
        ax.scatter([x], [y], s=2400, color=color, alpha=0.92, edgecolor="white", linewidth=2.0, zorder=3)
        ax.text(x, y, label, ha="center", va="center", color="white", fontsize=12, weight="bold", zorder=4)

    arrows = [
        ("智能体\nAgent", "动作\nAction", "选择动作"),
        ("动作\nAction", "环境\nEnvironment", "作用于环境"),
        ("环境\nEnvironment", "状态\nState", "产生新状态"),
        ("环境\nEnvironment", "奖励\nReward", "给出反馈"),
        ("状态\nState", "智能体\nAgent", "观察"),
        ("奖励\nReward", "智能体\nAgent", "学习信号"),
        ("策略\nPolicy", "智能体\nAgent", "控制行为"),
        ("价值函数\nValue / Q", "策略\nPolicy", "改进策略"),
        ("奖励\nReward", "价值函数\nValue / Q", "估计长期回报"),
    ]
    for start, end, label in arrows:
        sx, sy, _ = nodes[start]
        ex, ey, _ = nodes[end]
        arrow = FancyArrowPatch(
            (sx, sy),
            (ex, ey),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.8,
            color=INK,
            alpha=0.72,
            shrinkA=38,
            shrinkB=38,
            connectionstyle="arc3,rad=0.08",
        )
        ax.add_patch(arrow)
        mx = sx * 0.45 + ex * 0.55
        my = sy * 0.45 + ey * 0.55
        ax.text(mx, my, label, color=MUTED, fontsize=9, ha="center", va="center", bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 1.5})

    ax.text(
        5,
        5.72,
        "强化学习的核心：用奖励信号把试错经验转化为更好的行为策略",
        ha="center",
        va="center",
        fontsize=14,
        color=INK,
        weight="bold",
    )
    fig.tight_layout()
    return fig


def plot_bandit(result: BanditResult, epsilon: float) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=120)
    fig.patch.set_facecolor(PAPER)

    steps = np.arange(1, len(result.rewards) + 1)
    axes[0].plot(steps, np.cumsum(result.rewards) / steps, color=TEAL, label="平均奖励")
    axes[0].plot(steps, rolling_mean(result.rewards, max(10, len(result.rewards) // 20)), color=ROSE, alpha=0.78, label="滑动平均")
    axes[0].axhline(result.true_means[result.optimal_arm], color=AMBER, linestyle="--", linewidth=1.7, label="最优臂期望")
    axes[0].set_title(f"探索率 epsilon = {epsilon:.2f}")
    axes[0].set_xlabel("尝试次数")
    axes[0].set_ylabel("奖励")
    axes[0].grid(True)
    axes[0].legend()

    x = np.arange(len(result.true_means))
    axes[1].bar(x - 0.18, result.true_means, width=0.36, color=BLUE, alpha=0.78, label="真实期望")
    axes[1].bar(x + 0.18, result.estimates, width=0.36, color=TEAL, alpha=0.86, label="估计值")
    axes[1].scatter([result.optimal_arm], [result.true_means[result.optimal_arm]], s=140, color=AMBER, edgecolor=INK, zorder=4, label="最优臂")
    for arm, count in enumerate(result.counts):
        axes[1].text(arm, min(result.true_means.min(), result.estimates.min()) - 0.22, str(count), ha="center", va="top", fontsize=8, color=MUTED)
    axes[1].set_title("每个臂的真实收益与学习估计")
    axes[1].set_xlabel("拉杆编号；底部数字是选择次数")
    axes[1].set_ylabel("期望奖励")
    axes[1].grid(True, axis="y")
    axes[1].legend()

    fig.tight_layout()
    return fig


def plot_q_rewards(rewards: np.ndarray, success: np.ndarray, upto: int | None = None) -> plt.Figure:
    upto = len(rewards) if upto is None else max(1, min(upto, len(rewards)))
    visible_rewards = rewards[:upto]
    visible_success = success[:upto].astype(float)
    fig, ax1 = new_fig(8.6, 3.7)
    x = np.arange(1, upto + 1)
    ax1.plot(x, visible_rewards, color=ROSE, alpha=0.34, linewidth=1.1, label="每回合奖励")
    ax1.plot(x, rolling_mean(visible_rewards, 25), color=ROSE, linewidth=2.6, label="奖励滑动平均")
    ax1.set_xlabel("训练回合")
    ax1.set_ylabel("奖励", color=ROSE)
    ax1.tick_params(axis="y", labelcolor=ROSE)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(x, rolling_mean(visible_success, 35), color=TEAL, linewidth=2.3, label="成功率滑动平均")
    ax2.set_ylabel("到达目标比例", color=TEAL)
    ax2.set_ylim(-0.04, 1.04)
    ax2.tick_params(axis="y", labelcolor=TEAL)

    lines, labels = ax1.get_legend_handles_labels()
    more_lines, more_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + more_lines, labels + more_labels, loc="lower right")
    ax1.set_title("训练奖励曲线实时更新视图")
    fig.tight_layout()
    return fig


def draw_gridworld(q_table: np.ndarray, path: tuple[tuple[int, int], ...], title: str) -> plt.Figure:
    env = GridWorld()
    fig, ax = new_fig(7.4, 5.4)
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13, weight="bold")

    visits: dict[tuple[int, int], int] = {}
    for pos in path:
        visits[pos] = visits.get(pos, 0) + 1

    values = np.max(q_table, axis=1).reshape(env.height, env.width)
    finite_values = values.copy()
    if np.allclose(finite_values.max(), finite_values.min()):
        finite_values = np.zeros_like(finite_values)
    else:
        finite_values = (finite_values - finite_values.min()) / (finite_values.max() - finite_values.min() + 1e-8)

    for row in range(env.height):
        for col in range(env.width):
            pos = (row, col)
            face = "#fffdf8"
            edge = LINE
            if pos in env.walls:
                face = "#29353b"
            elif pos == env.goal:
                face = "#dcefe6"
            elif pos in env.traps:
                face = "#f6d8de"
            else:
                intensity = finite_values[row, col]
                face = (1 - 0.18 * intensity, 1 - 0.04 * intensity, 1 - 0.22 * intensity)

            ax.add_patch(Rectangle((col, row), 1, 1, facecolor=face, edgecolor=edge, linewidth=1.2))

            if pos == env.start:
                ax.text(col + 0.5, row + 0.22, "S", ha="center", va="center", fontsize=10, color=INK, weight="bold")
            if pos == env.goal:
                ax.text(col + 0.5, row + 0.5, "G\n+1", ha="center", va="center", fontsize=11, color=GREEN, weight="bold")
            elif pos in env.traps:
                ax.text(col + 0.5, row + 0.5, "陷阱\n-1", ha="center", va="center", fontsize=10, color=ROSE, weight="bold")
            elif pos in env.walls:
                ax.text(col + 0.5, row + 0.5, "墙", ha="center", va="center", fontsize=10, color="white", weight="bold")
            else:
                state = env.state_index(pos)
                action = int(np.argmax(q_table[state]))
                if np.max(np.abs(q_table[state])) > 1e-8:
                    ax.text(col + 0.5, row + 0.58, ARROWS[action], ha="center", va="center", fontsize=17, color=INK, weight="bold")
                    ax.text(col + 0.5, row + 0.82, f"{np.max(q_table[state]):.2f}", ha="center", va="center", fontsize=7.5, color=MUTED)

            if pos in visits:
                ax.scatter(col + 0.5, row + 0.5, s=28 + 16 * min(visits[pos], 5), color=AMBER, alpha=0.42, edgecolor="none", zorder=4)

    if len(path) > 1:
        xs = [col + 0.5 for _, col in path]
        ys = [row + 0.5 for row, _ in path]
        ax.plot(xs, ys, color=AMBER, linewidth=3.0, alpha=0.86, zorder=5)
        ax.scatter(xs[-1], ys[-1], s=160, color=TEAL, edgecolor="white", linewidth=1.8, zorder=6)

    fig.tight_layout()
    return fig


def draw_policy_gradient_diagram(step_size: float) -> plt.Figure:
    fig, ax = new_fig(9.3, 4.8)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    stages = [
        ("策略 π(a|s)", "给每个动作分配概率", TEAL, 1.2),
        ("采样轨迹 τ", "按当前策略行动", BLUE, 3.0),
        ("累计回报 G", "把后续奖励加总", AMBER, 4.9),
        ("梯度估计", "好动作概率上调", ROSE, 6.9),
        ("更新参数 theta", "theta <- theta + alpha grad J", GREEN, 8.7),
    ]
    for title, subtitle, color, x in stages:
        ax.add_patch(Rectangle((x - 0.82, 2.0), 1.64, 1.35, facecolor=color, edgecolor="white", linewidth=1.8, alpha=0.92))
        ax.text(x, 2.77, title, ha="center", va="center", color="white", fontsize=11, weight="bold")
        ax.text(x, 2.34, subtitle, ha="center", va="center", color="white", fontsize=8.5)

    for left, right in zip(stages, stages[1:]):
        arrow = FancyArrowPatch(
            (left[3] + 0.86, 2.68),
            (right[3] - 0.86, 2.68),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=2.0,
            color=INK,
            alpha=0.72,
        )
        ax.add_patch(arrow)

    ax.text(
        5,
        4.42,
        "策略梯度不先学 Q 表，而是直接让高回报动作的概率变大",
        ha="center",
        va="center",
        fontsize=14,
        color=INK,
        weight="bold",
    )

    probs_before = np.array([0.25, 0.25, 0.25, 0.25])
    returns = np.array([1.2, -0.4, 0.2, -0.7])
    logits = np.log(probs_before) + step_size * returns
    probs_after = np.exp(logits) / np.exp(logits).sum()

    inset = ax.inset_axes([0.16, 0.06, 0.68, 0.28])
    x = np.arange(4)
    inset.bar(x - 0.17, probs_before, width=0.34, color=LINE, label="更新前")
    inset.bar(x + 0.17, probs_after, width=0.34, color=TEAL, label="更新后")
    inset.set_xticks(x, ACTION_NAMES)
    inset.set_ylim(0, max(0.62, probs_after.max() + 0.12))
    inset.set_ylabel("动作概率")
    inset.set_title("一次 REINFORCE 风格更新的直觉")
    inset.grid(True, axis="y")
    inset.legend(ncol=2, loc="upper right")

    fig.tight_layout()
    return fig


def run_manual_policy(env: GridWorld, policy: str, max_steps: int, seed: int) -> tuple[list[tuple[int, int]], float, bool]:
    rng = np.random.default_rng(seed)
    pos = env.reset()
    path = [pos]
    total_reward = 0.0
    done = False
    for _ in range(max_steps):
        if policy == "随机策略":
            action = int(rng.integers(env.n_actions))
        elif policy == "总是向右优先":
            action = 1 if pos[1] < env.goal[1] else 0
        else:
            if pos[0] > env.goal[0]:
                action = 0
            elif pos[1] < env.goal[1]:
                action = 1
            else:
                action = int(rng.integers(env.n_actions))
        pos, reward, done = env.step(pos, action)  # type: ignore[arg-type]
        total_reward += reward
        path.append(pos)
        if done:
            break
    return path, total_reward, done


def draw_environment_demo(path: list[tuple[int, int]], title: str) -> plt.Figure:
    env = GridWorld()
    q_table = np.zeros((env.n_states, env.n_actions))
    return draw_gridworld(q_table, tuple(path), title)


st.markdown(css(), unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero">
      <h1>强化学习入门模块</h1>
      <p>
      强化学习研究的是智能体如何通过试错学会行动：它不像监督学习那样每一步都有标准答案，
      而是在环境反馈的奖励里逐渐形成策略。这个页面用一个多臂赌博机和一个纯 Python 网格世界，
      把探索、利用、Q-Learning、策略梯度和奖励曲线连起来。
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("全局设置")
    seed = st.number_input("随机种子", min_value=0, max_value=9999, value=17, step=1)
    st.caption("同一个随机种子会复现赌博机收益、网格世界探索路径和训练结果。")
    st.divider()
    st.caption("所有环境和算法都是本页纯 Python / NumPy 实现，没有依赖 gym。图表使用 Matplotlib 渲染。")


tabs = st.tabs(["核心概念", "多臂赌博机", "Q-Learning 网格世界", "策略梯度", "环境 Demo"])

with tabs[0]:
    render_cards(
        [
            ("智能体与环境", "智能体选择动作，环境返回新状态和奖励。学习问题就藏在这个闭环里。"),
            ("探索与利用", "利用会选择当前看起来最好的动作；探索会主动尝试不确定的动作。"),
            ("价值与策略", "价值函数估计长期回报，策略负责把状态映射为动作或动作概率。"),
        ]
    )
    close_and_show(draw_concept_graph())
    render_table(
        [
            ("状态 s", "智能体看到的局面。例如网格中的当前位置。"),
            ("动作 a", "智能体可以做的选择。例如上、右、下、左。"),
            ("奖励 r", "环境给出的即时反馈。目标格是正奖励，陷阱是负奖励。"),
            ("回报 G", "从现在开始能拿到的折扣奖励总和。强化学习真正优化的是长期回报。"),
            ("策略 π", "行动规则。可以是确定性的，也可以是每个动作一个概率。"),
        ]
    )

with tabs[1]:
    render_cards(
        [
            ("问题", "有多台老虎机，每台的平均收益不同，但真实收益一开始未知。"),
            ("epsilon-greedy", "以 1-epsilon 的概率选当前估计最好的臂，以 epsilon 的概率随机探索。"),
            ("观察重点", "epsilon 太小容易早早押错；epsilon 太大又会长期浪费在差臂上。"),
        ]
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        epsilon = st.slider("探索率 epsilon", 0.0, 1.0, 0.12, 0.01)
    with c2:
        bandit_steps = st.slider("尝试次数", 100, 3000, 900, 100)
    with c3:
        arms = st.slider("拉杆数量", 3, 12, 6, 1)

    bandit = simulate_bandit(epsilon, bandit_steps, arms, 1.0, int(seed))
    optimal_rate = float(np.mean(bandit.choices == bandit.optimal_arm))
    left, mid, right = st.columns(3)
    left.metric("累计平均奖励", f"{np.mean(bandit.rewards):.3f}")
    mid.metric("选择最优臂比例", f"{optimal_rate:.1%}")
    right.metric("当前估计最优臂", f"{int(np.argmax(bandit.estimates))}")
    close_and_show(plot_bandit(bandit, epsilon))
    render_note("多臂赌博机是强化学习最小的探索-利用实验。它没有状态转移，只有动作和奖励，所以适合先理解 epsilon 的作用。")

with tabs[2]:
    render_cards(
        [
            ("环境", "从左下角出发，到右上角目标格；墙不能穿过，陷阱会立刻结束回合。"),
            ("Q-Learning", "用 Q(s,a) 估计在状态 s 做动作 a 后的长期价值，边试错边更新表格。"),
            ("动画", "拖动训练快照可以看到策略箭头和贪心路径如何从混乱变得稳定。"),
        ]
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        episodes = st.slider("训练回合", 50, 1200, 420, 50)
    with c2:
        q_epsilon = st.slider("Q-Learning epsilon", 0.0, 0.8, 0.18, 0.01)
    with c3:
        alpha = st.slider("学习率 alpha", 0.02, 0.8, 0.22, 0.02)
    with c4:
        gamma = st.slider("折扣因子 gamma", 0.50, 0.99, 0.92, 0.01)

    trace = train_q_learning(episodes, alpha, gamma, q_epsilon, 80, int(seed))
    final_success = float(np.mean(trace.success[-min(80, len(trace.success)) :]))
    final_reward = float(np.mean(trace.rewards[-min(80, len(trace.rewards)) :]))
    final_length = float(np.mean(trace.lengths[-min(80, len(trace.lengths)) :]))
    m1, m2, m3 = st.columns(3)
    m1.metric("最近成功率", f"{final_success:.1%}")
    m2.metric("最近平均奖励", f"{final_reward:.3f}")
    m3.metric("最近平均步数", f"{final_length:.1f}")

    left, right = st.columns([0.56, 0.44])
    with left:
        frame = st.slider("学习动画帧", 0, len(trace.snapshots) - 1, len(trace.snapshots) - 1, 1)
        title = f"第 {trace.snapshots[frame]} 回合后的贪心策略与路径"
        close_and_show(draw_gridworld(trace.q_tables[frame], trace.paths[frame], title))
    with right:
        close_and_show(plot_q_rewards(trace.rewards, trace.success))
        render_table(
            [
                ("更新公式", "Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]"),
                ("alpha", "新经验覆盖旧估计的速度。太大容易震荡，太小学习很慢。"),
                ("gamma", "未来奖励的重要性。越接近 1，越看重长期路径。"),
                ("epsilon", "探索概率。训练早期通常需要探索，后期可以逐步降低。"),
            ]
        )

    st.subheader("实时训练奖励曲线")
    live_col, text_col = st.columns([0.58, 0.42])
    with live_col:
        if st.button("播放一次实时更新", width="stretch"):
            plot_slot = st.empty()
            progress = st.progress(0)
            chunk_count = 18
            for index, upto in enumerate(np.linspace(5, len(trace.rewards), chunk_count, dtype=int), 1):
                fig = plot_q_rewards(trace.rewards, trace.success, int(upto))
                plot_slot.pyplot(fig, clear_figure=True, width="stretch")
                plt.close(fig)
                progress.progress(index / chunk_count)
                time.sleep(0.055)
        else:
            st.caption("点击按钮后，奖励曲线会按训练进度逐段刷新。")
    with text_col:
        render_note("这段实时更新复用同一批训练结果逐段播放，避免每次刷新都重新跑完整训练。曲线从噪声很大的单回合奖励，逐渐收敛到更稳定的成功路径。")

with tabs[3]:
    render_cards(
        [
            ("和 Q-Learning 的差别", "Q-Learning 先学动作价值；策略梯度直接调整动作概率。"),
            ("核心思想", "如果一条轨迹回报高，就提高这条轨迹里动作的对数概率。"),
            ("适用场景", "动作连续、策略需要随机性、或很难维护完整 Q 表时，策略梯度更自然。"),
        ]
    )
    step_size = st.slider("概念更新强度", 0.0, 1.5, 0.55, 0.05)
    close_and_show(draw_policy_gradient_diagram(step_size))
    render_table(
        [
            ("目标函数", "最大化期望回报 J(θ)=E[G]。θ 是策略网络或策略函数的参数。"),
            ("REINFORCE", "用采样轨迹估计 ∇J(θ)。高回报动作概率上调，低回报动作概率下调。"),
            ("baseline", "减去一个基线可以降低梯度方差，不改变期望梯度方向。"),
            ("Actor-Critic", "Actor 学策略，Critic 学价值估计，用价值估计来稳定策略更新。"),
        ]
    )
    st.markdown(
        '<div class="code-box">'
        'for trajectory in collect(policy):\n'
        '    G = discounted_return(trajectory.rewards)\n'
        '    loss = -sum(log_prob(action) * (G - baseline))\n'
        '    update(policy_parameters, gradient(loss))'
        "</div>",
        unsafe_allow_html=True,
    )

with tabs[4]:
    render_cards(
        [
            ("纯 Python 环境", "GridWorld 类只实现 reset、step、状态索引和终止判断。"),
            ("不依赖 gym", "这足够支撑 Q-Learning、随机策略、启发式策略和手写环境测试。"),
            ("可扩展", "把坐标换成图像、把动作换成连续控制，就是更复杂环境的同一接口思想。"),
        ]
    )
    env = GridWorld()
    c1, c2, c3 = st.columns(3)
    with c1:
        demo_policy = st.selectbox("策略", ["随机策略", "总是向右优先", "靠近目标的贪心启发式"], index=2)
    with c2:
        demo_steps = st.slider("最多步数", 5, 80, 28, 1)
    with c3:
        demo_seed = st.number_input("Demo 种子", min_value=0, max_value=9999, value=int(seed), step=1)

    path, total_reward, done = run_manual_policy(env, demo_policy, demo_steps, int(demo_seed))
    d1, d2, d3 = st.columns(3)
    d1.metric("路径长度", len(path) - 1)
    d2.metric("累计奖励", f"{total_reward:.2f}")
    d3.metric("是否终止", "是" if done else "否")
    left, right = st.columns([0.56, 0.44])
    with left:
        close_and_show(draw_environment_demo(path, f"{demo_policy} 的一次环境交互"))
    with right:
        render_table(
            [
                ("reset()", "返回起点状态。这里是左下角坐标。"),
                ("step(action)", "接收动作，返回 next_state、reward、done。"),
                ("done", "到达目标或踩到陷阱时为 True，当前回合结束。"),
                ("奖励设计", f"普通移动 {env.step_cost:+.2f}，目标 {env.goal_reward:+.1f}，陷阱 {env.trap_reward:+.1f}。"),
            ]
        )
        code = (
            "env = GridWorld()\n"
            "state = env.reset()\n"
            "done = False\n"
            "while not done:\n"
            "    action = policy(state)\n"
            "    state, reward, done = env.step(state, action)"
        )
        st.markdown(f'<div class="code-box">{e(code)}</div>', unsafe_allow_html=True)
