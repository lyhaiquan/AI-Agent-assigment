"""
AGENT02: Reinforcement-Learning-Based Agent vs Rule-Based Agent
================================================================

Course: Intelligent System Development

Run:
    python agent02_experiment.py

Dependencies:
    numpy
    matplotlib

Outputs (agent02_results/):
    Reference outputs (unchanged from the provided code):
        baseline_episode_rewards.csv
        baseline_evaluation.csv
        baseline_evaluation_episodes.csv
        learning_rate_comparison.csv
        exploration_comparison.csv
        multi_seed_summary.csv
        q_tables.txt
        episode_rewards.png
        learning_rate_comparison.png
        exploration_comparison.png

    Added outputs (student extensions, see section "EXTENSIONS"):
        rule_vs_rule_evaluation.csv        fair rule-based reference in slot 0
        learning_rate_curves.png           training curves for alpha in {0.01, 0.10, 0.50}
        exploration_curves.png             training curves for exploration settings
        exploration_training_vs_eval.csv  training reward vs greedy evaluation reward
        results_table.csv                  mean / std / discount rate over 5 seeds
        results_table.png
        two_learners.csv                   two simultaneous Q-learners (Q8)
        two_learners_curves.png
        two_learners_q_tables.txt
        modification_experiment.csv        modified environments (Q9)
        modification_experiment.png
        expected_rewards.csv               analytic one-step expected rewards

Agents:
    1. RL-based agent: tabular Q-learning
    2. Rule-based agent: fixed if-else policy

The RL agent is trained against a fixed rule-based opponent.
Evaluation freezes the learned Q-table and disables exploration.

Changes with respect to the reference implementation
(all documented, the reference baseline is reproduced exactly):
    - Config gains `demand_persistence` (default None = i.i.d. demand,
      identical to the reference). When set, demand follows a two-state
      Markov chain that keeps the current level with that probability.
    - ECommerceEnvironment uses the persistence parameter only when it
      is not None, so the random stream of the baseline is unchanged.
    - New opponents for robustness tests: AlwaysDiscountAgent,
      NeverDiscountAgent.
    - evaluate_pair accepts any policy in slot 0 and any opponent in
      slot 1, so the rule-based agent can be evaluated in the same slot,
      against the same opponent and with the same environment seed.
    - New experiments E-H at the end of the file.
"""

from dataclasses import dataclass, replace
from pathlib import Path
import csv

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

@dataclass
class Config:
    episodes: int = 5000
    steps_per_episode: int = 20

    alpha: float = 0.10
    gamma: float = 0.90

    epsilon_start: float = 0.20
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.995

    price: float = 10.0
    unit_cost: float = 5.0
    discount_rate: float = 0.20

    low_demand_customers: int = 10
    high_demand_customers: int = 20

    # Extension: None = i.i.d. demand (reference behaviour).
    demand_persistence: float | None = None

    seed: int = 42
    evaluation_episodes: int = 500

    output_dir: str = "agent02_results"


State = tuple[int, int]

SEEDS = [11, 22, 33, 44, 55]


# ============================================================
# 2. E-COMMERCE ENVIRONMENT
# ============================================================

class ECommerceEnvironment:
    """
    Two agents compete in a simplified e-commerce market.

    State:
        (demand, competitor_previous_action)

        demand:
            0 = low demand
            1 = high demand

        competitor_previous_action:
            0 = no discount
            1 = discount

    Action:
        0 = no discount
        1 = discount

    Reward:
        Profit = sales * profit_per_sale
    """

    def __init__(self, config, seed=None):
        self.config = config
        actual_seed = config.seed if seed is None else seed
        self.rng = np.random.default_rng(actual_seed)

        self.demand = 0
        self.previous_actions = [0, 0]

    def reset(self):
        self.demand = int(self.rng.integers(0, 2))
        self.previous_actions = [0, 0]
        return self.get_states()

    def get_states(self):
        return [
            (self.demand, self.previous_actions[1]),
            (self.demand, self.previous_actions[0])
        ]

    def next_demand(self):
        persistence = self.config.demand_persistence

        # Reference model: demand is i.i.d. with P(high) = 0.5.
        if persistence is None:
            return int(self.rng.integers(0, 2))

        # Extension: two-state Markov chain.
        if self.rng.random() < persistence:
            return self.demand

        return 1 - self.demand

    def step(self, actions):
        if len(actions) != 2:
            raise ValueError("Exactly two actions are required.")

        if any(action not in (0, 1) for action in actions):
            raise ValueError("Actions must be 0 or 1.")

        rewards = []
        sales_list = []
        conversion_list = []

        if self.demand == 0:
            customers = self.config.low_demand_customers
        else:
            customers = self.config.high_demand_customers

        # Both actions are selected before this function is called.
        # Neither agent observes the other's current action beforehand.
        for i in range(2):
            own_action = actions[i]
            competitor_action = actions[1 - i]

            # Illustrative conversion probability.
            conversion = (
                0.30
                + 0.25 * self.demand
                + 0.15 * own_action
                - 0.10 * competitor_action
            )

            conversion = float(np.clip(
                conversion, 0.05, 0.95
            ))

            # Stochastic number of sales.
            sales = int(self.rng.binomial(
                customers, conversion
            ))

            # Apply a discount only when action = 1.
            effective_price = self.config.price * (
                1.0 - self.config.discount_rate * own_action
            )

            profit_per_sale = (
                effective_price - self.config.unit_cost
            )

            profit = sales * profit_per_sale

            rewards.append(float(profit))
            sales_list.append(sales)
            conversion_list.append(conversion)

        # Store current actions for the next observation.
        self.previous_actions = list(actions)

        # Demand evolves independently of current actions.
        self.demand = self.next_demand()

        next_states = self.get_states()

        info = {
            "sales": sales_list,
            "conversion": conversion_list,
            "customers": customers
        }

        return next_states, rewards, info


# ============================================================
# 3. REINFORCEMENT-LEARNING-BASED AGENT
# ============================================================

class QLearningAgent:
    """
    Tabular Q-learning agent.

    Q[demand, competitor_previous_action, action]

    There are:
        2 demand states
        2 competitor-action states
        2 possible actions

    Total Q-values = 2 * 2 * 2 = 8
    """

    def __init__(
        self,
        alpha=0.10,
        gamma=0.90,
        epsilon=0.20,
        seed=123
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.rng = np.random.default_rng(seed)

        self.q = np.zeros((2, 2, 2), dtype=float)

    def act(self, state, explore=True):
        demand, competitor_previous_action = state

        # Exploration.
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, 2))

        # Exploitation.
        values = self.q[demand, competitor_previous_action]

        # Random tie-breaking when action values are equal.
        best_actions = np.flatnonzero(
            np.isclose(values, np.max(values))
        )

        return int(self.rng.choice(best_actions))

    def learn(
        self,
        state,
        action,
        reward,
        next_state,
        terminal=False
    ):
        demand, competitor_previous_action = state
        next_demand, next_competitor_action = next_state

        old_value = self.q[
            demand, competitor_previous_action, action
        ]

        if terminal:
            target = reward
        else:
            best_future_value = np.max(
                self.q[next_demand, next_competitor_action]
            )

            target = (
                reward + self.gamma * best_future_value
            )

        # Q-learning update.
        self.q[
            demand, competitor_previous_action, action
        ] = old_value + self.alpha * (
            target - old_value
        )

    def greedy_policy(self):
        policy = {}

        for demand in range(2):
            for competitor_action in range(2):
                state = (demand, competitor_action)

                policy[state] = int(
                    np.argmax(self.q[demand, competitor_action])
                )

        return policy


# ============================================================
# 4. RULE-BASED AGENT (+ fixed opponents used in extensions)
# ============================================================

class RuleBasedAgent:
    """
    Fixed policy:

    Rule 1:
        Low demand -> no discount.

    Rule 2:
        High demand and competitor previously discounted
        -> offer a discount.

    Rule 3:
        High demand and competitor did not previously discount
        -> no discount.

    This agent does not learn from rewards.
    """

    def act(self, state, explore=False):
        demand, competitor_previous_action = state

        if demand == 0:
            return 0

        elif demand == 1 and competitor_previous_action == 1:
            return 1

        else:
            return 0


class AlwaysDiscountAgent:
    """Extension: aggressive opponent that discounts in every state."""

    def act(self, state, explore=False):
        return 1


class NeverDiscountAgent:
    """Extension: passive opponent that never discounts."""

    def act(self, state, explore=False):
        return 0


# ============================================================
# 5. TRAIN THE RL AGENT
# ============================================================

def train_rl_against_rule(
    config,
    seed,
    alpha=None,
    epsilon_start=None,
    epsilon_min=None,
    epsilon_decay=None,
    opponent=None
):
    """
    Train one RL agent against a fixed rule-based opponent.

    The opponent's policy remains unchanged during training.
    `opponent` (extension) defaults to the reference RuleBasedAgent.
    """

    alpha = config.alpha if alpha is None else alpha

    epsilon_start = (
        config.epsilon_start
        if epsilon_start is None
        else epsilon_start
    )

    epsilon_min = (
        config.epsilon_min
        if epsilon_min is None
        else epsilon_min
    )

    epsilon_decay = (
        config.epsilon_decay
        if epsilon_decay is None
        else epsilon_decay
    )

    env = ECommerceEnvironment(config, seed=seed)

    rl_agent = QLearningAgent(
        alpha=alpha,
        gamma=config.gamma,
        epsilon=epsilon_start,
        seed=seed + 10_000
    )

    rule_agent = RuleBasedAgent() if opponent is None else opponent

    episode_results = []

    for episode in range(config.episodes):

        # Decrease exploration as training progresses.
        rl_agent.epsilon = max(
            epsilon_min,
            epsilon_start * epsilon_decay ** episode
        )

        states = env.reset()

        episode_rewards = [0.0, 0.0]
        episode_discounts = 0

        for step in range(config.steps_per_episode):

            # Both agents choose actions before the environment changes.
            rl_action = rl_agent.act(
                states[0], explore=True
            )

            rule_action = rule_agent.act(states[1])

            actions = [rl_action, rule_action]

            next_states, rewards, info = env.step(actions)

            terminal = (
                step == config.steps_per_episode - 1
            )

            rl_agent.learn(
                state=states[0],
                action=rl_action,
                reward=rewards[0],
                next_state=next_states[0],
                terminal=terminal
            )

            episode_rewards[0] += rewards[0]
            episode_rewards[1] += rewards[1]
            episode_discounts += rl_action

            states = next_states

        episode_results.append({
            "episode": episode + 1,
            "rl_episode_reward": episode_rewards[0],
            "rule_episode_reward": episode_rewards[1],
            "rl_mean_reward_per_step": (
                episode_rewards[0]
                / config.steps_per_episode
            ),
            "rule_mean_reward_per_step": (
                episode_rewards[1]
                / config.steps_per_episode
            ),
            "rl_discount_rate": (
                episode_discounts
                / config.steps_per_episode
            ),
            "epsilon": rl_agent.epsilon,
            "alpha": alpha
        })

    return rl_agent, episode_results


# ============================================================
# 6. SEPARATE POLICY EVALUATION
# ============================================================

def evaluate_pair(
    config,
    rl_agent,
    episodes=None,
    seed=900_000,
    opponent=None,
    names=("RL", "Rule-based")
):
    """
    Evaluate a frozen RL policy against the fixed rule-based agent.

    Important:
        - No Q-table updates occur.
        - RL exploration is disabled.
        - Evaluation uses a separate environment random seed.

    Extension: `rl_agent` may be any object with act(state, explore),
    e.g. a RuleBasedAgent, so that both approaches are evaluated in the
    same slot, against the same opponent, with the same seed.
    """

    if episodes is None:
        episodes = config.evaluation_episodes

    env = ECommerceEnvironment(config, seed=seed)

    rule_agent = RuleBasedAgent() if opponent is None else opponent

    total_rewards = np.zeros(2, dtype=float)
    action_counts = np.zeros((2, 2), dtype=int)

    episode_results = []

    for episode in range(episodes):

        states = env.reset()
        rewards_this_episode = [0.0, 0.0]

        for _ in range(config.steps_per_episode):

            # Greedy RL policy: no exploration.
            rl_action = rl_agent.act(
                states[0], explore=False
            )

            rule_action = rule_agent.act(states[1])

            actions = [rl_action, rule_action]

            next_states, rewards, info = env.step(actions)

            for i in range(2):
                total_rewards[i] += rewards[i]
                rewards_this_episode[i] += rewards[i]
                action_counts[i, actions[i]] += 1

            states = next_states

        episode_results.append({
            "episode": episode + 1,
            "rl_reward": rewards_this_episode[0],
            "rule_reward": rewards_this_episode[1],
            "rl_mean_reward_per_step": (
                rewards_this_episode[0]
                / config.steps_per_episode
            ),
            "rule_mean_reward_per_step": (
                rewards_this_episode[1]
                / config.steps_per_episode
            )
        })

    total_steps = episodes * config.steps_per_episode

    summary = []

    for i, name in enumerate(names):
        summary.append({
            "agent": name,
            "total_reward": total_rewards[i],
            "mean_reward_per_step": (
                total_rewards[i] / total_steps
            ),
            "no_discount_actions": int(action_counts[i, 0]),
            "discount_actions": int(action_counts[i, 1]),
            "discount_action_rate": (
                action_counts[i, 1] / total_steps
            )
        })

    return summary, episode_results


# ============================================================
# 7. CSV AND Q-TABLE OUTPUT
# ============================================================

def write_csv(filepath, rows):
    if not rows:
        return

    with open(
        filepath, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def save_q_table(filepath, agent, mode="w", title=None):
    with open(filepath, mode, encoding="utf-8") as file:
        if title:
            file.write(f"{title}\n")
        file.write(
            "Q-table: Q[demand, competitor_previous_action, action]\n"
        )
        file.write(
            "Action 0 = no discount; Action 1 = discount\n\n"
        )

        for demand in range(2):
            for competitor_action in range(2):

                q0 = agent.q[demand, competitor_action, 0]
                q1 = agent.q[demand, competitor_action, 1]

                best_action = int(
                    np.argmax(
                        agent.q[demand, competitor_action]
                    )
                )

                file.write(
                    f"State=({demand}, {competitor_action}) | "
                    f"Q0={q0:.6f} | Q1={q1:.6f} | "
                    f"Greedy action={best_action}\n"
                )

        file.write("\n")


# ============================================================
# 8. PLOTS
# ============================================================

def moving_average(values, window=100):
    values = np.asarray(values, dtype=float)

    if len(values) < window:
        return values

    kernel = np.ones(window) / window

    return np.convolve(
        values, kernel, mode="valid"
    )


def plot_training_rewards(rows, filepath):
    episodes = [
        row["episode"] for row in rows
    ]

    rl_rewards = [
        row["rl_mean_reward_per_step"] for row in rows
    ]

    rule_rewards = [
        row["rule_mean_reward_per_step"] for row in rows
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        episodes, rl_rewards,
        alpha=0.25, label="RL training reward"
    )

    plt.plot(
        episodes, rule_rewards,
        alpha=0.25, label="Rule-based reward"
    )

    window = min(100, max(10, len(rows) // 20))

    if len(rows) >= window:
        plt.plot(
            episodes[window - 1:],
            moving_average(rl_rewards, window),
            linewidth=2,
            label=f"RL moving average ({window})"
        )

        plt.plot(
            episodes[window - 1:],
            moving_average(rule_rewards, window),
            linewidth=2,
            label=f"Rule moving average ({window})"
        )

    plt.xlabel("Training episode")
    plt.ylabel("Mean reward per step")
    plt.title("RL Training Against a Rule-Based Agent")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=160)
    plt.close()


def plot_bar_comparison(
    rows, label_key, value_key, filepath, title
):
    labels = [str(row[label_key]) for row in rows]
    values = [row[value_key] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)

    plt.xlabel(label_key.replace("_", " ").title())
    plt.ylabel(value_key.replace("_", " ").title())
    plt.title(title)

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=160)
    plt.close()


def plot_curves(curves, filepath, title, window=100, key="rl_mean_reward_per_step"):
    """Extension: overlay moving-average training curves of several runs."""

    plt.figure(figsize=(10, 6))

    for label, rows in curves:
        values = [row[key] for row in rows]
        smoothed = moving_average(values, window)
        x = np.arange(window, len(values) + 1)
        plt.plot(x, smoothed, linewidth=1.8, label=label)

    plt.xlabel("Training episode")
    plt.ylabel(f"Mean reward per step (moving average, {window})")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=160)
    plt.close()


# ============================================================
# 9. BASELINE EXPERIMENT
# ============================================================

def run_baseline(config, output_dir):
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: BASELINE")
    print("=" * 60)

    agent, training_rows = train_rl_against_rule(
        config, seed=config.seed
    )

    write_csv(
        output_dir / "baseline_episode_rewards.csv",
        training_rows
    )

    save_q_table(
        output_dir / "q_tables.txt", agent
    )

    plot_training_rewards(
        training_rows,
        output_dir / "episode_rewards.png"
    )

    summary, eval_rows = evaluate_pair(
        config,
        agent,
        episodes=config.evaluation_episodes,
        seed=config.seed + 500_000
    )

    write_csv(
        output_dir / "baseline_evaluation.csv",
        summary
    )

    write_csv(
        output_dir / "baseline_evaluation_episodes.csv",
        eval_rows
    )

    print("\nSeparate evaluation results:")

    for row in summary:
        print(
            f"{row['agent']:12s} | "
            f"total reward = {row['total_reward']:.0f} | "
            f"mean reward/step = "
            f"{row['mean_reward_per_step']:.4f} | "
            f"discount rate = "
            f"{row['discount_action_rate']:.3f}"
        )

    print("\nLearned greedy policy:")

    for state, action in agent.greedy_policy().items():
        action_name = (
            "DISCOUNT" if action == 1
            else "NO DISCOUNT"
        )
        print(f"State {state}: {action_name}")

    return agent


# ============================================================
# 10. LEARNING-RATE EXPERIMENT
# ============================================================

def run_learning_rate_experiment(config, output_dir):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: LEARNING RATE")
    print("=" * 60)

    results = []
    curves = []

    for alpha in [0.01, 0.10, 0.50]:

        agent, training_rows = train_rl_against_rule(
            config,
            seed=config.seed + 1000,
            alpha=alpha
        )

        curves.append((f"alpha = {alpha}", training_rows))

        summary, _ = evaluate_pair(
            config,
            agent,
            episodes=config.evaluation_episodes,
            seed=config.seed + 600_000
        )

        rl_result = next(
            row for row in summary
            if row["agent"] == "RL"
        )

        last = training_rows[-1000:]

        results.append({
            "alpha": alpha,
            "rl_mean_reward_per_step":
                rl_result["mean_reward_per_step"],
            "rl_discount_action_rate":
                rl_result["discount_action_rate"],
            "train_last1000_mean": float(np.mean(
                [r["rl_mean_reward_per_step"] for r in last]
            )),
            "train_last1000_std": float(np.std(
                [r["rl_mean_reward_per_step"] for r in last], ddof=1
            )),
            "greedy_policy": str(agent.greedy_policy())
        })

        print(
            f"alpha={alpha:.2f} | "
            f"evaluation reward/step="
            f"{rl_result['mean_reward_per_step']:.4f} | "
            f"discount rate={rl_result['discount_action_rate']:.3f}"
        )

    write_csv(
        output_dir / "learning_rate_comparison.csv",
        results
    )

    plot_bar_comparison(
        results,
        "alpha",
        "rl_mean_reward_per_step",
        output_dir / "learning_rate_comparison.png",
        "Effect of Learning Rate"
    )

    plot_curves(
        curves,
        output_dir / "learning_rate_curves.png",
        "Training reward for different learning rates (seed 1042)"
    )

    return results


# ============================================================
# 11. EXPLORATION EXPERIMENT
# ============================================================

EXPLORATION_SETTINGS = [
    # name, epsilon_start, epsilon_min, epsilon_decay
    ("low", 0.05, 0.02, 0.999),
    ("baseline", 0.20, 0.02, 0.995),
    ("high", 0.50, 0.05, 0.995),
    # Extension: fixed epsilon, as suggested in Experiment D.
    ("fixed_0.05", 0.05, 0.05, 1.0),
    ("fixed_0.30", 0.30, 0.30, 1.0),
]


def run_exploration_experiment(config, output_dir):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: EXPLORATION")
    print("=" * 60)

    results = []
    curves = []

    for name, eps_start, eps_min, eps_decay in EXPLORATION_SETTINGS:

        agent, training_rows = train_rl_against_rule(
            config,
            seed=config.seed + 2000,
            epsilon_start=eps_start,
            epsilon_min=eps_min,
            epsilon_decay=eps_decay
        )

        curves.append((name, training_rows))

        summary, _ = evaluate_pair(
            config,
            agent,
            episodes=config.evaluation_episodes,
            seed=config.seed + 700_000
        )

        rl_result = next(
            row for row in summary
            if row["agent"] == "RL"
        )

        last = training_rows[-1000:]

        results.append({
            "exploration_setting": name,
            "epsilon_start": eps_start,
            "epsilon_min": eps_min,
            "epsilon_decay": eps_decay,
            "rl_mean_reward_per_step":
                rl_result["mean_reward_per_step"],
            "rl_discount_action_rate":
                rl_result["discount_action_rate"],
            "train_mean_all": float(np.mean(
                [r["rl_mean_reward_per_step"] for r in training_rows]
            )),
            "train_last1000_mean": float(np.mean(
                [r["rl_mean_reward_per_step"] for r in last]
            )),
            "train_last1000_discount_rate": float(np.mean(
                [r["rl_discount_rate"] for r in last]
            )),
            "greedy_policy": str(agent.greedy_policy())
        })

        print(
            f"{name:10s} | "
            f"train(last 1000)={results[-1]['train_last1000_mean']:.4f} | "
            f"evaluation reward/step="
            f"{rl_result['mean_reward_per_step']:.4f}"
        )

    write_csv(
        output_dir / "exploration_comparison.csv",
        results
    )

    write_csv(
        output_dir / "exploration_training_vs_eval.csv",
        [
            {
                "setting": r["exploration_setting"],
                "train_mean_all": r["train_mean_all"],
                "train_last1000_mean": r["train_last1000_mean"],
                "eval_mean": r["rl_mean_reward_per_step"],
            }
            for r in results
        ]
    )

    plot_bar_comparison(
        results,
        "exploration_setting",
        "rl_mean_reward_per_step",
        output_dir / "exploration_comparison.png",
        "Effect of Exploration Schedule"
    )

    plot_curves(
        curves,
        output_dir / "exploration_curves.png",
        "Training reward for different exploration settings (seed 2042)"
    )

    return results


# ============================================================
# 12. MULTIPLE-SEED EXPERIMENT
# ============================================================

def run_multi_seed_experiment(config, output_dir):
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: MULTIPLE RANDOM SEEDS")
    print("=" * 60)

    seeds = SEEDS
    results = []

    for seed in seeds:

        agent, _ = train_rl_against_rule(
            config, seed=seed
        )

        summary, _ = evaluate_pair(
            config,
            agent,
            episodes=config.evaluation_episodes,
            seed=seed + 800_000
        )

        for row in summary:
            results.append({
                "seed": seed,
                "agent": row["agent"],
                "mean_reward_per_step":
                    row["mean_reward_per_step"],
                "discount_action_rate":
                    row["discount_action_rate"]
            })

    write_csv(
        output_dir / "multi_seed_summary.csv",
        results
    )

    print("\nMean and standard deviation across seeds:")

    for name in ["RL", "Rule-based"]:

        values = [
            row["mean_reward_per_step"]
            for row in results
            if row["agent"] == name
        ]

        print(
            f"{name:12s} | "
            f"mean={np.mean(values):.4f} | "
            f"std={np.std(values, ddof=1):.4f} | "
            f"n={len(values)}"
        )

    return results


# ============================================================
# EXTENSIONS (student work)
# ============================================================

def expected_one_step_reward(config, demand, own_action, competitor_action):
    """Analytic E[r] = N * p * (P(1 - delta*a) - C)."""

    customers = (
        config.high_demand_customers if demand == 1
        else config.low_demand_customers
    )

    p = np.clip(
        0.30 + 0.25 * demand + 0.15 * own_action - 0.10 * competitor_action,
        0.05, 0.95
    )

    margin = config.price * (1 - config.discount_rate * own_action) - config.unit_cost

    return customers * p * margin


def run_expected_reward_table(config, output_dir):
    """Extension: analytic check of which action is myopically better."""

    print("\n" + "=" * 60)
    print("EXTENSION A: ANALYTIC ONE-STEP EXPECTED REWARD")
    print("=" * 60)

    rows = []

    for discount_rate in [0.20, 0.10]:
        cfg = replace(config, discount_rate=discount_rate)

        for demand in range(2):
            for competitor_action in range(2):
                r0 = expected_one_step_reward(cfg, demand, 0, competitor_action)
                r1 = expected_one_step_reward(cfg, demand, 1, competitor_action)

                rows.append({
                    "discount_rate": discount_rate,
                    "demand": demand,
                    "competitor_current_action": competitor_action,
                    "E_reward_no_discount": round(float(r0), 4),
                    "E_reward_discount": round(float(r1), 4),
                    "better_action": int(r1 > r0)
                })

                print(
                    f"delta={discount_rate:.2f} d={demand} c={competitor_action} | "
                    f"E[r|a=0]={r0:6.2f}  E[r|a=1]={r1:6.2f}"
                )

    write_csv(output_dir / "expected_rewards.csv", rows)

    return rows


def run_rule_vs_rule(config, output_dir):
    """
    Extension (Experiment E): the reference evaluation measures the RL
    agent in slot 0 and the rule agent in slot 1, i.e. they face
    different opponents. For a fair comparison we also put the
    rule-based agent in slot 0, against the same rule-based opponent,
    with the same evaluation seed as the baseline.
    """

    print("\n" + "=" * 60)
    print("EXTENSION B: RULE-BASED AGENT IN THE RL SLOT (FAIR REFERENCE)")
    print("=" * 60)

    summary, _ = evaluate_pair(
        config,
        RuleBasedAgent(),
        episodes=config.evaluation_episodes,
        seed=config.seed + 500_000,
        names=("Rule-based (slot 0)", "Rule-based opponent")
    )

    write_csv(output_dir / "rule_vs_rule_evaluation.csv", summary)

    for row in summary:
        print(
            f"{row['agent']:22s} | "
            f"mean reward/step = {row['mean_reward_per_step']:.4f} | "
            f"discount rate = {row['discount_action_rate']:.3f}"
        )

    return summary


def multi_seed_rl(config, **train_kwargs):
    """Train + greedy-evaluate one RL configuration on every seed in SEEDS."""

    rewards = []
    discount_rates = []

    for seed in SEEDS:
        agent, _ = train_rl_against_rule(config, seed=seed, **train_kwargs)

        summary, _ = evaluate_pair(
            config, agent,
            episodes=config.evaluation_episodes,
            seed=seed + 800_000
        )

        rewards.append(summary[0]["mean_reward_per_step"])
        discount_rates.append(summary[0]["discount_action_rate"])

    return rewards, discount_rates


def multi_seed_fixed(config, policy):
    """Greedy-evaluate a fixed policy in slot 0 on the same seeds."""

    rewards = []
    discount_rates = []

    for seed in SEEDS:
        summary, _ = evaluate_pair(
            config, policy,
            episodes=config.evaluation_episodes,
            seed=seed + 800_000
        )

        rewards.append(summary[0]["mean_reward_per_step"])
        discount_rates.append(summary[0]["discount_action_rate"])

    return rewards, discount_rates


def run_results_table(config, output_dir):
    """
    Extension (Section 7.6): every configuration is trained on the same
    5 seeds and evaluated greedily in slot 0 against the rule-based
    opponent with the same evaluation seeds.
    """

    print("\n" + "=" * 60)
    print("EXTENSION C: RESULTS TABLE OVER 5 SEEDS")
    print("=" * 60)

    configs = [
        ("RL, baseline (alpha=0.10, eps 0.20->0.02)", {}),
        ("RL, low learning rate (alpha=0.01)", {"alpha": 0.01}),
        ("RL, high learning rate (alpha=0.50)", {"alpha": 0.50}),
        ("RL, fixed eps=0.05", {"epsilon_start": 0.05, "epsilon_min": 0.05, "epsilon_decay": 1.0}),
        ("RL, fixed eps=0.30", {"epsilon_start": 0.30, "epsilon_min": 0.30, "epsilon_decay": 1.0}),
        ("RL, high exploration (0.50->0.05)", {"epsilon_start": 0.50, "epsilon_min": 0.05, "epsilon_decay": 0.995}),
    ]

    rows = []

    for name, kwargs in configs:
        rewards, discount_rates = multi_seed_rl(config, **kwargs)
        rows.append((name, rewards, discount_rates))

    for name, policy in [
        ("Rule-based", RuleBasedAgent()),
        ("Never discount (reference)", NeverDiscountAgent()),
        ("Always discount (reference)", AlwaysDiscountAgent()),
    ]:
        rewards, discount_rates = multi_seed_fixed(config, policy)
        rows.append((name, rewards, discount_rates))

    table = []

    for name, rewards, discount_rates in rows:
        table.append({
            "configuration": name,
            "mean_reward": float(np.mean(rewards)),
            "std_dev": float(np.std(rewards, ddof=1)),
            "discount_rate": float(np.mean(discount_rates)),
            "per_seed_rewards": " ".join(f"{r:.3f}" for r in rewards)
        })

        print(
            f"{name:45s} | mean={table[-1]['mean_reward']:.4f} | "
            f"std={table[-1]['std_dev']:.4f} | "
            f"discount rate={table[-1]['discount_rate']:.3f}"
        )

    write_csv(output_dir / "results_table.csv", table)

    plt.figure(figsize=(11, 6))
    labels = [row["configuration"] for row in table]
    means = [row["mean_reward"] for row in table]
    stds = [row["std_dev"] for row in table]
    colors = ["#2b6cb0"] * 6 + ["#c05621", "#718096", "#718096"]
    y = np.arange(len(labels))
    plt.barh(y, means, xerr=stds, color=colors, capsize=4)
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.xlim(0, max(means) + 5)
    for yi, (m_, s_) in enumerate(zip(means, stds)):
        plt.text(m_ + s_ + 0.4, yi, f"{m_:.2f} ± {s_:.2f}", va="center", fontsize=8)
    plt.xlabel("Evaluation mean reward per step (mean +/- std over 5 seeds)")
    plt.title("Greedy evaluation in slot 0 against the rule-based opponent")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "results_table.png", dpi=160)
    plt.close()

    return table


def train_two_learners(config, seed):
    """
    Extension (Q8): both agents are independent Q-learners that learn
    simultaneously. Each one is part of the other's environment, so the
    environment is non-stationary from each agent's point of view.
    """

    env = ECommerceEnvironment(config, seed=seed)

    agents = [
        QLearningAgent(config.alpha, config.gamma, config.epsilon_start, seed + 10_000),
        QLearningAgent(config.alpha, config.gamma, config.epsilon_start, seed + 20_000),
    ]

    rows = []

    for episode in range(config.episodes):
        eps = max(config.epsilon_min, config.epsilon_start * config.epsilon_decay ** episode)

        for agent in agents:
            agent.epsilon = eps

        states = env.reset()
        totals = [0.0, 0.0]
        discounts = [0, 0]

        for step in range(config.steps_per_episode):
            actions = [agents[i].act(states[i], explore=True) for i in range(2)]
            next_states, rewards, _ = env.step(actions)
            terminal = step == config.steps_per_episode - 1

            for i in range(2):
                agents[i].learn(states[i], actions[i], rewards[i], next_states[i], terminal)
                totals[i] += rewards[i]
                discounts[i] += actions[i]

            states = next_states

        rows.append({
            "episode": episode + 1,
            "agent1_mean_reward_per_step": totals[0] / config.steps_per_episode,
            "agent2_mean_reward_per_step": totals[1] / config.steps_per_episode,
            "agent1_discount_rate": discounts[0] / config.steps_per_episode,
            "agent2_discount_rate": discounts[1] / config.steps_per_episode,
        })

    return agents, rows


def run_two_learners_experiment(config, output_dir):
    print("\n" + "=" * 60)
    print("EXTENSION D: TWO SIMULTANEOUS Q-LEARNERS (Q8)")
    print("=" * 60)

    results = []
    first_rows = None

    open(output_dir / "two_learners_q_tables.txt", "w").close()

    for seed in SEEDS:
        agents, rows = train_two_learners(config, seed)

        if first_rows is None:
            first_rows = rows

        save_q_table(
            output_dir / "two_learners_q_tables.txt", agents[0],
            mode="a", title=f"=== seed {seed}, agent 1 ==="
        )
        save_q_table(
            output_dir / "two_learners_q_tables.txt", agents[1],
            mode="a", title=f"=== seed {seed}, agent 2 ==="
        )

        # Greedy evaluation of the two learned policies against each other.
        summary, _ = evaluate_pair(
            config, agents[0],
            episodes=config.evaluation_episodes,
            seed=seed + 800_000,
            opponent=_GreedyWrapper(agents[1]),
            names=("Q-learner 1", "Q-learner 2")
        )

        # Training-time discount rate early vs late shows policy drift.
        first = rows[:500]
        last = rows[-500:]

        results.append({
            "seed": seed,
            "agent1_eval_reward": summary[0]["mean_reward_per_step"],
            "agent2_eval_reward": summary[1]["mean_reward_per_step"],
            "agent1_eval_discount_rate": summary[0]["discount_action_rate"],
            "agent2_eval_discount_rate": summary[1]["discount_action_rate"],
            "agent1_train_discount_first500": float(np.mean([r["agent1_discount_rate"] for r in first])),
            "agent1_train_discount_last500": float(np.mean([r["agent1_discount_rate"] for r in last])),
            "agent1_policy": str(agents[0].greedy_policy()),
            "agent2_policy": str(agents[1].greedy_policy()),
        })

        print(
            f"seed {seed} | agent1 eval={summary[0]['mean_reward_per_step']:.3f} "
            f"(disc {summary[0]['discount_action_rate']:.3f}) | "
            f"agent2 eval={summary[1]['mean_reward_per_step']:.3f} "
            f"(disc {summary[1]['discount_action_rate']:.3f})"
        )

    write_csv(output_dir / "two_learners.csv", results)

    _plot_two_learners(first_rows, output_dir / "two_learners_curves.png")

    return results


class _GreedyWrapper:
    """Lets a QLearningAgent act as the slot-1 opponent (always greedy)."""

    def __init__(self, agent):
        self.agent = agent

    def act(self, state, explore=False):
        return self.agent.act(state, explore=False)


def _plot_two_learners(rows, filepath, window=100):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for key, label in [
        ("agent1_mean_reward_per_step", "Q-learner 1"),
        ("agent2_mean_reward_per_step", "Q-learner 2"),
    ]:
        values = [row[key] for row in rows]
        axes[0].plot(np.arange(window, len(values) + 1), moving_average(values, window), label=label)

    for key, label in [
        ("agent1_discount_rate", "Q-learner 1"),
        ("agent2_discount_rate", "Q-learner 2"),
    ]:
        values = [row[key] for row in rows]
        axes[1].plot(np.arange(window, len(values) + 1), moving_average(values, window), label=label)

    axes[0].set_ylabel(f"Reward per step (MA {window})")
    axes[1].set_ylabel(f"Discount rate (MA {window})")
    axes[1].set_xlabel("Training episode")
    axes[0].set_title("Two simultaneous Q-learners (seed 11)")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(filepath, dpi=160)
    plt.close()


def run_modification_experiment(config, output_dir):
    """
    Extension (Q9): change the environment and compare the RL agent
    (retrained in the new environment) with the unchanged rule-based
    agent, both in slot 0 against the rule-based opponent.
    """

    print("\n" + "=" * 60)
    print("EXTENSION E: MODIFIED ENVIRONMENTS (Q9)")
    print("=" * 60)

    scenarios = [
        ("Reference (delta=0.20, i.i.d. demand)", config, None),
        ("Smaller discount (delta=0.10)", replace(config, discount_rate=0.10), None),
        ("Persistent demand (stay prob 0.9)", replace(config, demand_persistence=0.9), None),
        ("Aggressive opponent (always discounts)", config, AlwaysDiscountAgent),
    ]

    rows = []

    for name, cfg, opponent_cls in scenarios:
        rl_rewards, rl_disc = [], []
        rule_rewards, rule_disc = [], []
        policies = []

        for seed in SEEDS:
            opponent = RuleBasedAgent() if opponent_cls is None else opponent_cls()

            agent, _ = train_rl_against_rule(cfg, seed=seed, opponent=opponent)
            policies.append(agent.greedy_policy())

            summary, _ = evaluate_pair(cfg, agent, seed=seed + 800_000, opponent=opponent)
            rl_rewards.append(summary[0]["mean_reward_per_step"])
            rl_disc.append(summary[0]["discount_action_rate"])

            summary, _ = evaluate_pair(cfg, RuleBasedAgent(), seed=seed + 800_000, opponent=opponent)
            rule_rewards.append(summary[0]["mean_reward_per_step"])
            rule_disc.append(summary[0]["discount_action_rate"])

        rows.append({
            "scenario": name,
            "rl_mean": float(np.mean(rl_rewards)),
            "rl_std": float(np.std(rl_rewards, ddof=1)),
            "rl_discount_rate": float(np.mean(rl_disc)),
            "rule_mean": float(np.mean(rule_rewards)),
            "rule_std": float(np.std(rule_rewards, ddof=1)),
            "rule_discount_rate": float(np.mean(rule_disc)),
            "rl_minus_rule": float(np.mean(rl_rewards) - np.mean(rule_rewards)),
            "rl_policy_seed11": str(policies[0]),
        })

        print(
            f"{name:42s} | RL={rows[-1]['rl_mean']:.3f}+/-{rows[-1]['rl_std']:.3f} "
            f"(disc {rows[-1]['rl_discount_rate']:.2f}) | "
            f"Rule={rows[-1]['rule_mean']:.3f}+/-{rows[-1]['rule_std']:.3f} "
            f"(disc {rows[-1]['rule_discount_rate']:.2f})"
        )

    write_csv(output_dir / "modification_experiment.csv", rows)

    labels = [row["scenario"] for row in rows]
    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(11, 6))
    plt.bar(x - width / 2, [r["rl_mean"] for r in rows], width,
            yerr=[r["rl_std"] for r in rows], capsize=4, label="RL (retrained)", color="#2b6cb0")
    plt.bar(x + width / 2, [r["rule_mean"] for r in rows], width,
            yerr=[r["rule_std"] for r in rows], capsize=4, label="Rule-based (unchanged)", color="#c05621")
    plt.xticks(x, labels, rotation=12, ha="right")
    plt.ylabel("Evaluation mean reward per step (5 seeds)")
    plt.title("Robustness to changes in the environment")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "modification_experiment.png", dpi=160)
    plt.close()

    return rows


class ImprovedQLearningAgent(QLearningAgent):
    """
    Extension (Q10): Q-learning with two optional changes.

    augmented=True
        State = (demand, competitor_previous_action, own_previous_action).
        The rule-based opponent reacts to *our* previous action, so the
        reference state hides the variable that determines the
        opponent's next move. Q then has 2 * 2 * 2 * 2 = 16 entries.

    decaying_alpha=True
        Per-(state, action) learning rate alpha = max(0.005, 1 / n^0.6),
        where n is the visit count. Rewards are binomial and noisy; a
        constant alpha = 0.10 keeps the Q-values fluctuating by several
        units, which is larger than the gap between the two actions in
        several states. A decaying rate averages that noise out.
    """

    def __init__(
        self, alpha=0.10, gamma=0.90, epsilon=0.20, seed=123,
        augmented=True, decaying_alpha=True
    ):
        super().__init__(alpha, gamma, epsilon, seed)
        self.augmented = augmented
        self.decaying_alpha = decaying_alpha
        self.q = np.zeros((2,) * (4 if augmented else 3), dtype=float)
        self.visits = np.zeros_like(self.q)

    def observe(self, env, states):
        if self.augmented:
            return states[0] + (env.previous_actions[0],)

        return states[0]

    def act(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, 2))

        values = self.q[state]
        best_actions = np.flatnonzero(np.isclose(values, np.max(values)))

        return int(self.rng.choice(best_actions))

    def learn(self, state, action, reward, next_state, terminal=False):
        index = state + (action,)
        self.visits[index] += 1

        alpha = (
            max(0.005, 1.0 / self.visits[index] ** 0.6)
            if self.decaying_alpha else self.alpha
        )

        old_value = self.q[index]
        target = reward if terminal else reward + self.gamma * np.max(self.q[next_state])
        self.q[index] = old_value + alpha * (target - old_value)

    def greedy_policy(self):
        return {
            index: int(np.argmax(self.q[index]))
            for index in np.ndindex(self.q.shape[:-1])
        }


def run_improved(config, agent, seed, opponent, train):
    """Train (train=True) or greedily evaluate an ImprovedQLearningAgent."""

    env = ECommerceEnvironment(config, seed=seed)
    episodes = config.episodes if train else config.evaluation_episodes
    total, discounts = 0.0, 0

    for episode in range(episodes):
        if train:
            agent.epsilon = max(
                config.epsilon_min,
                config.epsilon_start * config.epsilon_decay ** episode
            )

        states = env.reset()
        state = agent.observe(env, states)

        for step in range(config.steps_per_episode):
            action = agent.act(state, explore=train)
            opponent_action = opponent.act(states[1])
            states, rewards, _ = env.step([action, opponent_action])
            next_state = agent.observe(env, states)

            if train:
                terminal = step == config.steps_per_episode - 1
                agent.learn(state, action, rewards[0], next_state, terminal)

            total += rewards[0]
            discounts += action
            state = next_state

    steps = episodes * config.steps_per_episode

    return total / steps, discounts / steps


IMPROVED_VARIANTS = [
    ("Augmented state only", True, False),
    ("Decaying alpha only", False, True),
    ("Improved (augmented + decaying alpha)", True, True),
]


def run_improved_agent_experiment(config, output_dir):
    """
    Extension (Q10): ablation of the improved agent on the same four
    scenarios and seeds as the modification experiment (Q9).
    """

    print("\n" + "=" * 60)
    print("EXTENSION F: IMPROVED AGENT, ABLATION (Q10)")
    print("=" * 60)

    scenarios = [
        ("Reference (delta=0.20, i.i.d. demand)", config, RuleBasedAgent),
        ("Smaller discount (delta=0.10)", replace(config, discount_rate=0.10), RuleBasedAgent),
        ("Persistent demand (stay prob 0.9)", replace(config, demand_persistence=0.9), RuleBasedAgent),
        ("Aggressive opponent (always discounts)", config, AlwaysDiscountAgent),
    ]

    rows = []

    for name, cfg, opponent_cls in scenarios:
        for variant, augmented, decaying_alpha in IMPROVED_VARIANTS:
            rewards, discount_rates = [], []
            policies = []

            for seed in SEEDS:
                agent = ImprovedQLearningAgent(
                    cfg.alpha, cfg.gamma, cfg.epsilon_start, seed + 10_000,
                    augmented=augmented, decaying_alpha=decaying_alpha
                )
                run_improved(cfg, agent, seed, opponent_cls(), train=True)
                reward, discount_rate = run_improved(
                    cfg, agent, seed + 800_000, opponent_cls(), train=False
                )
                rewards.append(reward)
                discount_rates.append(discount_rate)
                policies.append(agent.greedy_policy())

            rows.append({
                "scenario": name,
                "variant": variant,
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards, ddof=1)),
                "discount_rate": float(np.mean(discount_rates)),
                "per_seed_rewards": " ".join(f"{r:.3f}" for r in rewards),
                "policy_seed11": str(policies[0]),
            })

            print(
                f"{name:40s} | {variant:38s} | {rows[-1]['mean']:.3f}"
                f" +/- {rows[-1]['std']:.3f} "
                f"(disc {rows[-1]['discount_rate']:.2f})"
            )

    write_csv(output_dir / "improved_agent.csv", rows)

    return rows


# ============================================================
# 13. MAIN PROGRAM
# ============================================================

def main():
    config = Config()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AGENT02: RL AGENT VS RULE-BASED AGENT")
    print("=" * 60)

    print(f"Episodes: {config.episodes}")
    print(f"Steps per episode: {config.steps_per_episode}")
    print(f"Evaluation episodes: {config.evaluation_episodes}")
    print(f"Random seed: {config.seed}")
    print(f"Output directory: {output_dir.resolve()}")

    run_baseline(config, output_dir)

    run_learning_rate_experiment(
        config, output_dir
    )

    run_exploration_experiment(
        config, output_dir
    )

    run_multi_seed_experiment(
        config, output_dir
    )

    # Student extensions.
    run_expected_reward_table(config, output_dir)
    run_rule_vs_rule(config, output_dir)
    run_results_table(config, output_dir)
    run_two_learners_experiment(config, output_dir)
    run_modification_experiment(config, output_dir)
    run_improved_agent_experiment(config, output_dir)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 60)

    print(f"Results saved to: {output_dir.resolve()}")

    print("\nUse the CSV files and PNG figures in the AGENT02 report.")


if __name__ == "__main__":
    main()
