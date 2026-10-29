"""
AGENT02 interactive demo (Streamlit).

Run:
    pip install -r requirements.txt
    streamlit run demo_app.py

The demo imports the environment, agents, training and evaluation
functions directly from agent02_experiment.py, so every number it shows
is produced by the same code as the report.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import agent02_experiment as exp


RESULTS_DIR = Path(__file__).parent / "agent02_results"

ACTION_NAME = {0: "Giữ giá", 1: "Giảm giá"}
DEMAND_NAME = {0: "Thấp", 1: "Cao"}

SCENARIOS = {
    "Thị trường gốc (giảm 20 %, nhu cầu i.i.d.)": ({}, exp.RuleBasedAgent),
    "Giảm giá ít hơn (giảm 10 %)": ({"discount_rate": 0.10}, exp.RuleBasedAgent),
    "Nhu cầu lưu trì (giữ nguyên với xác suất 0.9)": ({"demand_persistence": 0.9}, exp.RuleBasedAgent),
    "Đối thủ hung hăng (luôn giảm giá)": ({}, exp.AlwaysDiscountAgent),
}

AGENT_TYPES = [
    "Q-learning gốc (state = nhu cầu, hành động trước của đối thủ)",
    "Q-learning cải tiến (thêm hành động trước của mình + α giảm dần)",
]


# ------------------------------------------------------------
# Training / evaluation wrappers around agent02_experiment.py
# ------------------------------------------------------------

def observe(agent, env, states):
    """Slot-0 observation for any agent type."""

    if isinstance(agent, exp.ImprovedQLearningAgent):
        return agent.observe(env, states)

    return states[0]


@st.cache_resource(show_spinner=False)
def train(scenario, agent_type, alpha, eps_start, eps_min, eps_decay, episodes, seed):
    overrides, opponent_cls = SCENARIOS[scenario]
    config = replace(
        exp.Config(), episodes=episodes, alpha=alpha,
        epsilon_start=eps_start, epsilon_min=eps_min, epsilon_decay=eps_decay,
        **overrides
    )

    if agent_type == AGENT_TYPES[0]:
        agent, rows = exp.train_rl_against_rule(
            config, seed=seed, opponent=opponent_cls()
        )
        curve = pd.DataFrame({
            "RL": [r["rl_mean_reward_per_step"] for r in rows],
            "Đối thủ": [r["rule_mean_reward_per_step"] for r in rows],
        })
    else:
        agent = exp.ImprovedQLearningAgent(
            alpha, config.gamma, eps_start, seed + 10_000,
            augmented=True, decaying_alpha=True
        )
        curve = train_improved_with_curve(config, agent, seed, opponent_cls())

    return config, agent, curve


def train_improved_with_curve(config, agent, seed, opponent):
    """Same loop as exp.run_improved(train=True), but records rewards per episode."""

    env = exp.ECommerceEnvironment(config, seed=seed)
    rl_rewards, opp_rewards = [], []

    for episode in range(config.episodes):
        agent.epsilon = max(
            config.epsilon_min,
            config.epsilon_start * config.epsilon_decay ** episode
        )

        states = env.reset()
        state = agent.observe(env, states)
        totals = [0.0, 0.0]

        for step in range(config.steps_per_episode):
            action = agent.act(state, explore=True)
            states, rewards, _ = env.step([action, opponent.act(states[1])])
            next_state = agent.observe(env, states)
            agent.learn(state, action, rewards[0], next_state,
                        step == config.steps_per_episode - 1)
            totals[0] += rewards[0]
            totals[1] += rewards[1]
            state = next_state

        rl_rewards.append(totals[0] / config.steps_per_episode)
        opp_rewards.append(totals[1] / config.steps_per_episode)

    return pd.DataFrame({"RL": rl_rewards, "Đối thủ": opp_rewards})


def evaluate(config, policy, opponent, seed, episodes):
    """Greedy evaluation of `policy` in slot 0 (no learning)."""

    env = exp.ECommerceEnvironment(config, seed=seed)
    total, discounts, steps = 0.0, 0, 0

    for _ in range(episodes):
        states = env.reset()

        for _ in range(config.steps_per_episode):
            action = policy.act(observe(policy, env, states), explore=False)
            states, rewards, _ = env.step([action, opponent.act(states[1])])
            total += rewards[0]
            discounts += action
            steps += 1

    return total / steps, discounts / steps


def trace_episode(config, policy, opponent, seed):
    """One greedy 20-step episode, step by step."""

    env = exp.ECommerceEnvironment(config, seed=seed)
    states = env.reset()
    rows = []

    for t in range(config.steps_per_episode):
        demand = env.demand
        competitor_prev = states[0][1]
        action = policy.act(observe(policy, env, states), explore=False)
        opponent_action = opponent.act(states[1])
        states, rewards, info = env.step([action, opponent_action])

        rows.append({
            "Bước": t + 1,
            "Nhu cầu": DEMAND_NAME[demand],
            "Đối thủ bước trước": ACTION_NAME[competitor_prev],
            "Agent (slot 0)": ACTION_NAME[action],
            "Đối thủ": ACTION_NAME[opponent_action],
            "Tỉ lệ chuyển đổi": round(info["conversion"][0], 2),
            "Số đơn": info["sales"][0],
            "Lợi nhuận agent": rewards[0],
            "Lợi nhuận đối thủ": rewards[1],
        })

    return pd.DataFrame(rows)


def q_table_frame(agent):
    rows = []

    for index, values in np.ndenumerate(agent.q[..., 0]):
        q0, q1 = agent.q[index]
        row = {
            "Nhu cầu": DEMAND_NAME[index[0]],
            "Đối thủ bước trước": ACTION_NAME[index[1]],
        }

        if len(index) == 3:
            row["Mình bước trước"] = ACTION_NAME[index[2]]

        row.update({
            "Q(giữ giá)": round(float(q0), 2),
            "Q(giảm giá)": round(float(q1), 2),
            "Hành động greedy": (
                "— (chưa thăm)" if q0 == 0 and q1 == 0
                else ACTION_NAME[int(np.argmax(agent.q[index]))]
            ),
        })
        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Page
# ------------------------------------------------------------

st.set_page_config(page_title="AGENT02 Demo", page_icon="🛒", layout="wide")

st.title("🛒 AGENT02 – Agent học tăng cường vs Agent dựa trên luật")
st.caption(
    "Lý Hải Quân – B23DCCN676 – Lớp 05. "
    "Mọi số liệu được tính trực tiếp bằng code trong agent02_experiment.py."
)

with st.sidebar:
    st.header("⚙️ Cấu hình")
    scenario = st.selectbox("Kịch bản thị trường", list(SCENARIOS))
    agent_type = st.radio("Loại agent RL", AGENT_TYPES)

    st.subheader("Siêu tham số")
    alpha = st.select_slider("Learning rate α", [0.01, 0.05, 0.10, 0.20, 0.50], value=0.10)
    eps_start = st.select_slider("ε ban đầu", [0.05, 0.10, 0.20, 0.30, 0.50], value=0.20)
    eps_min = st.select_slider("ε tối thiểu", [0.0, 0.02, 0.05, 0.10, 0.30], value=0.02)
    eps_decay = st.select_slider("Hệ số giảm ε mỗi episode", [0.99, 0.995, 0.999, 1.0], value=0.995)
    episodes = st.select_slider("Số episode huấn luyện", [500, 1000, 2000, 5000], value=5000)
    seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42, step=1)

    run = st.button("🚀 Huấn luyện & đánh giá", type="primary", width="stretch")

if run:
    st.session_state["params"] = (
        scenario, agent_type, alpha, eps_start, eps_min, eps_decay, episodes, int(seed)
    )

tab_train, tab_step, tab_report = st.tabs(
    ["📈 Huấn luyện & so sánh", "🔍 Mô phỏng từng bước", "📊 Kết quả trong báo cáo"]
)

params = st.session_state.get("params")

with tab_train:
    if params is None:
        st.info("Chọn cấu hình ở thanh bên trái rồi bấm **Huấn luyện & đánh giá**.")
    else:
        with st.spinner("Đang huấn luyện agent RL..."):
            config, agent, curve = train(*params)

        opponent_cls = SCENARIOS[params[0]][1]
        eval_seed = params[-1] + 500_000

        rl_reward, rl_disc = evaluate(config, agent, opponent_cls(), eval_seed, 500)
        rule_reward, rule_disc = evaluate(config, exp.RuleBasedAgent(), opponent_cls(), eval_seed, 500)

        st.subheader("Đánh giá greedy (500 episode, ε = 0, cùng seed, cùng đối thủ)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Agent RL – lợi nhuận / bước", f"{rl_reward:.3f}",
                  f"{rl_reward - rule_reward:+.3f} so với agent luật")
        c2.metric("Agent luật – lợi nhuận / bước", f"{rule_reward:.3f}")
        c3.metric("Tỉ lệ giảm giá (RL / luật)", f"{rl_disc:.2f} / {rule_disc:.2f}")

        st.subheader("Đường cong huấn luyện (trung bình trượt 100 episode)")
        st.line_chart(curve.rolling(100, min_periods=1).mean(),
                      x_label="Episode", y_label="Lợi nhuận trung bình / bước")

        st.subheader("Bảng Q học được")
        st.dataframe(q_table_frame(agent), hide_index=True, width="stretch")
        st.caption(
            "Các trạng thái 'đối thủ bước trước = Giảm giá' có thể không bao giờ xuất hiện khi "
            "đánh giá, vì mỗi episode bắt đầu với cả hai bên chưa giảm giá (xem mục 4.1 báo cáo)."
        )

with tab_step:
    if params is None:
        st.info("Huấn luyện một agent trước để xem mô phỏng từng bước.")
    else:
        config, agent, _ = train(*params)
        opponent_cls = SCENARIOS[params[0]][1]
        episode_seed = st.number_input("Seed của episode", 0, 1_000_000, 2026, key="episode_seed")

        left, right = st.columns(2)

        for column, title, policy in [
            (left, "Agent RL ở slot 0", agent),
            (right, "Agent luật ở slot 0", exp.RuleBasedAgent()),
        ]:
            trace = trace_episode(config, policy, opponent_cls(), int(episode_seed))
            with column:
                st.markdown(f"**{title}** – tổng lợi nhuận: **{trace['Lợi nhuận agent'].sum():.0f}**")
                st.dataframe(
                    trace.style.map(
                        lambda v: "background-color: #fde2c8" if v == "Giảm giá" else "",
                        subset=["Agent (slot 0)", "Đối thủ"]
                    ),
                    hide_index=True, width="stretch", height=740
                )

        st.caption("Cùng seed nên cùng chuỗi nhu cầu; khác biệt chỉ đến từ chính sách. Ô cam = giảm giá.")

with tab_report:
    st.markdown("Kết quả đầy đủ từ lần chạy `python agent02_experiment.py` (5 seed, 5 000 episode).")

    table = RESULTS_DIR / "results_table.csv"
    if table.exists():
        st.subheader("Bảng kết quả 5 seed (thí nghiệm E)")
        st.dataframe(pd.read_csv(table).drop(columns=["per_seed_rewards"]),
                     hide_index=True, width="stretch")

    improved = RESULTS_DIR / "improved_agent.csv"
    if improved.exists():
        st.subheader("Agent cải tiến – ablation (câu 10)")
        frame = pd.read_csv(improved)
        st.dataframe(
            frame.pivot(index="scenario", columns="variant", values="mean").round(3),
            width="stretch"
        )

    for name, caption in [
        ("results_table.png", "So sánh công bằng trên 5 seed"),
        ("modification_experiment.png", "RL và agent luật trong các môi trường sửa đổi"),
        ("learning_rate_curves.png", "Ảnh hưởng của learning rate"),
    ]:
        path = RESULTS_DIR / name
        if path.exists():
            st.image(str(path), caption=caption)
