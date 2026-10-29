# AGENT02 – Reinforcement-Learning Agent vs Rule-Based Agent

Laboratory Experiment 02, *Intelligent System Development* (2026–2027).

Two sellers compete in a simplified e-commerce market and decide each step whether to offer a discount.
Agent A learns with tabular Q-learning; agent B follows fixed if-else rules.
This repository contains the experiment code, all raw results and the written report.

## Repository layout

| Path | Content |
|---|---|
| `agent02_experiment.py` | Full experiment program: reference code (sections 1–13) + documented extensions |
| `agent02_results/` | All CSV files, Q-tables and figures produced by one run |
| `run_log.txt` | Console output of the run used in the report |
| `report/report.md` | Report source (Markdown) |
| `report/agent02_class_team_name.pdf` | Report as PDF (rename `class` / `team_name` before submitting) |
| `report/build_pdf.sh` | Rebuilds the PDF from the Markdown (pandoc + Edge/Chrome, no LaTeX) |

## How to run

```bash
pip install -r requirements.txt
python agent02_experiment.py
```

The run takes about 14 minutes on a laptop CPU (no GPU needed) and writes everything to `agent02_results/`.
All random seeds are fixed, so the numbers are reproducible:
environment seed 42 for the baseline, training seeds 11, 22, 33, 44, 55 for multi-seed results,
evaluation seed = training seed + 800 000.

Tested with Python 3.14.2, NumPy 2.4.2, Matplotlib 3.11.1.

## Experiments

| # | Experiment | Output files |
|---|---|---|
| 1 | Baseline: RL trained vs rule agent, separate greedy evaluation (Exp. A, B) | `baseline_*.csv`, `q_tables.txt`, `episode_rewards.png` |
| 2 | Learning rate α ∈ {0.01, 0.10, 0.50} (Exp. C) | `learning_rate_comparison.*`, `learning_rate_curves.png` |
| 3 | Exploration: 3 decay schedules + fixed ε = 0.05 / 0.30 (Exp. D) | `exploration_comparison.*`, `exploration_curves.png`, `exploration_training_vs_eval.csv` |
| 4 | Reference multi-seed experiment | `multi_seed_summary.csv` |
| A | Analytic one-step expected rewards | `expected_rewards.csv` |
| B | Rule-based agent evaluated in the RL slot (fair reference) | `rule_vs_rule_evaluation.csv` |
| C | Results table: every configuration × 5 seeds (Exp. E, §7.6) | `results_table.csv`, `results_table.png` |
| D | Two simultaneous Q-learners (Q8) | `two_learners.csv`, `two_learners_curves.png`, `two_learners_q_tables.txt` |
| E | Modified environments: δ = 0.10, persistent demand, aggressive opponent (Q9) | `modification_experiment.*` |
| F | Improved agent + ablation (Q10) | `improved_agent.csv` |

## Rebuilding the report PDF

```bash
bash report/build_pdf.sh
```

Needs `pandoc` and Microsoft Edge or Google Chrome on the PATH / default install location.
