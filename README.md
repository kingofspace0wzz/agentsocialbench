<div align="center">

# AgentSocialBench

### Evaluating Privacy Risks in Human-Centered Agentic Social Networks

[![arXiv](https://img.shields.io/badge/arXiv-2604.01487-b31b1b.svg)](https://arxiv.org/abs/2604.01487)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-AgentSocialBench-yellow)](https://huggingface.co/datasets/kingofspace0wzz/AgentSocialBench)
[![Website](https://img.shields.io/badge/🌐_Website-agent--social--bench.github.io-blue)](https://agent-social-bench.github.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

</div>

---

**AgentSocialBench** is the first benchmark for evaluating privacy preservation in *human-centered agentic social networks* — settings where teams of AI agents serve individual users across multiple domains, coordinate on shared tasks, and must protect sensitive personal information throughout. It comprises **352 scenarios** across **7 categories** spanning dyadic and multi-party interactions, evaluated on **8 LLM backbones** under multiple privacy instruction levels.

<div align="center">
<img src="assets/agentsocialbench.png" width="85%">
</div>

<div align="center">
<img src="assets/scenario_examples_filled.png" width="85%">
</div>

## Key Findings

- **Cross-domain coordination creates the strongest leakage pressure.** Intra-team coordination across domain boundaries (CDLR ≈ 0.40–0.67) produces 2–3× more leakage than mediated or cross-user interactions.
- **Multi-party settings reshape but do not uniformly amplify privacy risks.** Group chat leakage is comparable to dyadic mediation, competitive pressure suppresses self-disclosure, and affinity-tier compliance is near-perfect.
- **The Abstraction Paradox.** Privacy instructions that teach agents *how* to abstract sensitive information paradoxically cause them to discuss it *more* — sanctioned privacy language increases the surface area of partial disclosure.

## 🏆 Leaderboard

Results under **L0 (Unconstrained)** — agents receive no privacy guidance and must infer boundaries from social norms alone.

> Privacy metrics (↓ lower is better): **CDLR** (Cross-Domain), **MLR** (Mediated Comm.), **CULR** (Cross-User), **MPLR** (Group Chat), **HALR** (Hub-and-Spoke), **CSLR** (Competitive Self-Leakage).
> Quality metrics (↑ higher is better): **ACS** (Affinity Compliance), **IAS** (Info. Abstraction), **TCQ** (Task Completion Quality).

| Model | CDLR↓ | MLR↓ | CULR↓ | MPLR↓ | HALR↓ | CSLR↓ | ACS↑ | IAS↑ | TCQ↑ | Task%↑ |
|:------|:-----:|:----:|:-----:|:-----:|:-----:|:-----:|:----:|:----:|:----:|:------:|
| DeepSeek V3.2 | 0.51 | **0.21** | 0.19 | 0.22 | 0.14 | 0.10 | **1.00** | 0.76 | 0.77 | 83.6 |
| Qwen3-235B | 0.49 | 0.26 | **0.14** | 0.22 | **0.06** | 0.08 | 1.00 | 0.75 | 0.73 | 74.9 |
| Kimi K2.5 | 0.67 | 0.30 | 0.29 | 0.30 | 0.12 | 0.09 | 1.00 | 0.69 | 0.86 | 93.3 |
| MiniMax M2.1 | 0.62 | 0.25 | 0.17 | 0.20 | 0.20 | 0.10 | 0.99 | 0.75 | 0.77 | 80.8 |
| GPT-5 Mini | **0.40** | 0.23 | 0.16 | 0.18 | 0.11 | 0.09 | 1.00 | 0.75 | 0.69 | 68.2 |
| Claude Haiku 4.5 | 0.57 | 0.27 | 0.19 | 0.24 | 0.15 | 0.09 | 0.99 | 0.75 | 0.73 | 69.6 |
| Claude Sonnet 4.5 | 0.52 | 0.24 | 0.19 | **0.16** | 0.10 | 0.08 | 1.00 | 0.79 | 0.83 | 87.4 |
| Claude Sonnet 4.6 | 0.50 | 0.21 | 0.19 | 0.18 | 0.10 | **0.08** | 1.00 | **0.85** | **0.87** | **94.1** |

> **Bold** = best per column. See the [paper](https://arxiv.org/abs/2604.01487) for defense-level results (L0 → L1 → L2) and the full analysis.

## Quick Start

### Installation

```bash
git clone https://github.com/kingofspace0wzz/agentsocialbench.git
cd agentsocialbench
pip install -r requirements.txt
```

### Set Up API Keys

```bash
cp .env.example .env
# Edit .env with your API keys (OpenAI, Gemini, AWS Bedrock, etc.)
```

### Run a Scenario

```bash
# Simulate a cross-domain scenario
python -m prism.scripts.simulate \
  --scenario prism/data/samples/cd_sample_01.json \
  --llm openai --privacy-mode explicit

# Evaluate the simulation
python -m prism.scripts.evaluate \
  --simulation prism/simulations/cross_domain/openai/gpt-5-mini/explicit/sim_CD_*.json \
  --scenario prism/data/samples/cd_sample_01.json \
  --llm gemini
```

## Benchmark Categories

| Category | Code | N | Description |
|:---------|:----:|:-:|:------------|
| Cross-Domain | CD | 100 | Intra-team coordination across domain boundaries |
| Mediated Comm. | MC | 100 | Agent brokers human-to-human interaction |
| Cross-User | CU | 50 | Agents from different users interact via A2A protocol |
| Group Chat | GC | 28 | 3–6 users' agents in shared group chat |
| Hub-and-Spoke | HS | 23 | Coordinator aggregates from multiple participants |
| Competitive | CM | 23 | Agents compete for a resource under pressure |
| Affinity-Modulated | AM | 28 | Asymmetric affinity tiers modulate per-recipient sharing rules |

## Pipeline

AgentSocialBench follows a four-stage pipeline: **Generate → Simulate → Evaluate → Analyze**.

```bash
# 1. Generate scenarios
python -m prism.scripts.generate --category cd --count 10 --llm gemini

# 2. Simulate interactions
python -m prism.scripts.simulate \
  --batch-dir prism/data/scenarios/cross_domain \
  --llm openai --privacy-mode explicit

# 3. Evaluate privacy, abstraction, task completion, and behavior
python -m prism.scripts.evaluate \
  --batch-sim-dir prism/simulations/cross_domain/openai/gpt-5-mini/explicit \
  --batch-scenario-dir prism/data/scenarios/cross_domain \
  --llm gemini

# 4. Analyze results (generate tables and figures)
python -m prism.analysis.generate_all
```

## Privacy Instruction Levels

| Level | Mode | Description |
|:-----:|:-----|:------------|
| L0 | Unconstrained | No privacy guidance; agents infer norms from social context |
| L1 | Explicit | Hard privacy rules + acceptable abstractions |
| L2 | Full Defense | L1 + Domain Boundary Prompting + Abstraction Templates + Minimal Information Principle |

## Citation

If you use AgentSocialBench in your research, please cite our paper:

```bibtex
@misc{wang2026agentsocialbenchevaluatingprivacyrisks,
      title={AgentSocialBench: Evaluating Privacy Risks in Human-Centered Agentic Social Networks}, 
      author={Prince Zizhuang Wang and Shuli Jiang},
      year={2026},
      eprint={2604.01487},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.01487}, 
}
```

## Star History

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=kingofspace0wzz/agentsocialbench&type=Date&theme=dark" />
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=kingofspace0wzz/agentsocialbench&type=Date" />
  <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=kingofspace0wzz/agentsocialbench&type=Date" width="70%" />
</picture>
</div>
