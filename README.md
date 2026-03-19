# AgentElect: Governance of the Commons Simulation with Elections

AgentElect extends the [GovSim](https://github.com/giorgiopiatti/GovSim) framework to study how leadership structures affect collective resource management. This project is primarily designed to produce **reproducible results** for comparing three population leadership variants:

- **Elected-Leader**: Leaders are chosen through an election process among agents.
- **Fixed-Leader**: A pre-assigned leader governs the population throughout the simulation.
- **No-Leader**: Agents self-govern without any designated leadership.

![AgentElect overview](imgs/agent_elect.png)

## Code Setup

First, clone the repository:

```bash
git clone --recurse-submodules https://github.com/rfaulkner/GovSimElect.git
cd GovSimElect
```

### Environment Setup

There are two setup scripts depending on your environment:

| Script | Environment | Description |
|--------|-------------|-------------|
| `setup.sh` | **Default (Conda)** | Installs dependencies via Conda and pip. Use this for standard local development. |
| `setup_venv.sh` | **Virtual Environment** | Creates a Python virtual environment (`.venv`) and installs dependencies via pip. Intended for HPC / cluster environments where Conda is unavailable. |

**Default setup (Conda):**
```bash
bash ./setup.sh
```

**Virtual environment setup:**
```bash
bash ./setup_venv.sh
source .venv/bin/activate
```

If you plan to run **open-weight models** locally via [vLLM](https://github.com/vllm-project/vllm), also run:
```bash
bash ./setup_vllm.sh
```

### Environment Variables

Copy the example `.env` file and fill in your API keys as needed:
```bash
cp .env_example .env
```

## Running Simulations

All launch scripts are located in `scripts/`. They are designed as SLURM job scripts (`sbatch`) but can also be run directly with `bash`.

### Open-Weight Models (Local GPU)

Use `launch_job.sh` to run a simulation with a locally-hosted open-weight model (e.g. via vLLM / HuggingFace Transformers). Requires GPU resources.

```bash
bash scripts/launch_job.sh [seed] [population] [disinformation]
```

**Arguments:**

| Argument | Default | Options |
|----------|---------|---------|
| `seed` | `1` | Any integer |
| `population` | `balanced` | `balanced`, `none`, `one_prosocial`, `one_altruistic`, `one_competitive`, `one_individualistic`, `lean_altruistic`, `lean_competitive` |
| `disinformation` | `true` | `true`, `false` |

The model is configured by editing the `model_id` variable inside the script. Supported models include Qwen, Llama-3, and Mistral variants.

**Example:**
```bash
bash scripts/launch_job.sh 3 none false
```

### API-Based Models

Use `launch_job_api.sh` to run simulations with API-hosted models (via OpenRouter, OpenAI, or Anthropic). No local GPU required for inference.

```bash
bash scripts/launch_job_api.sh [seed] [population] [disinformation]
```

Arguments are the same as above. The `model_id` variable in the script selects the API model. Supported providers include:

- **Google**: Gemini, Gemma-3 (via OpenRouter)
- **Meta**: Llama-3 (via OpenRouter)
- **Qwen**: Qwen-2.5 (via OpenRouter)
- **OpenAI**: GPT-4-Turbo, GPT-4o, GPT-3.5-Turbo
- **Anthropic**: Claude 3.5 Sonnet, Claude 3.5 Haiku (via OpenRouter)

There is also `launch_job_api_vanilla.sh` which runs the original GovSim simulation (without the election mechanism) using API-based models.

### Batch Experiments

Use batch scripts to launch sweeps over multiple seeds, population types, and disinformation settings. Jobs are submitted via `sbatch`.

**Simulation batch:**
```bash
bash scripts/launch_job_batch.sh
```
This iterates over populations (`balanced`, `none`) × disinformation (`true`, `false`) × seeds (`1..4`), launching a job for each combination.

**Social analysis batch:**
```bash
bash scripts/launch_social_batch.sh
```
Sweeps over all 8 population types × disinformation settings for social value orientation analysis.

**Judge analysis batch:**
```bash
bash scripts/launch_judge_batch.sh
```
Sweeps over populations (`one_prosocial`, `one_altruistic`, `one_competitive`, `one_individualistic`) × disinformation settings for LLM-judge sentiment analysis.

## Running Analyses

### Social Value Orientation Analysis

Computes social value orientation metrics from simulation outputs:

```bash
bash scripts/launch_social.sh [population] [disinformation] [model_name]
```

| Argument | Default | Example |
|----------|---------|---------|
| `population` | `balanced` | `one_prosocial` |
| `disinformation` | `true` | `false` |
| `model_name` | `Qwen/Qwen1.5-110B-Chat-GPTQ-Int4` | `openrouter-google/gemini-2.5-flash` |

### LLM-Judge Sentiment Analysis

Uses an LLM-as-judge to evaluate leader sentiment in simulation conversations:

```bash
bash scripts/launch_judge.sh [population] [disinformation] [sentiment_type]
```

| Argument | Default | Options |
|----------|---------|---------|
| `population` | `one_prosocial` | Any population variant |
| `disinformation` | `true` | `true`, `false` |
| `sentiment_type` | `persuasion` | `cooperation`, `persuasion`, `svo` |

See the [llm_judge/README.md](llm_judge/README.md) for more details on configuring the LLM judge.

### General Analysis

Run aggregate analysis scripts directly:

```bash
bash scripts/launch_analysis.sh
```

The analysis modules in `simulation/analysis/` include plotting (`plots.py`), preprocessing (`preprocessing.py`), per-group analysis (`group.py`), detailed result inspection (`details.py`), and social metrics (`social.py`).

## Project Structure

```
GovSimElect/
├── scripts/               # Launch scripts for simulations and analyses
├── simulation/
│   ├── main.py            # Original GovSim simulation entrypoint
│   ├── main_elect.py      # GovSimElect simulation entrypoint (with elections)
│   ├── analysis/          # Analysis and plotting modules
│   ├── conf/              # Hydra configuration files
│   ├── persona/           # Agent persona definitions
│   ├── scenarios/         # Scenario configurations
│   └── utils/             # Simulation utilities
├── llm_judge/             # LLM-as-judge evaluation framework
├── utils/                 # General utilities
├── setup.sh               # Default setup (Conda)
├── setup_venv.sh          # Virtual environment setup
├── setup_vllm.sh          # vLLM setup for local open-weight models
├── requirements.txt       # Pip dependencies (default)
└── requirements_venv.txt  # Pip dependencies (venv / cluster)
```
