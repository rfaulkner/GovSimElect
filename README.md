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
│   ├── main.py            # Simulation entrypoint (Hydra-based)
│   ├── run.py             # Simulation loop — builds and executes phases
│   ├── analysis/          # Analysis and plotting modules
│   ├── conf/              # Hydra configuration files
│   │   ├── config.yaml    # Top-level Hydra config
│   │   └── experiment/    # Experiment YAML variants
│   ├── environment/       # Environment classes (ConcurrentEnv, PerturbationEnv)
│   ├── persona/           # Agent state, cognition components, memory
│   │   ├── persona.py     # PersonaAgent — stateful agent container
│   │   ├── cognition/     # Act, Converse, Reflect, Leaders, etc.
│   │   ├── memory/        # Associative memory and scratch
│   │   └── common.py      # Shared data structures
│   ├── phases/            # Modular simulation phase classes
│   │   ├── base.py        # Phase ABC, PhaseContext, shared helpers
│   │   ├── policy_making.py
│   │   ├── election.py
│   │   ├── harvesting.py
│   │   ├── discussion.py
│   │   └── reflection.py
│   └── utils/             # Logger, model wrappers
├── llm_judge/             # LLM-as-judge evaluation framework
├── utils/                 # General utilities
├── setup.sh               # Default setup (Conda)
├── setup_venv.sh          # Virtual environment setup
├── setup_vllm.sh          # vLLM setup for local open-weight models
├── requirements.txt       # Pip dependencies (default)
└── requirements_venv.txt  # Pip dependencies (venv / cluster)
```

## Simulation Configuration

Simulation behavior is controlled by [Hydra](https://hydra.cc/) YAML files in `simulation/conf/`. The top-level `config.yaml` sets global defaults (LLM path, debug flags), while experiment-specific files in `conf/experiment/` define agent populations, environment parameters, and phase ordering.

### Experiment YAML Structure

Each experiment file (e.g. `fish_baseline_concurrent_leaders.yaml`) configures:

| Section | Purpose |
|---------|---------|
| `phases` | Ordered list of simulation phase names (see below) |
| `env` | Environment parameters — resource pool, regeneration, harvesting order, perturbations |
| `personas` | Agent definitions — names, traits, number of agents |
| `agent` | Agent cognition settings — prompts, conversation limits, leader population type |

**Example experiment config:**

```yaml
name: fishing_${code_version}/${group_name}
scenario: fishing
debug: true
seed: 0

phases:
  - policy_making
  - election
  - harvesting
  - discussion
  - reflection

env:
  max_num_rounds: 6
  initial_resource_in_pool: 100
  num_agents: 8
  disinformation: false
  harvesting_order: concurrent
  regen_factor_range: [1.0, 3.0]

agent:
  leader_population: balanced    # balanced, none, one_prosocial, etc.
  act:
    harvest_strategy: one_step
  converse:
    max_conversation_steps: 50
```

### Key Configuration Options

**`env.disinformation`** — When `true`, elected leaders can inject disinformation into harvest reports.

**`agent.leader_population`** — Controls the Social Value Orientation (SVO) distribution of leader candidates: `balanced`, `none` (no leaders), `one_prosocial`, `one_competitive`, etc.

**`env.harvesting_order`** — Either `concurrent` (all agents harvest simultaneously) or `random-sequential`.

## Simulation Phases

The simulation is organized into configurable **phases** that execute in sequence each round. Phase ordering is defined by the `phases` list in the experiment YAML.

### Default Phase Order

```
PolicyMaking → Election → Harvesting(+Report) → Discussion → Reflection
```

### Built-in Phases

| Phase Name | Class | Description |
|------------|-------|-------------|
| `policy_making` | `PolicyMakingPhase` | Leaders generate policy agendas based on SVO, past harvest data, and the previous winning agenda |
| `election` | `ElectionPhase` | Non-leader agents vote on leader agendas; winner is determined and announced |
| `harvesting` | `HarvestingPhase` | Agents decide how many fish to catch; harvest report and public memories are generated at the end |
| `discussion` | `DiscussionPhase` | Group conversation at the restaurant — agents discuss strategy and share information |
| `reflection` | `ReflectionPhase` | Individual reflection at home — agents process the round's events into memories |

### Customizing Phase Order

To change the phase order, edit the `phases` list in your experiment YAML. For example, to run discussion before harvesting:

```yaml
phases:
  - policy_making
  - election
  - discussion
  - harvesting
  - reflection
```

To run without elections (e.g. for a no-leader experiment):

```yaml
phases:
  - harvesting
  - discussion
  - reflection
```

### Creating Custom Phases

You can add new phases by implementing the `Phase` abstract base class and registering them in `run.py`.

#### Step 1: Create the phase class

Create a new file in `simulation/phases/`, e.g. `my_phase.py`:

```python
"""MyPhase — description of what this phase does."""

from simulation.phases.base import Phase
from simulation.phases.base import PhaseContext


class MyPhase(Phase):
    """One-line summary."""

    @property
    def name(self) -> str:
        return "my_phase"

    def execute(self, ctx: PhaseContext) -> PhaseContext:
        # Access agents via ctx.personas (dict of PersonaAgent)
        # Access environment via ctx.env
        # Access config via ctx.cfg
        # Access the current observation via ctx.obs

        for agent_id, agent in ctx.personas.items():
            # Do something with each agent...
            pass

        # If stepping the environment:
        # ctx.agent_id, ctx.obs, _, termination = ctx.env.step(action)

        return ctx
```

The `PhaseContext` dataclass carries all mutable state between phases:
- **`ctx.personas`** — dict of agent ID → `PersonaAgent`
- **`ctx.env`** — the environment instance
- **`ctx.obs`** — current observation from the environment
- **`ctx.wrapper`** — LLM model wrapper for prompting
- **`ctx.round_num`** — current round number
- **`ctx.winner`** / **`ctx.agenda`** — current election results
- **`ctx.round_harvest_stats`** — per-round harvest data
- **`ctx.terminated`** — set to `True` to end the simulation

#### Step 2: Register the phase

In `simulation/run.py`, import your phase and add it to `PHASE_REGISTRY`:

```python
from simulation.phases.my_phase import MyPhase

PHASE_REGISTRY = {
    "policy_making": PolicyMakingPhase,
    "election": ElectionPhase,
    "harvesting": HarvestingPhase,
    "discussion": DiscussionPhase,
    "reflection": ReflectionPhase,
    "my_phase": MyPhase,  # Add your phase here
}
```

#### Step 3: Use it in config

Add the phase name to your experiment YAML:

```yaml
phases:
  - policy_making
  - election
  - harvesting
  - my_phase
  - discussion
  - reflection
```

#### Tips for Phase Development

- **Environment coordination**: The environment has internal sub-phases (`lake`, `pool_after_harvesting`, `restaurant`, `home`). If your phase needs to drive environment steps, use `while ctx.env.phase == "phase_name":` loops to stay in sync.
- **Termination**: Always check for termination after `env.step()` calls using `check_terminated(ctx, termination)` from `simulation.phases.base`.
- **Agent state sync**: Call `sync_agent_state(agent, ctx)` to push the current agenda, harvest report, and leader info onto an agent before it acts.
- **Logging**: Use `log_step(ctx, action)` after each `env.step()` to record game metrics.

