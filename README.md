# ICRL Safety

Constrained fine-tuning of LLM orchestrators using Inverse Constraint Reinforcement Learning on ST-WebAgentBench.

## Setup

**1. Clone repos (one-time, place alongside this repo)**
```bash
git clone https://github.com/ServiceNow/BrowserGym ../BrowserGym
git clone https://github.com/segev-shlomov/ST-WebAgentBench ../ST-WebAgentBench
```

**2. Create venv and install Python dependencies**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# BrowserGym core (newer version — not on PyPI as compatible version)
pip install -e ../BrowserGym/browsergym/core --no-deps
pip install pyparsing lxml beautifulsoup4 requests pydantic

# ST-WebAgentBench BrowserGym integration
pip install -e ../ST-WebAgentBench/browsergym/stwebagentbench --no-deps
pip install PyPDF2 rapidfuzz beartype nltk aiolimiter tiktoken tqdm text-generation

# Make the stwebagentbench root package importable
echo "$(realpath ../ST-WebAgentBench)" > venv/lib/python3.*/site-packages/stwebagentbench.pth

# Download NLTK tokenizer data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Install Chromium for Playwright
playwright install chromium
```

**3. SuiteCRM (Docker) — required to run benchmark tasks**

SuiteCRM is the web app the benchmark operates on. It runs locally via Docker.

```bash
# Pull images (Apple Silicon needs explicit platform flag)
docker pull --platform linux/amd64 public.ecr.aws/bitnami/mariadb:11.4
docker pull --platform linux/amd64 public.ecr.aws/bitnami/suitecrm:8

# Start containers (takes ~2-3 min on first boot)
cd ../ST-WebAgentBench/suitecrm_setup
docker compose up -d

# Wait until this returns HTTP 200:
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/public

# Load demo data (run once after first boot)
docker exec -i suitecrm_setup-mariadb-1 \
    mysql -u bn_suitecrm -pbitnami123 bitnami_suitecrm < init-db/demo_data.sql
```

To stop: `docker compose -f ../ST-WebAgentBench/suitecrm_setup/docker-compose.yaml down`

**4. API keys** — copy `.env.example` to `.env` and fill in:
```
OPENROUTER_API_KEY   # required — Qwen 72B via OpenRouter (actor + verifier)
WA_SUITECRM=http://localhost:8080/public   # required — SuiteCRM URL
HUGGINGFACE_TOKEN    # required for fine-tuning (Llama-3-8B)
WANDB_API_KEY        # optional — experiment tracking
```

**5. Verify everything works**
```bash
# Check 375 tasks are registered
python -c "import browsergym.stwebagentbench, gymnasium as gym; print(len([e for e in gym.envs.registry if 'STWebAgent' in e]), 'tasks registered')"

# Test Qwen API + data saving (no browser needed)
python scripts/smoke_collection.py

# Run one live episode with Qwen on SuiteCRM (browser required)
python scripts/run_demo.py --task-id 47 --max-steps 15
```

## Running a demo

```bash
# One episode, headless
python scripts/run_demo.py --task-id 47

# Watch the browser
python scripts/run_demo.py --task-id 47 --no-headless

# Different task
python scripts/run_demo.py --task-id 55
```

Output saved to `data/demos/single_run.jsonl`. Each line is one trajectory with:
- `is_safe`: ground truth from benchmark's `safety_report`
- `constraint_score`: 1.0 if no violations, 0.0 if any
- `steps`: list of `{action, observation}` pairs
- `reward`: 1.0 if task completed, 0.0 otherwise

## Full demo collection (250 safe demos)

```bash
# Discover task IDs for each safety category
python scripts/discover_task_ids.py --benchmark-root ../ST-WebAgentBench

# Collect 50 safe + all unsafe per task type
python scripts/collect_demos.py +demos=collection +compute=local \
    paths.benchmark_root=../ST-WebAgentBench run_name=collect_v1
```

Output: `data/demos/{task_type}_safe.jsonl` and `data/demos/{task_type}_unsafe.jsonl`

## Structure

```
src/          — constraint encoder, Lagrangian PPO, probing, data pipeline
configs/      — Hydra configs (compute targets, constraint, finetune, demos)
scripts/      — entry-point scripts for each pipeline stage
slurm/        — SLURM job templates (Carleton cluster)
gridworld/    — original gridworld ICRL implementation (reference)
```
