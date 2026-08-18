# ABOUTME: Killarney environment for ICRL — code and model caches on /project, trace outputs on /scratch
# ABOUTME: Source before sbatch: set -a; source scratch/killarney_environment.sh; source .env; set +a
export ICRL_ROOT=/project/aip-s2ganapa/kunwar/icrl
export ICRL_VENV=/project/aip-s2ganapa/kunwar/venvs/icrl_v4
export HF_HOME=/project/aip-s2ganapa/kunwar/hf_cache
export PLAYWRIGHT_BROWSERS_PATH=/project/aip-s2ganapa/kunwar/playwright-browsers
export REPOS_ROOT=/project/aip-s2ganapa/kunwar
export STWEBAGENT_ROOT=/project/aip-s2ganapa/kunwar/ST-WebAgentBench
export BROWSERGYM_ROOT=/project/aip-s2ganapa/kunwar/BrowserGym
export SCRATCH=/scratch/kunwar
export ICRL_ACCOUNT=aip-s2ganapa
