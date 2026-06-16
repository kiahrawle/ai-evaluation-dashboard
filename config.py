"""Central configuration. Edit here, not in the modules."""
from pathlib import Path
import yaml

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# The model used to judge truthfulness. Use the strongest you can afford.
# A model should NEVER judge its own outputs — keep this distinct from the
# models under test to avoid self-preference bias.
JUDGE_MODEL = "claude-opus-4-8"

# Default model under test when the CLI is invoked without --models.
# Overridable via config/config.yaml (model.name).
MODEL_NAME = "claude-haiku-4-5-20251001"

# Generation settings for the models under test.
GEN_TEMPERATURE = 0.0          # deterministic so runs are reproducible
GEN_MAX_TOKENS = 256

# Local embedding model for the cheap semantic baseline scorer.
EMBED_MODEL = "all-MiniLM-L6-v2"

# Optional system prompt for models under test. Leave None for the raw
# zero-shot behavior TruthfulQA was designed to probe. Set a guardrail prompt
# here later to measure how much it reduces hallucination.
SYSTEM_PROMPT = None

# ---- Defaults for YAML-overridable settings (defined up front so a missing
# or malformed config/config.yaml still leaves every name bound). ----
RAG_ENABLED = True
GUARDRAILS_ENABLED = True
RISK_REALTIME = False
DEFAULT_LIMIT = 20
JUDGE_MODELS = [JUDGE_MODEL]
# Self-consistency sampling is OFF by default: it makes N extra generations per
# question. Turn it on via config.yaml (self_consistency.enabled) when you want
# the reference-free hallucination signal.
SELF_CONSISTENCY_ENABLED = False
SELF_CONSISTENCY_N = 5

# Load optional YAML config to override the defaults above.
_yaml_path = ROOT / "config" / "config.yaml"
if _yaml_path.exists():
	try:
		with _yaml_path.open("r", encoding="utf-8") as fh:
			_cfg = yaml.safe_load(fh) or {}
		# Default model under test (does NOT change the judge model).
		if isinstance(_cfg.get("model"), dict):
			_m = _cfg["model"].get("name")
			if _m:
				MODEL_NAME = _m
		# Feature toggles
		RAG_ENABLED = bool(_cfg.get("rag", {}).get("enabled", RAG_ENABLED))
		GUARDRAILS_ENABLED = bool(_cfg.get("guardrails", {}).get("enabled", GUARDRAILS_ENABLED))
		RISK_REALTIME = bool(_cfg.get("risk", {}).get("realtime", RISK_REALTIME))
		DEFAULT_LIMIT = int(_cfg.get("run", {}).get("default_limit", DEFAULT_LIMIT))
		# Optional list of judge models for multi-judge verification
		jm = _cfg.get("judges")
		if isinstance(jm, list) and jm:
			JUDGE_MODELS = jm
		# Self-consistency sampling
		_sc = _cfg.get("self_consistency", {})
		if isinstance(_sc, dict):
			SELF_CONSISTENCY_ENABLED = bool(_sc.get("enabled", SELF_CONSISTENCY_ENABLED))
			SELF_CONSISTENCY_N = int(_sc.get("n", SELF_CONSISTENCY_N))
	except Exception:
		# Keep the defaults defined above if the YAML is unreadable/malformed.
		pass
