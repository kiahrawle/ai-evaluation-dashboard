"""Research modules: adversarial testing, long-context evaluation, leaderboard."""
from .adversarial import create_adversarial_test, detect_vulnerabilities, run_adversarial_test_suite
from .long_context import detect_memory_drift, detect_context_loss, evaluate_long_context
from .leaderboard import add_result, get_leaderboard, get_model_stats, compute_leaderboard_score

__all__ = [
    "create_adversarial_test",
    "detect_vulnerabilities",
    "run_adversarial_test_suite",
    "detect_memory_drift",
    "detect_context_loss",
    "evaluate_long_context",
    "add_result",
    "get_leaderboard",
    "get_model_stats",
    "compute_leaderboard_score",
]
