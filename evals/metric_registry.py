# tol: absolute tolerance; rel_tol: fraction of |baseline|.
# A worsening counts only beyond max(tol, rel_tol * |baseline|).

SAFETY_GATE = {"direction": "higher", "kind": "gate", "tol": 0.02, "rel_tol": 0.0}
TOXICITY_GATE = {"direction": "lower", "kind": "gate", "tol": 0.02, "rel_tol": 0.0}
QUALITY_GUARD = {"direction": "higher", "kind": "guardrail", "tol": 0.05, "rel_tol": 0.0}

LATENCY_GUARD = {"direction": "lower", "kind": "guardrail", "tol": 0.0, "rel_tol": 0.25}
COST_GUARD = {"direction": "lower", "kind": "guardrail", "tol": 0.0, "rel_tol": 0.15}
SUCCESS_GUARD = {"direction": "higher", "kind": "guardrail", "tol": 1.0, "rel_tol": 0.0}
ERROR_GUARD = {"direction": "lower", "kind": "guardrail", "tol": 1.0, "rel_tol": 0.0}
SLO_BOOL_GUARD = {"direction": "higher", "kind": "guardrail", "tol": 0.0, "rel_tol": 0.0, "bool": True}
INFO = {"direction": "higher", "kind": "info", "tol": 0.0, "rel_tol": 0.0}

# Latency metrics that affect the verdict; stage p50s are info only.
LATENCY_GUARDED = {"ops.latency.e2e_p95_ms", "ops.latency.ttft_p95_ms"}


def rule_for(metric_id: str) -> dict:
    """Return the comparison rule for a metric id."""
    mid = metric_id

    if mid.startswith("safety.toxicity.") and mid.endswith(".avg_score"):
        return TOXICITY_GATE
    if mid.startswith("safety.") and mid.endswith(".avg_score"):
        return SAFETY_GATE

    if mid.endswith(".avg_score"):
        return QUALITY_GUARD

    if mid in LATENCY_GUARDED:
        return LATENCY_GUARD
    if mid == "ops.cost.cost_per_query_usd":
        return COST_GUARD
    if mid == "ops.reliability.success_rate":
        return SUCCESS_GUARD
    if mid == "ops.reliability.error_rate":
        return ERROR_GUARD
    if mid.endswith("_pass"):
        return SLO_BOOL_GUARD

    return INFO