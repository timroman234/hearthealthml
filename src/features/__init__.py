"""Feature engineering and selection modules."""

from .build_features import (
    calculate_cardiac_risk_score,
    calculate_heart_rate_reserve,
    create_age_groups,
    create_bp_category,
    create_cholesterol_risk,
    create_interaction_features,
    engineer_features,
)
from .selection import (
    remove_correlated_features,
    select_by_importance,
    select_by_rfe,
    select_by_variance,
)

__all__ = [
    "engineer_features",
    "create_age_groups",
    "create_bp_category",
    "create_cholesterol_risk",
    "calculate_heart_rate_reserve",
    "calculate_cardiac_risk_score",
    "create_interaction_features",
    "remove_correlated_features",
    "select_by_variance",
    "select_by_rfe",
    "select_by_importance",
]
