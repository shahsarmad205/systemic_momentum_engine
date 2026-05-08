"""Institutional model-selection validation components."""

from model_selection.factor_neutralization import (
    NeutralizationConfig,
    neutralize_scores_by_date,
    factor_variance_explained,
    compute_neutralization_diagnostics,
    format_neutralization_report,
)
from model_selection.direction_model import (
    DirectionModelConfig,
    DirectionMetaModel,
    DirectionPrediction,
    evaluate_direction_stability,
    format_direction_report,
)
from model_selection.signal_stability import (
    StabilityConfig,
    cross_section_normalize,
    compute_stability_metrics,
    learn_shrinkage_lambda,
    apply_signal_shrinkage,
    compute_stability_diagnostics,
    format_stability_report,
)
from model_selection.latent_regime import (
    LatentRegimeConfig,
    LatentRegimeModel,
    extract_latent_features,
    compute_soft_ic_by_regime,
    format_latent_regime_report,
)

__all__ = [
    # factor neutralization
    "NeutralizationConfig",
    "neutralize_scores_by_date",
    "factor_variance_explained",
    "compute_neutralization_diagnostics",
    "format_neutralization_report",
    # direction meta-model
    "DirectionModelConfig",
    "DirectionMetaModel",
    "DirectionPrediction",
    "evaluate_direction_stability",
    "format_direction_report",
    # signal stability
    "StabilityConfig",
    "cross_section_normalize",
    "compute_stability_metrics",
    "learn_shrinkage_lambda",
    "apply_signal_shrinkage",
    "compute_stability_diagnostics",
    "format_stability_report",
    # latent regime
    "LatentRegimeConfig",
    "LatentRegimeModel",
    "extract_latent_features",
    "compute_soft_ic_by_regime",
    "format_latent_regime_report",
]
