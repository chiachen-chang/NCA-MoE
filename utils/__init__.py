"""
NCA-MoE Utils Package
=====================

This package contains utility functions for the NCA-MoE framework:
- Data loading and preprocessing
- N-gram representation
- Evaluation metrics

Author: Jiachen Zhang
Institution: Beijing University of Posts and Telecommunications
"""

from .data_loader import (
    WebSecurityDataset,
    get_dataloaders,
    select_replay_samples,
    create_replay_buffer,
    get_dataset_statistics
)

from .ngram_representation import (
    NgramRepresentation,
    preprocess_web_request
)

from .evaluation_metrics import (
    ContinualLearningMetrics,
    calculate_classification_metrics,
    evaluate_model_performance,
    create_performance_matrix,
    print_evaluation_summary
)

__all__ = [
    # Data loading
    'WebSecurityDataset',
    'get_dataloaders',
    'select_replay_samples',
    'create_replay_buffer',
    'get_dataset_statistics',
    
    # N-gram representation
    'NgramRepresentation',
    'preprocess_web_request',
    
    # Evaluation metrics
    'ContinualLearningMetrics',
    'calculate_classification_metrics',
    'evaluate_model_performance',
    'create_performance_matrix',
    'print_evaluation_summary'
]

__version__ = "1.0.0"
