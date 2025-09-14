"""
NCA-MoE Models Package
======================
- NCA-Net: N-gram Channel Attention Network
- Mixture of Experts: Dynamic expert expansion and routing

"""

from .nca_net import NCANet, SKLayer, AttentionBlock, create_nca_net
from .mixture_of_experts import NCA_MoE, ExpertNet, GateNet, create_nca_moe

__all__ = [
    'NCANet',
    'SKLayer', 
    'AttentionBlock',
    'create_nca_net',
    'NCA_MoE',
    'ExpertNet',
    'GateNet',
    'create_nca_moe'
]

__version__ = "1.0.0"
