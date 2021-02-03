from dataclasses import dataclass
import os
from typing import Optional

@dataclass
class ENV:
    subscription_id: Optional[str] = os.environ.get('SUBSCRIPTION_ID')
    cpu_vm_size_scoring: Optional[str] = os.environ.ge('CPU_VM_SIZE_SCORING')
    cpu_vm_size: Optional[str] = os.environ.get('CPU_VM_SIZE')
    gpu_vm_size_scoring: Optional[str] = os.environ.get('GPU_VM_SIZE_SCORING')
    gpu_vm_size: Optional[str] = os.environ.get('GPU_VM_SIZE')
    vm_priority_scoring: Optional[str] = os.environ.get('VM_PRIORITY_SCORING')
    vm_priority: Optional[str] = os.environ.get('VM_PRIORITY')
    max_nodes_scoring: Optional[int] = os.environ.get('MAX_NODES_SCORING')
    max_nodes: Optional[int] = os.environ.get('MAX_NODES')
    min_nodes_scoring: Optional[int] = os.environ.get('MIN_NODES_SCORING')
    min_nodes: Optional[int] = os.environ.get('MIN_NODES')
    
#%%