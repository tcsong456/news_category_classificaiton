from dataclasses import dataclass
import os
from typing import Optional

@dataclass
class ENV:
    subscription_id: Optional[str] = os.environ.get('SUBSCRIPTION_ID')
    workspace: Optional[str] = os.environ.get('WORKSPACE')
    resource_group: Optional[str] = os.environ.get('RESOURCE_GROUP')
    build_id: Optional[str] = os.environ.get('BUILD_ID')
    build_url: Optional[str] = os.environ.get('BUILD_URL')
    cpu_vm_size_scoring: Optional[str] = os.environ.get('CPU_VM_SIZE_SCORING')
    cpu_vm_size: Optional[str] = os.environ.get('CPU_VM_SIZE')
    gpu_vm_size_scoring: Optional[str] = os.environ.get('GPU_VM_SIZE_SCORING')
    gpu_vm_size: Optional[str] = os.environ.get('GPU_VM_SIZE')
    vm_priority_scoring: Optional[str] = os.environ.get('VM_PRIORITY_SCORING')
    vm_priority: Optional[str] = os.environ.get('VM_PRIORITY')
    max_nodes_scoring: Optional[int] = int(os.environ.get('MAX_NODES_SCORING'))
    max_nodes: Optional[int] = int(os.environ.get('MAX_NODES'))
    min_nodes_scoring: Optional[int] = int(os.environ.get('MIN_NODES_SCORING'))
    min_nodes: Optional[int] = int(os.environ.get('MIN_NODES'))
    datastore_name: Optional[str] = os.environ.get('DATASTORE_NAME')
    pipeline_name: Optional[str] = os.environ.get('PIPELINE_NAME')
    vocab: Optional[str] = os.environ.get('VOCAB')
    experiment_name: Optional[str] = os.environ.get('EXPERIMENT_NAME')
    
    model_name: Optional[str] = os.environ.get('MODEL_NAME')
    model_version: Optional[str] = os.environ.get('MODEL_VERSION')
    cuda: Optional[str] = os.environ.get('CUDA')
    batch_size_train: Optional[str] = os.environ.get('BATCH_SIZE_TRAIN')
    batch_size_eval: Optional[str] = os.environ.get('BATCH_SIZE_EVAL')
    tokenizer: Optional[str] = os.environ.get('TOKENIZER')
    is_sentence: Optional[str] = os.environ.get('IS_SENTENCE')
    max_seq_len: Optional[int] = int(os.environ.get('MAX_SEQ_LEN'))
    fasttext_mode: Optional[str] = os.environ.get('FASSTTEXT_MODE')
    hidden_size: Optional[int] = int(os.environ.get('HIDDEN_SIZE'))
    num_layers: Optional[int] = int(os.environ.get('NUM_LAYERS'))
    dropout: Optional[float] = float(os.environ.get('DROPOUT'))
    embedding_size: Optional[int] = int(os.environ.get('EMBEDDING_SIZE'))
    embedding_trainable: Optional[str] = os.environ.get('EMBEDDING_TRAINABLE')
    use_word_embedding: Optional[str] = os.environ.get('USE_WORD_EMBEDDING')
    bidirectional: Optional[bool] = bool(os.environ.get('BIDIRECTIONAL'))
    learning_rate: Optional[float] = float(os.environ.get('LEARNING_RATE'))
    epochs: Optional[int] = int(os.environ.get('EPOCHS'))
    
#%%
