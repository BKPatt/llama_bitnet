from dataclasses import asdict, dataclass, field
import json

@dataclass
class BitNetB158Config:
    vocab_size: int = 128256
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    bos_token_id: int = 128000
    eos_token_id: list = field(default_factory=lambda: [128001, 128008, 128009])
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    mlp_bias: bool = False
    pretraining_tp: int = 1
    rope_theta: float = 500000.0
    rope_scaling: dict = field(default_factory=lambda: {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    })
    
    # BitNet-specific parameters
    quantization_bits: float = 1.58
    quant_type: str = "absmean"
    output_attentions: bool = False
    output_hidden_states: bool = False

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.head_dim * self.num_attention_heads == self.hidden_size, "hidden_size must be divisible by num_attention_heads"

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_json(self, json_file):
        with open(json_file, 'w') as f:
            json.dump(asdict(self), f, indent=2)