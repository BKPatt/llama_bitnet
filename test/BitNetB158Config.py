import json
from dataclasses import dataclass, asdict

@dataclass
class BitNetB158Config:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    quantization_bits: float = 1.58
    rope_theta: float = 10000.0
    rope_scaling: dict = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    quant_type: str = "absmean"
    output_attentions: bool = False
    output_hidden_states: bool = False

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.head_dim * self.num_attention_heads == self.hidden_size, "hidden_size must be divisible by num_attention_heads"
        if self.rope_scaling is None:
            self.rope_scaling = {"type": "linear", "factor": 1.0}

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_json(self, json_file):
        with open(json_file, 'w') as f:
            json.dump(asdict(self), f, indent=2)