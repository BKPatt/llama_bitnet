from plantuml import PlantUML

uml = '''
@startuml

skinparam classFontSize 12
skinparam classFontName Helvetica
skinparam classAttributeFontSize 10
skinparam classAttributeFontName Helvetica
skinparam classAttributeIconSize 0

class BitNetB158Model {
  - layers: List[BitNetB158Layer]
  - embed_tokens: nn.Embedding
  - norm: RMSNorm
  - lm_head: nn.Linear
  + __init__(config: BitNetB158Config)
  + forward(input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor
}

class BitNetB158Layer {
  - self_attn: BitNetAttention
  - mlp: BitNetMLP
  - input_layernorm: RMSNorm
  - post_attention_layernorm: RMSNorm
  + __init__(config: BitNetB158Config)
  + forward(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]
}

class BitNetAttention {
  - hidden_size: int
  - num_heads: int
  - head_dim: int
  - q_proj: QuantizedLinear
  - k_proj: QuantizedLinear
  - v_proj: QuantizedLinear
  - o_proj: QuantizedLinear
  - rotary_emb: RotaryEmbedding
  + __init__(config: BitNetB158Config)
  + forward(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]
}

class BitNetMLP {
  - gate_proj: QuantizedLinear
  - up_proj: QuantizedLinear
  - down_proj: QuantizedLinear
  - act_fn: Callable[[torch.Tensor], torch.Tensor]
  + __init__(config: BitNetB158Config)
  + forward(x: torch.Tensor) -> torch.Tensor
}

class QuantizedLinear {
  - in_features: int
  - out_features: int
  - weight: torch.Tensor
  - scale: torch.Tensor
  + __init__(in_features: int, out_features: int)
  + forward(input: torch.Tensor) -> torch.Tensor
  + quantize() -> None
}

class AbsmeanQuantization {
  + {static} quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
  + {static} dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor
}

class RMSNorm {
  - weight: torch.Tensor
  - variance_epsilon: float
  + __init__(hidden_size: int, eps: float = 1e-6)
  + forward(hidden_states: torch.Tensor) -> torch.Tensor
}

class RotaryEmbedding {
  - dim: int
  - max_position_embeddings: int
  - base: float
  - inv_freq: torch.Tensor
  + __init__(dim: int, max_position_embeddings: int = 2048, base: float = 10000.0)
  + forward(q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]
}

class BitNetB158Config {
  + vocab_size: int
  + hidden_size: int
  + intermediate_size: int
  + num_hidden_layers: int
  + num_attention_heads: int
  + max_position_embeddings: int
  + rms_norm_eps: float
  + __init__(**kwargs)
}

BitNetB158Model "1" *-- "n" BitNetB158Layer
BitNetB158Model "1" *-- "1" RMSNorm
BitNetB158Layer "1" *-- "1" BitNetAttention
BitNetB158Layer "1" *-- "1" BitNetMLP
BitNetB158Layer "1" *-- "2" RMSNorm
BitNetAttention "1" *-- "4" QuantizedLinear
BitNetAttention "1" *-- "1" RotaryEmbedding
BitNetMLP "1" *-- "3" QuantizedLinear
QuantizedLinear ..> AbsmeanQuantization : uses
BitNetB158Model ..> BitNetB158Config : uses
BitNetB158Layer ..> BitNetB158Config : uses
BitNetAttention ..> BitNetB158Config : uses
BitNetMLP ..> BitNetB158Config : uses

@enduml
'''

puml = PlantUML(url='http://www.plantuml.com/plantuml/png/')
png_data = puml.processes(uml)

with open('bitnet_quant_aware_training.png', 'wb') as f:
    f.write(png_data)

print("UML diagram saved as 'bitnet_quant_aware_training.png'")