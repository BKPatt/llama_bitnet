from transformers import PretrainedConfig

class BitNetB158Config(PretrainedConfig):
    model_type = "bitnet-b158"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        intermediate_size=16384,
        hidden_act="gelu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        rotary_emb_base=10000,
        hidden_dropout_prob=0.1,  # Added attribute
        attention_dropout=0.1,    # Added attribute
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.rotary_emb_base = rotary_emb_base
        self.hidden_dropout_prob = hidden_dropout_prob  # Initialize hidden_dropout_prob
        self.attention_dropout = attention_dropout      # Initialize attention_dropout
        self.use_cache = kwargs.get("use_cache", True)

    @classmethod
    def from_llama_config(cls, llama_config):
        return cls(
            vocab_size=llama_config.vocab_size,
            hidden_size=llama_config.hidden_size,
            num_hidden_layers=llama_config.num_hidden_layers,
            num_attention_heads=llama_config.num_attention_heads,
            intermediate_size=llama_config.intermediate_size,
            hidden_act=llama_config.hidden_act,
            max_position_embeddings=llama_config.max_position_embeddings,
            initializer_range=llama_config.initializer_range,
            layer_norm_eps=llama_config.layer_norm_eps,
            rotary_emb_base=llama_config.rotary_emb_base,
            hidden_dropout_prob=llama_config.hidden_dropout_prob,  # Include in from_llama_config method
            attention_dropout=llama_config.attention_dropout,      # Include in from_llama_config method
            use_cache=llama_config.use_cache,
        )
