from transformers import PreTrainedModel, PreTrainedTokenizer
import torch

from BitNetB158Model import BitNetB158Model

class BitNetB158Pipeline:
    def __init__(self, model: BitNetB158Model, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, input_texts, max_length=50, num_return_sequences=1):
        # Tokenize the input text
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        # Generate sequences
        outputs = self.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, num_return_sequences=num_return_sequences)

        # Decode the generated sequences
        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        return decoded_outputs

    def generate(self, input_ids, attention_mask=None, max_length=50, num_return_sequences=1):
        # Prepare inputs
        inputs = self.model.prepare_inputs_for_generation(input_ids, attention_mask)

        # Generate sequences
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )

        return outputs

    def save_pretrained(self, save_directory):
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, model_name_or_path, tokenizer_name_or_path=None):
        model = BitNetB158Model.from_pretrained(model_name_or_path)
        tokenizer = PreTrainedTokenizer.from_pretrained(tokenizer_name_or_path or model_name_or_path)
        return cls(model=model, tokenizer=tokenizer)
