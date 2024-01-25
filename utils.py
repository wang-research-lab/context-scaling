from dataclasses import dataclass, asdict
from typing import Optional, List, Union

import torch
from transformers import (
    StoppingCriteria, 
)

class StopWordsCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer):
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self._cache_str = ''

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self._cache_str += self.tokenizer.decode(input_ids[0, -1])
        for stop_words in self.stop_words:
            if stop_words in self._cache_str:
                return True
        return False


def print_rank_0(message, debug=False, end="\n"):
    """Print from rank 0 only."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True, end=end)
    else:
        print(message, flush=True, end=end)
        

@dataclass
class Instance:
    instance_id: int
    input_text: str
    reference_output: Union[str, List[str]]
    predicted_output: Optional[str] = None
    prompt_length: Optional[int] = None
    generation_time: Optional[int] = None
    generation_length: Optional[int] = None
    metric_value: Optional[float] = None
    

def construct_prompt_single(
    instance: Instance, 
    prefix_sequence: str,
    suffix_sequence: str,
    add_reference_output: Optional[bool] = False,
    add_trailing_separator: Optional[bool] = False,
    separator_sequence: Optional[str] = None,
    ) -> str:
        
    prompt = prefix_sequence + instance.input_text + suffix_sequence 
    if add_reference_output:
        prompt += instance.reference_output[0] if isinstance(instance.reference_output, list) else instance.reference_output
    if add_trailing_separator:
        prompt += separator_sequence
    return prompt
    

def validate_instance(
    instance: Instance, 
    prefix_sequence: str,
    suffix_sequence: str,
    separator_sequence: str,
    max_length_per_example: int,
    tokenizer,
    skip_validation: bool = False,
) -> bool:
    
    if skip_validation:
        return True
    
    prompt = construct_prompt_single(
        instance=instance,
        prefix_sequence=prefix_sequence,
        suffix_sequence=suffix_sequence,
        add_reference_output=True,
        add_trailing_separator=True,
        separator_sequence=separator_sequence,
    )
    tokenized_prompt = tokenizer(prompt, add_special_tokens=False).input_ids
    if len(tokenized_prompt) <= max_length_per_example:  
        return True
    return False


class Dataset:
    
    def __init__(
        self,
        name: str,
        max_new_tokens: int,
        metric: str,
        stop_tokens: List[str],
        prefix_sequence: str,
        suffix_sequence: str,
        separator_sequence: str,
        n_shots: int,
        global_seed: int,
        local_seed: int,
        instances: List[Instance],
        shots: Optional[List[Instance]] = None,
        metric_value: Optional[float] = None,
    ):
    
        self.name = name
        self.max_new_tokens = max_new_tokens
        self.metric = metric
        self.stop_tokens = stop_tokens
        self.prefix_sequence = prefix_sequence
        self.suffix_sequence = suffix_sequence
        self.separator_sequence = separator_sequence
        self.n_shots = n_shots
        self.global_seed = global_seed
        self.local_seed = local_seed
        self.instances = instances
        self.shots = shots
        self.metric_value = metric_value
        
    
    def get_prompts(self):
        
        selected_shots = "".join(
            [construct_prompt_single(
                instance=shot,
                prefix_sequence=self.prefix_sequence,
                suffix_sequence=self.suffix_sequence,
                add_reference_output=True,
                add_trailing_separator=True,
                separator_sequence=self.separator_sequence,
            ) for shot in self.shots]
        ) if self.n_shots > 0 else ""
        
        prompts = [
            selected_shots + construct_prompt_single(
                instance=instance,
                prefix_sequence=self.prefix_sequence,
                suffix_sequence=self.suffix_sequence,
                add_reference_output=False,
                add_trailing_separator=False,
                separator_sequence=self.separator_sequence,
            ).rstrip() for instance in self.instances
        ]
        
        return prompts
        
    
    def to_dict(self):
        self.instances = [asdict(instance) for instance in self.instances]
        if self.shots is not None:
            self.shots = [asdict(shot) for shot in self.shots]
        return self.__dict__
    

def patch_vllm_rope(scale_factor):
    """
    This patch fixes an illegal memory access error which arises when using the 
    unscaled base model on long sequences (>8k). Simply, we increase the size of the 
    cache by a factor of x>1.0 which turns out to be sufficient up to 128k context.
    """
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(scale_factor * self.max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache
    RotaryEmbedding._compute_cos_sin_cache = _compute_cos_sin_cache