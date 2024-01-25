import os
from typing import List
import json

import hydra
import numpy as np
from vllm import LLM, SamplingParams
from transformers import (
    set_seed,
    AutoTokenizer,
)
from datasets import get_dataset_config_names

from dataset_constructors import dataset_name_to_constructor
from metrics import metric_to_metric_fn
from utils import (
    print_rank_0,
    Dataset,
    validate_instance,
    patch_vllm_rope,
)
    
def construct_dataset(
    name: str,
    max_new_tokens: int,
    metric: str,
    stop_tokens: List[str],
    prefix_sequence: str,
    suffix_sequence: str,
    separator_sequence: str,
    n_shots: int,
    max_length_per_example: int,
    tokenizer,
    max_eval_samples: int,
    global_seed: int,
    local_seed: int,
    config: str,
    skip_validation: bool = False,
) -> Dataset:
    
    if n_shots > 0:
        training_instances = dataset_name_to_constructor[name](split='train', seed=global_seed, config=config)
        training_instances_filtered = []
        for instance in training_instances:
            if validate_instance(instance, prefix_sequence, suffix_sequence, separator_sequence, max_length_per_example, tokenizer, skip_validation):
                training_instances_filtered.append(instance)
        print_rank_0(f"[Training Instances] Before filtering: {len(training_instances)}, \
            After filtering: {len(training_instances_filtered)}, \
            % remaining {100*len(training_instances_filtered)/len(training_instances)}")
        assert len(training_instances_filtered) >= n_shots, "Not enough training examples after filtering."
        local_rng = np.random.default_rng(local_seed)
        local_rng.shuffle(training_instances_filtered)
        training_instances_filtered = training_instances_filtered[:n_shots]
    
    testing_instances = dataset_name_to_constructor[name](split='test', seed=global_seed, config=config)
    testing_instances_filtered = []
    for instance in testing_instances:
        if validate_instance(instance, prefix_sequence, suffix_sequence, separator_sequence, max_length_per_example, tokenizer, skip_validation):
            testing_instances_filtered.append(instance)
    print_rank_0(f"[Testing Instances] Before filtering: {len(testing_instances)}, \
        After filtering: {len(testing_instances_filtered)}, \
        % remaining {100*len(testing_instances_filtered)/len(testing_instances)}")
    np.random.shuffle(testing_instances_filtered)
    testing_instances_filtered = testing_instances_filtered[:max_eval_samples]
    
    return Dataset(
        name=name,
        max_new_tokens=max_new_tokens,
        metric=metric,
        stop_tokens=list(stop_tokens) + [prefix_sequence] if prefix_sequence else list(stop_tokens),
        prefix_sequence=prefix_sequence,
        suffix_sequence=suffix_sequence,
        separator_sequence=separator_sequence,
        n_shots=n_shots,
        global_seed=global_seed,
        local_seed=local_seed,
        instances=testing_instances_filtered,
        shots=training_instances_filtered if n_shots > 0 else None,
        metric_value=None,
    )
    
    
def run_inference(
    model,
    dataset,
) -> None:

    prompts = dataset.get_prompts()
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=dataset.max_new_tokens,
        stop=dataset.stop_tokens,
    )

    responses = model.generate(prompts, sampling_params)

    for response, instance in zip(responses, dataset.instances):
        instance.prompt_length = len(response.prompt_token_ids)
        instance.predicted_output = response.outputs[0].text
        instance.generation_length = len(response.outputs[0].token_ids)


def run_metrics(dataset: Dataset) -> None:
    if dataset.metric == 'None':
        dataset.metric_value = None
        return
    else:
        metric_fn = metric_to_metric_fn[dataset.metric]
        for instance in dataset.instances:
            if isinstance(instance.reference_output, list):
                instance.metric_value = float(max([metric_fn(gold, instance.predicted_output) for gold in instance.reference_output]))
            else:
                instance.metric_value = float(metric_fn(instance.reference_output, instance.predicted_output))
        dataset.metric_value = float(np.mean([instance.metric_value for instance in dataset.instances]))
    
    
@hydra.main(version_base="1.3", config_path="./configs/eval", config_name="eval")
def main(cfg):
    
    set_seed(cfg.global_seed)

    output_dir = os.path.join(cfg.output_dir, cfg.dataset.name)
    os.makedirs(output_dir, exist_ok=True)
    
    if os.getenv("PATCH_VLLM_ROPE"):
        patch_vllm_rope(32.0)
        
    model_name = cfg.model.split("/")[-1]

    model = LLM(
        model=cfg.model,
        max_model_len=131072,
        **cfg.vllm,
    )
    
    if cfg.dataset.name == "deepmind-math":
        configs = get_dataset_config_names("deepmind/math_dataset")
    else:
        configs = [""]
    for config in configs:
        for n_shots in cfg.n_shots:

            if cfg.dataset.name == "deepmind-math":
                run_name = f"dataset={cfg.dataset.name},model={model_name},config={config},shots={n_shots},local_seed={cfg.local_seed}.json"
                if os.path.exists(os.path.join(output_dir, run_name)) and not cfg.override_existing_runs:
                    print_rank_0(f"Skipping {model_name} on {cfg.dataset.name} using config {config} with {n_shots} shots and local seed {cfg.local_seed}.")
                    continue
                print_rank_0(f"Running {model_name} on {cfg.dataset.name} using config {config} with {n_shots} shots and local seed {cfg.local_seed}.")
            else:
                run_name = f"dataset={cfg.dataset.name},model={model_name},shots={n_shots},local_seed={cfg.local_seed}.json"
                if os.path.exists(os.path.join(output_dir, run_name)) and not cfg.override_existing_runs:
                    print_rank_0(f"Skipping {model_name} on {cfg.dataset.name} with {n_shots} shots and local seed {cfg.local_seed}.")
                    continue
                print_rank_0(f"Running {model_name} on {cfg.dataset.name} with {n_shots} shots and local seed {cfg.local_seed}.")

            
            tokenizer = AutoTokenizer.from_pretrained(cfg.model)

            dataset = construct_dataset(
                **cfg.dataset,
                n_shots=n_shots,
                max_length_per_example=cfg.max_length_per_example,
                tokenizer=tokenizer,
                max_eval_samples=cfg.max_eval_samples,
                global_seed=cfg.global_seed,
                config=config,
                local_seed=cfg.local_seed,
            )

            run_inference(
                model=model,
                dataset=dataset,
            )
            
            run_metrics(dataset)
            
            with open(os.path.join(output_dir, run_name), "w") as f:
                json.dump(dataset.to_dict(), f, indent=4)

if __name__ == "__main__":
    main()
