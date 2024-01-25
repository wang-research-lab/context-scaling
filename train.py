# Adapted from https://github.com/jquesnelle/yarn/blob/995db5b575e75230b3384d658f8b944c9662f775/finetune.py

from datetime import timedelta
import os

import hydra
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, 
    StateDictType, 
    FullStateDictConfig,
)
from accelerate import Accelerator
from accelerate.utils import (
    set_seed, 
    InitProcessGroupKwargs,
)
from datasets import load_dataset
from transformers import (
    default_data_collator, 
    get_linear_schedule_with_warmup, 
)

from scaled_rope.configuration_llama import LlamaConfig
from scaled_rope.modeling_llama_yarn import LlamaForCausalLM

@hydra.main(version_base="1.3", config_path="./configs/training", config_name="train")
def main(cfg):
    
    os.makedirs(cfg.accelerator.project_dir, exist_ok=True)
    set_seed(cfg.seed)
    
    accelerator = Accelerator(
        **cfg.accelerator,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))]
    )
    
    config = LlamaConfig.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
    )
    config.rope_scaling = dict(cfg.rope_scaling)
    config.max_position_embeddings = int(cfg.rope_scaling.factor * cfg.rope_scaling.original_max_position_embeddings)
    config.use_cache = False
    
    model = LlamaForCausalLM.from_pretrained(
        **cfg.model,
        config=config,
        torch_dtype=torch.bfloat16,
    )

    with accelerator.local_main_process_first():
        train_dataset = load_dataset(cfg.data.path)["train"].map(
            lambda x: {k: v[:config.max_position_embeddings] for k, v in x.items()}, 
            desc=f"Truncating to {config.max_position_embeddings} tokens.", 
            **cfg.data.map,
        )
            
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        **cfg.data.dataloader,
    )
    
    model = accelerator.prepare(model)

    optim = AdamW(
        model.parameters(),
        **cfg.optim.params,
        betas=tuple(cfg.optim.betas),
    )
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optim,
        **cfg.scheduler,
    )
    
    optim, train_dataloader, lr_scheduler = accelerator.prepare(
        optim, train_dataloader, lr_scheduler
    )

    accelerator.print(f"Begin training for {cfg.num_steps} steps on sequences of {config.max_position_embeddings} tokens.")
    accelerator.init_trackers(
        project_name="context-scaling",
    )
    pbar = tqdm(
        range(cfg.num_steps), 
        disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    
    model.train()
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            loss = model(**batch).loss
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.optim.max_grad_norm)
                accelerator.log({"loss": loss.item(), "grad_norm": grad_norm.item()}, step=completed_steps)
            
            optim.step()
            lr_scheduler.step()
            optim.zero_grad()
        
        if accelerator.sync_gradients:
            pbar.update(1)
            completed_steps += 1
        
        if completed_steps >= cfg.num_steps:
            break
    
    accelerator.print(f"Finished training for {cfg.num_steps} steps. Saving model to {cfg.accelerator.project_dir}.")
    
    accelerator.wait_for_everyone()
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        state_dict = accelerator.get_state_dict(model, unwrap=False)
    accelerator.unwrap_model(model).save_pretrained(
        cfg.accelerator.project_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
    )
    
    accelerator.end_training()
    
if __name__ == "__main__":
    main()
