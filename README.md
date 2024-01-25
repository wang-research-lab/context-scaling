# Predicting Task Performance with Context-aware Scaling Laws
This repo contains the source code for the paper: [Predicting Task Performance with Context-aware Scaling Laws](https://arxiv.org/abs/2510.14919).

<p align="center">
  📃 <a href="https://arxiv.org/abs/2510.14919" target="_blank">[Paper]</a> • 💻 <a href="https://github.com/wang-research-lab/context-scaling" target="_blank">[GitHub]</a> • 🤗 <a href="https://huggingface.co/collections/WangResearchLab/context-aware-scaling-laws-67d9d0a7968288a4788d6dea" target="_blank">[Hugging Face]</a>
</p>

## Installation
Clone this repository and install the dependencies. This codebase was built and tested with Python 3.10. 
```python
git clone https://github.com/wang-research-lab/context-scaling.git
cd context-scaling

conda create --name context-scaling python=3.10
conda activate context-scaling

pip install -r requirements.txt
pip install flash-attn==2.4.3 --no-build-isolation
```

## Usage
Our codebase supports both training (context extension via YaRN) and evaluation.

### Training
We reimplement the YaRN context extension method in `train.py` closely following the [official implementation](https://github.com/jquesnelle/yarn). We use Accelerate and FSDP for distributed training. Configuration is managed via Hydra, and the base configuration can be found at `configs/training/train.yaml`. For example, to extend the context limit of Llama-2-7b to 8k tokens, run:
```bash
accelerate launch --config_file configs/training/accelerate/fsdp_8gpu.yaml train.py run_name=Yarn-Llama-2-7b-8k model.pretrained_model_name_or_path=meta-llama/Llama-2-7b-hf rope_scaling.factor=2.0
```
Extended checkpoints of Llama-2 7b and 13b can be found [here](https://huggingface.co/collections/WangResearchLab/context-aware-scaling-laws-67d9d0a7968288a4788d6dea).

### Evaluation
Evaluation code can be found in `eval.py`. Configuration is managed via Hydra, and the base configuration can be found at `configs/eval/eval.yaml`. Supported datasets can be found at `configs/eval/dataset`. For example, to evaluate Yarn-Llama-2-7b-128k on GSM8K, run:
```bash
python eval.py dataset=gsm model=WangResearchLab/Yarn-Llama-2-7b-128k
```
> Note: To enable vLLM to process inputs that exceed the model's context window, set the environment variable **VLLM_ALLOW_LONG_MAX_MODEL_LEN** to 1. Furthermore, when using the base 7b and 13b models, set the environment variable **PATCH_VLLM_ROPE** to 1 to avoid an illegal memory access error.

## Citation
If you find this work useful or relevant to your work, please kindly cite our paper:
```bibtex
@misc{contextscaling2025,
  title={Predicting Task Performance with Context-aware Scaling Laws},
  author={Kyle Montgomery* and David Park* and Jianhong Tu and Michael Bendersky and Beliz Gunel and Dawn Song and Chenguang Wang},
  year={2025},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2510.14919}
}
```
