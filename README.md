# QLoRA Fine-Tuning: Qwen2.5-1.5B-Instruct → Hint-Only Coding Assistant

This fine-tunes `Qwen/Qwen2.5-1.5B-Instruct` with **QLoRA** (4-bit NF4 quantization + LoRA adapters) so the model gives **hints instead of full solution code** for a coding-assessment context.

Pipeline:
1. Install deps
2. Define **all hyperparameters** up front
3. Quantize base model with `bitsandbytes` (4-bit)
4. Load & format `dataset.jsonl` (chat-format, 300 examples)
5. Attach LoRA adapters, train with `SFTTrainer`
6. Evaluate: base model vs fine-tuned model, side-by-side, plus loss/leakage metrics
