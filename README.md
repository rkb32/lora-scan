# lora-scan

Weight-only backdoor screening for LoRA adapters on Hugging Face. Nothing is executed:
it reads `adapter_model.safetensors` (pickle `.bin` adapters are refused) and computes
spectral statistics of each attention update, following
[Weight Space Detection of Backdoors in LoRA Adapters](https://arxiv.org/abs/2602.15195).

```bash
pip install -e .
lora-scan munkhbayar-batkhuu/Llama-3.2-3B-Instruct-Mongolian
```

Status: feature extraction only. No calibrated detector yet, so there is no risk score.

Differences from the paper's code:
- Finds q/k/v/o updates on any layer and any PEFT key prefix (the paper hard-codes one layer).
- Kurtosis is computed scale-invariantly; the paper's `1e-12` epsilon dominates real-sized updates.
- Records the `lora_alpha / r` scale separately from the raw spectral stats.
