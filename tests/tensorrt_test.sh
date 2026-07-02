python3 - <<'PY'
import sys
print("Python:", sys.version)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
except Exception as e:
    print("torch check failed:", repr(e))

try:
    import tensorrt_llm
    print("tensorrt_llm:", tensorrt_llm.__version__)
except Exception as e:
    print("trtllm check failed:", repr(e))

try:
    import transformers
    print("transformers:", transformers.__version__)
except Exception as e:
    print("transformers check failed:", repr(e))

from tensorrt_llm import LLM
from transformers import AutoConfig
model_type = AutoConfig.from_pretrained(
    "google/gemma-4-E2B-it", token="", trust_remote_code=True
).model_type
llm = LLM(model="google/gemma-4-E2B-it")

print("===============done===============")

PY
