### first start
```shell
  pip install -r requirements.txt
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### llm judge usge

```shell
  uv run llm_judge.py --judge_model "gpt-oss-20b" | "gpt-oss-120b" | "Google-Gemma-3-27B" | "Llama-3.1-70B" | "Llama-3.1-405B-Instruct-FP8"
```
