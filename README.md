# VLM Eval Benchmark

這是一個用於測試影片 VLM (Vision-Language Model) 效能並使用 LLM 進行自動評分的基準測試框架。

### Option A

```shell
# 1. 建立虛擬環境並安裝依賴
uv sync

# 2. 硬體測試 (確認 GPU 可用)
uv run python scripts/gpu_test.py
```

### Option B

```shell
# 1. 建立並啟用虛擬環境
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# 2. 安裝依賴
pip install --upgrade pip
pip install -r requirements.txt

# 若要進行開發，建議安裝此專案為可編輯模式
pip install -e .

# 3. 硬體測試 (確認 GPU 可用)
python3 scripts/gpu_test.py
```

---

## 執行批次測試

如果你有多個模型與多個影片目錄需要測試，請使用批次腳本。這會自動完成「推論」與「評分」的完整流程。

### 使用批次腳本
你可以編輯 `scripts/run_batch_eval.py` 裡面的 `video_dirs` 與 `model_ids` 清單，然後執行：

```shell
# 指定 GPU 並執行批次測試
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python scripts/run_batch_eval.py
```

### 腳本內容說明
該腳本會遍歷所有組合並生成對應的報告於 `runs/` 目錄下：
- **Inference**: 產出 `predictions.jsonl`
- **Judge**: 產出 `judge_results.jsonl` 與 `judge_summary.json`

---

## 單項執行指令

如果你只想單獨測試特定的模型或影片：

### 1. 執行基準測試 (Benchmark)
```shell
uv run python scripts/run_benchmark.py \
  --video_dir ./dataset/climbing_stair \
  --model_id google/gemma-4-E4B-it \
  --sample_fps 5.0
```

### 2. 執行評分 (Judge)
```shell
uv run python scripts/run_judge.py \
  --video_dir ./dataset/climbing_stair \
  --model_id google/gemma-4-E4B-it \
  --judge_model gpt-4o
```

`judge_results.jsonl` 會替每筆 prediction 寫入 `bleu`、`rouge_l`、`cider`，並同步保留在 `text_metrics`；`judge_summary.json` 會寫入平均 BLEU、corpus BLEU、ROUGE-L、CIDEr。

如果只想補 BLEU/ROUGE/CIDEr，不呼叫 Judge LLM：

```shell
uv run python scripts/run_judge.py \
  --video_dir ./dataset/climbing_stair \
  --model_id google/gemma-4-E4B-it \
  --skip_llm_judge
```

---

## 專案結構

- `src/vlm_eval/`: 核心邏輯代碼。
- `scripts/`:
    - `run_batch_eval.py`: **主要入口**，用於跑多模型、多數據集的批次任務。
    - `run_benchmark.py`: 單次執行影片推論任務。
    - `run_judge.py`: 單次執行 LLM 評分任務。
    - `gpu_test.py`: GPU 效能基準測試與環境檢查。
- `dataset/`: 存放影片數據集的目錄。
- `runs/`: 存放所有測試結果與 Log。
