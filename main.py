import os
import random
import glob
import time
import logging
import argparse
import subprocess

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.utils import logging as transformers_logging
import decord
import numpy as np
from PIL import Image
from dotenv import load_dotenv


load_dotenv()


os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
transformers_logging.set_verbosity_error()


def get_hardware_name():
	"""
	取得硬體名稱(GPU優先)
	"""
	if torch.cuda.is_available():
		return torch.cuda.get_device_name(0)
	return os.uname().machine


def get_gpu_power_watts():
	"""
	讀取目前 GPU 功耗（W）
	若系統沒有 nvidia-smi 或不是 NVIDIA GPU，回傳 None
	"""
	try:
		result = subprocess.check_output(
				[
					"nvidia-smi",
					"--query-gpu=power.draw",
					"--format=csv,noheader,nounits"
					],
				stderr=subprocess.DEVNULL
				).decode("utf-8").strip().splitlines()

		if len(result) > 0:
			return float(result[0])
		return None
	except Exception:
		return None


def get_video_duration(video_reader):
	"""
	從 decord VideoReader 取得影片長度（秒）
	"""
	try:
		fps = video_reader.get_avg_fps()
		total_frames = len(video_reader)
		if fps and fps > 0:
			return total_frames / fps
	except Exception:
		pass
	return None


def sample_frames(video_path, num_frames=16):
	"""
	使用 decord 讀取影片並均勻抽幀
	回傳:
	  pil_frames: List[PIL.Image]
	  video_duration_sec: float 或 None
	  total_frames: int
	  original_fps: float 或 None
	"""
	try:
		vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
	except Exception as e:
		logging.error(f"⚠️ 無法讀取影片 {video_path}: {e}")
		return None, None, None, None

	total_frames = len(vr)
	if total_frames == 0:
		return None, None, None, None

	try:
		original_fps = vr.get_avg_fps()
	except Exception:
		original_fps = None

	video_duration_sec = get_video_duration(vr)

	indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
	frames = vr.get_batch(indices).asnumpy()

	pil_frames = [Image.fromarray(f) for f in frames]
	return pil_frames, video_duration_sec, total_frames, original_fps


def main():
	parser = argparse.ArgumentParser(description="Run VLM inference benchmark on videos")
	parser.add_argument("--video_dir", type=str, default="./dataset/climbing_stair",
					 help="Directory containing mp4/mkv files")
	parser.add_argument("--model_id", type=str, default="google/gemma-3-4b-it",
					 help="Hugging Face model ID")
	parser.add_argument("--num_frames", type=int, default=16,
					 help="Fixed number of sampled frames per video")
	parser.add_argument("--sample_size", type=int, default=20,
					 help="Number of videos to randomly sample")
	args = parser.parse_args()

	clean_video_dir = os.path.normpath(args.video_dir)
	ground_truth_name = os.path.basename(clean_video_dir)
	if not ground_truth_name or ground_truth_name == ".":
		ground_truth_name = "default_ground_truth"

	model_id = args.model_id
	model_name = model_id.split("/")[-1].replace("-it", "")

	log_dir = f"{model_name}-{args.num_frames}frames"
	os.makedirs(log_dir, exist_ok=True)
	main_log_file = os.path.join(log_dir, f"{ground_truth_name}.log")

	logging.basicConfig(
			level=logging.INFO,
			format="%(message)s",
			handlers=[
				logging.FileHandler(main_log_file, encoding="utf-8", mode="a"),
				logging.StreamHandler()
				]
			)
	logging.getLogger().setLevel(logging.INFO)

	hf_token = os.environ.get("HF_TOKEN")
	hardware_name = get_hardware_name()

	logging.info(f"載入模型與處理器: {model_id} ... \n")
	try:
		processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
		model = AutoModelForCausalLM.from_pretrained(
				model_id,
				token=hf_token,
				torch_dtype=torch.bfloat16,
				device_map="auto"
				)
	except Exception as e:
		logging.error(f"載入模型失敗: {e}")
		return

	if torch.cuda.is_available():
		torch.cuda.reset_peak_memory_stats()

	video_dir = args.video_dir
	video_paths = glob.glob(os.path.join(video_dir, "**/*.mp4"), recursive=True)
	video_paths += glob.glob(os.path.join(video_dir, "**/*.mkv"), recursive=True)

	logging.info(f"\n從 {video_dir} 尋找，共找到 {len(video_paths)} 支影片。\n")
	if len(video_paths) == 0:
		logging.error("❌ 找不到任何影片檔案。")
		return

	sample_size = min(args.sample_size, len(video_paths))
	sampled_videos = random.sample(video_paths, sample_size)

	logging.info(f"開始測試，隨機抽取 {sample_size} 支影片進行推論...\n")
	logging.info(f"Hardware Name: {hardware_name}")
	logging.info(f"Fixed Frames : {args.num_frames}")

	prompt_text = "Describe the main action accurately in under 10 words."

	results = []
	total_time = 0.0
	total_generated_tokens = 0
	successful_runs = 0
	total_video_duration = 0.0
	valid_duration_count = 0
	power_readings = []

	for i, v_path in enumerate(sampled_videos):
		logging.info(f"\n{'=' * 60}")
		logging.info(f"[{i+1}/{sample_size}] 處理影片: {v_path}")

		frames, video_duration_sec, total_video_frames, original_fps = sample_frames(
				v_path, num_frames=args.num_frames
				)

		if frames is None:
			continue

		content_items = [{"type": "image"} for _ in range(len(frames))]
		content_items.append({"type": "text", "text": prompt_text})

		messages = [
				{
					"role": "user",
					"content": content_items
					}
				]

		try:
			formatted_prompt = processor.apply_chat_template(
					messages,
					add_generation_prompt=True
					)

			inputs = processor(
					text=formatted_prompt,
					images=frames,
					return_tensors="pt"
					).to(model.device)

			if "pixel_values" in inputs:
				inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

			start_power = get_gpu_power_watts()
			start_time = time.time()

			with torch.no_grad():
				output_ids = model.generate(
						**inputs,
						max_new_tokens=150,
						do_sample=False,
						temperature=None,
						top_p=None
						)

			end_time = time.time()
			end_power = get_gpu_power_watts()

			generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
			response = processor.decode(generated_ids, skip_special_tokens=True)

			elapsed = end_time - start_time
			num_generated_tokens = len(generated_ids)
			tps = num_generated_tokens / elapsed if elapsed > 0 else 0.0
			sampled_fps = len(frames) / elapsed if elapsed > 0 else 0.0

			if start_power is not None and end_power is not None:
				power_readings.append((start_power + end_power) / 2.0)
			elif start_power is not None:
				power_readings.append(start_power)
			elif end_power is not None:
				power_readings.append(end_power)

			if video_duration_sec is not None:
				total_video_duration += video_duration_sec
				valid_duration_count += 1

			logging.info(f"影片長度: {video_duration_sec:.2f} sec" if video_duration_sec is not None else "影片長度: N/A")
			logging.info(f"Average Query Latency: {elapsed:.2f} sec")
			logging.info(f"Frames Per Second: {sampled_fps:.2f}")
			logging.info(f"Throughput: {tps:.2f} tokens/sec")
			logging.info(f"模型回答: {response.strip()}")
			logging.info("=" * 60)

			results.append({
				"video": v_path,
				"query_latency_sec": elapsed,
				"video_duration_sec": video_duration_sec,
				"tokens": num_generated_tokens,
				"throughput_tps": tps,
				"frames_per_second": sampled_fps,
				"response": response.strip()
				})

			total_time += elapsed
			total_generated_tokens += num_generated_tokens
			successful_runs += 1

		except Exception as e:
			logging.error(f"❌ 推論過程中發生錯誤: {e}")

	if successful_runs > 0:
		avg_query_latency = total_time / successful_runs
		avg_throughput = total_generated_tokens / total_time if total_time > 0 else 0.0
		avg_frames_per_second = (args.num_frames * successful_runs) / total_time if total_time > 0 else 0.0

		avg_video_duration = (
				total_video_duration / valid_duration_count
				if valid_duration_count > 0 else None
				)

		# Equivalent Real-time Latency = 平均推論時間 / 平均影片長度
		# < 1 代表 faster than real-time
		equivalent_real_time_latency = (
				avg_query_latency / avg_video_duration
				if avg_video_duration and avg_video_duration > 0 else None
				)

		peak_vram = None
		if torch.cuda.is_available():
			peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

		avg_power = (
				sum(power_readings) / len(power_readings)
				if len(power_readings) > 0 else None
				)

		logging.info(f"\n{'=' * 20} Benchmark Summary {'=' * 20}\n")
		logging.info(f"Input")
		logging.info(f"	Model: {model_id}")
		logging.info(f"	Hardware Name  : {hardware_name}")
		logging.info(f"	Video Dir      : {video_dir}")
		logging.info(f"	Fixed Frames   : {args.num_frames}")
		logging.info("\n")
		logging.info("Output")
		logging.info(f"	Average Query Latency: {avg_query_latency:.4f} sec")
		logging.info(f"	Frames Per Second: {avg_frames_per_second:.4f}")
		if equivalent_real_time_latency is not None:
			logging.info(f"	Equivalent Real-time Latency (RTF): {equivalent_real_time_latency:.4f}")
		else:
			logging.info(f"	Equivalent Real-time Latency (RTF): N/A")
		if peak_vram is not None:
			logging.info(f"	Peak VRAM Usage: {peak_vram:.4f} GB")
		else:
			logging.info(f"	Peak VRAM Usage: N/A")
		logging.info(f"	Throughput: {avg_throughput:.4f} tokens/sec")
		if avg_power is not None:
			logging.info(f"	Power Consumption: {avg_power:.2f} W")
		else:
			logging.info(f"Power Consumption: N/A")

		logging.info("\n")
		logging.info(f"成功處理影片數: {successful_runs} / {sample_size}")
		if avg_video_duration is not None:
			logging.info(f"平均影片長度: {avg_video_duration:.4f} sec")
		logging.info(f"詳細測試結果已記錄於: {main_log_file}")
	else:
		logging.info("沒有成功完成任何影片推論。")


if __name__ == "__main__":
	main()
