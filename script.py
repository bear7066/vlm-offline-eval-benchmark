import os

video_dirs = [
	#"climbing_stair",
	#"face_planting",
	#"falling_off_bike",
    "falling_off_chair"
]

model_id = [
	# "google/gemma-3-4b-it",
	"google/gemma-4-E4B-it",
	# "bear7011/gemma3-4b-kinetic3K_FT"
]

for v in video_dirs:
    video_path = f"./dataset/{v}"

    for model in model_id:
        os.system(f"python3 main.py --video_dir={video_path} --model_id={model} --sample_fps=5")
        os.system(f"python3 llm_judge.py --video_dir={video_path} --model_id={model}")
