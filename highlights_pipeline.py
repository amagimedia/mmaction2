import json
import torch
from mmaction.apis import init_recognizer, inference_recognizer
from os import listdir
from os.path import isfile, join, abspath
from pathlib import Path
import time
import typer
from subprocess import Popen, run
import numpy as np


def get_highlights(
    input_video: str = typer.Option(
        ..., "--input-video", "-i", help="Input video file"
    ),
    # config_file: str = typer.Option(...,"--config","-c", help="Config file of model"),
    # use_gpu: bool = typer.Option(False,"--use-gpu","-g", help="Use gpu for inference"),
    # ckpt_file: str = typer.Option(...,"--model","-m",help="Model ckpt file for inference"),
    # model_dir: str = typer.Option(...,"--model-dir","-md", help="Videoken models directory"),
):
    # REMOVE THESE AFTER AUTOMATION
    use_gpu = True
    model_dir = "/home/varun/videoken/scratch/models"
    op_json = Path(input_video).stem + "_highlights.json"
    # run videoken -t on video

    # input_video = abspath(input_video)

    path_to_videoken_environment = "/home/varun/videoken/venv/bin/videoken"
    cmd = ["-i", input_video, "-o", op_json, "-t", "-md", model_dir, "-bs", "500"]
    if use_gpu:
        cmd.append("-g")
    # run([path_to_videoken_environment,"video-analyze"]+cmd)

    # initialize inference
    # remove these after automation
    config_file = "configs/task/finetuning/ipcsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py"
    ckpt_file = "work_dirs/ipcsn_sports1m_pretrained_soccerdatafull/epoch_6.pth"

    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    device = torch.device(device)
    print(device)

    model = init_recognizer(config_file, ckpt_file, device=device)

    # read json and do inference

    labels = ["ZOOM", "NPLAY", "ATTK", "IDLE"]

    highlights_dict = {key: list() for key in labels}
    highlights_list = []
    data = None

    with open(op_json, "r") as f:
        data = json.load(f)
        # add shot vs scene
        frame_rate = data["video_meta"]["frame_rate"]
        shots = data["video_annotations"]["shot_annotations"]
        prev_class = None
        for shot in shots:
            start_frame = round(shot["start_time_offset"] * frame_rate)
            end_frame = round(shot["end_time_offset"] * frame_rate)
            frame_diff = end_frame - start_frame + 1

            clip_len = 32
            frame_interval = 2
            max_clips_per_shot = 2
            num_clips_float = frame_diff / (clip_len * frame_interval)
            num_clips = int(num_clips_float)
            if num_clips_float > max_clips_per_shot + 0.5:
                # split shots into smaller chunks

                shot_len_frms = clip_len * frame_interval * max_clips_per_shot
                clip_intervals = (
                    np.arange(num_clips // max_clips_per_shot + 1) * shot_len_frms
                )
                clip_intervals = clip_intervals + start_frame
                clip_intervals = np.append(clip_intervals, end_frame + 1)
                split_shots_list = [
                    (clip_intervals[i], clip_intervals[i + 1] - 1)
                    for i in range(len(clip_intervals) - 1)
                ]
            else:
                split_shots_list = [(start_frame, end_frame)]
                if num_clips == 0:
                    num_clips = 1

            for split_shot in split_shots_list:
                (start_frame, end_frame) = split_shot
                frame_diff = end_frame - start_frame + 1
                print(start_frame, end_frame)
                results = inference_recognizer(
                    model, input_video, start_index=start_frame, total_frames=frame_diff
                )
                top_result = results[0]
                label = labels[top_result[0]]
                confidence = float(top_result[1])

                if prev_class == label:
                    end_frame

                highlights_list.append(
                    {
                        "start_time_offset": round(start_frame / frame_rate, 2),
                        "end_time_offset": round(end_frame / frame_rate, 2),
                        "confidence": confidence,
                        "label": label,
                    }
                )

        for i, shot in enumerate(highlights_list):
            pass

    data["video_annotations"].update({"highlights": highlights_dict})

    with open(op_json, "w") as f:
        json_object = json.dumps(data, indent=2)
        f.write(json_object)


if __name__ == "__main__":
    typer.run(get_highlights)
