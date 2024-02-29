# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import os.path as osp

import cv2
import time
import tempfile
import subprocess
import json
from tqdm import tqdm
from typing import List, Tuple, Optional
from tempfile import NamedTemporaryFile
from cog import BasePredictor, Input, Path, BaseModel

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.system("pip install -e .")

import argparse

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.dataset import Compose
from mmyolo.registry import RUNNERS

import torch
import shutil

import numpy as np
import supervision as sv
from torchvision.ops import nms
from mmengine.runner.amp import autocast


WEIGHTS_DIR_PATH = "./weights"
DEFAULT_YOLOW_WEIGHT_URL = "https://weights.replicate.delivery/default/yolo-world/yolow-v8_l_clipv2_frozen_t2iv2_bn_o365_goldg_pretrain.pth"
WEIGHTS_FILE_PATH = (
    "./weights/yolow-v8_l_clipv2_frozen_t2iv2_bn_o365_goldg_pretrain.pth"
)

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator(text_color=sv.Color.BLACK)
JSON_RESP = []


class Output(BaseModel):
    media_path: Optional[Path]
    json_str: Optional[str]


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO-World Replicate Cog")
    parser.add_argument(
        "--config",
        default="configs/pretrain/yolo_world_l_t2i_bn_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py",
    )
    parser.add_argument(
        "--checkpoint",
        default="./weights/yolow-v8_l_clipv2_frozen_t2iv2_bn_o365_goldg_pretrain.pth",
    )
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    args, unknown = parser.parse_known_args()

    # Manually add the unknown arguments to the args namespace
    # This example simply adds them as a list under the attribute 'unknown_args'
    # You can adjust the attribute name as needed
    setattr(args, "unknown_args", unknown)

    return args


def download_weights(url: str, dest: str, extract: bool = False) -> None:
    start = time.time()
    print("[!] downloading url: ", url)
    print("[!] downloading to: ", dest)
    command = [
        "pget",
        "-v",
        url,
        dest,
    ]
    if extract:
        command.insert(1, "-x")
    subprocess.check_call(command, close_fds=False)
    print("[!] downloading took: ", time.time() - start)


def get_available_gpu_memory():
    torch.cuda.empty_cache()
    return (
        torch.cuda.get_device_properties(0).total_memory / 1024**2
        - torch.cuda.memory_allocated(0) / 1024**2
    )


def estimate_memory_usage_per_image(
    image_size: Tuple[int, int, int] = (640, 640, 3), dtype: torch.dtype = torch.float32
) -> float:
    """
    Estimate memory usage per image based on its size and data type.
    """
    # Calculate the memory usage in bytes and convert to megabytes
    image_memory = (
        torch.tensor(image_size).prod().item()
        * torch.tensor([], dtype=dtype).element_size()
        / 1024**2
    )
    # Estimate model and overhead memory usage per image (this is a rough estimate and should be adjusted based on actual measurements)
    model_overhead_per_image = (
        50  # Adjust based on your model's actual memory usage per image
    )
    total_memory_per_image = image_memory + model_overhead_per_image
    return total_memory_per_image


def find_safe_batch_size(
    initial_batch_size: int = 1, increment: int = 1, safety_margin: int = 1024
) -> int:
    """
    Dynamically finds a safe batch size based on the estimated memory usage per image and available GPU memory.
    """
    batch_size = initial_batch_size
    while True:
        # Estimate memory usage for the current batch size
        estimated_memory_usage_per_image = estimate_memory_usage_per_image()
        estimated_total_memory_usage = estimated_memory_usage_per_image * batch_size

        available_memory = get_available_gpu_memory()

        if estimated_total_memory_usage + safety_margin >= available_memory:
            # If the estimated total memory usage with the safety margin exceeds available memory,
            # return the last safe batch size.
            return max(1, batch_size - increment)  # Ensure batch size is at least 1

        batch_size += increment


def run_image(
    runner: Runner,
    image: np.ndarray,
    text: str,
    max_num_boxes: int = 100,
    score_thr: float = 0.05,
    nms_thr: float = 0.5,
):
    global JSON_RESP
    with NamedTemporaryFile(suffix=".jpeg") as f:
        cv2.imwrite(f.name, image)
        texts = [[t.strip()] for t in text.split(",")] + [[" "]]
        data_info = dict(img_id=0, img_path=f.name, texts=texts)
        data_info = runner.pipeline(data_info)
        data_batch = dict(
            inputs=data_info["inputs"].unsqueeze(0),
            data_samples=[data_info["data_samples"]],
        )

        with autocast(enabled=False), torch.no_grad():
            output = runner.model.test_step(data_batch)[0]
            pred_instances = output.pred_instances

        keep_idxs = nms(
            pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr
        )

        pred_instances = pred_instances[keep_idxs]
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

        if len(pred_instances.scores) > max_num_boxes:
            indices = pred_instances.scores.float().topk(max_num_boxes)[1]
            pred_instances = pred_instances[indices]

        pred_instances = pred_instances.cpu().numpy()

        detections = sv.Detections(
            xyxy=pred_instances["bboxes"],
            class_id=pred_instances["labels"],
            confidence=pred_instances["scores"],
            data={
                "class_name": np.array(
                    [texts[class_id][0] for class_id in pred_instances["labels"]]
                )
            },
        )
        JSON_RESP.append(get_output_json_from_detections(detections))

        labels = [
            f"{class_name} {confidence:0.2f}"
            for class_name, confidence in zip(
                detections["class_name"], detections.confidence
            )
        ]
        annotated_image = image.copy()
        annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
        annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections, labels)
        return annotated_image


def get_output_json_from_detections(detections):
    output_dict = {}
    for i, (bbox, score, class_id, class_name) in enumerate(
        zip(
            detections.xyxy,
            detections.confidence,
            detections.class_id,
            detections.data["class_name"],
        )
    ):
        output_dict[f"Det-{i}"] = {
            "x0": float(bbox[0]),
            "y0": float(bbox[1]),
            "x1": float(bbox[2]),
            "y1": float(bbox[3]),
            "score": float(score),
            "cls": class_name,
        }

    return json.dumps(output_dict)


def extract_audio(video_path: str, output_audio_path: str) -> bool:
    print(f"[~] Extracting audio from {video_path} to {output_audio_path}")
    # Updated ffmpeg command to extract and convert the audio
    command = f"ffmpeg -i {video_path} -vn -ar 44100 -ac 2 -ab 192k -f mp3 {output_audio_path}"
    os.system(command)

    # Check if the output audio file exists and has content
    if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
        return True
    else:
        print("[!] No audio stream found in the video or extraction failed.")
        return False


def get_fps(video_path: str) -> float:
    print(f"[~] Getting FPS for {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def run_video(
    runner: Runner,
    video_path: str,
    text: str,
    max_num_boxes: int,
    score_thr: float,
    nms_thr: float,
    output_video_path: str,
    batch_size: int,
) -> Path:
    print(f"[!] Using batch size {batch_size}")
    fps = get_fps(video_path)
    temp_dir = tempfile.mkdtemp()
    print(f"[!] Temporary directory for frames: {temp_dir}")
    audio_path = os.path.join(temp_dir, "audio.mp3")
    has_audio = extract_audio(video_path, audio_path)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = tqdm(total=frame_count, desc="Processing frames")
    frames_batch = []
    frame_paths = []
    frame_counter = 0  # Initialize a frame counter before the loop

    while True:
        ret, frame = cap.read()
        if not ret:
            if frames_batch:  # Process the last batch if there are any frames left
                process_batch(
                    frames_batch,
                    frame_paths,
                    runner,
                    text,
                    max_num_boxes,
                    score_thr,
                    nms_thr,
                )
            break
        frames_batch.append(frame)
        frame_counter += 1  # Increment the frame counter for each frame
        frame_path = os.path.join(temp_dir, f"frame_{frame_counter:05d}.png")
        frame_paths.append(frame_path)
        if len(frames_batch) == batch_size:
            process_batch(
                frames_batch,
                frame_paths,
                runner,
                text,
                max_num_boxes,
                score_thr,
                nms_thr,
            )
            frames_batch = []
            frame_paths = []
        pbar.update(1)

    cap.release()
    pbar.close()

    print("[~] All frames processed, generating video...")
    # Adjust ffmpeg command based on whether audio was extracted
    frames_pattern = os.path.join(temp_dir, "frame_%05d.png")
    if has_audio:
        ffmpeg_cmd = f"ffmpeg -y -r {fps} -i {frames_pattern} -i {audio_path} -c:v libx264 -pix_fmt yuv420p -c:a aac -strict experimental {output_video_path}"
    else:
        ffmpeg_cmd = f"ffmpeg -y -r {fps} -i {frames_pattern} -c:v libx264 -pix_fmt yuv420p {output_video_path}"

    print(f"[!] Running ffmpeg command: {ffmpeg_cmd}")
    os.system(ffmpeg_cmd)
    shutil.rmtree(temp_dir)

    return Path(output_video_path)


def process_batch(
    frames_batch: List[np.ndarray],
    frame_paths: List[str],
    runner: Runner,
    text: str,
    max_num_boxes: int,
    score_thr: float,
    nms_thr: float,
) -> None:
    # This function processes a batch of frames
    for i, frame in enumerate(frames_batch):
        annotated_frame = run_image(
            runner, frame, text, max_num_boxes, score_thr, nms_thr
        )
        cv2.imwrite(frame_paths[i], annotated_frame)


class Predictor(BasePredictor):
    def setup(self) -> None:

        if not os.path.exists(WEIGHTS_DIR_PATH):
            os.makedirs(WEIGHTS_DIR_PATH)
        if not os.path.exists(WEIGHTS_FILE_PATH):
            download_weights(DEFAULT_YOLOW_WEIGHT_URL, WEIGHTS_FILE_PATH)

        args = parse_args()

        # load config
        cfg = Config.fromfile(args.config)
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)

        if args.work_dir is not None:
            cfg.work_dir = args.work_dir
        elif cfg.get("work_dir", None) is None:
            cfg.work_dir = osp.join(
                "./work_dirs", osp.splitext(osp.basename(args.config))[0]
            )

        cfg.load_from = args.checkpoint

        if "runner_type" not in cfg:
            print("Creating runner using Runner.from_cfg")
            runner = Runner.from_cfg(cfg)
        else:
            print("Creating runner using RUNNERS.build")
            runner = RUNNERS.build(cfg)

        runner.call_hook("before_run")
        runner.load_or_resume()
        pipeline = cfg.test_dataloader.dataset.pipeline
        runner.pipeline = Compose(pipeline)
        runner.model.eval()
        self.runner = runner

    def predict(
        self,
        input_media: Path = Input(
            description="Path to the input image or video",
        ),
        class_names: str = Input(
            default="dog, eye, tongue, ear, leash, backpack, person, nose",
            description="Enter the classes to be detected, separated by comma",
        ),
        max_num_boxes: int = Input(
            default=100,
            description="Maximum number of bounding boxes to display",
            ge=1,
            le=300,
        ),
        score_thr: float = Input(
            default=0.05,
            description="Score threshold for displaying bounding boxes",
            ge=0,
            le=1,
        ),
        nms_thr: float = Input(
            default=0.5,
            description="NMS threshold",
            ge=0,
            le=1,
        ),
        return_json: bool = Input(
            description="Return results in json format", default=False
        ),
    ) -> Output:
        global JSON_RESP
        JSON_RESP = []

        image_path = str(input_media)

        if image_path.endswith(".mp4"):
            output_video_path = "output_video.mp4"
            # Calculating a safe batch size for video processing, adjusted to 80% of the maximum possible value and rounded to the nearest lower power of 2 for efficiency
            safe_batch_size = 2 ** int(
                np.floor(
                    np.log2(
                        int(
                            find_safe_batch_size(
                                initial_batch_size=1, increment=1, safety_margin=1024
                            )
                            * 0.8
                        )
                    )
                )
            )
            run_video(
                runner=self.runner,
                video_path=image_path,
                text=class_names,
                max_num_boxes=max_num_boxes,
                score_thr=score_thr,
                nms_thr=nms_thr,
                output_video_path=output_video_path,
                batch_size=safe_batch_size,
            )
            if return_json:
                frame_json = {
                    f"Frame-{i}": json.loads(JSON_RESP[i])
                    for i in range(len(JSON_RESP))
                }
                return Output(json_str=json.dumps(frame_json))
            return Output(media_path=Path(output_video_path))
        else:
            image = run_image(
                runner=self.runner,
                image=cv2.imread(image_path),
                text=class_names,
                max_num_boxes=max_num_boxes,
                score_thr=score_thr,
                nms_thr=nms_thr,
            )
            output_image_path = "output.png"
            cv2.imwrite(output_image_path, image)

            if return_json:
                return Output(json_str=JSON_RESP[0])
            return Output(media_path=Path(output_image_path))
