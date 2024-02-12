# Copyright (c) Tencent Inc. All rights reserved.
import os
import argparse
import os.path as osp
from functools import partial
from io import BytesIO
from copy import deepcopy

import onnx
import onnxsim
import torch
import gradio as gr
import numpy as np
from PIL import Image
from torchvision.ops import nms
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmdet.visualization import DetLocalVisualizer
from mmdet.datasets import CocoDataset
from mmyolo.registry import RUNNERS

from yolo_world.easydeploy.model import DeployModel, MMYOLOBackend


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO-World Demo")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
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
    args = parser.parse_args()
    return args


def run_image(
    runner,
    image,
    text,
    max_num_boxes,
    score_thr,
    nms_thr,
    image_path="./work_dirs/demo.png",
):
    os.makedirs("./work_dirs", exist_ok=True)
    image.save(image_path)
    texts = [[t.strip()] for t in text.split(",")] + [[" "]]
    data_info = dict(img_id=0, img_path=image_path, texts=texts)
    data_info = runner.pipeline(data_info)
    data_batch = dict(
        inputs=data_info["inputs"].unsqueeze(0),
        data_samples=[data_info["data_samples"]],
    )

    with autocast(enabled=False), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances

    keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    pred_instances = pred_instances[keep_idxs]
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    if len(pred_instances.scores) > max_num_boxes:
        indices = pred_instances.scores.float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]
    output.pred_instances = pred_instances

    image = np.array(image)
    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta["classes"] = [t[0] for t in texts]
    visualizer.add_datasample(
        "image",
        np.array(image),
        output,
        draw_gt=False,
        out_file=image_path,
        pred_score_thr=score_thr,
    )
    image = Image.open(image_path)
    return image


def export_model(runner, checkpoint, text, max_num_boxes, score_thr, nms_thr):
    backend = MMYOLOBackend.ONNXRUNTIME
    postprocess_cfg = ConfigDict(
        pre_top_k=10 * max_num_boxes,
        keep_top_k=max_num_boxes,
        iou_threshold=nms_thr,
        score_threshold=score_thr,
    )

    base_model = deepcopy(runner.model)
    texts = [[t.strip() for t in text.split(",")] + [" "]]
    base_model.reparameterize(texts)
    deploy_model = DeployModel(
        baseModel=base_model, backend=backend, postprocess_cfg=postprocess_cfg
    )
    deploy_model.eval()

    device = (next(iter(base_model.parameters()))).device
    fake_input = torch.ones([1, 3, 640, 640], device=device)
    # dry run
    deploy_model(fake_input)

    os.makedirs("work_dirs", exist_ok=True)
    save_onnx_path = os.path.join("work_dirs", "yolow-l.onnx")
    # export onnx
    with BytesIO() as f:
        output_names = ["num_dets", "boxes", "scores", "labels"]
        torch.onnx.export(
            deploy_model,
            fake_input,
            f,
            input_names=["images"],
            output_names=output_names,
            opset_version=12,
        )
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)
    onnx_model, check = onnxsim.simplify(onnx_model)
    onnx.save(onnx_model, save_onnx_path)

    del base_model
    del deploy_model
    del onnx_model
    return gr.update(visible=True), save_onnx_path


def demo(runner, args, cfg):
    with gr.Blocks(title="YOLO-World") as demo:
        with gr.Row():
            gr.Markdown(
                "<h1><center>YOLO-World: Real-Time Open-Vocabulary "
                "Object Detector</center></h1>"
            )
        with gr.Row():
            with gr.Column(scale=0.3):
                with gr.Row():
                    image = gr.Image(type="pil", label="input image")
                input_text = gr.Textbox(
                    lines=7,
                    label="Enter the classes to be detected, " "separated by comma",
                    value=", ".join(CocoDataset.METAINFO["classes"]),
                    elem_id="textbox",
                )
                with gr.Row():
                    submit = gr.Button("Submit")
                    clear = gr.Button("Clear")
                with gr.Row():
                    export = gr.Button("Deploy and Export ONNX Model")
                out_download = gr.File(
                    label="Download link", visible=True, height=30, interactive=False
                )
                max_num_boxes = gr.Slider(
                    minimum=1,
                    maximum=300,
                    value=100,
                    step=1,
                    interactive=True,
                    label="Maximum Number Boxes",
                )
                score_thr = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.05,
                    step=0.001,
                    interactive=True,
                    label="Score Threshold",
                )
                nms_thr = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    step=0.001,
                    interactive=True,
                    label="NMS Threshold",
                )
            with gr.Column(scale=0.7):
                output_image = gr.Image(type="pil", label="output image")

        submit.click(
            partial(run_image, runner),
            [image, input_text, max_num_boxes, score_thr, nms_thr],
            [output_image],
        )
        clear.click(lambda: [[], "", ""], None, [image, input_text, output_image])
        export.click(
            partial(export_model, runner, args.checkpoint),
            [input_text, max_num_boxes, score_thr, nms_thr],
            [out_download, out_download],
        )
        demo.launch(share=True)


if __name__ == "__main__":
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    os.makedirs("./work_dirs", exist_ok=True)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    cfg.load_from = args.checkpoint

    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.call_hook("before_run")
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    runner.pipeline = Compose(pipeline)
    runner.model.eval()
    demo(runner, args, cfg)
