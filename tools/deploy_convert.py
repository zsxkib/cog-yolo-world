# Copyright (c) Lin Song. All rights reserved.
import os
import json
import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.dataset import Compose
from mmyolo.registry import RUNNERS


def get_caption_embed(runner, caption, prompt_template):
    captions = json.load(open(caption, 'r'))
    captions = [[prompt_template.format(c[0])] for c in captions]
    with torch.no_grad():
        embed = runner.model.backbone.text_model(captions)
        embed = F.normalize(embed[:, 0, :], dim=1, p=2)
        embed = embed.detach().cpu()
        embed = embed[:, :, None, None]
    return embed


def convert(runner, caption, checkpoint, prompt_template):
    checkpoint = torch.load(checkpoint, map_location='cpu')
    state_dict = checkpoint['state_dict']
    embed = get_caption_embed(runner, caption, prompt_template)
    import ipdb; ipdb.set_trace()

    new_state_dict = {}
    for key in list(state_dict.keys()):
        if key.startswith('backbone.text_model'):
            continue
        elif key.startswith('backbone.image_model'):
            new_key = key.replace('backbone.image_model', 'backbone')
            new_state_dict[new_key] = state_dict[key].clone()
        elif key.startswith('bbox_head.head_module.cls_contrasts'):
            module_key = '.'.join(key.split('.')[:4])
            logit_scale = state_dict[module_key + '.logit_scale']
            bias = state_dict[module_key + '.bias']
            conv_weight = embed * logit_scale.exp()
            conv_bias = bias.repeat(conv_weight.shape[0])
            new_state_dict[module_key + '.conv.weight'] = conv_weight
            new_state_dict[module_key + '.conv.bias'] = conv_bias
        else:
            new_state_dict[key] = state_dict[key].clone()

    new_checkpoint = {'state_dict': new_state_dict}
    return new_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('caption', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--prompt-template', type=str,
                        default='{}')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    # replace the ${key} with the value of cfg.key
    # cfg = replace_cfg_vals(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    runner.call_hook('before_run')
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    runner.pipeline = Compose(pipeline)
    runner.model.eval()

    new_checkpoint = convert(runner, args.caption, args.checkpoint,
                             args.prompt_template)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(new_checkpoint, args.output)
