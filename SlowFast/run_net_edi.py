#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""

import argparse
import sys, os
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.multiprocessing as mpu
from slowfast.config.defaults import get_cfg

from tools.demo_net_edi import demo


def load_config(cfg_path):
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    assert os.path.isfile(cfg_path), 'SlowFast config file path is wrong: %s' % cfg_path
    cfg.merge_from_file(cfg_path)

    return cfg

def slowfast_run(cfg_path):
    """
    Main function to spawn the train and test process.
    """
    cfg = load_config(cfg_path)

    if cfg.NUM_GPUS > 1:
        torch.multiprocessing.spawn(
            mpu.run,
            nprocs=cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                demo,
                "tcp://localhost:9999",
                cfg.SHARD_ID,
                cfg.NUM_SHARDS,
                cfg.DIST_BACKEND,
                cfg,
            ),
            daemon=False,
        )
    else:
        demo(cfg=cfg)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("forkserver")
    #print(os.getcwd())
    cfg_path = 'ava_SLOWFAST_32x2_R101_50_50.yaml'
    slowfast_run()
