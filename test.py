from prettytable import PrettyTable
import os
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--config_file", default='../logs/CUHK-PEDES/20240810_110716_baseline/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)

    args.training = False
    logger = setup_logger('dm-adapter', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    test_img_loader, test_txt_loader = build_dataloader(args)
    model = build_model(args)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)
    do_inference(model, test_img_loader, test_txt_loader)