# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse

import torch
import torch.backends.cudnn as cudnn
from tabulate import tabulate

import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.function import test
from utils.train_init import get_data, get_loader


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate segmentation network')
    
    parser.add_argument(
        '--cfg',
        help='experiment configure file name',
        default="configs/lpcv/pidnet_small_lpcv.yaml",
        type=str,
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help="Path to model.pt file",
    )
    parser.add_argument(
        '--model-name',
        help="Model name. For example, pidnet_s or pidnet_m",
        default="pidnet_s",
    )
    parser.add_argument(
        '--device',
        help="cuda or cpu",
        default="cpu",
    )
    parser.add_argument(
        'opts',
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    # cudnn related setting
    cudnn.benchmark = False
    cudnn.deterministic = True

    # build model
    print("Load Model")
    model = models.pidnet.get_seg_model(
        model_name=args.model_name,
        num_classes=config.DATASET.NUM_CLASSES,
        aux_heads=False,
        checkpoint_path=args.checkpoint,
        strict_state_dict=False,
    )
    print("Model Loaded")

    # prepare data
    print("Load test dataset")
    train_dataset, test_dataset = get_data(config)
    _, testloader = get_loader(config, train_dataset, test_dataset)
    print("Test dataset loaded")

    org_mean_F1 = test(
        model,
        testloader,
        args.device,
        config,
    )

    print(f"Org mean F1: {org_mean_F1}")
    print("Done")


if __name__ == "__main__":
    main()
