#!/usr/bin/env python
import argparse
from pathlib import Path

import torch

from src.predictors import PREDICTORS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("--input", type=str, help="Path to image or text file")
    parser.add_argument("--input-dir", type=str, help="Path to directory of inputs")
    args = parser.parse_args()
    predictor = PREDICTORS.instantiate("classification", ckpt_path=args.ckpt_path, device="cuda:0" if torch.cuda.is_available() else "cpu")
    result = predictor.predict(args.input)
    print(result)


if __name__ == "__main__":
    main()
