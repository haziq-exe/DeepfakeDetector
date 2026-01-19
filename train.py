import argparse
import random
import numpy as np
import torch

from training.trainer import train_frame_detector, train_video_aggregator


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['frame', 'video'], required=True,
                        help='Training mode: frame or video')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to use')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--output', type=str, default='model',
                        help='Output model prefix')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained weights (for video mode)')
    parser.add_argument('--kframes', type=int, default=20,
                        help='Number of frames to sample (for video mode)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    if args.mode == 'frame':
        train_frame_detector(
            data_path=args.data,
            split=args.split,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            output_prefix=args.output
        )
    elif args.mode == 'video':
        train_video_aggregator(
            data_path=args.data,
            kframes=args.kframes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            output_prefix=args.output,
            pretrained_path=args.pretrained
        )


if __name__ == '__main__':
    main()