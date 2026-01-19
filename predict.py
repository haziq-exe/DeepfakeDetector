import argparse
from inference.predict import predict_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True,
                        help='Path to video file')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to model weights (uses latest if not specified)')
    parser.add_argument('--kframes', type=int, default=8,
                        help='Number of frames to sample')
    
    args = parser.parse_args()
    
    video_prob, frame_indices, frame_probs = predict_video(
        video_path=args.video,
        weights_path=args.weights,
        kframes=args.kframes
    )
    
    print("\nSummary:")
    print(f"Video probability: {video_prob:.4f}")
    print(f"Verdict: {'DEEPFAKE' if video_prob > 0.5 else 'AUTHENTIC'}")


if __name__ == '__main__':
    main()