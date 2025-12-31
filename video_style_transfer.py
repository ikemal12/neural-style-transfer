import cv2  # type: ignore
import torch
import numpy as np
import argparse
import time
import torchvision.transforms as transforms
from transformer_net import TransformerNetOld
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k: v for k, v in state_dict.items() 
                 if 'running_mean' not in k and 'running_var' not in k}
    model = TransformerNetOld().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def stylize_frame(frame, model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    img_tensor: torch.Tensor = transform(img)  # type: ignore
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
    
    output = output.squeeze(0).clamp(0, 255).cpu().numpy()
    output = output.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    output = output.astype(np.uint8)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output_bgr

def process_video(video_path, model_path, output_path):
    print("Loading model...")
    model = load_model(model_path)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        stylized_frame = stylize_frame(frame, model)
        out.write(stylized_frame)
        frame_count += 1
        
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps_processed = frame_count / elapsed
            remaining = (total_frames - frame_count) / fps_processed if fps_processed > 0 else 0
            remaining_mins = int(remaining // 60)
            remaining_secs = int(remaining % 60)
            print(f"Processed {frame_count}/{total_frames} frames "
                  f"({frame_count/total_frames*100:.1f}%) - "
                  f"ETA: {remaining_mins} minutes {remaining_secs} seconds")
            
    cap.release()
    out.release()
    total_time = time.time() - start_time
    total_mins = int(total_time // 60)
    total_secs = int(total_time % 60)
    print(f"\nCompleted! Processed {frame_count} frames in {total_mins} minutes {total_secs} seconds")
    print(f"Output saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply neural style transfer to video')
    parser.add_argument('--input', required=True, help='Path to input video')
    parser.add_argument('--model', required=True, help='Path to pretrained model')
    parser.add_argument('--output', required=True, help='Path to output video')
    args = parser.parse_args()
    process_video(args.input, args.model, args.output)