import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ----- Setup SAM -----
# Set path to your SAM checkpoint (e.g., 'sam_vit_h_4b8939.pth')
sam_checkpoint = "./checkpoint/sam_vit_h_4b8939.pth"
model_type = "vit_h"  # or the appropriate model type for your checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the SAM model and set up the automatic mask generator
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# ----- Extract frames from the video -----
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

print(f"Extracted {len(frames)} frames.")

# ----- Function to select the subject mask -----
def select_subject_mask(masks, frame):
    """
    A heuristic to choose the subject mask:
    Here we check which mask contains the center of the frame
    and then choose the one with the largest area.
    Adjust this function as needed.
    """
    center_y, center_x = frame.shape[0] // 2, frame.shape[1] // 2
    candidate = None
    max_area = 0
    for mask in masks:
        segmentation = mask["segmentation"]
        # If the center of the frame is inside the mask
        if segmentation[center_y, center_x]:
            if mask["area"] > max_area:
                max_area = mask["area"]
                candidate = segmentation
    # If no mask covers the center, just return the largest mask overall
    if candidate is None and masks:
        candidate = max(masks, key=lambda x: x["area"])["segmentation"]
    return candidate

# ----- Process each frame with SAM -----
segmented_frames = []
for idx, frame in enumerate(frames):
    # Generate masks for the current frame
    masks = mask_generator.generate(frame)
    subject_mask = select_subject_mask(masks, frame)
    
    # Create an output image where the background is removed
    # (Here we assume a binary mask: foreground=1, background=0)
    if subject_mask is not None:
        # Convert boolean mask to uint8 format expected by cv2
        mask_uint8 = (subject_mask.astype(np.uint8)) * 255
        # Use the mask to extract the subject from the frame
        subject = cv2.bitwise_and(frame, frame, mask=mask_uint8)
    else:
        # If no mask was found, keep the original frame
        subject = frame
    
    segmented_frames.append(subject)
    print(f"Processed frame {idx+1}/{len(frames)}")

# ----- Reassemble segmented frames into a video -----
output_path = "segmented_test.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for frame in segmented_frames:
    out_video.write(frame)
out_video.release()

print(f"Segmented video saved as {output_path}")
