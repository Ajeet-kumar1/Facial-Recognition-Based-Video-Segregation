# Import requirements
import os
import cv2
import shutil
import tempfile
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace
from config import matched_dir, unmatched_dir, log_file
from config import frame_interval_sec, model_name, detector_backend, align, threshold


def driver_code(video_dir, reference_image):
    # ---------- List all video files ----------
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    results = []

    # Open each video
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(video_dir, video_file)
        if not os.path.exists(video_path):
            print(f"File does not exist, skipping: {video_file}")
            continue

        # Read the video and check it's first frame.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_file}")
            cap.release()
            continue
        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            print(f"Cannot read first frame of {video_file}. Skipping.")
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame

        # Check the FPS of video
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None or fps != fps:  # If fps is NaN then take 25
            fps = 25.0
        
        # Make a frame interval to avoid unecessary calculations
        frame_interval_frames = max(1, int(fps * frame_interval_sec)) 
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {video_file}, FPS: {fps:.2f}, Total Frames: {total_frames}, Frame Interval: {frame_interval_frames}")

        matched = False
        min_distance = None
        frames_checked = 0
        frame_num = 0
        
        # Now read the video's frame with given interval
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_num % frame_interval_frames == 0:
                try:
                    # Save frame temporarily to disk
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        temp_frame_path = tmp.name
                        cv2.imwrite(temp_frame_path, frame)

                    # Detect the face and verify it with reference image.
                    result = DeepFace.verify(
                        img1_path=reference_image,
                        img2_path=temp_frame_path,
                        model_name=model_name,
                        detector_backend=detector_backend,
                        align=align,
                        enforce_detection=False
                    )
                    # Remove the temporary file and compare with threshold
                    os.remove(temp_frame_path)
                    distance = result.get("distance", 1.0)
                    verified = distance <= threshold  # <-- Using custom threshold

                    print(f"Frame {frame_num}: distance={distance:.4f}, verified={verified}")
                    frames_checked += 1
                    
                    # If it is matched then print name of image
                    if verified:
                        matched = True
                        min_distance = distance
                        print(f"Match found in frame {frame_num} with distance {distance:.4f}.")
                        frames_checked += 1

                except Exception as e:
                    print(f"Error at frame {frame_num}: {e}")
            frame_num += 1

        cap.release()

        # Copy video to matched/unmatched folder
        dest_folder = matched_dir if matched else unmatched_dir
        shutil.copy(video_path, os.path.join(dest_folder, video_file))

        results.append({
            "video": video_file,
            "match": "Yes" if matched else "No",
            "min_distance": round(min_distance, 4) if min_distance is not None else "N/A",
            "frames_checked": frames_checked
        })

    # ---------- Save log ----------
    df = pd.DataFrame(results)
    df.to_csv(log_file, index=False)
    print(f"Log saved to: {log_file}")

# Boiler plate code.
if __name__=='__main__':
    # Take the reference image path
    reference_image = input("Enter the reference image path: ")
    video_dir = input("Enter the videos directory: ")
    driver_code(video_dir, reference_image)

