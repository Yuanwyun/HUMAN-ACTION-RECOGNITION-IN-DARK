import os
import cv2
from glob import glob


def save_frames(video_path, save_dir, gap=10):
    #
    name = os.path.splitext(os.path.basename(video_path))[0]

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the frame interval based on the desired frame rate
    frame_interval = int(fps / gap)

    idx = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()     #results are stored in the ret and frame

        if not ret:
            break

        # Save frames at the specified interval
        if frame_count % frame_interval == 0:
            cv2.imwrite(os.path.join(save_dir, f"{name}_{idx}.png"), frame)

            idx += 1

        frame_count += 1

    cap.release()

actionList = ['Jump', 'Run', 'Sit', 'Stand', 'Turn', 'Walk']
for i in actionList:
  video_paths = glob(f"/content/drive/MyDrive/EE6222/EE6222_train_and_validate_2023/train/{i}/*.mp4")
  save_dir = "/content/drive/MyDrive/EE6222/train"
  print(f"Processing action: {i}")
  # Process each video file
  for video_path in video_paths:
    save_frames(video_path, save_dir, gap=10)

