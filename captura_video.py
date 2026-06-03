import pyrealsense2 as rs
import numpy as np
import cv2
import time
from pathlib import Path
 
def main():
    out_dir = Path("video")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"color_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
 
    width, height, fps = 640, 480, 30

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
 
    profile = pipeline.start(config)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # MP4
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
 
    if not writer.isOpened():
        pipeline.stop()
        raise RuntimeError("error video")
 
    print("q fin")
 
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
 
            color_image = np.asanyarray(color_frame.get_data())
            cv2.putText(color_image, time.strftime("%H:%M:%S"),
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
 
            writer.write(color_image)
            cv2.imshow("RealSense Color (REC)", color_image)
 
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
 
    finally:
        writer.release()
        pipeline.stop()
        cv2.destroyAllWindows()
 
if __name__ == "__main__":
    main()