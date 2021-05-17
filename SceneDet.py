import os
import cv2
from tqdm import tqdm

# Standard PySceneDetect imports:
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors.content_detector import ContentDetector


def find_scenes(video_path):
    # type: (str) -> List[Tuple[FrameTimecode, FrameTimecode]]
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    # Construct our SceneManager and pass it our StatsManager.
    scene_manager = SceneManager(stats_manager)

    # Add ContentDetector algorithm (each detector's constructor
    # takes detector options, e.g. threshold).
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()

    try:
        # Set downscale factor to improve processing speed.
        video_manager.set_downscale_factor()

        # Start video_manager.
        video_manager.start()

        # Perform scene detection on video_manager.
        scene_manager.detect_scenes(frame_source=video_manager)

        # Obtain list of detected scenes.
        scene_list = scene_manager.get_scene_list(base_timecode)

    finally:
        video_manager.release()

    scene_list_new = []
    for cur_scene in scene_list:
        start_frame = cur_scene[0].get_frames()
        end_frame = cur_scene[1].get_frames()
        scene_list_new.append((start_frame, end_frame))

    return scene_list_new


def demo_sceneDet(video_path, scene_list):
    cap = cv2.VideoCapture(video_path)
    frame_len = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    save_path = os.path.splitext(video_path)[0] + '_SceDet.mp4'
    _, frame = cap.read()
    h, w = frame.shape[:2]
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    if len(scene_list) <= 1:
        print("No scene change!")
    else:
        cur_scene_id = 0
        next_scene = scene_list[cur_scene_id + 1][0].get_frames()
        for i in tqdm(range(frame_len)):
            if i == next_scene:
                cur_scene_id += 1
                if cur_scene_id + 1 < len(scene_list):
                    next_scene = scene_list[cur_scene_id+1][0].get_frames()

            frame = cv2.putText(frame, 'Current Scene: %d' % cur_scene_id, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            vid_writer.write(frame)
            _, frame = cap.read()

        vid_writer.release()
        print('Scene Detection video saved:', save_path)

if __name__ == '__main__':
    vid_path = 'MIB_1.mov'
    scene_list = find_scenes(vid_path)
    demo_sceneDet(vid_path, scene_list)
