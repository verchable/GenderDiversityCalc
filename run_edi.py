import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import pickle


from Tracking import ReIDTrack
from SlowFast.tools.demo_net_edi import Demo
from deep_sort import DeepSort
from InsightFace import faceDet
from SceneDet import find_scenes


def parse_args():
    parser = argparse.ArgumentParser(description='Verchable video understanding system')
    parser.add_argument("--video", default=None, help="file path of source video")
    parser.add_argument("--demo", action="store_true", help="produce tagged videos/frames & idFace")
    return parser.parse_args()

def main(vid_path, demo=False, tar_act=None):
    # make output dir
    dir_name, _ = os.path.splitext(vid_path)

    # run SceneDetection on the whole video
    pkl_scene_path = dir_name+'_sceneDet.pkl'
    if os.path.isfile(pkl_scene_path):
        with open(pkl_scene_path, 'rb') as pkl_scene:
            scene_list = pickle.load(pkl_scene)
        print('Scene Detection file found, skipped!')
    else:
        print('Running Scene Detection ...')
        scene_list = find_scenes(vid_path)
        with open(pkl_scene_path, 'wb') as pkl_scene:
            pickle.dump(scene_list, pkl_scene)

    path_bbox_npy = dir_name + '_obj_bbox.npy'
    path_score_npy = dir_name + '_obj_score.npy'
    assert os.path.isfile(path_score_npy), 'Wrong path of score npy file: %s' % path_score_npy
    assert os.path.isfile(path_bbox_npy), 'Wrong path of bbox npy file: %s' % path_bbox_npy
    bbox_npy = np.load(path_bbox_npy, allow_pickle=True)
    score_npy = np.load(path_score_npy, allow_pickle=True)

    ############# Face Det&Recog Module ############
    face_det = faceDet.FaceDet()
    ##############################################

    ##############  Action Recog Module ##########
    act_cfg_path = 'SlowFast/ava_SLOWFAST_32x2_R101_50_50.yaml'
    act_model_path = 'SlowFast/checkpoints/SLOWFAST_32x2_R101_50_50.pkl'
    act_label_path = 'SlowFast/demo/AVA/ava.names'
    act_demo = Demo(act_cfg_path, act_model_path, act_label_path)
    seq_len = act_demo.cfg.DATA.NUM_FRAMES * act_demo.cfg.DATA.SAMPLING_RATE
    ##############################################

    cap = cv2.VideoCapture(vid_path)
    vid_fps = round(cap.get(cv2.CAP_PROP_FPS))
    vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     assert vid_len == scene_list[-1][1], 'the video length is not equal to scene lengthen'
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    pbar = tqdm(total=vid_len)  # manual visualize

    vid_writer = cv2.VideoWriter(dir_name + '_edi.mp4', cv2.VideoWriter_fourcc(*"mp4v"),
                                 vid_fps, (width, height))

    global_ids = {'cur_id':0, 'id_set':[]}
    bbox_npy_new = []
    all_tracks = []
    actions_fps = []
    vid_idx = 0

    for cur_scene in scene_list:
        cur_start = cur_scene[0]
        cur_end = cur_scene[1]
        assert cur_start == vid_idx, 'cur_start not equal to vid_idx: %d != %d' % (cur_start, vid_idx)
        if (cur_end - cur_start) < seq_len:
            #print('\nWarning: the total frames in current scene {} is less than {}\n'.format(cur_scene, seq_len))
            frame_shortage = True
        else:
            frame_shortage = False
        cur_bbox_npy = bbox_npy[cur_start : cur_end]
        cur_score_npy = score_npy[cur_start : cur_end]
        frames = []  # store frames feeding into model
        img_set = []  # store original frames for demo purpose
        # Deep SORT
        N_INIT = 3
        deepsort = DeepSort("./deep_sort/deep/checkpoint/ckpt.t7", n_init=N_INIT)
        # initialization
        reid_track = ReIDTrack(cur_bbox_npy, N_INIT, dir_name, cur_start, global_ids, vid_writer, demo=demo)
        for idx in range(cur_end - cur_start):
            flag, frame = cap.read()
            assert flag is True, 'flag of CAP is False at: %d' % (cur_start + idx)
            ######### deep-sort tracking ##############
            inp_xyxy = cur_bbox_npy[idx].astype(int)
            track_conf = cur_score_npy[idx]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tracker = deepsort.update(inp_xyxy, track_conf, frame_rgb)
            # output bbox identities
            reid_track.run(frame, tracker, idx)
            #####################################

            if len(frames) != seq_len:
                frame_processed = act_demo.process(frame_rgb)
                frames.append(frame_processed)
                img_set.append(frame)
            if idx == cur_end - cur_start - 1:
                if frame_shortage:
                    mid_frame_id_shortage = idx - (len(frames)//2)
                    frames = [np.zeros_like(frame_processed)] * (seq_len - len(frames)) + frames
                else:
                    frames = prev_frmes[-(seq_len - len(frames)):] + frames
            if len(frames) == seq_len:
                mid_frame_id = idx - (seq_len // 2)
                if frame_shortage:
                    mid_frame_id = mid_frame_id_shortage
                mid_frame_bbox = reid_track.check_midframe_conflict(mid_frame_id)
                pred_labels = act_demo.run(frames, mid_frame_bbox, height, width, tar_act=tar_act)

                # for all frames in a second:
                # for writing txt to image purpose
                if idx == cur_end - cur_start - 1:
                    draw_imgs = img_set
                    draw_idx = idx - len(draw_imgs) + 1  # we retrieve the last batch (seq_len) <<<---------------
                else:
                    draw_imgs = img_set[:vid_fps]
                    draw_idx = idx - seq_len + 1  # we retrieve the last batch (seq_len) <<<---------------

                # check missing tracks
                reid_track.draw_track(face_det, pred_labels, draw_imgs, draw_idx)

                actions_fps.append(pred_labels)

                # when the last batch in current scene is coming:
                if idx >= cur_end - cur_start - seq_len:
                    prev_frmes = frames
                frames = frames[vid_fps:]
                img_set = img_set[vid_fps:]

                pbar.n = cur_start + idx + 1
                pbar.refresh()

            vid_idx += 1

        bbox_npy_new += reid_track.bbox_npy_new
        all_tracks += reid_track.all_tracks

    pbar.close()
    


if __name__ == '__main__':
    args = parse_args()
    assert os.path.isfile(args.video), 'Wrong path of video file: %s' % args.video
    dir_name, _ = os.path.splitext(args.video)

    # transform video to frames (imgs)
    if args.demo:
        if os.path.isdir(dir_name):
            assert len(os.listdir(dir_name)) > 0, 'Empty dir of video frames: %s' % dir_name
            print('Video frames directory exists, skipped!')
        else:
            os.mkdir(dir_name)
            os.system("ffmpeg -r 1 -i %s -r 1 %s" % (args.video, os.path.join(dir_name, '%05d.jpg')))
        assert len(os.listdir(dir_name)) > 0, 'Empty dir of video frames: %s' % dir_name

    # use yolov5 to get all obj bbox info
    objDet_vid_path = os.path.join('..', args.video)
    if os.path.isfile(dir_name + '_obj_bbox.npy'):
        print('Yolov5 detection results exist, skipped!')
    else:
        # start running yolov5
        print('Running yolov5')
        #obj_detect(args)
        if args.demo:
            detect_cmd = "cd yolov5 && python detect.py --video %s --demo" % objDet_vid_path
        else:
            detect_cmd = "cd yolov5 && python detect.py --video %s" % objDet_vid_path
        os.system(detect_cmd)
        # output of detection: img folder of results, two numpy files

    print('Start tracking and action recognition...')
    main(args.video, demo=args.demo, tar_act=['talk to'])
