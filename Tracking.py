import cv2
import os
import numpy as np

from vid_process import draw_frame


class ReIDTrack(object):
    def __init__(self, bbox_npy, n_init, dir_name, cur_start, global_id, vid_writer, demo=False):
        self.bbox_npy = bbox_npy
        self.N_INIT = n_init
        self.out_dir = dir_name
        self.cur_start = cur_start
        self.vid_writer = vid_writer
        self.demo = demo
        if self.demo:
            self.id_dir = self.out_dir + '_idFace'  # record detected 'talking' faces
            if not os.path.isdir(self.id_dir):
                os.mkdir(self.id_dir)

            self.out_dir = dir_name + '_edi'
            if not os.path.isdir(self.out_dir):
                os.mkdir(self.out_dir)

        self.bbox_npy_new = []
        self.all_tracks = []
        self.global_id = global_id
        self.local_id = {}
        self.track_missing = []
        self.gender_cls = None
        (_, self.id_height), _ = cv2.getTextSize('ID:99', cv2.FONT_HERSHEY_SIMPLEX, .7, 2)
        (_, self.act_height), _ = cv2.getTextSize('[] action', cv2.FONT_HERSHEY_SIMPLEX, .5, 1)

    def run(self, frame, tracker, idx):
        cur_track = []
        cur_bboxs = []
        # check whether inp_xyxy is valid
        for bbox_id, track in enumerate(tracker.tracks):
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            box = track.to_tlwh()
            x, y, w, h = box
            x1 = max(int(x), 0)
            x2 = min(int(x + w), frame.shape[1] - 1)
            y1 = max(int(y), 0)
            y2 = min(int(y + h), frame.shape[0] - 1)

            # record the ids in current frame
            cur_bboxs.append([x1, y1, x2, y2])
            cur_track.append(track.track_id)

        if len(self.all_tracks) < self.N_INIT - 1:
            # we assume tracker was initialized
            self.track_missing.append(idx)

        # assert len(cur_bboxs) == len(cur_track)
        self.bbox_npy_new.append(cur_bboxs)
        self.all_tracks.append(cur_track)

    def check_midframe_conflict(self, mid_frame_id):
        if len(self.bbox_npy_new[mid_frame_id - 1]) != len(self.bbox_npy_new[mid_frame_id]) != len(
                self.bbox_npy_new[mid_frame_id + 1]):
            print('Warning at %d: mid_frame bbox is NOT consistent with other frames!' % mid_frame_id)

        return np.array(self.bbox_npy_new[mid_frame_id], dtype=np.float32)

    def draw_track(self, face_det, pred_labels, draw_imgs, draw_idx):
    # Parameters:
    #   face_det: a function for face detection and embedding generation
    #   pred_labels: a list of action predictions for a frame, [[(prob, label), ...], ...], shape: Bbox x 5
    #   draw_imgs: list of images in current batch for drawing

        while self.track_missing:
            miss_id = self.track_missing.pop()
            assert miss_id + 2 < len(self.all_tracks), 'miss_id is beyond all_track range: %d' % (
                        self.cur_start + miss_id)
            self.all_tracks[miss_id] = self.all_tracks[miss_id + 2]
            self.bbox_npy_new[miss_id] = self.bbox_npy_new[miss_id + 2]

        for draw_k, frame in enumerate(draw_imgs):
            idx = draw_idx + draw_k
            cur_boxes = self.bbox_npy_new[idx]
            cur_track = self.all_tracks[idx]
            opt_len = min(len(cur_boxes), len(pred_labels))
            for box, track_id, act_pred in zip(cur_boxes[:opt_len], cur_track[:opt_len], pred_labels[:opt_len]):
                # act_pred is a list of five tuples like (prob, label) or None
                act_not_valid = [None] * 5 == act_pred
                if not act_not_valid:
                    x1, y1, x2, y2 = box
                    if track_id not in self.local_id:
                        face_inp = frame[y1:(min(y1 + (x2 - x1), y2)), x1:x2]
                        face_embed,self.gender_cls = face_det.gen_feature(face_inp)
                        if len(face_embed) == 0:
                            # no face is detected
                            continue
                        face_id = self.reID(face_embed)
                        if face_id:
                            self.local_id[track_id] = face_id
                            frame = draw_frame(frame, box, act_pred, face_id, self.id_height, self.act_height,self.gender_cls)
                        else:
                            new_global_id = self.global_id['cur_id'] + 1
                            self.global_id['cur_id'] = new_global_id
                            self.global_id['id_set'].append((new_global_id, face_embed))
                            if self.demo:
                                id_path = os.path.join(self.id_dir,
                                                       'id_{:03d}_{:05d}.jpg'.format(new_global_id, self.cur_start+idx))
                                cv2.imwrite(id_path, face_inp)
                            self.local_id[track_id] = new_global_id
                            frame = draw_frame(frame, box, act_pred, new_global_id, self.id_height, self.act_height,self.gender_cls)
                    else:
                        face_id = self.local_id[track_id]
                        frame = draw_frame(frame, box, act_pred, face_id, self.id_height, self.act_height,self.gender_cls)

            # Display the frame
            self.vid_writer.write(frame)
            if self.demo:
                img_path = os.path.join(self.out_dir, '{:05d}.jpg'.format(self.cur_start + idx))
                cv2.imwrite(img_path, frame)

    def reID(self, face_embed):
        if not self.global_id['id_set']:
            return None
        sim = [np.dot(ref, face_embed.T) for id, ref in self.global_id['id_set']]
        best_match = np.argmax(sim)
        face_prob = sim[best_match]
        if face_prob > 0.4:
            return self.global_id['id_set'][best_match][0]
        else:
            return None
