import cv2
import os


class VideoReader(object):

    def __init__(self, vid_path):
        self.source = vid_path
        assert os.path.isfile(vid_path)

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.source)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.source))

        self.fps = round(self.cap.get(cv2.CAP_PROP_FPS))
        self.vid_len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self

    def __next__(self):
        was_read, frame = self.cap.read()
        return was_read, frame


def draw_frame(frame, box, act_pred, face_id, id_height, act_height,gender_cls):
    x1, y1 = box[:2]
    x1 += 5
    y1 += 20

    # draw obj_person bbox
    cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 0, 255), thickness=2)
    ## add track id to frame

    cv2.putText(frame, 'ID: ' + str(face_id)+'Gender:'+str(gender_cls),
                    (int(x1), int(y1)),
                    0, 0.7, (0, 255, 0), 2)

    y1 += id_height + 10

    for pred in act_pred:
        if pred is None:
            continue
        prob, label = pred
        txt = '[{:.2f}] {}'.format(prob, label)
        cv2.putText(
            frame, txt, (int(x1), int(y1)),
            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1
        )
        y1 += act_height + 12

    return frame
