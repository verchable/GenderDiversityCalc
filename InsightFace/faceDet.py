from sklearn.preprocessing import normalize
import insightface


class FaceDet(object):
    def __init__(self):
        ############### face recog module (RetinaFace & ArcFace) ###################
        self.det_model = insightface.model_zoo.get_model('retinaface_r50_v1')
        self.det_model.prepare(ctx_id=0)
        self.rec_model = insightface.model_zoo.get_model('arcface_r100_v1')
        self.rec_model.prepare(ctx_id=0)
        self.ga_model = insightface.model_zoo.get_model('genderage_v1')
        self.ga_model.prepare(ctx_id=0)
        #########################################################################

    def gen_feature(self, face_inp):
        face_bboxes, face_landmarks = self.det_model.detect(face_inp, threshold=0.8)
        # bbox = face_bboxes[0, 0:4]
        if len(face_bboxes)==0:
            return [],None

        aligned = insightface.utils.face_align.norm_crop(face_inp, landmark=face_landmarks[0])
        face_embedding = self.rec_model.get_embedding(aligned).flatten()
        face_embedding = normalize(face_embedding.reshape(-1, 512))
        gender_cls,_ =self.ga_model.get(aligned)
        return face_embedding,gender_cls