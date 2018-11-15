import numpy as np

SIZE_DIST_THRESHOLD = 400

class Face:
    def __init__(self):
        self.bounding_rect = None
        self.left_eye = None
        self.right_eye = None

    def set_face(self, face):
        self.bounding_rect = face

    def update_face(self, source_image, candidate_faces):
        if self.bounding_rect == None:
            return -1

        has_possible_candidate = False
        distances = np.zeros((len(candidate_faces)))
        (current_x, current_y, current_w, current_h) = self.bounding_rect
        face_index = 0
        for (fx, fy, fw, fh) in candidate_faces:
            width_dist = np.square(fw - current_w)
            height_dist = np.square(fh - current_h)

            if width_dist > SIZE_DIST_THRESHOLD or height_dist > SIZE_DIST_THRESHOLD:
                distances[i] = np.inf
            else:
                distances[i] = np.square(fx - current_x) + np.square(fy - current_y)
                has_possible_candidate = True

            face_index += 1

        if has_possible_candidate:
            print(np.argmin(distances))
            return 0
        else:
            return -1


    # def detect_eyes(self, gray_face):        
    #     roi_gray = gray_face[y:y+int(self.bounding_rect[3] * 0.6), x:x+]
    #     roi_color = frame[y:y+h, x:x+w]
        
    #     new_eyes = eye_cascade.detectMultiScale(roi_gray)



    #     for (ex,ey,ew,eh) in eyes:
    #         cv.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
