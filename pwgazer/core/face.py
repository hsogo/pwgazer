import numpy as np
import cv2
import dlib
from pathlib import Path
from .util import get_euler_angles, get_rotation_matrix, rect
try:
    import mediapipe as mp
    facemesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=3,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
except:
    pass

supported_face_detectors = ['dlib', 'mediapipe']

module_dir = Path(__file__).parent

dlib_face_detector = dlib.get_frontal_face_detector()
dlib_face_predictor = dlib.shape_predictor(str(module_dir/'resources'/'shape_predictor_68_face_landmarks.dat'))

# 3D face model points.
default_face_model = np.array([
    (  0.0,   0.0,  0.0),    # Nose tip
    ( 48.0, -39.0, 30.0),    # Left eye left (outer) corner
    ( 18.0, -37.0, 23.0),    # Left eye right (inner) corner
    (-18.0, -37.0, 23.0),    # Right eye left (inner) corner
    (-48.0, -39.0, 30.0),    # Right eye right (outer) corne
    ( 25.0,  35.0, 20.0),    # Left Mouth corner 
    (-25.0,  35.0, 20.0),    # Right mouth corner
    (  0.0,  12.1,  6.4),    # subnasale
    (  0.0, -44.4, 14.6),    # nose root
    #(  0.0,  82.0, 30.0),    # Chin
    #( 19.8,   2.3, 11.0),    # Left nose
    #(-19.8,   2.3, 11.0)     # Right nose
])

default_eye_params = np.array([
     24.0, # diameter
      0.0, # offset LX
      0.0, # offset LY
    -12.0, # offset LZ
      0.0, # offset RX
      0.0, # offset RY
    -12.0  # offset RZ
])

n_face_model = default_face_model.shape[0]
FACE_CONFIDENCE_THRESHOLD = 0.5

def detect_face(frame, detector='mediapipe', aoi=None, scale=1.0):

    if detector == 'dlib':
        # monochrome
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # prepare scaled image
        if scale == 1.0:
            scaled_frame = frame
        else:
            scaled_frame = cv2.resize(frame, None, fx=scale, fy=scale)

        # detect faces
        dlib_dets, _, _ = dlib_face_detector.run(scaled_frame, 0) # detections, scores, weight_indices
        # convert dlib.rectangle to rect
        detections = [rect(d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top()) for d in dlib_dets]

        # restore rectangles if scaled
        if scale != 0:
            for r in detections:
                r.scale(1/scale)

        # check whether faces are in AOI
        face_detected = False
        if aoi is None:
            if len(detections) > 0:
                face_detected = True
                target_idx = 0
        else:
            for target_idx in range(len(detections)):
                if aoi.contains(detections[target_idx]):
                    face_detected = True
                    break
        
        if face_detected:
            shape = dlib_face_predictor(frame, dlib_dets[target_idx])
            '''
            30: Nose tip
            45: Left eye left (outer) corner
            42: Left eye right (inner) corner
            39: Right eye left (inner) corner
            36: Right eye right (outer) corner
            54: Left mouth corner
            48: Right mouth corner
            33: Subnasale
            27: Nose root
            '''
            fitting_points = np.zeros((n_face_model,2), dtype=np.float32)
            for i,j in enumerate([30,45,42,39,36,54,48,33,27]):
                fitting_points[i] = (shape.part(j).x, shape.part(j).y)

            '''
            42-47: Left eyelid (clockwize viewing from the front of the face)
            36-41: Right eyelid (clockwize viewing from the front of the face)
            '''
            eyelid_points = np.zeros((12,2), dtype=np.float32)
            for i,j in enumerate([42,43,44,45,46,47,36,37,38,39,40,41]):
                eyelid_points[i] = (shape.part(j).x, shape.part(j).y)

            return True, fitting_points, eyelid_points

        else:
            return False, None, None
    
    elif detector == 'mediapipe':
        # Because facemesh returns landmarks directly, "scale" parameter is ignored.
        fm_results = facemesh.process(frame)

        if fm_results.multi_face_landmarks is None:
            return False, None, None

        detections = []
        eyelids_detections = []
        for lm in fm_results.multi_face_landmarks:
            '''
            4:   Nose tip
            263: Left eye left (outer) corner
            362: Left eye right (inner) corner
            133: Right eye left (inner) corner
            33:  Right eye right (outer) corner
            375: Left mouth corner
            61:  Right mouth corner
            2:   Subnasale
            168: Nose root
            '''
            fitting_points = np.zeros((n_face_model,2),dtype=np.float32)
            for i, j in enumerate([4, 263, 362, 133, 33, 375, 61, 2, 168]):
                fitting_points[i] = (int(lm.landmark[j].x*frame.shape[1]), int(lm.landmark[j].y*frame.shape[0]))
            detections.append(fitting_points)

            '''
            362, 384, 387, 263, 373, 380: Left eyelid (clockwize viewing from the front of the face)
            33, 160, 157, 133, 153, 144: Right eyelid (clockwize viewing from the front of the face)
            '''
            eyelid_points = np.zeros((12,2),dtype=np.float32)
            for i, j in enumerate([362, 384, 387, 263, 373, 380, 33, 160, 157, 133, 153, 144]):
                eyelid_points[i] = (int(lm.landmark[j].x*frame.shape[1]), int(lm.landmark[j].y*frame.shape[0]))
            eyelids_detections.append(eyelid_points)

        # check whether faces are in AOI
        face_detected = False
        if aoi is None:
            if len(detections) > 0:
                face_detected = True
                target_idx = 0
        else:
            for target_idx in range(len(detections)):
                xmin = detections[target_idx][:,0].min()
                xmax = detections[target_idx][:,0].max()
                ymin = detections[target_idx][:,1].min()
                ymax = detections[target_idx][:,1].max()
                if aoi.contains(rect(xmin, ymin, xmax-xmin, ymax-ymin)):
                    face_detected = True
                    break
        
        if face_detected:
            return True, detections[target_idx], eyelids_detections[target_idx]

        else:
            return False, None, None

    else:
        msg = '{} is not supported. available detectors are: {}'.format(detector, supported_face_detectors)
        raise ValueError(msg)


def get_face_boxes(frame, detector='mediapipe'):
   
    if detector == 'dlib':
        dets, _, _ = dlib_face_detector.run(frame, 0) # detections, scores, weight_indices
        detections = [rect(d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top()) for d in dets]
        return detections
    
    elif detector == 'mediapipe':
        fm_results = facemesh.process(frame)
        detection = []
        for fm in fm_results.multi_face_landmarks:
            px, py = [], []
            for lm in fm.landmark:
                px.append(int(lm.x * frame.shape[1]))
                py.append(int(lm.y * frame.shape[0]))
            xmin = np.min(px)
            xmax = np.max(px)
            ymin = np.min(py)
            ymax = np.max(py)
            detection.append(rect(xmin, ymin, xmax-xmin, ymax-ymin))
        return detection

    else:
        msg = '{} is not supported. available detectors are: {}'.format(detector, supported_face_detectors)
        raise ValueError(msg)



class facedata(object):
    """
    landmarks = None
    rotation_matrix = None
    translation_vector = None
    euler_angles = None
    rotX = None
    rotY = None
    rotZ = None
    fitting_pts = np.zeros((n_model_points,2))
    marker_p1 = np.zeros(2)
    marker_p2 = np.zeros(2)
    model_points = None
    left_eye_camera_coord = np.zeros(3)
    right_eye_camera_coord = np.zeros(3)
    """

    # screen_size = 640x480
    # focal_length = 480
    # center = (640/2, 480/2)
    camera_matrix = np.array(
                             [[480,   0, 640/2],
                              [  0, 480, 480/2],
                              [  0,   0,     1]], dtype = "double"
                             )
    dist_coeffs = np.zeros((4,1)) # no lens distortion
    
    def __init__(self, landmarks, eyelid_points, camera_matrix=None, dist_coeffs=None, face_model=None, eye_params=None, prev_vec=(None, None), filter=(None, None)):
        """
        Initialize face model.

            :param landmarks: 

            :param camera_matrix:
            :param dist_coeffs:
            :face_model:
        """
        if face_model is None:
            face_model = default_face_model
        if eye_params is None:
            eye_params = default_eye_params

        prev_rvec = prev_vec[0]
        prev_tvec = prev_vec[1]
        filter_rot = filter[0]
        filter_tr = filter[1]

        self.face_model = face_model
        self.eye_diameter = eye_params[0]
        self.eye_offset_L = eye_params[1:4]
        self.eye_offset_R = eye_params[4:]

        self.left_eye_center = (face_model[1] + face_model[2])/2.0 + self.eye_offset_L
        self.right_eye_center = (face_model[3] + face_model[4])/2.0 + self.eye_offset_R

        self.fitting_pts = landmarks
        self.eyelid_pts = eyelid_points

        if camera_matrix is not None:
            self.camera_matrix = camera_matrix
        
        if dist_coeffs is not None:
            self.dist_coeffs = dist_coeffs

        if prev_rvec is not None:
            self.rotation_vector = prev_rvec.copy()
        else:
            self.rotation_vector = np.array((0.0,0.0,0.0)).reshape((3,1))
        if prev_tvec is not None:
            self.translation_vector = prev_tvec.copy()
        else:
            self.translation_vector = np.array((0.0,0.0,600.0)).reshape((3,1))

        self.estimate_face_posture()
        self.euler_angles = get_euler_angles(self.rotation_matrix)

        if filter_rot is not None:
            self.euler_angles = filter_rot.update(self.euler_angles)
            # update rotation matrix ()
            self.rotation_matrix = get_rotation_matrix(self.euler_angles)
        if filter_tr is not None:
            self.translation_vector = filter_tr.update(self.translation_vector)

        self.rotX, self.rotY, self.rotZ = self.euler_angles

        self.calc_marker_2D()
        self.get_eye_center()
    
    def estimate_face_posture(self):
        """
        Calculate rotation matrix and translation vector of the face
        """
        # get rotation vector and translation vector
        (_, self.rotation_vector, self.translation_vector, _) = cv2.solvePnPRansac(
            self.face_model, self.fitting_pts, self.camera_matrix, self.dist_coeffs, 
            useExtrinsicGuess=True, rvec=self.rotation_vector, tvec=self.translation_vector,
            flags=cv2.SOLVEPNP_ITERATIVE)

         # get rotation matrix and projection matrix
        self.rotation_matrix, _ = cv2.Rodrigues(self.rotation_vector)
        self.projection_matrix = np.hstack((self.rotation_matrix, self.translation_vector))
        
    def calc_marker_2D(self):
        # calculate marker points to draw face direction vector
        pts = np.vstack((
            (self.face_model[2]+self.face_model[3])/2,
            [(25.0, 0.0, 0.0)],
            [(0.0, 25.0, 0.0)],
            [(0.0, 0.0, -100.0)]))
        (nose_end_pts, _) = cv2.projectPoints(
            pts, self.rotation_vector, self.translation_vector, self.camera_matrix, self.dist_coeffs)
        
        self.marker_pts = np.squeeze(nose_end_pts, axis=1).astype(np.int32)

    def get_eye_center(self):
        self.left_eye_camera_coord = np.dot(self.rotation_matrix, self.left_eye_center.reshape(3,1)) + self.translation_vector
        self.right_eye_camera_coord = np.dot(self.rotation_matrix, self.right_eye_center.reshape(3,1)) + self.translation_vector

    def draw_eyelids_landmarks(self, image):
        for (x, y) in self.eyelid_pts:
            cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)
    
    def draw_marker(self, image):
        for p in self.fitting_pts:
            cv2.circle(image, (int(p[0]), int(p[1])), 3, (0,255,0), -1)
        cv2.line(image, self.marker_pts[0], self.marker_pts[1], (0,0,255), 1)
        cv2.line(image, self.marker_pts[0], self.marker_pts[2], (0,255,0), 1)
        cv2.line(image, self.marker_pts[0], self.marker_pts[3], (255,0,0), 2)

        #debug
        """
        for i in range(3):
            cv2.putText(image, "{}".format(180*self.rotation_vector[i,0]/np.pi), 
                (10,100+24*i), cv2.FONT_HERSHEY_TRIPLEX, 1.0,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_8)
        """
    
    def update_model_points(self, model_points):
        self.model_points = model_points

    def get_fitting_error(self):
        """
        Get fitting error in pixel.
        """
        diff = []
        for i, p in enumerate(self.model_points):
            (p2d, jacobian) = cv2.projectPoints(
                p.reshape((1,3)), self.rotation_vector, self.translation_vector, self.camera_matrix, self.dist_coeffs)
            diff.append(np.linalg.norm(p2d - self.fitting_pts[i]))
        return diff
    
    def get_distance_nosetip(self):
        return np.linalg.norm(self.translation_vector)

    def get_distance_between_eyes(self):
        return np.linalg.norm(self.left_eye_center - self.right_eye_center)

