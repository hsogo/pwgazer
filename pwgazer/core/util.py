import numpy as np
import warnings
from dlib import rectangle as dlib_rect
import os

debug_mode = False

def stretch(img):
    # input must be gray
    # https://qiita.com/satoyoshiharu/items/d33c4f6b2c80c87e0074
    inImg = img.astype('float64')
    maxv = np.amax(inImg)
    minv = np.amin(inImg)
    factor = 255.0 / (maxv - minv)
    out = (inImg - minv) * factor
    # out = cv2.blur(out.astype('uint8'), (3, 3))
    return out.astype('uint8')

def get_euler_angles(R):
    """
    get XYZ-Euler angle from rotation matrix.

    :param R: rotation matrix
    :return: XYZ-Euler angle
    """
    # https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    
    #assert(isRotationMatrix(R))
    #To prevent the Gimbal Lock it is possible to use
    #a threshold of 1e-6 for discrimination
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])    
    singular = sy < 1e-6

    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0     
    
    return np.array((x, y, z))

def get_rotation_matrix(Q):
    """
    get rotation matrix from XYZ-Euler angle

    :param Q: XYZ-Euler angle.
    :return: Rotation matrix.
    """
    if hasattr(Q,'shape') and Q.shape == (3,1):
        Q = np.ravel(Q)
    R_x = np.array([[1,         0,            0             ],
                    [0,         np.cos(Q[0]), -np.sin(Q[0]) ],
                    [0,         np.sin(Q[0]), np.cos(Q[0])  ]
                    ])

    R_y = np.array([[np.cos(Q[1]),    0,      np.sin(Q[1])  ],
                    [0,               1,      0             ],
                    [-np.sin(Q[1]),   0,      np.cos(Q[1])  ]
                    ])

    R_z = np.array([[np.cos(Q[2]),    -np.sin(Q[2]),    0],
                    [np.sin(Q[2]),    np.cos(Q[2]),     0],
                    [0,               0,                1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R


def get_float_image(im):
    minval = im.min()
    maxval = im.max()
    return (im-minval)/(maxval-minval)


def get_gaze_vector(point, eye_point):
    v = point - eye_point
    return v/np.linalg.norm(v)


def get_eye_rotation(face, eye):
    """
    get eye rotation from face and eye objects.

    Note: the success of this function heavily depends on user-defined parameters such as eye center.

    :param face: pwgazer.core.face.facedata object.
    :param eye: pwgazer.core.eye.eyedata object.
    :return: 2D iris vector.
    """
    iris_center_2D = eye.iris_center/eye.image_scale + eye.image_origin
    ec = face.left_eye_center if eye.eye == 'L' else face.right_eye_center
    eye_center_3D = (np.dot(face.rotation_matrix, ec.reshape((3,1))) + face.translation_vector).reshape((3,))

    Fx = face.camera_matrix[0,0] #focal length X (pix)
    Fy = face.camera_matrix[1,1] #focal length Y (pix)
    Cx = face.camera_matrix[0,2] #image center X (pix)
    Cy = face.camera_matrix[1,2] #image center Y (pix)
    iris_image_3D = np.array((iris_center_2D[0]-Cx, iris_center_2D[1]-Cy, (Fx+Fy)/2)) # 

    a = np.dot(iris_image_3D, iris_image_3D)
    b = -2*np.dot(iris_image_3D, eye_center_3D)
    c = np.dot(eye_center_3D, eye_center_3D)-(face.eye_diameter/2)**2
    if b*b-4*a*c < 0: # no answer
        return np.array((np.nan,np.nan))
    k = (-b+np.sqrt(b*b-4*a*c))/(2*a)
    iris_center_3D = iris_image_3D*k

    iris_vector = iris_center_3D - eye_center_3D
    iris_vector_norm = np.dot(np.linalg.inv(face.rotation_matrix), iris_vector)/face.eye_diameter # normalize

    return(iris_vector_norm[:2])


def calc_gaze_position(eye, rmat, eye_center, eye_norm, screen, fitting_param, filter=None):
    """
    get gaze position on the screen.

    """
    # normalized 2D iris center
    (nix, niy) = eye_norm
    #(nix, niy) = get_eye_rotation(face, eye)

    if filter is not None:
        (nix, niy) = filter.update((nix, niy))

    # calc 3D iris center
    if eye == 'L':
        if fitting_param is None:
            tx = nix
            ty = niy
        else:
            tx = np.dot(np.array((nix, niy, 1)), fitting_param[0])
            ty = np.dot(np.array((nix, niy, 1)), fitting_param[1])
        
    elif eye == 'R':
        if fitting_param is None:
            tx = nix
            ty = niy
        else:
            tx = np.dot(np.array((nix, niy, 1)), fitting_param[2])
            ty = np.dot(np.array((nix, niy, 1)), fitting_param[3])
    else:
        raise ValueError('Eye must be L or R')

    vec = np.dot(rmat, np.array([tx, ty, -1*(1-(tx**2+ty**2))]))
    sp = screen.get_screen_point_from_gaze_vector(vec.reshape(3), eye_center.reshape(3))

    return screen.convert_camera_coordinate_to_screen_coordinate(sp)


class MA_filter(object):
    def __init__(self, dim=2, order=3):
        if not isinstance(dim, int) or (dim not in (2, 3, 4)):
            raise ValueError('MA filter dim must be 2, 3 or 4.')
        if not isinstance(order, int) or order < 2:
            raise ValueError('MA filter order must be an integer greater than 1.')
        self.dim = dim
        self.order = order
        self.buffer = np.zeros((dim, order)) * np.nan # generate NaN array
    
    def update(self, measurement):
        if hasattr(measurement, 'shape') and measurement.shape != (self.dim,):
            orig_shape = measurement.shape
            measurement = measurement.reshape((self.dim,))
        else:
            orig_shape = None

        self.buffer[:,:(self.order-1)] = self.buffer[:,-(self.order-1):]
        self.buffer[:,-1] = measurement

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            if orig_shape is None:
                return np.nanmean(self.buffer, axis=1)
            else:
                return np.nanmean(self.buffer, axis=1).reshape(orig_shape)


def get_filter(face_filter_type, face_filter_param, iris_filter_type, iris_filter_param):
    if face_filter_type == '':
        face_filter_rot = None
        face_filter_tr = None
    elif face_filter_type == 'MA':
        if isinstance(face_filter_param, str):
            rot_order, tr_order = list(map(int, face_filter_param.split(',')))
        else:
            rot_order = face_filter_param[0]
            tr_order = face_filter_param[1]
        face_filter_rot = MA_filter(dim=3, order=rot_order)
        face_filter_tr = MA_filter(dim=3, order=tr_order)
    elif os.path.isfile(face_filter_type):
        # read codes from file
        with open(face_filter_type, 'r') as fp:
            code = fp.read()
        # execute codes
        try:
            exec(code)
        except:
            raise RuntimeError('Could not read filter from file: {}'.format(face_filter_type))
        if 'iri_filter_l' not in locals() or 'iris_filter_r' not in locals():
            raise RuntimeError('"face_filter_rot" and "face_filter_tr" must be defined in {}'.format(face_filter_type))
    else:
        raise RuntimeError('Unknown filter: {}'.format(face_filter_type))
    
    if iris_filter_type == '':
        iris_filter_l = None
        iris_filter_r = None
    elif iris_filter_type == 'MA':
        if not isinstance(iris_filter_param, int):
            iris_order = int(iris_filter_param)
        iris_filter_l = MA_filter(dim=2, order=iris_order)
        iris_filter_r = MA_filter(dim=2, order=iris_order)
    elif os.path.isfile(iris_filter_type):
        # read codes from file
        with open(iris_filter_type, 'r') as fp:
            code = fp.read()
        # execute code
        try:
            exec(code)
        except:
            raise RuntimeError('Could not read filter from file: {}'.format(iris_filter_type))
        if 'iri_filter_l' not in locals() or 'iris_filter_r' not in locals():
            raise RuntimeError('"iris_filter_l" and "iris_filter_r" must be defined in {}'.format(iris_filter_type))
    else:
        raise RuntimeError('Unknown filter: {}'.format(iris_filter_type))

    return face_filter_rot, face_filter_tr, iris_filter_l, iris_filter_r

def get_region_brightness_contrast(image, region, feature_region=None):
    if isinstance(region, dlib_rect):
        rt, rb, rl, rr = (region.top(), region.bottom(), region.left(), region.right())
    else:
        rt, rb, rl, rr = (region[1], region[1]+region[3], region[0], region[0]+region[2])

    if feature_region is not None:
        if isinstance(feature_region, dlib_rect):
            ft,  fb, fl, fr = (feature_region.top(), feature_region.bottom(), feature_region.left(), feature_region.right())
        else:
            ft, fb, fl, fr = (feature_region[1], feature_region[1]+feature_region[3], feature_region[0], feature_region[0]+feature_region[2])
        
        rt += 0.75*(ft-rt)
        rb += 0.75*(fb-rb)
        rl += 0.75*(fl-rl)
        rr += 0.75*(fr-rr)
    
    rt, rb, rl, rr = map(int, (rt, rb, rl, rr))

    region_image = image[rt:rb, rl:rr]

    #scene_average = np.mean(image)
    region_average = np.mean(region_image)
    region_rms_contrast = np.sqrt(np.mean((region_image - region_average) ** 2))
    
    return region_average, region_rms_contrast, (rl, rt, rr-rl, rb-rt)


