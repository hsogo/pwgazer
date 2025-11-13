import configparser
import numpy as np
import os
import warnings

application_params = {
    'FACE_DETECTOR':'mediapipe',
    'IRIS_DETECTOR':'ert',
    'CALIBRATED_OUTPUT':'1',
    'CALIBRATIONLESS_OUTPUT':'0',
    'DATAFILE_OPEN_MODE':'new',
}

camera_params = {
    'CAMERA_ID':'0',
    'RESOLUTION_HORIZ':'640',
    'RESOLUTION_VERT':'480',
    'DOWNSCALING':'1.0'
}

camera_matrix_params = {
    'CAMERA_MATRIX_R0C0':'480.0',
    'CAMERA_MATRIX_R0C1':'0.0',
    'CAMERA_MATRIX_R0C2':'320.0',
    'CAMERA_MATRIX_R1C0':'0.0',
    'CAMERA_MATRIX_R1C1':'480.0',
    'CAMERA_MATRIX_R1C2':'240.0',
    'CAMERA_MATRIX_R2C0':'0.0',
    'CAMERA_MATRIX_R2C1':'0.0',
    'CAMERA_MATRIX_R2C2':'1.0',
    'DIST_COEFFS_R0C0':'0.0',
    'DIST_COEFFS_R1C0':'0.0',
    'DIST_COEFFS_R2C0':'0.0',
    'DIST_COEFFS_R3C0':'0.0',
    'DIST_COEFFS_R4C0':'0.0'
}

screen_layout_params = {
    'WIDTH':'464.0',
    'HORIZ_RES':'1920',
    'OFFSET_X':'0.0',
    'OFFSET_Y':'140.0',
    'OFFSET_Z':'30.0',
    'ROT_X':'-10.0',
    'ROT_Y':'0.0',
    'ROT_Z':'0.0'
}

face_model_params = {
    'NOSE_TIP':'0.00,34.64,-24.15',
    'LEFT_EYE_OUTER':'47.44,-2.34,7.97',
    'LEFT_EYE_INNER':'17.51,0.00,0.00',
    'RIGHT_EYE_INNER':'-17.51,0.00,0.00',
    'RIGHT_EYE_OUTER':'-47.44,-2.34,7.97',
    'LEFT_MOUTH_CORNER':'24.36,68.82,-1.60',
    'RIGHT_MOUTH_CORNER':'-24.36,68.82,-1.60',
    'SUBNASALE':'0.00,46.27,-14.26',
    'NOSE_ROOT':'0.00,-7.20,-7.19'
}

eye_params = {
    'EYE_DIAMETER':'24.0',
    'EYE_OFFSET_LX':'0.0',
    'EYE_OFFSET_LY':'0.0',
    'EYE_OFFSET_LZ':'-12.0',
    'EYE_OFFSET_RX':'0.0',
    'EYE_OFFSET_RY':'0.0',
    'EYE_OFFSET_RZ':'-12.0'
}

filter_params = {
    'FACE_FILTER':'MA',
    'FACE_FILTER_PARAM':'3,3',
    'IRIS_FILTER':'MA',
    'IRIS_FILTER_PARAM':'3'
}

class config(object):
    def __init__(self):
        self.face_detector = 'mediapipe'
        self.iris_detector = 'ert'
        self.calibrated_output = True
        self.calibrationless_output = False
        self.datafile_open_mode = 'new'
        self.camera_matrix = None
        self.dist_coeffs = None
        self.screen_width = None
        self.screen_h_res = None
        self.screen_offset = None
        self.screen_rot = None
        self.face_model = None
        self.eye_params = None
        self.face_filter = None
        self.face_filter_param = '0,0'
        self.iris_filter = None
        self.iris_filter_param = '0'

        self.application_param_file=''
        self.camera_param_file = ''
        self.face_model_file = ''
        self.filter_param_file = ''

    def load_application_param(self, filename):
        if not os.path.isfile(filename):
            msg = 'Configuration file ({}) is not found.'.format(filename)
            raise RuntimeError(msg)
        cfgp = configparser.ConfigParser()
        cfgp.optionxform = str
        cfgp.read(filename)

        for option in application_params.keys():
            try:
                s = cfgp.get('Application', option)
            except:
                s = application_params[option]
                msg = '"{}" is not defined in [Application]. Default value ({}) is used.'.format(option, s)
                warnings.warn(msg)

            if option == 'IRIS_DETECTOR':
                self.iris_detector = s
            if option == 'FACE_DETECTOR':
                self.face_detector = s
            elif option == 'CALIBRATED_OUTPUT':
                if s == 'False' or s == '0':
                    self.calibrated_output = False
                elif s == 'True' or s  == '1':
                    self.calibrated_output = True
                else:
                    msg = 'CALIBRATED_OUTPUT must be (False, True, 0, 1)'
                    raise ValueError(msg)
            elif option == 'CALIBRATIONLESS_OUTPUT':
                if s == 'False' or s == '0':
                    self.calibrationless_output = False
                elif s == 'True' or s  == '1':
                    self.calibrationless_output = True
                else:
                    msg = 'CALIBRATIONLESS_OUTPUT must be (False, True, 0, 1)'
                    raise ValueError(msg)
            elif option == 'DATAFILE_OPEN_MODE':
                if s in ('new','overwrite','rename'):
                    self.datafile_open_mode = s
                else:
                    msg = 'DATAFILE_OPEN_MODE must be (\'new\', \'overwrite\', \'rename\')'
                    raise ValueError(msg)

        if not (self.calibrated_output or self.calibrationless_output):
            msg = 'Either CALIBRATED_OUTPUT or CALIBRATIONLESS_OUTPUT must be True.'
            raise ValueError(msg)
        
        self.application_param_file = filename

    def load_camera_param(self, filename):
        if not os.path.isfile(filename):
            msg = 'Configuration file ({}) is not found.'.format(filename)
            raise RuntimeError(msg)
        cfgp = configparser.ConfigParser()
        cfgp.optionxform = str
        cfgp.read(filename)

        values = []
        for option in camera_params.keys():
            try:
                s = cfgp.get('Basic Parameters', option)
            except:
                s = camera_params[option]
                warnings.warn('"{}" is not defined in [Basic Parameters]. Default value ({}) is used.'.format(option, s))

            try:
                values.append(float(s))
            except:
                msg = 'Invalid value: {}={}'.format(option,s)
                raise ValueError(msg)

        self.camera_id = int(values[0])
        self.camera_resolution_h = int(values[1])
        self.camera_resolution_v = int(values[2])
        self.downscaling_factor = values[3]

        values = []
        for option in camera_matrix_params.keys():
            try:
                s = cfgp.get('Calibration Parameters', option)
            except:
                s = camera_params[option]
                msg = '"{}" is not defined in [Calibration Parameters]. Default value ({}) is used.'.format(option, s)
                warnings.warn(msg)

            try:
                values.append(float(s))
            except:
                msg = 'Invalid value: {}={}'.format(option,s)
                raise ValueError(msg)

        self.camera_matrix = np.array(values[:9]).reshape((3,3))
        self.dist_coeffs = np.array(values[9:]).reshape((5,1))

        values = []
        for option in screen_layout_params.keys():
            try:
                s = cfgp.get('Screen Layout Parameters', option)
            except:
                s = screen_layout_params[option]
                msg = '"{}" is not defined in [Screen Layout Parameters]. Default value ({}) is used.'.format(option, s)
                warnings.warn(msg)

            try:
                values.append(float(s))
            except:
                msg = 'Invalid value: {}={}'.format(option,s)
                raise ValueError(msg)
        
        self.screen_width = values[0]
        self.screen_h_res = int(values[1])
        self.screen_offset = values[2:5]
        self.screen_rot = values[5:8]

        self.camera_param_file = filename

    def load_face_model(self, filename):
        if not os.path.isfile(filename):
            msg = 'Configuration file ({}) is not found.'.format(filename)
            raise RuntimeError(msg)
        cfgp = configparser.ConfigParser()
        cfgp.optionxform = str
        cfgp.read(filename)

        values = []
        for option in face_model_params.keys():
            try:
                s = cfgp.get('Face Model', option)
            except:
                s = face_model_params[option]
                msg = '"{}" is not defined in [Face Model]. Default value ({}) is used.'.format(option, s)
                warnings.warn(msg)

            try:
                v = s.split(',')
                values.append((float(v[0]),float(v[1]),float(v[2])))
            except:
                msg = 'Invalid value: {}={}'.format(option,s)
                raise ValueError(msg)

        self.face_model = np.array(values)

        values = []
        for option in eye_params.keys():
            try:
                s = cfgp.get('Eye Parameters', option)
            except:
                s = eye_params[option]
                msg = '"{}" is not defined in [Eye Parameters]. Default value ({}) is used.'.format(option, s)
                warnings.warn(msg)

            try:
                values.append(float(s))
            except:
                msg = 'Invalid value: {}={}'.format(option,s)
                raise ValueError(msg)

        self.eye_params = np.array(values)

        self.face_model_file = filename


    def load_filter_param(self, filename):
        if not os.path.isfile(filename):
            msg = 'Configuration file ({}) is not found.'.format(filename)
            raise RuntimeError(msg)
        cfgp = configparser.ConfigParser()
        cfgp.optionxform = str
        cfgp.read(filename)

        for option in filter_params.keys():
            try:
                s = cfgp.get('Filter', option)
            except:
                s = filter_params[option]
                msg = '"{}" is not defined in [Filter]. Default value ({}) is used.'.format(option, s)
                warnings.warn(msg)

            if option == 'FACE_FILTER':
                self.face_filter = s
            elif option == 'FACE_FILTER_PARAM':
                self.face_filter_param = s
            elif option == 'IRIS_FILTER':
                self.iris_filter = s
            elif option == 'IRIS_FILTER_PARAM':
                self.iris_filter_param = s

        self.filter_param_file = filename


    def save_camera_param(self, filename=None):
        if self.camera_matrix is None or self.dist_coeffs is None:
            msg = 'Camera parameters are not initialized'
            raise RuntimeError(msg)
        serialized_params = np.hstack((self.camera_matrix.ravel(), self.dist_coeffs.ravel()))

        if filename is None:
            filename = self.camera_param_file

        with open(filename, 'w') as fp:
            fp.write('[Basic Parameters]\n')
            fp.write('CAMERA_ID = {}\n'.format(self.camera_id))
            fp.write('RESOLUTION_HORIZ = {}\n'.format(self.camera_resolution_h))
            fp.write('RESOLUTION_VERT = {}\n'.format(self.camera_resolution_v))
            fp.write('DOWNSCALING = {}\n'.format(self.downscaling_factor))
            fp.write('\n')

            fp.write('[Calibration Parameters]\n')
            for i, option in enumerate(camera_matrix_params):
                fp.write('{} = {}\n'.format(option, serialized_params[i]))
            fp.write('\n')
            
            fp.write('[Screen Layout Parameters]\n')
            fp.write('WIDTH = {}\n'.format(self.screen_width))
            fp.write('HORIZ_RES= {}\n'.format(self.screen_h_res))
            for i, axis in enumerate(['X','Y','Z']):
                fp.write('OFFSET_{} = {}\n'.format(axis, self.screen_offset[i]))
            for i, axis in enumerate(['X','Y','Z']):
                fp.write('ROT_{} = {}\n'.format(axis, self.screen_rot[i]))

    def save_face_model(self, filename=None):
        if self.face_model is None:
            msg = 'Face model is not initialized'
            raise RuntimeError(msg)

        if filename is None:
            filename = self.face_model_file

        with open(filename, 'w') as fp:
            fp.write('[Face Model]\n')
            for i, option in enumerate(face_model_params):
                fp.write('{} = {},{},{}\n'.format(option, self.face_model[i,0], self.face_model[i,1], self.face_model[i,2]))

            fp.write('\n')
            fp.write('[Eye Parameters]\n')
            for i, option in enumerate(eye_params):
                fp.write('{} = {}\n'.format(option, self.eye_params[i]))
            
    def save_filter_param(self, filename=None):
        if filename is None:
            filename = self.filter_param_file

        with open(filename, 'w') as fp:
            fp.write('[Filter]\n')
            fp.write('FACE_FILTER = {}\n'.format(self.face_filter))
            fp.write('FACE_FILTER_PAARM = {}\n'.formt(self.face_filter_param))
            fp.write('IRIS_FILTER = {}\n'.format(self.iris_filter))
            fp.write('IRIS_FILTER_PAARM = {}\n'.format(self.iris_filter_param))

