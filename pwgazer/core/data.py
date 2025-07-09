import os
# import datetime

import numpy as np

#from .eye import eyedata, eye_filter
#from .face import facedata, get_face_boxes, get_face_landmarks
from .util import calc_gaze_position, get_gaze_vector

class gazedata(object):
    def __init__(self, filename, open_mode='new', calibrated_output=True, calibrationless_output=False, debug_mode=False):
        self.fp = None
        self.recording_data = []
        self.message_data = []
        self.calibrated_output = calibrated_output
        self.calibrationless_output = calibrationless_output
        self.debug_mode = debug_mode

        if not (calibrated_output or calibrationless_output):
            raise ValueError('No gaze output')

        if open_mode not in ('new', 'overwrite', 'rename'):
            raise ValueError('write_mode must be "new", "overwrite" or "rename".')

        if os.path.exists(filename):
            if open_mode == 'new':
                return
            elif open_mode == 'rename':
                counter = 0
                while True:
                    backup_name = '{}.{}'.format(filename, counter)
                    if not os.path.exists(backup_name):
                        os.rename(filename, backup_name)
                        break
                    counter += 1

        try:
            self.fp = open(filename, 'w')
        except:
            self.fp = None
            return
        
        if self.fp is None:
            return

        # output header
        self.fp.write('#pwgazerDataFile\n')

        format_string = '#DATA_FORMAT,t,'
        if self.calibrated_output:
            format_string += 'xL,yL,xR,yR,'
        if self.calibrationless_output:
            format_string += '_xL,_yL,_xR,_yR,'
        format_string += 'face.rX,face.rY,face.rZ,face.tX,face.tY,face.tZ,earL,earR,blinkL,blinkR'
        if self.debug_mode:
            format_string += ',nlx,nly,nrx,nry'
        format_string += '\n'

        self.fp.write(format_string)

    def insert_settings(self, settings_list):
        if self.fp is None or self.recording:
            return
        
        # TODO insert saccade-related information


    def append_data(self, t, face, left_eye, right_eye, screen, fitting_param):
        data = (t,)

        if self.calibrated_output:
            if not left_eye.blink:
                xL, yL = calc_gaze_position('L', face.rotation_matrix, face.left_eye_camera_coord, 
                                            left_eye.normalized_iris_center, screen, fitting_param, None)
            else:
                xL = yL = np.nan

            if not right_eye.blink:
                xR, yR = calc_gaze_position('R', face.rotation_matrix, face.right_eye_camera_coord, 
                                            right_eye.normalized_iris_center, screen, fitting_param, None)
            else:
                xR = yR = np.nan

            data += (xL, yL, xR, yR)

        if self.calibrationless_output:
            if not left_eye.blink:
                xL, yL = calc_gaze_position('L', face.rotation_matrix, face.left_eye_camera_coord, 
                                            left_eye.normalized_iris_center, screen, None, None)
            else:
                xL, yL = (np.nan, np.nan)

            if not right_eye.blink:
                xR, yR = calc_gaze_position('R', face.rotation_matrix, face.right_eye_camera_coord, 
                                            right_eye.normalized_iris_center, screen, None, None)
            else:
                xR, yR = (np.nan, np.nan)

            data += (xL, yL, xR, yR)

        data += (face.rotX, face.rotY, face.rotZ,
            face.translation_vector[0,0], face.translation_vector[1,0], face.translation_vector[2,0],
            left_eye.eye_aspect_ratio,right_eye.eye_aspect_ratio,
            left_eye.blink, right_eye.blink)

        if self.debug_mode:
            try:
                nlx, nly = left_eye.normalized_iris_center
            except:
                nlx, nly = (np.nan, np.nan)
            try:
                nrx, nry = right_eye.normalized_iris_center
            except:
                nrx, nry = (np.nan, np.nan)

            data += (nlx, nly, nrx, nry)

        self.recording_data.append(data)
    
    def append_message(self, t, message):
        self.message_data.append((t, message))

    def start_recording(self, timestamp):
        if self.fp is None:
            return False

        self.fp.write('#START_REC,{}\n'.format(timestamp))
        self.recording_data = []
        self.message_data = []
    
    def get_latest_gazepoint(self, ma=1):
        if self.recording_data == []:
            return
        
        if ma==1:
            return self.recording_data[-1][1:5]
        else:
            tmp = np.array(self.recording_data[-ma:], dtype=float)
            return np.nanmean(tmp[:,1:5], axis=0)

    def stop_recording(self):
        self.flush()
        self.fp.write('#STOP_REC\n\n')
        self.fp.flush()

    def is_opened(self):
        return False if self.fp is None else True

    def has_data(self):
        return True if len(self.recording_data) > 0 else False

    def flush(self):
        if self.fp is None:
            return

        for data in self.recording_data:
            line = ','.join(['{}']*len(data))+'\n'
            self.fp.write(line.format(*data))
        self.recording_data = []

        for data in self.message_data:
            self.fp.write('#MESSAGE,{:.3f},{}\n'.format(*data))
        self.message_data = []
        
        self.fp.flush()

    def close(self):
        if self.fp is None:
            return
        
        if self.recording_data != [] or self.message_data != []:
            self.flush()

        self.fp.close()
        self.fp = None

    def __del__(self):
        self.close()


class calibrationdata(object):
    def __init__(self, screen, offline=True, debug_mode=True):
        self.offline = offline
        self.screen = screen
        self.raw_data = []
        self.data = []
        self.debug_mode = debug_mode

    def add_raw_data(self, face, leye, reye):
        if not self.offline:
            raise RuntimeError('This method is for the offline mode')
        rvec = face.rotation_vector if face is not None else None
        tvec = face.translation_vector if face is not None else None
        rmat = face.rotation_matrix if face is not None else None
        euler = face.euler_angles if face is not None else None
        leye_center = face.left_eye_camera_coord if face is not None else None
        reye_center = face.right_eye_camera_coord if face is not None else None
        leye_norm = leye.normalized_iris_center if not leye.blink else None
        reye_norm = reye.normalized_iris_center if not reye.blink else None

        self.raw_data.append((
            rvec, tvec, rmat, euler, leye_center, reye_center,
            leye_norm, reye_norm
        ))
    
    def add_caldata_from_raw(self, calpoint, indices):
        if not self.offline:
            raise RuntimeError('This method is for the offline mode')
        if len(self.raw_data) == 0:
            raise RuntimeError('Raw data must be stored befor calling this method.')
        
        for i in indices:
            if self.raw_data[i][0] is not None and \
               self.raw_data[i][6] is not None and \
               self.raw_data[i][7] is not None: # face, leye_norm, reye_norm
                self.data.append((calpoint, ) + self.raw_data[i])


    def add_caldata(self, calpoint, face, leye, reye):
        if self.offline:
            raise RuntimeError('This method is for the realtime mode')

        self.data.append((
            calpoint, # 
            face.rotation_vector, face.translation_vector,
            face.rotation_matrix, face.euler_angles,
            face.left_eye_camera_coord, face.right_eye_camera_coord,
            leye.normalized_iris_center,
            reye.normalized_iris_center
        ))

    def remove_points(self, points):
        for p in points:
            # search p from tail to head of self.data
            for idx in range(len(self.data)-1, -1, -1):
                if (p[0] == self.data[idx][0][0]) and \
                   (p[1] == self.data[idx][0][1]):
                    # remove entry
                    self.data.pop(idx)

    def clear_data(self):
        self.data = []

    def clear_raw_data(self):
        self.raw_data = []
    
    def is_empty(self):
        return True if len(self.data)==0 else False

    def LM_calibration(self):
        if len(self.data) == 0:
            raise RuntimeError('No calibration data.')
        if self.data[0][0] is None:
            raise RuntimeError('Calibration point is empty.')

        s = len(self.data)
        LX = np.zeros((s,1))
        LY = np.zeros((s,1))
        RX = np.zeros((s,1))
        RY = np.zeros((s,1))
        IJ_L = np.zeros((s,3))
        IJ_R = np.zeros((s,3))

        if self.debug_mode:
            cal_debugdata_fp = open('cal_debugdata.csv','w')
            cal_debugdata_fp.write('face_tx,face_y,face_tz,face_rx,face_ry,face_rz,nix_l,niy_l,nix_r,niy_r,orig_sample_x,orig_sample_y,sample_x,sample_y,sample_z,vec_lx,vec_ly,vec_rx,vec_ry\n')

        for idx, (orig_point, rvec, tvec, rmat, euler, leye_center, reye_center, leye_norm, reye_norm) in enumerate(self.data):
            # get normaized iris center
            (nix_l, niy_l) = leye_norm
            (nix_r, niy_r) = reye_norm
            
            # get target position in camera coordinate
            sample_point = self.screen.convert_screen_points_to_camera_coordinate(orig_point)
        
            # get gaze vecter
            vec_l = np.dot(np.linalg.inv(rmat), get_gaze_vector(sample_point, leye_center.reshape(3)))
            vec_r = np.dot(np.linalg.inv(rmat), get_gaze_vector(sample_point, reye_center.reshape(3)))
            
            LX[idx,0] = vec_l[0]
            LY[idx,0] = vec_l[1]
            RX[idx,0] = vec_r[0]
            RY[idx,0] = vec_r[1]
            
            IJ_L[idx,:] = [nix_l, niy_l, 1.0]
            IJ_R[idx,:] = [nix_r, niy_r, 1.0]

            if self.debug_mode:
                cal_debugdata_fp.write('{},{},{},'.format(*tvec.reshape((3,))))
                cal_debugdata_fp.write('{},{},{},'.format(*euler))
                cal_debugdata_fp.write('{},{},{},{},'.format(nix_l, niy_l, nix_r, niy_r))
                cal_debugdata_fp.write('{},{},'.format(*orig_point))
                cal_debugdata_fp.write('{},{},{},'.format(*sample_point))
                cal_debugdata_fp.write('{},{},'.format(*vec_l))
                cal_debugdata_fp.write('{},{}\n'.format(*vec_r))

        px_L = np.dot(np.dot(np.linalg.inv(np.dot(IJ_L.T,IJ_L)),IJ_L.T), LX)
        py_L = np.dot(np.dot(np.linalg.inv(np.dot(IJ_L.T,IJ_L)),IJ_L.T), LY)
        px_R = np.dot(np.dot(np.linalg.inv(np.dot(IJ_R.T,IJ_R)),IJ_R.T), RX)
        py_R = np.dot(np.dot(np.linalg.inv(np.dot(IJ_R.T,IJ_R)),IJ_R.T), RY)

        fitting_param = [px_L, py_L, px_R, py_R]

        if self.debug_mode:
            cal_debugdata_fp.write('{},{},{}\n'.format(*px_L.reshape((3,))))
            cal_debugdata_fp.write('{},{},{}\n'.format(*py_L.reshape((3,))))
            cal_debugdata_fp.write('{},{},{}\n'.format(*px_R.reshape((3,))))
            cal_debugdata_fp.write('{},{},{}\n'.format(*py_R.reshape((3,))))
            cal_debugdata_fp.close()

        return fitting_param

    def calc_results(self, fitting_param):
        """
        
        """
        error_list = np.zeros((len(self.data),2)) # L, R
        detail = []
        for idx, (orig_sample_point, rvec, tvec, rmat, euler, leye_center, reye_center, leye_norm, reye_norm) in enumerate(self.data):
            x_l, y_l = calc_gaze_position('L', rmat, leye_center, leye_norm, self.screen, fitting_param, None)
            x_r, y_r = calc_gaze_position('R', rmat, reye_center, reye_norm, self.screen, fitting_param, None)
            error_list[idx,0] = np.sqrt((x_l - orig_sample_point[0])**2 + (y_l - orig_sample_point[1])**2)
            error_list[idx,1] = np.sqrt((x_r - orig_sample_point[0])**2 + (y_r - orig_sample_point[1])**2)
            detail.append('{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f}'.format(
                orig_sample_point[0],
                orig_sample_point[1],
                x_l, y_l, x_r, y_r))

        precision = error_list.mean(axis=0)
        accuracy = error_list.std(axis=0)
        max_error = error_list.max(axis=0)
        results_detail = ','.join(detail)

        if self.debug_mode:
            cal_debugdata_fp = open('cal_debugdata.csv','a')
            cal_debugdata_fp.write('{},{},'.format(*precision))
            cal_debugdata_fp.write('{},{},'.format(*accuracy,))
            cal_debugdata_fp.write('{},{}\n'.format(*max_error))
            for i in range(len(detail)):
                cal_debugdata_fp.write('{}\n'.format(detail[i]))
            cal_debugdata_fp.close()

        return(precision, accuracy, max_error, results_detail)

