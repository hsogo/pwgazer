import sys
import argparse
from pathlib import Path

import cv2

from ..core.config import config as configuration
from ..app._util import load_pwgazer_config
from ..core.eye import eyedata
from ..core.face import facedata, detect_face


if __name__ == '__main__':
    conf = configuration()
    arg_parser = argparse.ArgumentParser(description='pwgazer frame inspection tool')
    arg_parser.add_argument('movie', type=str, help='movie file')
    arg_parser.add_argument('frame', type=int, help='movie file')
    arg_parser.add_argument('--camera-param', type=str, help='camera parameters file')
    arg_parser.add_argument('--filter-param', type=str, help='filter parameters file')
    arg_parser.add_argument('--face-model', type=str, help='face model file')
    arg_parser.add_argument('--iris-detector', type=str, help='iris detector (ert, peak, enet or path to detector)')
    args = arg_parser.parse_args()

    camera_param_file, face_model_file, filter_param_file, iris_detector = load_pwgazer_config(conf, args)

    if iris_detector is None:
        print('Error:iris detector is invalid.')
        sys.exit()

    name = Path(Path(args.movie).name).stem

    with open('{}_{}.txt'.format(name, args.frame),'w') as rfp:
        cap = cv2.VideoCapture(args.movie)
        if not cap.isOpened():
            print('Error:movie({}) is not opened.'.format(args.movie))
            sys.exit()
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)

        ret, frame = cap.read()
        frame_mono = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        reye_img = None
        leye_img = None

        face_detected, landmarks, eyelids = detect_face(frame, scale=)

        # TODO? support rvecs?
        face_rvec = None
        face_tvec = None

        if face_detected: # face is found
            face_detected = True
            rfp.write('----- Face Detection -----\nResult:success\n')
            
            # only first face is used
            cv2.imwrite('{}_{}_mono.png'.format(name, args.frame), frame_mono)
            
            # create facedata
            face = facedata(landmarks, eyelids, camera_matrix=conf.camera_matrix, face_model=conf.face_model,
                eye_params=conf.eye_params, prev_rvec=face_rvec, prev_tvec=face_tvec)

            # create eyedata
            left_eye = eyedata(frame_mono, eyelids, eye='L', iris_detector=iris_detector)
            right_eye = eyedata(frame_mono, eyelids, eye='R', iris_detector=iris_detector)

            if not left_eye.blink:
                #left_eye.draw_marker(frame)
                leye_img = left_eye.draw_marker_on_eye_image()
            if not right_eye.blink:
                #right_eye.draw_marker(frame)
                reye_img = right_eye.draw_marker_on_eye_image()

            face.draw_marker(frame)
            face.draw_eyelids_landmarks(frame)

            cv2.imwrite('{}_{}_face.png'.format(name, args.frame),frame)

            cv2.imwrite('{}_{}_leye.png'.format(name, args.frame),leye_img)
            cv2.imwrite('{}_{}_reye.png'.format(name, args.frame),reye_img)

            rfp.write('----- Left Eye -----\n')
            if left_eye.detected:
                res = iris_detector(left_eye, debug=True)
                rfp.write('Result:{}\n'.format(res['status']))
                for key in res:
                    if key[:12] == 'ImageOutput_':
                        cv2.imwrite('{}_{}_left_{}.png'.format(name, args.frame, key[12:]),res[key])
            else:
                rfp.write('Result:not detected\n')

            rfp.write('----- Right Eye -----\n')
            if right_eye.detected:
                res = iris_detector(right_eye, debug=True)
                rfp.write('Result:{}\n'.format(res['status']))
                for key in res:
                    if key[:12] == 'ImageOutput_':
                        cv2.imwrite('{}_{}_right_{}.png'.format(name, args.frame, key[12:]),res[key])
            else:
                rfp.write('Result:not detected\n')

        else: # face is not found
            cv2.imwrite('{}_{}_mono.png'.format(name, args.frame), frame_mono)
            rfp.write('----- Face Detection -----\nResult:face was not detected\n')


