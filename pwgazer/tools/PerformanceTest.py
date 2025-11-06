import sys
import os
import numpy as np
import cv2
import argparse
import codecs
from ..core import config
from ..core.face import facedata, detect_face
from ..core.eye import eyedata
from ..core.util import calc_gaze_position, rect
from ..core.screen import screen
from ..core.util import get_region_brightness_contrast

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description='pwgazer performance test')
    arg_parser.add_argument('-i', '--image', type=str, help='Image file or folder', required=True)
    arg_parser.add_argument('-f', '--face-model', type=str, help='face model file')
    arg_parser.add_argument('-c', '--camera-caldata', type=str, help='camera calibration file')
    arg_parser.add_argument('-o', '--output', type=str, help='output file')
    arg_parser.add_argument('--face-detector', type=str, help='face detector (mediapipe, dlib)')
    arg_parser.add_argument('--iris-detector', type=str, help='iris detector (ert, peak, enet)')
    arg_parser.add_argument('--image-gain', type=float, help='pixel values are multiplied by this value')
    arg_parser.add_argument('--noise-sigma', type=float, help='sigma of Gaussian random noise')
    args = arg_parser.parse_args()

    conf = config.config()
    # image folder

    if os.path.isdir(args.image):
        files = [os.path.join(args.image,f) for f in os.listdir(args.image)]
    elif os.path.isfile(args.image):
        files = [args.image]
    else:
        print('{} must be a file or directory'.format(args.image))
        sys.exit()

    if args.camera_caldata is not None:
        conf.load_camera_param(args.camera_caldata)
    if args.face_model is not None:
        conf.load_face_model(args.face_model)

    dummy_screen = screen()
    dummy_screen.set_parameters(
        conf.screen_width/conf.screen_h_res, 
        conf.screen_rot,
        conf.screen_offset)    

    output_to_file = False
    if args.output is not None:
        outfp = codecs.open(args.output, 'w', 'utf-8')
        outfp.write('Image,Score,rX,rY,rZ,tX,tY,tZ,nLX,nLY,nRX,nRY,LX,LY,RX,RY\n')
        output_to_file = True
    
    if args.iris_detector == 'enet':
        from ..core.iris_detectors.enet_detector import enet_detector as iris_detector
    elif args.iris_detector == 'peak':
        from ..core.iris_detectors.peak_detector import peak_detector as iris_detector
    else: #default = ert detector
        from ..core.iris_detectors.ert_detector import ert_detector as iris_detector

    if args.face_detector is None:
        face_detector = 'dlib'
    else:
        face_detector = args.face_detector

    for f in files:
        try:
            frame = cv2.imread(f)
        except:
            print('{} is not a valid image file'.format(f))
            continue

        leye_img = None
        reye_img = None

        dtype = frame.dtype
        if args.image_gain is not None:
            frame *= args.image_gain
        if args.noise_sigma is not None:
            noises = np.random.normal(0, args.noise_sigma, frame.shape)
            frame = frame + noises
        if frame.dtype != dtype:
            frame[frame<0] = 0
            frame[frame>255] = 255
            frame = frame.astype(dtype)
        frame_mono = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_detected, landmarks, eyelids = detect_face(frame, detector=face_detector)

        if face_detected:
            # create facedata
            face = facedata(landmarks, eyelids, conf.camera_matrix, conf.dist_coeffs, conf.face_model)
            # calculate brightness and contrast
            fp_rect = cv2.boundingRect(face.fitting_pts.astype(np.int32))
            region_average, region_rms_contrast, target_rect = get_region_brightness_contrast(frame_mono, rect(*fp_rect))

            line = '{},'.format(f) # Image file
            line += ',' # Score is obsolate
            line += '{},{},{},'.format(*(180*face.euler_angles/np.pi)) # rX, rY, rZ
            line += '{},{},{},'.format(*np.ravel(face.translation_vector)) # tX, tY, tZ

            # create eyedata
            left_eye = eyedata(frame_mono, eyelids, eye='L', iris_detector=iris_detector)
            right_eye = eyedata(frame_mono, eyelids, eye='R', iris_detector=iris_detector)

            # normalized cood
            if not left_eye.blink:
                line += '{},{},'.format(*left_eye.normalized_iris_center) # nLX, nLY
                #line += '{},{},'.format(*get_eye_rotation(face, left_eye))
            else:
                line += ',,'
            if not right_eye.blink:
                line += '{},{},'.format(*right_eye.normalized_iris_center) # nRX, nRY
                #line += '{},{},'.format(*get_eye_rotation(face, right_eye))
            else:
                line += ',,'
            
            # screen position
            if not left_eye.blink:
                line += '{},{},'.format(*calc_gaze_position('L', face.rotation_matrix, face.left_eye_camera_coord, left_eye.normalized_iris_center, dummy_screen, None, None)) # LX, LY
            else:
                line += ',,'
            if not right_eye.blink:
                line += '{},{}'.format(*calc_gaze_position('R', face.rotation_matrix, face.right_eye_camera_coord, right_eye.normalized_iris_center, dummy_screen, None, None)) # LX, LY
            else:
                line += ','

            if output_to_file:
                outfp.write(line+'\n')
            else:
                print(line)
            
            if not left_eye.blink:
                #left_eye.draw_marker(frame)
                leye_img = left_eye.draw_marker_on_eye_image()
            if not right_eye.blink:
                #right_eye.draw_marker(frame)
                reye_img = right_eye.draw_marker_on_eye_image()

            face.draw_marker(frame)
            face.draw_eyelids_landmarks(frame)
            cv2.rectangle(frame, (target_rect.left, target_rect.top), (target_rect.right, target_rect.bottom),
                          (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'RA:{:.1f},CNT:{:.1f}'.format(region_average, region_rms_contrast),
                        (target_rect.left, target_rect.top), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        else: #face not found
            if output_to_file:
                outfp.write(f+','*14+'\n')
            else:
                print(f)
            
        cv2.imshow('original', frame)
        if leye_img is not None:
            cv2.imshow('left eye', leye_img)
        if reye_img is not None:
            cv2.imshow('right eye', reye_img)

        c = cv2.waitKey(1)
        if c == 27: #press ESC to exit
            break
    




