# pwgazer
## **P**ython-based, **W**ide-angle **GAZE** **R**ecoder

*pwgazer* is a python-based eye tracker for wide angle camera (such as web camera).
Postion and pose of human faces in the camera coordinate, relative position of pupils in the face, and gaze position on the PC screen are recorded.

### Setup

Before using pwgazer, camera calibration must be performed and screen layout must be set.
To calibrate the camera, first prepare a checkerboard.
An example of checkerboard can be found in the resoureces folder (pwgazer/core/resouces/checker8x5.pdf). 
To use checker8x5.pdf, print it out on A4 paper and paste it onto a rigid board.
With the camera to be used for the measurement available on the PC, execute the following command:

```
python -m pwgazer.app.CameraCalibration
```

Camera Calibration supports three sources: live camera, video file, and image file.  Add images of the checkerboard taken at various angles and select "Run" menu to perform calibration.
Calibration results are output as text and can be copied to the clipboard in the format of a configuration file. Prepare a blank text file (hereafter referred to as *camera_parameters.cfg*) and paste the calibration results into it.  Don't close the application until pasted.  Below is an example of the calibration results.

```
[Basic Parameters]
CAMERA_ID = 
RESOLUTION_HORIZ = 1920
RESOLUTION_VERT = 1080
DOWNSCALING = 0.5

[Calibration Parameters]
CAMERA_MATRIX_R0C0=1369.9104423575095
CAMERA_MATRIX_R0C1=0.0
CAMERA_MATRIX_R0C2=936.5660303951332
CAMERA_MATRIX_R1C0=0.0
CAMERA_MATRIX_R1C1=1371.2735355780137
CAMERA_MATRIX_R1C2=551.3859376282229
CAMERA_MATRIX_R2C0=0.0
CAMERA_MATRIX_R2C1=0.0
CAMERA_MATRIX_R2C2=1.0
DIST_COEFFS_R0C0=0.09818978049625876
DIST_COEFFS_R1C0=-0.048334067912787604
DIST_COEFFS_R2C0=0.0030492908222483615
DIST_COEFFS_R3C0=-2.1006330251313423e-05
DIST_COEFFS_R4C0=-2.1006330251313423e-05

[Screen Layout Parameters]
WIDTH=
HORIZ_RES=
OFFSET_X=
OFFSET_Y=
OFFSET_Z=
ROT_X=
ROT_Y=
ROT_Z=

[Filter]
FACE_FILTER = MA
FACE_FILTER_PARAM = 2,2
IRIS_FILTER = MA
IRIS_FILTER_PARAM = 2
```

As can be seen, "CAMERA_ID" in the "Basic Parameters" section and all entries in the "Screen Layout Parateters" section are not filled.
"CAMERA_ID" can be checked with the following command.  Note that the camera used for the measurement must be connected to the PC in advance.

```
python -m pwgazer.app.CameraSelector
```

This tool lists the cameras connected to the PC and allows you to see the images captured by each.
The number on the list is the CAMERA_ID.  The camera ID may change as cameras are added or removed.

To fill "Screen Layout Parameters" entries, run the following command.

```
python -m pwgazer.app.ScreenLayout
```

This command displays guide lines on the screen for measuring dimensions.  After entering the measured values and clicking OK, the parameters are displayed in the format of a configuration file.  Copy them and paste into the "Scrren Layout Parameters" section of *camera_parameters.cfg*.

If you want to use *camera_parameters.cfg* as your default configuration file, save it in pwgazer's configuration folder under the name *CameraParam.cfg*.
The configuration folder is as follows:

- %appdata%/pwgazer (Windows)
- $HOME/.pwgazer (Mac/Linux)

### Offline recording 

Offline recording mode measures the gaze of the participant in the video files.  Calibration task in which participants fixate on a known point on the screen must be recorded in the video.  The following command performs an example of calibration task.

```
python -m pwgazer.demo.OfflineCalibrationDemo
```

This demo draws calibration target on the screen while capturing camera image.  In each session, a CSV file and a MP4 movie file are created.  The CSV file records location and timing of the calibration target.  The MP4 file is a movie captured by the camera.  The following command calculates calibration parameters using these files.

```
python -m pwgazer.app.OfflineCalibration
```

The OfflineCalibration command has several options.  For example, CSV file, movie file and camear parameter configration file can be specified with command line options.

```
python -m pwgazer.app.OfflineCalibration --movie=cal.mp4 --cal_info=cal.csv --camera_param=camera_parameteres.cfg
```

Pwgazer requires 3D coordinates of facial feature points to estimate facial posture.
These coordinate values are defined in the FaceModel.cfg file in pwgazer's configuration folder.
To use other files, use --face_model option.

```
python -m pwgazer.app.OfflineCalibration --face_model=my_facemodel.cfg
```

The --iris_detector option specifies how the iris center is detected from the face image.  Three detectors, peak, ert, and enet are included in the pwgazer package.
The peak detector is fast in computation but very inaccurate especially in the vertical direction.
The enet detector requires the most time for computation but has the most accurate of the three.
The ert detector This detector is in the middle in terms of speed and accuracy.

```
python -m pwgazer.app.OfflineCalibration --iris_detector=enet
```

Calibration result is saved to a file with a .npz extension (e.g. cal_param.npz).

Batch mode is useful for processing multiple files at once.  In batch mode, calibration results can be output without any GUI manipulation.  Input movie file (--movie), calibration information (--cal_info), and output filename (--output) must be specified by commandline options.

```
python -m pwgazer.app.OfflineCalibration --movie=cal.mp4 --cal_info=cal.csv --output=cal_param.npz --batch
```

After the calibration result file is generated, the following command can be used to measure eye gaze for videos of the same participant in the same camera/screen configuration.  Most commandline options are common to the OfflineCalibration.

```
python -m pwgazer.app.OfflineTracker
```

### Realtime recording

Gaze measurement is a time-consuming process, taking much longer than one video frame.  Therefore, offline recording mode is recommended.
However, real-time recording is also supported if needed.  The following command starts the realtime tracker.  Similar to the offline tracker, 

```
python -m pwgazer.app.RealtimeTracker
```

The following command runs a demo of realtime recording.  PsychoPy and GazeParser are rquired to run the demo.

```
python -m pwgazer.demo.CalibrationDemo
```
