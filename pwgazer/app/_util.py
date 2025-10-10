import shutil
from pathlib import Path
import pwgazer
import numpy as np
import warnings
import wx

from ..core.iris_detectors import get_iris_detector

module_dir = Path(__file__).parent.parent

def load_pwgazer_config(conf, args):
    camera_param_file = None
    face_model_file = None
    filter_param_file = None

    appConfigDir = Path(pwgazer.configDir)

    if not appConfigDir.exists():
        Path.mkdir(appConfigDir)
        print('info: {} is created.'.format(appConfigDir))

    defaultconfig = appConfigDir/'tracker.cfg'
    if not defaultconfig.exists():
        shutil.copy(module_dir/'app'/'tracker.cfg',defaultconfig)
        print('info: default config file is created in {}.'.format(appConfigDir))
    conf.load_application_param(defaultconfig)

    if args.camera_param is None:
        # read default file
        cfgfile = appConfigDir/'CameraParam.cfg'
        if not cfgfile.exists():
            shutil.copy(module_dir/'core'/'resources'/'CameraParam.cfg', cfgfile)
            print('info: default camera parameter file is created in {}.'.format(appConfigDir))
        conf.load_camera_param(str(cfgfile))
        camera_param_file = str(cfgfile)
    else:
        conf.load_camera_param(args.camera_param)

    if args.face_model is None:
        cfgfile = appConfigDir/'FaceModel.cfg'
        if not cfgfile.exists():
            shutil.copy(module_dir/'core'/'resources'/'FaceModel.cfg',cfgfile)
            print('info: default face model file is created in {}.'.format(appConfigDir))
        conf.load_face_model(str(cfgfile))
        face_model_file = str(cfgfile)
    else:
        conf.load_face_model(args.face_model)

    if args.filter_param is None:
        cfgfile = appConfigDir/'FilterParam.cfg'
        if not cfgfile.exists():
            shutil.copy(module_dir/'core'/'resources'/'FilterParam.cfg',cfgfile)
            print('info: default filter parameter file is created in {}.'.format(appConfigDir))
        conf.load_filter_param(str(cfgfile))
        filter_param_file = str(cfgfile)
    else:
        conf.load_filter_param(args.filter_param)

    if args.iris_detector is None:
        iris_detector = get_iris_detector(conf.iris_detector)
    else:
        iris_detector = get_iris_detector(args.iris_detector)
    
    return camera_param_file, face_model_file, filter_param_file, iris_detector


def get_pwgazer_config_dir():
    return pwgazer.configDir


class recent_values(object):
    def __init__(self, shape):
        self._values = np.zeros(shape, dtype=np.float64)
        self._values.fill(np.nan)
        self._i = 0
    
    def append(self, value):
        self._values[self._i,:] = value
        self._i = (self._i+1) % self._values.shape[0]
    
    def values(self):
        return self._values
    
    def average(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanmean(self._values, axis=0)

    def std(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanstd(self._values, axis=0)


class CameraView(wx.StaticBitmap):
    def __init__(self, *args, **kwargs):
        super(CameraView, self).__init__(*args, **kwargs)
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.Bind(wx.EVT_PAINT, self.on_paint)

    def on_paint(self, event):
        try:
            image = self.GetBitmap()
            if not image:
                return
            dc = wx.AutoBufferedPaintDC(self)
            dc.Clear()
            dc.DrawBitmap(image, 0, 0, True)
        except:
            pass

