"""
.. Part of pwgazer package.
.. Copyright (C) 2025 Hiroyuki Sogo.
.. Distributed under the terms of the GNU General Public License (GPL).
"""

import os
import sys
import shutil
from .core import config as cfg

appDir = os.path.abspath(os.path.dirname(__file__))

if sys.platform == 'win32': #Windows
    homeDir = os.environ['USERPROFILE']
    appdataDir = os.environ['APPDATA']
    configDir = os.path.join(appdataDir,'pwgazer')
else: #MacOS X and Linux
    homeDir = os.environ['HOME']
    configDir = os.path.join(homeDir,'.pwgazer')

#create config directory if not exist.
if not os.path.exists(configDir):
    # Don't create directory if OS is not win32 and the user is root.
    if sys.platform == 'win32' or os.getuid() != 0:
        if not os.path.exists(configDir):
            os.mkdir(configDir)
            print('pwgazer: ConfigDir is successfully created.')
        
        src = []
        dst = []
        for fname in os.listdir(appDir):
            if fname[-4:] == '.cfg':
                src.append(os.path.join(appDir, fname))
                dst.append(os.path.join(configDir, fname[:-4]+'.cfg'))

        for i in range(len(src)):
            if not os.path.exists(dst[i]):
                print('%s -> %s' % (src[i], dst[i]))
                shutil.copyfile(src[i], dst[i])
            else:
                print('%s is existing.' % (dst[i]))

config = cfg.config()
