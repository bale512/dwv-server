# -*- coding: UTF-8 -*-
import os
from datetime import timedelta
basedir = os.path.abspath(os.path.dirname(__file__))

# Flask
DEBUG = True
SEND_FILE_MAX_AGE_DEFAULT = timedelta(seconds=1)

# 日志配置
LOG_DIR = os.path.join(basedir, 'logs')
LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s [in %(module)s.%(funcName)s:%(lineno)d]'
LOG_LEVEL = 'info'

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

# 节点配置
PID_FILE = 'dwv-server.pid'
SERVER_PORT = 7777
DICM_SAVE_PATH = 'static/_temp'
HOST_NAME = 'http://localhost:{port}'.format(port=SERVER_PORT)
