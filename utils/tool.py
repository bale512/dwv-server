# -*- coding: UTF-8 -*-
import codecs
import hashlib
import traceback
import json
import random
import config
from logger_manager import controller_logger as logger
import hashlib
import os
import sys

# 保存进程PID到PID文件


def save_pid(path, pid):
    with open(path, 'w') as fp:
        fp.write(str(pid))


# 非daemon方式启动服务端
def start_service(func, pid_file):
    pid = os.getpid()
    # saved PID
    logger.info('controller process started at PID: ' + str(pid))
    save_pid(pid_file, pid)
    func()


# daemon方式启动服务端
def start_daemon_service(func, pid_file):
    pid = os.fork()
    if pid == 0:
        func()
    else:
        # saved PID of child process
        logger.info('controller process started at PID: ' + str(pid))
        save_pid(pid_file, pid)

# 关闭服务进程


def shutdown_service(pid_path):
    command = 'kill -9 `cat ' + pid_path + '`;rm -f ' + pid_path
    os.system(command)


def CalcSha1(filepath):
    with open(filepath, 'rb') as f:
        sha1obj = hashlib.sha1()
        sha1obj.update(f.read())
        hash = sha1obj.hexdigest()
        return hash


def CalcMD5(filepath):
    with open(filepath, 'rb') as f:
        md5obj = hashlib.md5()
        md5obj.update(f.read())
        hash = md5obj.hexdigest()
        print(hash)
        return has
