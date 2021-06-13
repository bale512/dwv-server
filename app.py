import threading
import argparse
# import requests
import config as sys_config
import os
import zipfile
from flask import Flask, request, redirect, url_for, flash, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
import utils.tool as tool
from flask_cors import CORS
import RL.predict as Predict


UPLOAD_FOLDER = os.path.join(
    os.path.dirname(os.path.relpath(__file__)),
    sys_config.DICM_SAVE_PATH)
ALLOWED_EXTENSIONS = set(['zip'])

app = Flask(__name__, static_folder='public')
# 解决跨域
CORS(app, supports_credentials=True)
app.config.from_object('config')

# 创建一个锁
mu = threading.Lock()

# RL预测模型
# rlPredict = Predict(
#     model_with_crop_path=r"/home/jiayoutao/mailrlseg/segmention/rlseg/hsresult/pnet_high_level_enhanced/pnet_high_level_enhanced_499.pth")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 解压上传的zip文件，并放在服务器上
@app.route('/api/uploadZip', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

        filename = secure_filename(file.filename)
        # 保存zip文件
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        # 计算文件的hash值
        hash = tool.CalcSha1(os.path.join(UPLOAD_FOLDER, filename))
        # 解压文件
        zip_ref = zipfile.ZipFile(
            os.path.join(UPLOAD_FOLDER, filename), 'r')
        zip_ref.extractall(os.path.join(UPLOAD_FOLDER, hash))
        # 解压后各个文件的路径
        dcmList = [os.path.join(UPLOAD_FOLDER, hash, x.filename)
                   for x in zip_ref.filelist]
        zip_ref.close()

        # 返回json
        result = dict()
        result['msg'] = 'success'
        result['data'] = dcmList
        result['code'] = 0
        return jsonify(result)


# 强化预测
@app.route('/api/rlSegment', methods=['POST'])
def rl_predict():
    hints = request.json.get('hints')
    dir = request.json.get('dir')
    print(hints)
    print(dir)
    result = dict()
    result['msg'] = 'success'
    result['code'] = 0
    # 模型预测，以及返回预测结果
    if len(hints) == 0:
        result['data'] = rlPredict.predict_with_crop(dcm_list_path=dir, information_path="liyulong_t1c_information",
                                                     output_path=os.path.join(dir, 'predict'))
    else:
        result['data'] = rlPredict.second_model_inference(
            dir, os.path.join(dir, 'predict'), "liyulong_t1c_information", hints)

    return jsonify(result)


def run():
    app.run(debug=sys_config.DEBUG, host='0.0.0.0',
            port=sys_config.SERVER_PORT, threaded=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Object detection annotation service.')
    parser.add_argument('--start', action='store_true',
                        help='running background')
    parser.add_argument('--stop', action='store_true', help='shutdown process')
    parser.add_argument('--restart', action='store_true',
                        help='restart process')
    parser.add_argument('--daemon', action='store_true',
                        help='restart process')

    FLAGS = parser.parse_args()
    if FLAGS.start:
        if FLAGS.daemon:
            tool.start_daemon_service(run, sys_config.PID_FILE)
        else:
            tool.start_service(run, sys_config.PID_FILE)
    elif FLAGS.stop:
        tool.shutdown_service(sys_config.PID_FILE)
    elif FLAGS.restart:
        tool.shutdown_service(sys_config.PID_FILE)
        if FLAGS.daemon:
            tool.start_daemon_service(run, sys_config.PID_FILE)
        else:
            tool.start_service(run, sys_config.PID_FILE)
