import math

import cv2
import pyocr
import pyocr.builders
import numpy as np
from PIL import Image
from moviepy.editor import *
import re
from concurrent.futures import ProcessPoolExecutor
import time

file_path = 'C:/Users/Gold/Videos/2023-09-20 17-11-38.mp4'
color_list = {
    'blue': [250, 178, 91, 50],
    'yellow': [50, 177, 208, 90],
    'gray': [188, 189, 189, 70]
}

color = color_list['gray']

minBGR = np.array([color[0] - color[3], color[1] - color[3], color[2] - color[3]])
maxBGR = np.array([color[0] + color[3], color[1] + color[3], color[2] + color[3]])

tesseract_path = 'C:/Program Files/Tesseract-OCR'
if tesseract_path not in os.environ["PATH"].split(os.pathsep):
    os.environ["PATH"] += os.pathsep + tesseract_path


def get_text(frame):
    bgr2rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # ocrに渡せるpillowの形に変換
    to_PIL = Image.fromarray(bgr2rgb)
    # ocrのツールの読込
    tools = pyocr.get_available_tools()
    tool = tools[0]
    lang = 'eng'
    # pyocrを用いて画像から文字を取得
    text = tool.image_to_string(
        to_PIL,
        lang=lang,
        builder=pyocr.builders.TextBuilder(tesseract_layout=7)
    )
    return text


def split_movie():
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        sys.exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    start_millis1 = math.ceil(total_frames / 4)
    start_millis2 = math.ceil(total_frames / 4 * 2)
    start_millis3 = math.ceil(total_frames / 4 * 3)
    print(f'Total time:{total_frames / fps}')
    return start_millis1, start_millis2, start_millis3, total_frames


def analyze(start, stop):
    frame_index = start
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        sys.exit(-1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = 0
    flag = True
    while frame_index <= stop:
        frame_index += 1
        idx += 1
        ret, frame = cap.read()
        if ret:
            if idx < cap.get(cv2.CAP_PROP_FPS):
                continue
            else:
                resized_frame = frame[70:90, 30:1570]
                if not flag:
                    cv2.imshow('check', resized_frame)
                    if cv2.waitKey() & 0xFF == ord('q'):
                        cv2.destroyWindow('check')
                        break
                    else:
                        flag = True
                maskBGR = cv2.inRange(resized_frame, minBGR, maxBGR)
                text = get_text(maskBGR)
                print(text)
                text = re.sub('[sx]a[nm][il!]', 'xaml', text)
                text = re.sub('[ce]s.?', 'cs', text)
                text = re.sub('[a-zA-Z](?!\\.)xamlcs', '.xaml.cs', text)
                text = re.sub('[a-zA-Z]\\.xamlcs', '.xaml.cs', text)
                text = re.sub('[a-zA-Z](?!\\.)xaml', '.xaml', text)
                text = re.sub('[Ww]?indo[wW]?', 'Window', text)
                text = re.sub('[a-zA-Z]cs[a-zA-Z]', '', text)
                result = re.findall('[a-zA-Z]+.xaml.cs|[a-zA-Z]+.cs|[a-zA-Z]+.xaml', text)
                second = int(cap.get(cv2.CAP_PROP_POS_FRAMES) / idx)
                filled_second = str(second).zfill(4)
                result.append(filled_second)
                cv2.imwrite("{}_{}.{}".format('C:/Users/Gold/Videos/tmp/image', filled_second, '.jpg'),
                            maskBGR)
                # print(text)
                print('\033[32m' + str(result) + '\033[0m')
                idx = 0
        else:
            break


if __name__ == '__main__':
    t = time.time()
    start1, start2, start3, total_frames = split_movie()
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.submit(analyze, 0, start1)
        executor.submit(analyze, start1, start2)
        executor.submit(analyze, start2, start3)
        executor.submit(analyze, start3, total_frames)
    print(f'Time elapsed:{time.time() - t}')
