import math
import itertools
import cv2
import pyocr
import pyocr.builders
import numpy as np
from PIL import Image
from moviepy.editor import *
import re
from concurrent.futures import ProcessPoolExecutor
import time
import pandas

top = 70
height = 90
left = 1920
width = left + 1600
file_path = 'C:/Users/Gold/Videos/2023-09-21 14-41-11.mp4'
color_list = {
    'blue': [250, 178, 91, 50],
    'gold': [50, 177, 208, 90],
    'gray': [188, 189, 189, 70],
    'green': [164, 197, 107, 30],
    'brown': [111, 139, 183, 50]
}

color = color_list['brown']

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
    flag = False
    if not flag:
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames / 2)
        ret, frame = cap.read()
        resized_frame = frame[top:height, left:width]
        maskBGR = cv2.inRange(resized_frame, minBGR, maxBGR)
        cv2.imshow('check', resized_frame)
        cv2.imshow(get_text(maskBGR), maskBGR)
        if cv2.waitKey() & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            sys.exit()
        else:
            cv2.destroyAllWindows()
            flag = True
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    start_millis1 = math.ceil(total_frames / 4)
    start_millis2 = math.ceil(total_frames / 4 * 2)
    start_millis3 = math.ceil(total_frames / 4 * 3)
    print(f'Total time:{total_frames / fps}')
    return start_millis1, start_millis2, start_millis3, total_frames


def analyze(start, stop):
    analyze_result = []
    frame_index = start
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        sys.exit(-1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = 0
    while frame_index <= stop:
        frame_index += 1
        idx += 1
        ret, frame = cap.read()
        if ret:
            if idx < cap.get(cv2.CAP_PROP_FPS):
                continue
            else:
                resized_frame = frame[top:height, left:width]
                maskBGR = cv2.inRange(resized_frame, minBGR, maxBGR)
                text = get_text(maskBGR)
                # print(text)
                text = re.sub('[sx]a[nm][il!]', 'xaml', text)
                text = re.sub('[ce]s.?', 'cs', text)
                text = re.sub('[a-zA-Z](?!\\.)xamlcs', '.xaml.cs', text)
                text = re.sub('[a-zA-Z]\\.xamlcs', '.xaml.cs', text)
                text = re.sub('[a-zA-Z](?!\\.)xaml', '.xaml', text)
                text = re.sub('[Ww]?ind[oae][wW]?', 'Window', text)
                text = re.sub('[a-zA-Z]cs[a-zA-Z]', '', text)
                result = re.findall('[a-zA-Z]+.xaml.cs|[a-zA-Z]+.cs|[a-zA-Z]+.xaml', text)
                second = int(cap.get(cv2.CAP_PROP_POS_FRAMES) / idx)
                filled_second = str(second).zfill(4)
                if len(result) == 0:
                    result.append('-')
                result.append(filled_second)
                cv2.imwrite("{}_{}.{}".format('C:/Users/Gold/Videos/tmp/image', filled_second, '.jpg'),
                            maskBGR)
                # print(text)
                print('\033[32m' + str(result) + '\033[0m')
                analyze_result.append(result[0])
                idx = 0
        else:
            break
    return analyze_result


if __name__ == '__main__':
    total_result = []
    t = time.time()
    start1, start2, start3, total_frames = split_movie()
    with ProcessPoolExecutor(max_workers=4) as executor:
        f1 = executor.submit(analyze, 0, start1)
        f2 = executor.submit(analyze, start1, start2)
        f3 = executor.submit(analyze, start2, start3)
        f4 = executor.submit(analyze, start3, total_frames)
        total_result.append(f1.result())
        total_result.append(f2.result())
        total_result.append(f3.result())
        total_result.append(f4.result())
    print(f'Time elapsed:{time.time() - t}')
