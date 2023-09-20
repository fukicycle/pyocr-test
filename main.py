import cv2
import pyocr
import pyocr.builders
import numpy as np
from PIL import Image
from moviepy.editor import *
import re

# file_path = 'C:/Users/Gold/Videos/Session4_2.mp4'
file_path = 'C:/Users/Gold/Videos/2023-09-19 23-59-16.mp4'
delay = 1
window_name = 'frame'
thresh = 50
# bgr = [50, 177, 208]  # yellow
# bgr = [] #gray
bgr = [250, 178, 91]  # blue

# 色の閾値
minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])

tesseract_path = 'C:/Program Files/Tesseract-OCR'
if tesseract_path not in os.environ["PATH"].split(os.pathsep):
    os.environ["PATH"] += os.pathsep + tesseract_path


def get_text(frame):
    """
    画像を編集する部分
    """

    # BGRをRGBに変換←ここの部分グレースケールなど工夫することで精度に変化が出ます
    bgr2rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    """
    OCRに投げるための処理
    """
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
        builder=pyocr.builders.TextBuilder(tesseract_layout=6),

    )
    return text


# while文の場合
cap = cv2.VideoCapture(file_path)

if not cap.isOpened():
    sys.exit()
idx = 0
while cap.isOpened():
    idx += 1
    ret, frame = cap.read()
    if ret:
        if idx < cap.get(cv2.CAP_PROP_FPS):
            continue
        else:
            resized_frame = frame[100:150, 20:1570]
            maskBGR = cv2.inRange(resized_frame, minBGR, maxBGR)
            text = get_text(maskBGR)
            text = re.sub('[sx]a[nm][il!]', 'xaml', text)
            text = re.sub('[ce]s.?', 'cs', text)
            text = re.sub('[a-zA-Z](?!\\.)xamlcs', '.xaml.cs', text)
            text = re.sub('[a-zA-Z](?!\\.)xaml', '.xaml', text)
            result = re.findall('[a-zA-Z]+.xaml.cs|[a-zA-Z]+.cs|[a-zA-Z]+.xaml', text)
            second = int(cap.get(cv2.CAP_PROP_POS_FRAMES) / idx)
            filled_second = str(second).zfill(4)
            result.append(filled_second)
            cv2.imwrite("{}_{}.{}".format('C:/Users/Gold/Videos/tmp/image', filled_second, '.jpg'),
                        resized_frame)
            print(result)
            idx = 0
    else:
        break