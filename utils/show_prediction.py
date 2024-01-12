import streamlit as st
import os
from PIL import Image
import cv2
import pandas as pd

def show(dir_path, num):
    file_name = sorted(os.listdir(dir_path))[num]
    img_path = os.path.join(dir_path, file_name)

    st.write(st.session_state.num, file_name)
    st.image(Image.open(img_path))

def show_img(): # 단순히 이미지만 확인하는 함수
    # streamlit으로 확인해보고 싶은 이미지 폴더 경로 
    dir_path = '/data/ephemeral/home/level2-objectdetection-cv-03/yolo/runs/detect/predict'
    if not os.path.exists(dir_path):
        st.error('Directory does not exist.')
    else:
        min_num = 0
        max_num = len(os.listdir(dir_path)) - 1

        st.subheader('Go to...')
        input_num = st.number_input(f'Image Number ({min_num} ~ {max_num})', min_value=min_num, max_value=max_num, step=1)
        enter = st.button('ENTER')

        st.subheader('Move...')
        col1, col2 = st.columns(2)
        prev_button = col1.button('PREV')
        next_button = col2.button('NEXT')

        if 'num' not in st.session_state:
            st.session_state.num = 0

        if enter:
            st.session_state.num = input_num
            show(dir_path, st.session_state.num)

        if prev_button:
            if st.session_state.num > min_num:
                st.session_state.num -= 1
            show(dir_path, st.session_state.num)

        if next_button:
            if st.session_state.num < max_num:
                st.session_state.num += 1
            show(dir_path, st.session_state.num)

class_name = {0: 'General trash', 1: 'Paper', 2: 'Paper pack', 3: 'Metal', 4: 'Glass', 
              5: 'Plastic', 6: 'Styrofoam', 7: 'Plastic bag', 8: 'Battery', 9: 'Clothing'}

hexs = ("FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB", 
        "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7")

def hex2rgb(h):
    return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

def draw_bbox(file_path, prediction_str, conf_thr_s, conf_thr_e):
    palette = [hex2rgb(f"#{c}") for c in hexs]
    n_palette = len(palette)

    img = cv2.imread(file_path)

    pred_list = prediction_str.split(' ')
    bbox_num = len(pred_list) // 6

    for i in range(bbox_num):
        j = i * 6
        c, score, xmin, ymin, xmax, ymax = map(float, pred_list[j:j + 6])
        c, xmin, ymin, xmax, ymax = map(int, [c, xmin, ymin, xmax, ymax])

        if score < conf_thr_s or score > conf_thr_e:
            continue
        
        col = palette[int(c) % n_palette]
        color = (col[2], col[1], col[0])

        lw = max(round(sum(img.shape) / 2 * 0.003), 2) 
        tf = max(lw - 1, 1)  # font thickness
        sf = lw / 3  # font scale

        label = f'{class_name[c]}: {score:.2f}'
        w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]  # text width, height

        # bbox
        p1, p2 = (xmin, ymin), (xmax, ymax)
        cv2.rectangle(img, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

        # bbox label
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            sf,
            (255, 255, 255),
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img)

def main():
    st.title('Recycle Detection Testset Prediction')

    # test set 경로
    img_path = '/data/ephemeral/home/dataset/test'
    # prediction csv 경로
    csv_path = '/data/ephemeral/home/level2-objectdetection-cv-03/yolo/results/ensemble/dino_yolo_conf0.05_nms.csv'
    # conf_thr_s 이상 conf_thr_e 이하인 bbox만 시각화
    conf_thr_s = 0.0
    conf_thr_e = 1.0

    if not os.path.exists(img_path):
        st.error('Directory does not exist.')
    elif not os.path.exists(csv_path):
        st.error('CSV file does not exist.')
    else:
        min_num = 0
        max_num = len(os.listdir(img_path)) - 1

        file_name_list = sorted(os.listdir(img_path))
        df = pd.read_csv(csv_path)

        st.subheader('Go to...')
        input_num = st.number_input(f'Image Number ({min_num} ~ {max_num})', min_value=min_num, max_value=max_num, step=1)
        enter = st.button('ENTER')

        st.subheader('Move...')
        col1, col2 = st.columns(2)
        prev_button = col1.button('PREV')
        next_button = col2.button('NEXT')

        if 'num' not in st.session_state:
            st.session_state.num = 0

        if enter:
            st.session_state.num = input_num
            file_name = file_name_list[st.session_state.num]
            st.write(st.session_state.num, file_name)
            draw_bbox(os.path.join(img_path, file_name), df['PredictionString'][st.session_state.num], conf_thr_s, conf_thr_e)

        if prev_button:
            if st.session_state.num > min_num:
                st.session_state.num -= 1

            file_name = file_name_list[st.session_state.num]
            st.write(st.session_state.num, file_name)
            draw_bbox(os.path.join(img_path, file_name), df['PredictionString'][st.session_state.num], conf_thr_s, conf_thr_e)

        if next_button:
            if st.session_state.num < max_num:
                st.session_state.num += 1
            
            file_name = file_name_list[st.session_state.num]
            st.write(st.session_state.num, file_name)
            draw_bbox(os.path.join(img_path, file_name), df['PredictionString'][st.session_state.num], conf_thr_s, conf_thr_e)

if __name__ == '__main__':
    main()