import streamlit as st
import os
from PIL import Image

def show(dir_path, num):
    file_name = sorted(os.listdir(dir_path))[num]
    img_path = os.path.join(dir_path, file_name)

    st.write(st.session_state.num, file_name)
    st.image(Image.open(img_path))

def main():
    st.title('Recycle Detection Testset Prediction')

    # streamlit으로 확인해보고 싶은 이미지 폴더 경로 입력
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

if __name__ == '__main__':
    main()