import streamlit as st
import redis
import json, time
import pandas as pd

# nohup streamlit run redis_model_scheduler/redis_stremlit.py --server.port "port" > redis_model_scheduler/logs/redis_streamlit.log

tab1, tab2= st.tabs(['Redis Dashboard', 'Training Tab'])

with tab1:
    st.write('## Redis Queue List')
    
    redis_info_dict = dict()
    redis_server = redis.Redis(host='10.28.224.74', port=30158, db=0)
    key_list = redis_server.keys('*')
    key_list = [key.decode() for key in key_list]
    if len(key_list) > 0:
        for key in key_list:
            st.write(f'### {key}')
            tasks = redis_server.lrange(key, 0, -1)
            if len(tasks) > 0:
                redis_info_dict[key] = []
                df = pd.DataFrame()
                for i, task in enumerate(tasks):
                    redis_info_dict[key].append(task)
                    with st.container(border=True):
                        task_json = json.loads(task.decode('utf-8'))
                        task_dict = dict()
                        for k, v in task_json.items():
                            if k in ['ip', 'port']: continue
                            task_dict[k] = [str(v)]
                        df = df.append(pd.DataFrame(task_dict), ignore_index=True)
                st.table(df)
    else:
        st.caption('no data in redis queue')
    st.button('refresh', type='primary')

    st.write('## Redis Queue Delete')
    select_key = st.selectbox('select key', key_list)
    input_index = st.text_input(label='index', value='')
    is_clicked = st.button('delete', type='primary')
    
    if is_clicked:
        if input_index.isnumeric() and select_key in redis_info_dict and -1 < int(input_index) < len(redis_info_dict[select_key]):
            redis_server.lrem(select_key, 0, redis_info_dict[select_key][int(input_index)])
            st.success('delete complete..')
            
        else:
            st.error('delete fail..')

with tab2:
    st.write('## Training Setting')
    input_yolo_yaml_name, input_yolo_model_name, input_iteration = None, None, None
    select_redis_key = st.text_input('user(redis key)')
    select_library = st.selectbox('library', ['yolov5', 'detectron2', 'faster_rcnn'])
    if select_library == 'yolov5':
        input_yolo_yaml_name = st.text_input('yolo yaml name(ex : recycle_detection_56_0.yaml)')
        input_yolo_model_name = st.selectbox('yolo model name', ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'])
    elif select_library == 'detectron2':
        input_iteration = int(st.text_input('detectron iteration', 15000))

    input_seed = int(st.text_input('seed', 42))
    input_fold = int(st.selectbox('fold', [1, 2, 3, 4, 5]))
    input_epochs = int(st.text_input('epochs', 50))
    input_batch_size = int(st.text_input('batch size', 16))
    input_image_size = int(st.text_input('image size', 640))
    input_optimizer = st.selectbox('optimizer', ['SGD', 'Adam'])
    input_lr = float(st.text_input('learning rate', 1e-3))

    expriment_name = f"{select_redis_key}_{select_library}_{input_seed}_{input_fold}_ep{input_epochs}_bs{input_batch_size}_imgsz{input_image_size}_{input_optimizer}"
    if select_library == 'yolov5':
        expriment_name = f"{select_redis_key}_{input_yolo_model_name}_{input_seed}_{input_fold}_ep{input_epochs}_bs{input_batch_size}_imgsz{input_image_size}_{input_optimizer}"
    elif select_library == 'detectron2':
        expriment_name = f"{select_redis_key}_{select_library}_{input_seed}_{input_fold}_iter{input_iteration}_bs{input_batch_size}_imgsz{input_image_size}_{input_optimizer}"
    input_experiment_name = st.text_input(f'experiment name', expriment_name)
    

    is_pushed = st.button('push', type='primary')
    if is_pushed:
        config = {
            "user" : select_redis_key,
            "library" :  select_library,
            'yolo_yaml_name': input_yolo_yaml_name,
            'yolo_model_name': input_yolo_model_name,
            "epochs" : input_epochs,
            "iteration" : input_iteration,
            "batch_size" : input_batch_size,
            "image_size" : input_image_size,
            "optimizer" : input_optimizer,
            "lr" : input_lr,
            "experiment_name": input_experiment_name
        }
        redis_server.lpush(config["user"], json.dumps(config))
        st.success('push complete..')
