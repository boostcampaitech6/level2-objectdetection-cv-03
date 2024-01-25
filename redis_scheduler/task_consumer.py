import redis
import torch
import os, sys, json, time, traceback, argparse, logging, logging.handlers

work_dir = os.getcwd()
file_list = os.listdir(work_dir)
for f in file_list:
    import_path = os.path.join(work_dir, f)
    if os.path.isdir(import_path) and f not in ['redis_model_scheduler', '.git', '.github']:
        sys.path.append(import_path)

import train_faster_rcnn
import train_detectron2
# import train_mmdetection
import train_yolo


def is_queue_empty(redis_instance, queue_list):
    queue_length = 0
    for queue_name in queue_list:
        queue_length += redis_instance.llen(queue_name)
    return queue_length == 0

def set_logger(log_path):
    train_logger = logging.getLogger(log_path)
    train_logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] >> %(message)s')
    fileHandler = logging.handlers.TimedRotatingFileHandler(filename=log_path, encoding='utf-8')
    fileHandler.setFormatter(formatter)
    train_logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    train_logger.addHandler(streamHandler)
    return train_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ip", type=str, required=True
    )
    parser.add_argument(
        "--port", type=int, required=True
    )
    parser.add_argument(
        "--user", type=str, nargs='+', required=True
    )

    args = parser.parse_args()

    consumner_log_path = '/data/ephemeral/home/level2-objectdetection-cv-03/redis_model_scheduler/logs'
    if not os.path.exists(consumner_log_path):
        os.mkdir(consumner_log_path)

    redis_server = redis.Redis(host=args.ip, port=args.port, db=0)
    while True:
        if not is_queue_empty(redis_server, args.user): # queue에 message가 존재하면 진입
            element = redis_server.brpop(keys=args.user, timeout=None)
            config = json.loads(element[1].decode('utf-8'))

            logger = set_logger(f'{consumner_log_path}/{config["experiment_name"]}.log')
            logger.info(f'{config["experiment_name"]} Start')
            try:
                torch.cuda.empty_cache()
                if config['library'] == 'faster_rcnn':
                    train_faster_rcnn.main(config=config)
                elif config['library'] == 'detectron2':
                    train_detectron2.main(config=config)
                # elif config['library'] == 'mmdetection':
                #     train_mmdetection.main(config=config)
                elif config['library'] == 'yolov5':
                    train_yolo.yolo_main(config=config)
                else:
                    assert False, 'Invalid Library'

                logger.info(f'{config["experiment_name"]} Finish')
            except:
                logger.error(traceback.format_exc())
                torch.cuda.empty_cache()
        else:
            time.sleep(1)