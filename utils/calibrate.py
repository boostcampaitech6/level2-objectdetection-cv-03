import pandas as pd
from tqdm import tqdm

def main():
    # calibration 적용할 csv 파일 경로
    df = pd.read_csv('/data/ephemeral/home/level2-objectdetection-cv-03/submission.csv')

    df_score_list = []
    for pred_str in tqdm(df['PredictionString']):
        df_score_list += list(map(float, str(pred_str).split(' ')[1::6]))
    old_min = min(df_score_list)
    old_max = max(df_score_list)

    new_min = 0.05
    new_max = 1.0

    pred_str_list = []
    for pred_str in tqdm(df['PredictionString']):
        pred = str(pred_str).split(' ')
        pred[1::6] = [str((new_max - new_min) * (float(sc) - old_min) / (old_max - old_min) + new_min) for sc in pred[1::6]]
        pred_str_list.append(' '.join(pred))

    df['PredictionString'] = pred_str_list
    
    # 저장할 경로
    df.to_csv('/data/ephemeral/home/level2-objectdetection-cv-03/cal_submission.csv', index=False)

if __name__ == '__main__':
    main()