import shutil
import os


def copy_images_from_file(fold_txt_file_path, destination_folder):
    with open(fold_txt_file_path, 'r') as file:
        image_paths = [line.strip() for line in file.readlines()]

    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        destination_path = os.path.join(destination_folder, image_name)

        shutil.copyfile(image_path, destination_path)

def main():
    fold_txt_file_path = '/data/ephemeral/home/dataset/yolo/split/val_42_fold_1.txt'  # 이미지 폴더로 저장하고 싶은 fold.txt 파일 경로
    destination_folder = '/data/ephemeral/home/dataset/valid_split_1'  # fold 이미지를 저장할 폴더 경로, 폴더가 없으면 자동으로 생성됨
    os.makedirs(destination_folder, exist_ok=True)

    copy_images_from_file(fold_txt_file_path, destination_folder)

if __name__ == "__main__":
    main()
