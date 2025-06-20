import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os

input_dir = './dataset/original'
output_dir = './dataset/augmented'
os.makedirs(output_dir, exist_ok=True)

# 증강 파이프라인 정의
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    A.HueSaturationValue(p=0.5),
])

num_aug_per_image = 5

for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(input_dir, filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 원본 이미지 저장
        cv2.imwrite(os.path.join(output_dir, filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # 증강 이미지 저장
        for i in range(num_aug_per_image):
            augmented = transform(image=image)
            augmented_img = augmented['image']
            save_path = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_aug{i}.jpg')
            cv2.imwrite(save_path, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))

print("Albumentations 증강 완료")
