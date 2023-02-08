import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, dev = False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        if dev == True:
          self.images = self.images[0:10]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


if __name__ == "__main__":

  # після цього все непотрібне, просто нотатки...
  TRAIN_IMG_DIR = "data/train_images/"
  TRAIN_MASK_DIR = "data/train_masks/"
  import albumentations as A
  from albumentations.pytorch import ToTensorV2
  IMAGE_HEIGHT = 160
  IMAGE_WIDTH = 240
  transform= A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
  ds = CarvanaDataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir= TRAIN_MASK_DIR,
        transform=transform,
    )
  print(len(ds))

  for tup in ds:    
    image, mask = tup
    print(np.shape(image))
    print(mask.shape)
    break

  image, mask = ds[10]
  print(image.shape, mask.shape)

  print("loader:")
  from torch.utils.data import DataLoader
  BATCH_SIZE = 16
  NUM_WORKERS = 2

  dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=True)

  print(len(dl))

  for batch in dl:
    inputs, targets = batch
  
    print(inputs.shape, targets.shape)
    break





    