import torch
import torchvision
import os
from dataset import CarvanaDataset
from torch.utils.data import DataLoader


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("Saving checkpoint...")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("Loading checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers,
        pin_memory=True,
        dev = False,
):

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
        dev=dev,
    )    

    if dev == True:
      train_ds = val_ds
    else:
      train_ds = CarvanaDataset(
          image_dir=train_dir,
          mask_dir=train_maskdir,
          transform=train_transform,
          dev=dev,
      )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            result = model(x)
            preds = torch.sigmoid(result)
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (preds + y).sum() + 1e-8

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
        idx, epoch, folder, preds, y
):
    y = y.unsqueeze(1) # BCHW 16x1x100x100
    preds = preds.to("cpu")
    diff_mask = torch.abs(y - preds)
       
    all_masks = torch.cat((y, preds, diff_mask), 0) #48x1x100x100

    torchvision.utils.save_image(
    all_masks, f"{folder}/epoch_{epoch}_batch_{idx}.png")


def apply_model_and_save_results(
        loader, model, epoch, folder="predictions/", device="cuda", num_save = 3
):
    model.eval()
    os.makedirs(name=folder, exist_ok=True)
    for idx, (x,y) in enumerate(loader):
        if idx < num_save:
            x = x.to(device=device)# 16x3x100x100
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float() #16x1x100x100
            save_predictions_as_imgs(idx, epoch, folder, preds, y)


    model.train()





