import argparse
import logging
import time

import torch
import torchvision

from ..src import datasets, models


def get_logits(embeddings):
    out = projector(embeddings) + bias()
    return out / temp()


def mini_train_step(mini_batch):

    img = mini_batch["image"].to(DEVICE)
    msk = mini_batch["mask"].to(DEVICE)

    msk = msk - 1  # 0: unlabeled -> -1: ignore

    x = backbone(img)
    logits = get_logits(x)

    loss = models.heads.F.cross_entropy(logits, msk, ignore_index=-1)
    acc = (logits.detach().max(dim=1).indices == msk).float().mean().cpu()

    return loss, acc


def train_step(batch):

    this_batch_size = len(batch["image"])
    inds, ws = datasets.utils.shard_batch_indices(
        this_batch_size, DEVICE_BATCH_SIZE, GRAD_ACCU_STEPS
    )

    optimizer.zero_grad()

    acc = 0.0
    loss = 0.0
    for ind, w in zip(inds, ws):
        mini_loss, mini_acc = mini_train_step(
            dict(image=batch["image"][ind], mask=batch["mask"][ind])
        )
        acc += mini_acc.item() / len(ws)
        loss += mini_loss.item() / len(ws)
        mini_loss = w * mini_loss
        mini_loss.backward()

    optimizer.step()

    return loss, acc


def train_model(epochs):

    for edx in range(epochs):
        logging.info(f"epoch {edx}/{epochs} starts")
        for bdx, batch in enumerate(train_loader):
            loss, acc = train_step(batch)
            epoch_frac = int(100 * bdx / len(train_loader))
            acc *= 100
            logging.info(
                f"epoch{edx}: {epoch_frac}%, loss={loss:3.2f}, acc={acc:3.2f}%"
            )
        scheduler.step()

    checkpoint = {
        "backbone": backbone.state_dict(),
        "projector": projector.state_dict(),
        "bias": bias.state_dict(),
        "temperature": temp.state_dict(),
    }
    epoch = min(edx + 1, epochs)
    time_string = time.asctime().replace("  ", " ").replace(" ", "_").replace(":", ".")
    checkpoint_path = f"checkpoints/{time_string}-Epoch{epoch}.ckpt"
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a deeplabv3 model.")
    parser.add_argument("epochs", type=int)
    args = parser.parse_args()

    logging.basicConfig(
        filename="logs/train.log",
        filemode="w",
        format="%(asctime)s | %(message)s",
        level=logging.INFO,
    )

    # constants
    DIM = 128
    NUM_CLASSES = 182  # 0 is unlabeled will be ignored
    logging.info(f"DIM={DIM}, NUM_CLASSES={NUM_CLASSES}")

    DEVICE_BATCH_SIZE = 5
    GRAD_ACCU_STEPS = 12
    BATCH_SIZE = DEVICE_BATCH_SIZE * GRAD_ACCU_STEPS
    logging.info(
        f"DEVICE_BATCH_SIZE={DEVICE_BATCH_SIZE}, GRAD_ACCU_STEPS={GRAD_ACCU_STEPS}"
    )

    LR = 1e-3
    WD = 1e-5
    THERMO_LR = 1e-6 * LR
    GAMMA = 0.5
    logging.info(f"LR={LR}, WD={WD}, THERMO_LR={THERMO_LR}, GAMMA={GAMMA}")

    DEVICE = "cpu"
    BACKBONE = "deeplabv3_mobilenet_v3_large"
    logging.info(f"DEVICE={DEVICE}, BACKBONE={BACKBONE}")

    logging.info("loading datasets")
    train_cfg = datasets.transforms.TrainTransformsConfigs()
    eval_cfg = datasets.transforms.EvalTransformsConfigs()

    PATH_TO_TRAIN_DATA = "/home/scratch/2/cocostuff/train2017/"
    PATH_TO_TEST_DATA = "/home/scratch/2/cocostuff/val2017/"

    train_transforms = datasets.transforms.get_train_transforms(train_cfg)
    eval_transforms = datasets.transforms.get_eval_transforms(eval_cfg)

    train_data = datasets.SemanticSegmentationDataset(
        PATH_TO_TRAIN_DATA, transforms=train_transforms
    )
    eval_data = datasets.SemanticSegmentationDataset(
        PATH_TO_TEST_DATA, transforms=eval_transforms
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=BATCH_SIZE, shuffle=True
    )
    logging.info("dataset loaded")

    logging.info("loading models")
    backbone = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
        num_classes=DIM, pretrained_backbone=True
    )
    backbone.to(DEVICE)
    projector = models.heads.Projector(in_dim=DIM, out_dim=NUM_CLASSES)
    projector.to(DEVICE)
    bias = models.heads.Bias(dims=(1, DIM, 1, 1))
    bias.to(DEVICE)
    temp = models.heads.PositiveReal()
    temp.to(DEVICE)
    logging.info("models loaded")

    optimizer = torch.optim.AdamW(
        [
            dict(params=backbone.parameters()),
            dict(params=projector.parameters()),
            dict(params=bias.parameters()),
            dict(params=temp.parameters(), lr=THERMO_LR),
        ],
        lr=LR,
        weight_decay=WD,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, GAMMA, verbose=True)

    train_model(args.epoch)

    logging.info("Training was finished.")
