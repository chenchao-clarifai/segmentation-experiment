import argparse
import logging
import os
import time

import torch
import torchvision

import datasets
import models


def get_logits(embeddings):
    out = projector(embeddings) + bias()
    return out / temp()


def mini_train_step(mini_batch):

    img = mini_batch["image"].to(DEVICE)
    msk = mini_batch["mask"].to(DEVICE) % 255

    msk = msk - 1  # 0: unlabeled -> -1: ignore

    x = backbone(img)["out"]
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

    epoch_frac = 0
    edx = 0

    try:
        for edx in range(epochs):
            logging.info(f"epoch {edx}/{epochs} starts")
            for bdx, batch in enumerate(train_loader):
                loss, acc = train_step(batch)
                epoch_frac = int(100 * bdx / len(train_loader))
                acc *= 100
                thermo = temp().item()
                logging.info(
                    f"epoch{edx}: {epoch_frac}%, loss={loss:3.2f}, acc={acc:3.2f}%, temp={thermo:3.2f}"  # noqa
                )
            scheduler.step()
    except Exception as e:
        logging.exception(e)
    except KeyboardInterrupt:
        logging.info(f"Terminated manually at {edx + epoch_frac / 100}.")
    finally:
        checkpoint = {
            "backbone": backbone.state_dict(),
            "projector": projector.state_dict(),
            "bias": bias.state_dict(),
            "temperature": temp.state_dict(),
        }
        epoch = edx + epoch_frac / 100
        time_string = (
            time.asctime().replace("  ", " ").replace(" ", "_").replace(":", ".")
        )
        checkpoint_path = f"checkpoints/{time_string}-Epoch{epoch:3.2f}.ckpt"
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a deeplabv3 model.")
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["mobilenetv3", "resnet50", "resnet101"],
        default="resnet50",
    )
    parser.add_argument("--device", type=int)
    parser.add_argument("--minibatch", type=int, default=5)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--thermo_lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--from_checkpoint", type=str, default=None)
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

    DEVICE_BATCH_SIZE = max(1, args.minibatch)
    GRAD_ACCU_STEPS = max(1, args.accumulate)
    BATCH_SIZE = DEVICE_BATCH_SIZE * GRAD_ACCU_STEPS
    logging.info(
        f"DEVICE_BATCH_SIZE={DEVICE_BATCH_SIZE}, GRAD_ACCU_STEPS={GRAD_ACCU_STEPS}"
    )

    LR = args.lr
    WD = args.wd
    THERMO_LR = args.thermo_lr
    GAMMA = args.gamma
    logging.info(f"LR={LR}, WD={WD}, THERMO_LR={THERMO_LR}, GAMMA={GAMMA}")

    DEVICE = "cpu" if args.device == -1 else f"cuda:{args.device}"
    BACKBONE = args.backbone
    logging.info(f"DEVICE={DEVICE}, BACKBONE={BACKBONE}")

    backbone_factory = {
        "mobilenetv3": torchvision.models.segmentation.deeplabv3_mobilenet_v3_large,
        "resnet50": torchvision.models.segmentation.deeplabv3_resnet50,
        "resnet101": torchvision.models.segmentation.deeplabv3_resnet101,
    }

    logging.info("loading datasets")
    train_cfg = datasets.transforms.TrainTransformsConfigs()
    eval_cfg = datasets.transforms.EvalTransformsConfigs()

    PATH_TO_TRAIN_DATA = os.path.join(args.dataset_root, "train2017")
    PATH_TO_TEST_DATA = os.path.join(args.dataset_root, "val2017")

    train_transforms = datasets.transforms.get_train_transforms(train_cfg)
    eval_transforms = datasets.transforms.get_eval_transforms(eval_cfg)

    train_data = datasets.SemanticSegmentationDataset(
        PATH_TO_TRAIN_DATA, transforms=train_transforms
    )
    eval_data = datasets.SemanticSegmentationDataset(
        PATH_TO_TEST_DATA, transforms=eval_transforms
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    logging.info("dataset loaded")

    logging.info("loading models")
    backbone = backbone_factory[BACKBONE](num_classes=DIM, pretrained_backbone=True)
    backbone.to(DEVICE)
    projector = models.heads.Projector(in_dim=DIM, out_dim=NUM_CLASSES)
    projector.to(DEVICE)
    bias = models.heads.Bias(dims=(1, NUM_CLASSES, 1, 1))
    bias.to(DEVICE)
    temp = models.heads.PositiveReal()
    temp.to(DEVICE)

    if args.from_checkpoint:
        checkpoint = torch.load(args.from_checkpoint)
        backbone.load_state_dict(checkpoint["backbone"])
        projector.load_state_dict(checkpoint["projector"])
        bias.load_state_dict(checkpoint["bias"])
        temp.load_state_dict(checkpoint["temperature"])

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

    train_model(args.epochs)

    logging.info("Training was finished.")
