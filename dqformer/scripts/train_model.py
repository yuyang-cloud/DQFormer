import os
from os.path import join

import click
import torch
import yaml
from easydict import EasyDict as edict
from dqformer.datasets.semantic_dataset import SemanticDatasetModule
from dqformer.models.mask_model import MaskPS
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler


@click.command()
@click.option("--w", type=str, default=None, required=False)
@click.option("--ckpt", type=str, default=None, required=False)
@click.option("--nuscenes", is_flag=True)
@click.option("--train_for_test", is_flag=True) # semantic-kitti.yaml   train+8
def main(w, ckpt, nuscenes, train_for_test):
    model_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml")))
    )
    backbone_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml")))
    )
    decoder_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml")))
    )
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    if nuscenes:
        cfg.MODEL.DATASET = "NUSCENES"

    data = SemanticDatasetModule(cfg)
    model = MaskPS(cfg)
    if w:
        w = torch.load(w, map_location="cpu")
        model.load_state_dict(w["state_dict"])

    tb_logger = pl_loggers.TensorBoardLogger(
        "experiments/" + cfg.EXPERIMENT.ID, default_hp_metric=False
    )

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    if not train_for_test:
        iou_ckpt = ModelCheckpoint(
            monitor="metrics/iou",
            filename=cfg.EXPERIMENT.ID + "_epoch{epoch:02d}_iou{metrics/iou:.2f}",
            auto_insert_metric_name=False,
            mode="max",
            save_last=True,
        )
        pq_ckpt = ModelCheckpoint(
            monitor="metrics/pq",
            filename=cfg.EXPERIMENT.ID + "_epoch{epoch:02d}_pq{metrics/pq:.2f}",
            auto_insert_metric_name=False,
            mode="max",
            save_last=True,
        )
    elif train_for_test:
        inter_ckpt = ModelCheckpoint(
            filename=cfg.EXPERIMENT.ID + "_epoch{epoch:02d}",
            auto_insert_metric_name=False,
            every_n_epochs=1,  # 每隔 5 个 epoch 保存一次
            save_top_k=-1,     # 不覆盖
            save_last=True,
        )

    profiler = AdvancedProfiler(tb_logger.log_dir, 'profile')
    trainer = Trainer(
        gpus=cfg.TRAIN.N_GPUS,
        accelerator="ddp",
        logger=tb_logger,
        max_epochs=cfg.TRAIN.MAX_EPOCH,
        callbacks=[lr_monitor, inter_ckpt] if train_for_test else [lr_monitor, pq_ckpt, iou_ckpt],
        log_every_n_steps=1,
        gradient_clip_val=0.5,
        accumulate_grad_batches=cfg.TRAIN.BATCH_ACC,
        resume_from_checkpoint=ckpt,
        # num_sanity_val_steps=0,
        limit_val_batches=0 if train_for_test else 1.0,    # 关闭validation
        profiler=profiler,
    )

    trainer.fit(model, data)


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


if __name__ == "__main__":
    main()
