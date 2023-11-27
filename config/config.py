import ml_collections
import torch
from torch import nn


def get_config():
    config = ml_collections.ConfigDict()

    # dataset config
    config.csv_path = "E:\\UBC-Ovarian-Cancer\\train_thumbnails_augs"
    config.img_folder = "E:\\UBC-Ovarian-Cancer\\train_thumbnails"
    config.aug_imgs_folder = "E:\\UBC-Ovarian-Cancer\\train_thumbnails_augs"
    config.img_size = (224, 224)
    config.aug_imgs_num = 4
    config.split = 0.8
    # mode select:
    # use_origin_datasets, use_augs_datasets, use_fuse_datasets
    config.mode = "use_origin_datasets"

    # dataset transforms config
    config.use_augmixDataset = True
    config.use_randEraseing = True
    config.mean = []
    config.std = []

    # dataloader
    config.num_workers = 4
    config.pin_memory = True

    # mixup and cutmix config
    config.use_mixup = True
    config.mixup_alpha = 0.8
    config.cutmix_alpha = 0.1
    config.cutmix_minmax = 10
    config.prob = None
    config.switch_prob = None
    config.mode = None


    # model config
    config.label_smoothing = None
    config.num_classes = 5

    # lr_scheduler config
    config.use_cosine = False
    config.warmup_epoches = 0
    config.decay_epoches = 0
    config.mix_lr = 1e-5
    config.warmup_lr = 1e-5

    # training config
    config.model_name = "convnext_tiny"
    config.use_labelsmooth_loss = True
    config.train_epoches  = 50
    config.logger_name = None
    config.batch_size = 16
    config.lr = 1e-5
    config.weight_decay = 1e-4
    # select from:
    # adam, adamw,
    config.optimizer = "adam"
    config.momentum = 4e-3

    return config


if __name__ == "__main__":
    config = get_config()
    print(config)

