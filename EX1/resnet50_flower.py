# ex1_config.py
_base_ = 'mmpretrain::resnet/resnet50_8xb32_in1k.py'

# 定义模型相关参数
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        style='pytorch',
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    )
)

# 设定数据集的根目录
data_root = './data/flower_dataset'

# 配置训练数据的加载方式
train_dataloader = dict(
    dataset=dict(
        type='ImageNet',
        data_root=data_root,  # 明确指定花卉数据集的位置
        ann_file='train.txt',
        data_prefix='train',
    ),
    batch_size=32,
    num_workers=4,
)

# 配置验证数据的加载方式
val_dataloader = dict(
    dataset=dict(
        type='ImageNet',
        data_root=data_root,  # 明确指定花卉数据集的位置
        ann_file='val.txt',
        data_prefix='val',
    ),
    batch_size=32,
    num_workers=4,
)

# 设定训练的相关参数
train_cfg = dict(
    max_epochs=15,
)

# 配置优化器的相关参数
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.0008,
        momentum=0.92,
        weight_decay=0.00015,
    )
)

# 指定预训练模型权重的文件路径
load_from = './checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth'

# 调整数据相关配置，确保指向花卉数据集
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=train_dataloader['dataset'],
    val=val_dataloader['dataset'],
)