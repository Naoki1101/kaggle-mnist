#!/bin/bash

cd ../
# python -m experiments.resnet18 -c "test"
# python -m experiments.resnet50 -c "test" --debug
# python -m experiments.resnext50_32x4d -c "test" --debug
# python -m experiments.seresnext50_32x4d -c "test" --debug
# python -m experiments.efficientnet_b0 -c "test" --debug
# python -m experiments.tf_efficientnetv2_b0 -c "test" --debug
# python -m experiments.resnest50d -c "test" --debug
# python -m experiments.vit_small_patch16_224 -c "test" --debug
# python -m experiments.swin_base_patch4_window7_224 -c "test" --debug
python -m experiments.nfnet_f1 -c "test" --debug