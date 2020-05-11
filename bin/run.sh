cd ../src
# python train.py -m 'resnet18' -c 'test'
# python train.py -m 'resnet34' -c 'test'
# python train.py -m 'resnet50' -c 'test'
# python train.py -m 'resnet101' -c 'test'
# python train.py -m 'resnet152' -c 'test'
# python train.py -m 'resnext50_32x4d' -c 'test'
# python train.py -m 'resnext101_32x8d' -c 'test'
# python train.py -m 'wide_resnet50_2' -c 'test'
# python train.py -m 'wide_resnet101_2' -c 'test'

# python train.py -m 'se_resnext50_32x4d' -c 'test'
# python train.py -m 'se_resnext101_32x4d' -c 'test'

# python train.py -m 'densenet121' -c 'test'

# python train.py -m 'mobilenet' -c 'test'

# python train.py -m 'efficientnet_b0' -c 'test'
# python train.py -m 'efficientnet_b1' -c 'test'
# python train.py -m 'efficientnet_b2' -c 'test'
# python train.py -m 'efficientnet_b3' -c 'test'
# python train.py -m 'efficientnet_b4' -c 'test'
# python train.py -m 'efficientnet_b5' -c 'test'
# python train.py -m 'efficientnet_b6' -c 'test'
# python train.py -m 'efficientnet_b7' -c 'test'

# python train.py -m 'arcface_resnet50' -c 'test'

# python train.py -m 'resnest50' -c 'test'
# python train.py -m 'resnest101' -c 'test'
# python train.py -m 'resnest200' -c 'test'
# python train.py -m 'resnest269' -c 'test'

python train.py -m 'ghostnet' -c 'test'

cd ../
git add -A
git commit -m '...'
git push origin master