for case in resnet18_imagenet vgg19_imagenet; do # inception_v3_imagenet mobilenet_v2_imagenet resnext101_32x8d_imagenet convnext_base_imagenet; do
        for optim in sgd singd singd+tn; do
            for dev in cuda; do # cpu; do
                for metric in time peakmem; do #time peakmem; do
                    for seed in `seq 0 1 4`; do
                        python profile_training.py \
                               --optimizer=$optim \
                               --case=$case \
                               --device=$dev \
                               --metric=$metric \
                               --seed=$seed \
                               --batch_size=128
                    done
                    echo ""
                done
            done
        done
    done
