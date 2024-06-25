for optim in sgd singd+tn singd; do
    for case in vgg19_cifar100 resnet18_cifar100 convnext_base_imagenet; do
            for dev in cuda; do # cpu; do
                for metric in time peakmem; do
                    for seed in `seq 0 9 1`; do
                        python profile_training.py \
                               --optimizer=$optim \
                               --case=$case \
                               --device=$dev \
                               --metric=$metric \
                               --seed=$seed
                    done
                done
            done
        done
    done
