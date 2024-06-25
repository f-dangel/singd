for optim in singd+tn; do #sgd singd+tn; do
    for case in vgg19_cifar100; do # resnet18_cifar100 convnext_base_imagenet; do
            for dev in cpu; do # cuda; do
                for metric in time peakmem; do
                    for seed in `seq 0 1 10`; do
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
