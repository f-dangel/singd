"""Per-iteration time and peak memory comparison."""

from argparse import ArgumentParser
from time import time
from typing import List, Tuple

from memory_profiler import memory_usage
from torch import Tensor, cuda, device, manual_seed, rand, randint
from torch.nn import Conv2d, CrossEntropyLoss, Linear, Module
from torch.optim import SGD, Optimizer
from torchvision.models import convnext_base, resnet18, vgg19

from singd.optim.optimizer import SINGD


def set_up_data(case: str, batch_size: int) -> Tuple[Tensor, Tensor]:
    """Create synthetic data for the training case.

    Args:
        case: The training case to set up data for.
        batch_size: The batch size to use.

    Returns:
        A tuple of synthetic data X and labels y.
    """
    feature_shape = {
        "vgg19_cifar100": (3, 32, 32),
        "resnet18_cifar100": (3, 32, 32),
        "convnext_base_imagenet": (3, 256, 256),
    }[case]
    num_classes = {
        "vgg19_cifar100": 100,
        "resnet18_cifar100": 100,
        "convnext_base_imagenet": 1000,
    }[case]

    X = rand(batch_size, *feature_shape)
    y = randint(0, num_classes, (batch_size,))

    return X, y


def set_up_model(case: str) -> Module:
    """Set up the model for the training case.

    Args:
        case: The training case to set up the model for.

    Returns:
        The model for the training case.
    """
    model_fn = {
        "resnet18_cifar100": resnet18,
        "vgg19_cifar100": vgg19,
        "convnext_base_imagenet": convnext_base,
    }[case]
    num_classes = {
        "vgg19_cifar100": 100,
        "resnet18_cifar100": 100,
        "convnext_base_imagenet": 1000,
    }[case]
    return model_fn(num_classes=num_classes)


def set_up_optimizers(optimizer: str, model: Module) -> List[Optimizer]:
    """Set up the optimizers for the training case.

    Args:
        optimizer: The optimizer to use.
        model: The model to train.

    Returns:
        The optimizers for the training case.
    """
    sgd_lr = 0.01
    if optimizer == "sgd":
        params = [p for p in model.parameters() if p.requires_grad]
        optimizers = [SGD(params, lr=sgd_lr)]
    elif optimizer in {"singd", "singd+tn"}:
        # train supported layers with SINGD, others with SGD
        supported_params = [
            p
            for m in model.modules()
            if isinstance(m, (Linear, Conv2d))
            for p in m.parameters()
            if p.requires_grad
        ]
        optimizers = [
            SINGD(
                model,
                params=supported_params,
                T=1,  # update pre-conditioners every iteration
                structures=("diagonal", "diagonal"),  # use diagonal pre-conditioners
                kfac_approx="reduce",  # use KFAC-redue
                normalize_lr_cov=True,  # improves stability
            )
        ]

        supported_ids = [p.data_ptr() for p in supported_params]
        if unsupported_params := [
            p
            for p in model.parameters()
            if p.data_ptr() not in supported_ids and p.requires_grad
        ]:
            optimizers.append(SGD(unsupported_params, lr=sgd_lr))

        if optimizer == "singd":
            raise NotImplementedError
    else:
        raise NotImplementedError

    return optimizers


def maybe_synchronize(dev: device):
    """Synchronize the device if it is a CUDA device.

    Args:
        dev: The device to synchronize.
    """
    if "cuda" in str(dev):
        cuda.synchronize()


if __name__ == "__main__":
    parser = ArgumentParser("Parse parameters for training with SINGD")

    SUPPORTED_CASES = ["resnet18_cifar100", "convnext_base_imagenet", "vgg19_cifar100"]
    SUPPORTED_OPTIMIZERS = ["sgd", "singd", "singd+tn"]
    SUPPORTED_DEVICES = ["cuda", "cpu"]
    SUPPORTED_METRICS = ["time", "peakmem"]

    parser.add_argument(
        "--case",
        type=str,
        choices=SUPPORTED_CASES,
        help="Training scenario",
        required=True,
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=SUPPORTED_OPTIMIZERS,
        help="Optimizer to use",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=SUPPORTED_DEVICES,
        help="Device to use",
        required=True,
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=SUPPORTED_METRICS,
        help="Metric to measure",
        required=True,
    )
    parser.add_argument("--batch_size", type=int, help="Batch size", default=128)
    parser.add_argument("--seed", type=int, help="Random seed", default=0)

    args = parser.parse_args()

    manual_seed(args.seed)  # make deterministic
    DEV = device(args.device)

    X, y = set_up_data(args.case, args.batch_size)
    X, y = X.to(DEV), y.to(DEV)

    model = set_up_model(args.case)
    model = model.to(DEV)
    loss_func = CrossEntropyLoss().to(DEV)

    optimizers = set_up_optimizers(args.optimizer, model)
    maybe_synchronize(DEV)

    def step():
        """Perform a training step."""
        for optimizer in optimizers:
            optimizer.zero_grad()

        loss = loss_func(model(X), y)
        loss.backward()

        for optimizer in optimizers:
            optimizer.step()

    # warm-up
    step()

    num_steps = {"time": 10, "peakmem": 3}[args.metric]

    def f():
        """Run the training loop."""
        for _ in range(num_steps):
            step()

    description = (
        f"[{args.case}, {args.optimizer}, batch_size={args.batch_size}, "
        + f"device={args.device}, seed={args.seed}]"
    )

    if args.metric == "time":
        t_start = time()
        f()
        maybe_synchronize(DEV)
        t_end = time()
        print(f"{description} Time taken: {(t_end - t_start)/ num_steps:.2e} s / iter")

    elif args.metric == "peakmem":
        if "cuda" in str(DEV):
            f()
            peakmem_bytes = cuda.max_memory_allocated() / 2**10
        else:
            peakmem_bytes = memory_usage(f, interval=1e-4, max_usage=True)

        print(f"{description} Memory usage: {peakmem_bytes / 2**10 :.2e} GiB")

    else:
        raise NotImplementedError
