# SINGD: Structured Inverse-Free Natural Gradient Descent

This package contains the official PyTorch implementation of our
**memory-efficient and numerically stable KFAC** variant, termed SINGD
([paper](TODO Insert arXiv link)).

The main feature is a `torch.optim.Optimizer` which works like most PyTorch optimizers and is compatible with:

- [Per-parameter
  options](https://pytorch.org/docs/stable/optim.html#per-parameter-options)
  (`param_groups`)
- Using a learning rate
  [scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

- [Checkpointing](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training)

- [Gradient
  scaling](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler) &
  [mixed-precision training](https://pytorch.org/docs/stable/notes/amp_examples.html#typical-mixed-precision-training)

- [Gradient
  accumulation](https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation)
  (multiple forward-backwards, then take a step)

- [Distributed
  data-parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
  (DDP) training[^1]

The pre-conditioner matrices support different structures that allow to reduce
cost
([overview](https://singd.readthedocs.io/en/latest/generated/gallery/example_05_structures/)).

## Installation

- Stable (recommended):
  ```bash
  pip install singd
  ```

- Latest version from GitHub `main` branch:
  ```bash
  pip install git+https://github.com/f-dangel/singd.git@main
  ```

## Usage

 - [Basic
   example](https://singd.readthedocs.io/en/latest/generated/gallery/example_01_basic/)
 - Examples for [supported
   features](https://singd.readthedocs.io/en/latest/generated/gallery/)
 - [Advanced
   example](https://singd.readthedocs.io/en/latest/generated/gallery/example_04_advanced/)
 - [Supported
   structures](https://singd.readthedocs.io/en/latest/generated/gallery/example_05_structures/)

## Limitations

- `SINGD` does not support graph neural networks (GNN)

- The code has stabilized only recently. Expect things to break and help us
  improve by filing issues.

## Citation

If you find this code useful for your research, consider citing the paper:

```bib

@article{lin2023structured,
  title =        {Structured Inverse-Free Natural Gradient: Memory-Efficient &
  Numerically-Stable KFAC for Large Neural Nets},
  author =       {Lin, Wu and Dangel, Felix and Eschenhagen, Runa and Neklyudov,
  Kirill and Kristiadi, Agustinus and Turner, Richard E and Makhzani, Alireza},
  year =         2023,
}

```

[^1]: We do support standard DDP with one crucial difference: The model should
    not be wrapped with the DDP wrapper, but the rest, e.g. using the `torchrun`
    command stays the same.
