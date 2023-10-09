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

- Distributed computing

The pre-conditioner matrices support different structures that allow to reduce cost ([overview](TODO Insert link to example)).

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

 - [Basic example](TODO Insert link to example)
 - Examples for [supported features](TODO Insert link to gallery)
 - [Advanced example](TODO Insert link to example)
 - [Supported structures](TODO Insert link to example)

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
