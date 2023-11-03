"""SINGD with a model that uses in-place activations."""

from contextlib import nullcontext

from pytest import mark, raises
from torch import manual_seed, rand
from torch.nn import Conv2d, ReLU, Sequential

from singd.optim.optimizer import SINGD


@mark.parametrize("inplace", [True, False], ids=["inplace=True", "inplce=False"])
def test_bug_inplace_activations_not_supported(inplace: bool):
    """Test that SINGD does not support in in-place activations.

    See https://github.com/f-dangel/singd/issues/56.

    Args:
        inplace: Whether to use in-place activations.
    """
    manual_seed(0)

    X = rand(2, 1, 5, 5)
    model = Sequential(Conv2d(1, 1, 3), ReLU(inplace=inplace))
    SINGD(model)  # install hooks

    with raises(RuntimeError) if inplace else nullcontext():
        model(X)
