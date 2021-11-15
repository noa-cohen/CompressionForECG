from typing import List, NamedTuple


class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and number of correct classifications.
    """

    loss: float
    rms: float
    prd: float
    prdn: float
    snr: float
    qs: float


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch.
    """
    losses: List[float]
    avg_rms: float
    avg_prd: float
    avg_prdn: float
    avg_snr: float
    avg_qs: float


class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """

    num_epochs: int
    train_loss: List[float]
    test_loss: List[float]