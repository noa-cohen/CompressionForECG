import os
import abc
import sys
import h5py
import tqdm
import torch.nn as nn
from typing import Any, Callable
import matplotlib
from torch.nn.modules.loss import _Loss  # for implementing loss with regularization
import torch.nn.functional as F

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np
import json

from train_by_hdf5 import gen_dataset_from_hdf5, ECGdataset_w_disease_hdf5, create_data_loaders_hdf5
from dataloader import get_window, create_train_validation_loader_for_one_patient
from training_result import FitResult, BatchResult, EpochResult
from model import *
from getpass import getuser
from evaluation import *
import wandb


def convert_results_to_dic(results):
    dic = {
    "avg_rms": results.avg_rms,
    "avg_prd": results.avg_prd,
    "avg_prdn": results.avg_prdn,
    "avg_snr:": results.avg_snr,
    "avg_qs" : results.avg_qs
    }
    return dic


def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def plot_ecg_pred(orig, reconstructed, name):
    plt.plot(orig, "-b", label="orig")
    plt.plot(reconstructed, "-r", label="reconstructed")
    plt.legend(loc="upper left")
    # plt.show()
    plt.savefig(os.path.join('training_figs', f'{name}.png'))
    plt.close()


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, use_gpu=True):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        use_gpu = use_gpu and torch.cuda.is_available()
        if use_gpu:
            current_device = torch.cuda.current_device()
            # torch.cuda.synchronize  # DEBUG
            print("Device: ", torch.cuda.get_device_name(current_device))
        else:
            print("Device: CPU")
        self.device = torch.device('cuda' if use_gpu else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # torch.cuda.synchronize  # DEBUG
        model.to(self.device)
        # self.model = torch.nn.DataParallel(model).cuda()


    def fit(
            self,
            dl_train: DataLoader,
            dl_test: DataLoader,
            num_epochs,
            checkpoints: str = None,
            early_stopping: int = None,
            print_every=1,
            post_epoch_fn=None,
            train_pts=None,
            valid_pts=None,
            ewi: int = 0,
            initial_epoch: int = 0,
            initial_loss: int = None,
            initial_rms: int = 1000,
            initial_test_results = None,
            seed: int = None,
            data_percents = None,
            act_epoch: int = 0,
            initialize_weights: bool = False,
            **kw,
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set loss improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = act_epoch
        train_loss, test_loss = [], []
        train_avg_loss, test_avg_loss = [], []
        avg_epoch_rms_train, avg_epoch_rms_validation = [], []
        avg_epoch_snr_train, avg_epoch_snr_validation = [], []
        avg_epoch_prd_train, avg_epoch_prd_validation = [], []
        epochs, new_seed = [], []

        print(f'starting train with seed = {torch.initial_seed()}')

        best_loss = initial_loss
        epochs_without_improvement = ewi
        best_rms = initial_rms
        best_test_results = initial_test_results
        checkpoint_filename = None
        if checkpoints is not None:
            checkpoint_filename = f"{checkpoints}.pt"

        for epoch in range(num_epochs):
            save_checkpoint = False
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f"--- EPOCH {initial_epoch + epoch + 1}/{initial_epoch + num_epochs} ---", verbose)

            #  Train & evaluate for one epoch
            #  - Use the train/test_epoch methods.
            #  - Save losses and accuracies in the lists above.
            #  - Implement early stopping. This is a very useful and
            #    simple regularization technique that is highly recommended.
            # ========================
            train_result = self.train_epoch(dl_train, epoch, **kw)  #train EPOCH RESULTS
            test_result = self.test_epoch(dl_test, epoch, **kw) #validation EPOCH RESULTS
            # train_result.losses list loss of every batch in current epoch
            # train_loss list of every loss of every batch in every epoch
            train_loss.extend(train_result.losses)
            test_loss.extend(test_result.losses)

            #  train_epoch_avg_loss average of all batch losses in epoch
            train_epoch_avg_loss = np.average(train_result.losses)
            validation_epoch_avg_loss = np.average(test_result.losses)
            # train_avg_loss list of averge loss of every epoch
            train_avg_loss.append(train_epoch_avg_loss)
            test_avg_loss.append(validation_epoch_avg_loss)

            # print('------------------DEBUG----------------------------')
            # print(f'list  size = {len(train_result.losses)} train_result.losses = {train_result.losses}')
            # print(f'avg loss epoch train_epoch_avg_loss = {train_epoch_avg_loss}')
            # print(f'list size = {len(train_avg_loss)}   train_avg_loss = {train_avg_loss}')
            # print(f'train_result.avg_rms = {train_result.avg_rms}')
            # print('----------------------------------------------------')

            if epoch > 1:
                if round(test_loss[-1], 4) >= round(test_loss[-2], 4):
                #if round(avg_epoch_rms_validation[-1], 2) >= round(avg_epoch_rms_validation[-2], 2):
                    epochs_without_improvement += 1
                else:
                    epochs_without_improvement = 0
            # Beginning on checking average loss improvement
            if checkpoints:
                if epoch > 250:
                    save_checkpoint = True
                elif test_result.avg_rms < best_rms:
                    print("new best avg rms")
                    save_checkpoint = True
                    best_rms = test_result.avg_rms
                    best_test_results = test_result

                if save_checkpoint:
                    # plot window from train
                    dl_iter_train = iter(dl_train)
                    x_train, idx, rlab = next(dl_iter_train)
                    x_train = x_train.to(self.device)
                    idx = idx[0].item()
                    rlab = [lab for lab in rlab[0]]
                    plot_orig_and_reconstruct(x_train, self.model, initial_epoch + epoch + 1, "train", "train_normal_temp_h", idx, rlab)
                    # plot window from test
                    dl_iter_test = iter(dl_test)
                    x_test, idx, rlab = next(dl_iter_test)
                    x_test = x_test.to(self.device)
                    idx = idx[0].item()
                    rlab = [lab for lab in rlab[0]]
                    plot_orig_and_reconstruct(x_test, self.model, initial_epoch + epoch + 1, "test", "train_normal_temp_h", idx, rlab)
                else:
                    print(f'no better result, best avg rms = {best_rms}')

            # print(f'DEBUG: {epochs_without_improvement} > {early_stopping}')
            if initialize_weights:
                if epoch % 20 == 0:
                    print(f' epoch {epoch} % 20 == 0 train_result.avg_rms ={train_result.avg_rms}')
                    if train_epoch_avg_loss > 5e-3:
                        seed = randomize_seed()
                        new_seed.append((initial_epoch + epoch + 1, seed))
                        print(f'****\n****initializing weights with seed = {seed}\n****')
                        lr = self.optimizer.param_groups[0]['lr']
                        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, )  # initialize optimizer
                        model.apply(weight_reset)
                        actual_num_epochs = 0

            # ========================
            # Save model checkpoint if requested
            if save_checkpoint and checkpoint_filename is not None or actual_num_epochs == 1:
                if seed is None:
                    print("Saved checkpoint with seed = None")
                saved_state = dict(
                    best_loss=best_loss,
                    ewi=epochs_without_improvement,
                    epoch=initial_epoch + epoch + 1,
                    actual_num_epochs=actual_num_epochs,
                    model_state=self.model.state_dict(),
                    optimizer_state=self.optimizer.state_dict(),
                    trained_patients=train_pts,
                    validation_patients=valid_pts,
                    seed=seed,
                    test_result=convert_results_to_dic(best_test_results),
                    data_percents=data_percents
                )

                split_filename = os.path.splitext(os.path.basename(checkpoint_filename))
                filename = f'{split_filename[0]}_e{initial_epoch + epoch + 1}{split_filename[1]}'
                torch.save(saved_state, os.path.join(os.path.dirname(checkpoint_filename), filename))
                print(
                    f"*** Saved checkpoint {filename} " f"at epoch {initial_epoch + epoch + 1}"
                )

            if post_epoch_fn:
                post_epoch_fn(initial_epoch + epoch, train_result, test_result, verbose)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            actual_num_epochs += 1
            avg_epoch_rms_train.append(train_result.avg_rms)
            avg_epoch_rms_validation.append(test_result.avg_rms)
            avg_epoch_snr_train.append(train_result.avg_snr)
            avg_epoch_snr_validation.append(test_result.avg_snr)
            avg_epoch_prd_train.append(train_result.avg_prd)
            avg_epoch_prd_validation.append(test_result.avg_prd)
            epochs.append(initial_epoch + epoch + 1)
            learning_curve_data = {'epochs': epochs, 'init_seed': new_seed,
                    'loss_train': train_avg_loss, 'loss_validation': test_avg_loss,
                    'rms_train': avg_epoch_rms_train, 'rms_validation': avg_epoch_rms_validation,
                    'snr_train': avg_epoch_snr_train, 'snr_validation': avg_epoch_snr_validation,
                    'prd_train': avg_epoch_prd_train, 'prd_validation': avg_epoch_prd_validation,
                    }
            with open('./figs/learning_curve_data.json', "w") as write_file:
                json.dump(learning_curve_data, write_file)

            wandb.log({'train_avg_rms': train_result.avg_rms, 'validation_avg_rms': test_result.avg_rms,  # wandb add
                       'Train  Avg. Loss': train_epoch_avg_loss, 'Validation Avg. Loss': validation_epoch_avg_loss}, step=epoch)

        return FitResult(actual_num_epochs, train_loss, test_loss)

    def train_epoch(self, dl_train: DataLoader, epoch: int, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        fb = self._foreach_batch(dl_train, self.train_batch, epoch, **kw)
        return fb

    def test_epoch(self, dl_test: DataLoader, epoch: int, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, epoch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch, plot_flag, epoch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
            plot_flag: plot original and constructed signal when true
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
            dl: DataLoader,
            forward_fn: Callable[[Any], BatchResult],
            epoch: int,
            verbose=True,
            max_batches=None,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        rmss = []
        prds = []
        prdns = []
        snrs = []
        qss = []

        num_batches = len(dl.batch_sampler)
        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, "w")

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            plot_flag = True
            for batch_idx in range(1,num_batches):
                data, _, _ = next(dl_iter)
                batch_res = forward_fn(data, plot_flag, epoch)
                plot_flag = False
                pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                pbar.update()

                losses.append(batch_res.loss)
                rmss.append(batch_res.rms)
                prds.append(batch_res.prd)
                prdns.append(batch_res.prdn)
                snrs.append(batch_res.snr)
                qss.append(batch_res.qs)
                # print('------------BATCH DEBUG--------------------')
                # print(f' batch_res.loss = {batch_res.loss}  batch_res.rms = {batch_res.rms}')
                # print('--------------------------------------------')
            avg_loss = sum(losses) / len(losses)
            pbar.set_description(
                f"{pbar_name} "
                f"(Avg. Loss {avg_loss:.3g}"
            )
        avg_rms = sum(rmss) / len(rmss)
        avg_prds = sum(prds) / len(prds)
        avg_prdns = sum(prdns) / len(prdns)
        avg_snrs = sum(snrs) / len(snrs)
        avg_qs = sum(qss) / len(qss)
        print(
            f'Avg rms = {avg_rms:.3g} prd = {avg_prds:.3g}% prdn ={avg_prdns:.3g}% snr = {avg_snrs:.3g} qs ={avg_qs:.3g}')
        er = EpochResult(losses=losses, avg_rms=avg_rms, avg_prd=avg_prds, avg_prdn=avg_prdns, avg_snr=avg_snrs,
                    avg_qs=avg_qs)

        # print('--------------------------------------------')
        # print(f' batches avg_loss = {avg_loss}  batches avg_rms = {avg_rms}')
        # print('--------------------------------------------')
        return er


class AutoEncoderTrainer(Trainer):
    def train_batch(self, batch, plot_flag, epoch) -> BatchResult:
        x = batch
        x = x.to(self.device)
        # ========================
        # forward
        out = self.model(x)
        out = torch.squeeze(out, dim=1)
        out = out.to(self.device)
        # loss
        loss = self.loss_fn(x, out)

        # zero old gradients
        self.optimizer.zero_grad()

        # backward
        loss.backward()

        # step
        self.optimizer.step()

        # evaluation criteria
        rms = calc_RMS(x, out)
        prd = calc_PRD(x, out)
        prdn = calc_PRDN(x, out)
        snr = calc_SNR(x, out)
        cr = calc_CR_DEEP()
        qs = calc_QS(x, out,cr)
        # ========================

        # avrage on all batch
        avg_rms = torch.mean(rms)
        avg_prd = torch.mean(prd)
        avg_prdn = torch.mean(prdn)
        avg_snr = torch.mean(snr)
        avg_qs = torch.mean(qs)

        return BatchResult(loss.item(), avg_rms.item(), avg_prd.item(), avg_prdn.item(), avg_snr.item(), avg_qs.item())

    def test_batch(self, batch, plot_flag, epoch) -> BatchResult:
        x = batch
        x = x.to(self.device)  # Image batch (N,C,H,W)

        with torch.no_grad():
            # ========================
            # forward
            out = self.model(x)
            out = torch.squeeze(out, dim=1)

            loss = self.loss_fn(x, out)

            # evaluation criteria for each window
            rms = calc_RMS(x, out)
            prd = calc_PRD(x, out)
            prdn = calc_PRDN(x, out)
            snr = calc_SNR(x, out)
            cr = calc_CR_DEEP()
            qs = calc_QS(x, out,cr)

            #avrage on all batch
            avg_rms = torch.mean(rms)
            avg_prd = torch.mean(prd)
            avg_prdn = torch.mean(prdn)
            avg_snr = torch.mean(snr)
            avg_qs  = torch.mean(qs)

            # print(f'rms = {rms} ')
            # print(f'avg_rms = {avg_rms}')
            # ========================


        return BatchResult(loss.item(), avg_rms.item(), avg_prd.item(), avg_prdn.item(), avg_snr.item(), avg_qs.item())


def train_one_patient():
    print('Training on one patient')
    # Data
    num_epochs = 100
    batch_size = 32
    validation_ratio = 0.1

    db = ECGdataset(window_len=2000, num_of_patients=4, json_filename="patient_ecg_length.json")
    dl_train, dl_test = create_train_validation_loader_for_one_patient(db, validation_ratio, db.patients_to_use[2],
                                                                       batch_size)

    # Model
    Enc = Encoder()
    Dec = Decoder()
    model = EncoderDecoder(Enc, Dec)
    model = model.double()  # for double precision

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)

    # Loss
    loss_fn = torch.nn.MSELoss()

    # Trainer
    trainer = AutoEncoderTrainer(model, loss_fn, optimizer, use_gpu=True)
    res = trainer.fit(dl_train, dl_test, num_epochs, early_stopping=2)

    # Plot images from best model
    checkpoint_file = 'checkpoints/AutoEncoder'
    saved_state = torch.load(
        f'/home/{getuser()}/git/ECG_compression/{checkpoint_file}.pt')  # map_location=torch.cuda.current_device())
    model.load_state_dict(saved_state['model_state'])
    print('*** Images Generated from best model:')
    frequency, ecg_window = get_window(patient_id='0001', start=2000, length=2000)

    # plot original ecg
    plt.plot(ecg_window)
    plt.savefig('ecg_window.png')

    ecg_window = torch.from_numpy(ecg_window)
    # use_gpu = True
    ecg_window = ecg_window.to(torch.device('cuda'))
    ecg_window = torch.unsqueeze(ecg_window, dim=0)
    print(f'ecg_window size ={ecg_window.shape}')
    # run ecg_window in Autoencoder
    reconstruct = model(ecg_window)

    # For plotting
    torch.Tensor.ndim = property(lambda self: len(self.shape))
    reconstruct = torch.squeeze(reconstruct, dim=0)
    reconstruct = reconstruct.cpu()
    reconstruct = reconstruct.detach().numpy()

    plt.plot(reconstruct)
    plt.savefig('original_and_reconstructed_ecg.png')
    print('Done One.')

def plot_orig_and_reconstruct(orig, model, epoch, str, dir='no_dir_specified', idx=None, rlab=None):
    torch.no_grad()
    reconstruct = model(orig)
    plt.figure(figsize=(10, 8))
    plt.plot(orig.cpu().detach().numpy()[1], "-b", label="orig")
    plt.plot(reconstruct.cpu().detach().numpy()[1], "-r", label="reconstructed")
    plt.legend(loc="upper left")
    plt.title('')
    plt.ylabel('Amplitude')
    plt.xlabel('Samples')
    # if None not in (idx, patient, rlab):
    if idx != None:
        # plt.figtext(0.5, 0.01, "index={}, patient={}, rlab={}".format(idx, patient, rlab), ha="center", fontsize=18,
        #             bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
        plt.figtext(0.5, 0.01, "index={}".format(idx), ha="center", fontsize=18,
                    bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, f'ecg_{str}_e{epoch}.png'))
    plt.close()


def add_indexes_to_dict(new_dictionary, full_dictionary_fname, index_list):
    import json
    try:
        with open(full_dictionary_fname, "r") as read_file:
            full_dict = json.load(read_file)
    except OSError:
        exit(f"Could not open/read {full_dictionary_fname}.")
    temp_dict = {str(k):full_dict[str(k)] for k in index_list if str(k) in full_dict}
    new_dictionary.update(temp_dict)


def train_evaluate(parameterization):
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=parameterization.get('lr',0.001), betas=(0.9, 0.999), weight_decay=1e-5)

    trainer = AutoEncoderTrainer(model, loss_fn, optimizer, use_gpu=use_gpu)

    #train and this updated the model
    res = trainer.fit(dl_train, dl_test, num_epochs-saved_epoch, early_stopping=4, checkpoints=checkpoint_file, seed=random_seed,data_percents=data_percents, act_epoch=actual_num_epochs, initial_epoch=saved_epoch, initialize_weights=initialize_weights)


    # return the accuracy of the model as it was trained in this run
    return sum(res.test_loss)/len(res.test_loss)


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        mean_weights_before = torch.mean(m.weight)
        std_weights_before = torch.std(m.weight)
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
        mean_weights_after = torch.mean(m.weight)
        std_weights_after = torch.std(m.weight)
        print(f'mean before = {mean_weights_before} mean after = {mean_weights_after}')
        print(f'std before = {std_weights_before} std after = {std_weights_after}')
    if isinstance(m, nn.Linear):
        mean_weights_before = torch.mean(m.weight)
        std_weights_before = torch.std(m.weight)
        y = m.in_features
        m.weight.data.normal_(0.0, 1 / np.sqrt(y))
        torch.nn.init.zeros_(m.bias)
        mean_weights_after = torch.mean(m.weight)
        std_weights_after = torch.std(m.weight)
        print(f'mean linear before = {mean_weights_before} mean after = {mean_weights_after}')
        print(f'std linear before = {std_weights_before} std after = {std_weights_after}')

def randomize_seed(seed = None):
    if seed == None:
        random_seed = random.randint(0, 2 ** 32 - 1)
    else:
        random_seed = seed
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    return random_seed

def reg_l1():
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(abs(param))
    return  regularization_loss

def reg_tv(x):
    """
    Calculating Total Variation Regularization
    :param x: the reconstructed signal
    :return: TV regularization
    """
    return torch.sum(torch.abs(x[:-1] - x[1:]))

def calc_tv_loss(x, y, weight=1):
    """
    Calculating Total Variation Loss for TV Regularization
    :param x: the reconstructed signal
    :param y: the original signal (gt)
    :param weight: how much weight should be given to the Total Variation
    :return: Loss
    """
    tv = sum(abs(x[:-1] - x[1:]))
    diff_fn = torch.nn.MSELoss()
    return diff_fn(x, y) + weight * tv

class MSETVLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', lamda=0.0, threshold=0.5):
        super(MSETVLoss, self).__init__(size_average, reduce, reduction)
        self.lamda = lamda
        self.threshold = threshold

    def forward(self, label, reconstruction):
        diffs = torch.abs(reconstruction[:, :-1] - reconstruction[:, 1:])
        label_diffs = torch.abs(label[:, :-1] - label[:, 1:])
        mask = (diffs < self.threshold * label_diffs.max(dim=1, keepdim=True)[0]).to(dtype=diffs.dtype)
        # max_diff = torch.max(diffs, dim=1)[0]
        # masked = torch.where(diffs < 0.2*max_diff, diffs, torch.zeros_like(diffs))
        # print('input shape = ', reconstruction.shape, 'target shape = ', label.shape)
        # print('diffs shape = ', diffs.shape)
        # print('masked shape = ', mask.shape)
        # print('diffs = ', diffs)
        # print('max_diff = ', max_diff, '0.5*max_diff = ', 0.5*max_diff)
        # print('masked = ', masked)
        # print('max masked = ', torch.max(masked))
        mse = F.mse_loss(reconstruction, label, reduction=self.reduction)
        # tv = torch.sum(torch.mul(mask,diffs)) / diffs.shape[1]
        tv = torch.sum(torch.mul(mask,diffs),dim=1) / diffs.shape[1]


        # # if torch.max(diffs) > 0.5 * torch.max(label_diffs):
        # plt.figure(figsize=(10, 8))
        # plt.plot(reconstruction.cpu().detach().numpy()[0], label='reconstruction')
        # plt.plot(label.cpu().detach().numpy()[0], label='Label')
        # plt.plot(mask.cpu().detach().numpy()[0], ',', label='Mask')
        # plt.plot(diffs.cpu().detach().numpy()[0], ',', label='Diffs')
        # plt.legend(loc="upper left")
        # plt.savefig(f'tv_mask/fig_{reconstruction.cpu().detach().numpy()[0, 4]}.png', dpi=150)
        # plt.close()
        # print('mse = ', mse, 'shape = ', mse.shape)
        # print('tv = ', tv, 'shape = ', tv.shape)
        return mse + self.lamda * tv
        # print(len(input))
        # return F.mse_loss(input, target, reduction=self.reduction) + self.lamda * torch.sum(torch.mul(mask,diffs)) / len(input)

if __name__ == '__main__':
    # ******* PARAMS *******
    lamda = 0.01  # if you do not want to run with regularization then you should run with lamda = 0
    tv_threshold = 1
    initialize_weights = False
    use_gpu = True
    num_epochs = 100
    batch_size = 64
    lr = 0.0005
    weight_decay = 1e-7
    seed = 4263309716
    random_seed = randomize_seed(seed)
    total_num_of_patients = 48
    validation_ratio = 0.2
    afib_perc = 0
    sbr_perc = 0
    svta_perc = 0
    train_hdf5_fname = 'train_scaled.hdf5'
    checkpoint_file = 'checkpoints/AutoEncoder_h'
    # **********************
    config = dict(
        epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=lr,
        wieght_decay = weight_decay,
        total_num_of_patients=total_num_of_patients,
        seed=random_seed,
        lamda=lamda,
        tv_threshold=tv_threshold
    )
    with wandb.init(project="projectB", config=config,entity="noambenmoshe"):
        # Model
        Enc = Encoder()
        Dec = Decoder()
        model = EncoderDecoder(Enc, Dec)
        model = model.double()  # for double precision

        wandb.watch(model,log='all', log_freq=10)
        config = wandb.config

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), weight_decay=config.wieght_decay)

        # Loss
        loss_fn = torch.nn.MSELoss()
        # loss_fn = MSETVLoss(lamda=config.lamda, threshold=config.tv_threshold)

        # Trainer
        trainer = AutoEncoderTrainer(model, loss_fn, optimizer, use_gpu=use_gpu)

        # Data
        normal_perc = 1 - afib_perc - sbr_perc - svta_perc
        data_percents = {'normal': normal_perc,
                         'afib': afib_perc,
                         'sbr': sbr_perc,
                         'svta': svta_perc}

        validation_num_patients = int(config.total_num_of_patients * validation_ratio)
        train_num_patients = config.total_num_of_patients - validation_num_patients
        print(
            f'Starting train with {train_num_patients} train patients and {validation_num_patients} validation patients')

        checkpoint_file_cont = f'{checkpoint_file}_continue'
        if os.path.isfile(f'{checkpoint_file_cont}.pt'):
            # Continue training from saved epoch
            print('Loading checkpoint')
            saved_state = torch.load(f'{checkpoint_file_cont}.pt')
            model.load_state_dict(saved_state['model_state'])
            #optimizer.load_state_dict(saved_state['optimizer_state'])
            saved_best_loss = saved_state.get('best_loss')
            saved_ewi = saved_state.get('ewi')
            saved_epoch = saved_state.get("epoch")
            random_seed = saved_state.get("seed")
            data_percents_check = saved_state.get("data_percents")
            evaluation_data = saved_state.get("test_result")
            actual_num_epochs = saved_state.get("actual_num_epochs")

            data_percents = data_percents_check
            print(f'data_percents_check ={data_percents_check}')
            print(f'evaluation_data = {evaluation_data}')
        else:
            print('Starting new model.')
            actual_num_epochs = 0
            saved_epoch = 0
            model.apply(weights_init)

        hdf5_train_file = h5py.File(train_hdf5_fname, "r")
        train_windows, train_rlabs, amount_dic = gen_dataset_from_hdf5(afib_perc, sbr_perc, svta_perc, normal_perc,
                                                                            train_num_patients,
                                                                            hdf5_train_file, type='train', amount_dic=None)
        validation_windows, validation_rlabs, _ = gen_dataset_from_hdf5(afib_perc, sbr_perc, svta_perc, normal_perc,
                                                                         validation_num_patients,
                                                                         hdf5_train_file, type='validation', amount_dic=amount_dic)

        db = ECGdataset_w_disease_hdf5(train_windows, train_rlabs, validation_windows, validation_rlabs)
        dl_train, dl_test = create_data_loaders_hdf5(db, len(train_rlabs), validation_windows.shape[0], config.batch_size, num_workers=1)

        print(f'**********************batch_size = {config.batch_size} lr={config.learning_rate}*****************')
        res = trainer.fit(dl_train, dl_test, config.epochs-saved_epoch, early_stopping=4, checkpoints=checkpoint_file, seed=random_seed,data_percents=data_percents, act_epoch=actual_num_epochs, initial_epoch=saved_epoch, initialize_weights=initialize_weights)

        wandb.finish()
        print('Done.')