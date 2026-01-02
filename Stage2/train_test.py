import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import confusion_matrix


def _expand_time_dimension(x, n_steps):
    """
    Ensure input tensor has time dimension [B, C, H, W, T].
    """
    if x.dim() < 5:
        x = x.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps)
    return x


def _compute_spike_prediction(outputs):
    """
    Sum spikes over time and return predicted class indices.
    """
    spike_counts = torch.sum(outputs, dim=4)
    spike_counts = spike_counts.squeeze(-1).squeeze(-1)
    return spike_counts.detach().cpu().numpy()


def train(
    network,
    trainloader,
    optimizer,
    epoch,
    states,
    network_config,
    layers_config,
    criterion,
    syn_a,
):
    network.train()

    n_steps = network_config["n_steps"]
    n_class = network_config["n_class"]
    loss_type = network_config["loss"]

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    device = next(network.parameters()).device

    # -------------------------------------------------------
    # Pre-compute desired spike kernel if needed
    # -------------------------------------------------------
    if loss_type == "kernel":
        if n_steps >= 10:
            desired_spikes = torch.tensor(
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            ).repeat(n_steps // 10)
        else:
            desired_spikes = torch.tensor([0, 1, 1, 1, 1]).repeat(n_steps // 5)

        desired_spikes = desired_spikes.view(1, 1, 1, 1, n_steps).to(device)

        from data_utils import psp
        desired_spikes = psp(desired_spikes, network_config)
        desired_spikes = desired_spikes.view(1, 1, 1, n_steps)

    # -------------------------------------------------------
    # Training loop
    # -------------------------------------------------------
    for inputs, labels in trainloader:
        inputs = _expand_time_dimension(inputs, n_steps).float().to(device)
        labels = labels.to(device)

        outputs = network.forward(inputs, epoch, True, syn_a)

        # ---------------------------------------------------
        # Loss computation
        # ---------------------------------------------------
        if loss_type == "count":
            desired = network_config["desired_count"]
            undesired = network_config["undesired_count"]

            targets = torch.full(
                (outputs.shape[0], outputs.shape[1], 1, 1),
                undesired,
                device=device,
            )
            for i, lbl in enumerate(labels):
                targets[i, lbl] = desired

            loss = criterion.spike_count(
                outputs,
                targets,
                network_config,
                layers_config[list(layers_config.keys())[-1]],
            )

        elif loss_type == "kernel":
            targets = torch.zeros_like(outputs)
            for i, lbl in enumerate(labels):
                targets[i, lbl] = desired_spikes

            loss = criterion.spike_kernel(outputs, targets, network_config)

        elif loss_type == "softmax":
            loss = criterion.spike_soft_max(outputs, labels)

        else:
            raise ValueError("Unrecognized loss function.")

        # ---------------------------------------------------
        # Backpropagation
        # ---------------------------------------------------
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(network.get_parameters(), 1.0)
        optimizer.step()
        network.weight_clipper()

        # ---------------------------------------------------
        # Statistics
        # ---------------------------------------------------
        predictions = _compute_spike_prediction(outputs)
        labels_np = labels.cpu().numpy()

        total_correct += (predictions.argmax(axis=1) == labels_np).sum()
        total_samples += labels_np.shape[0]
        total_loss += loss.item()

        states.training.correct_samples = total_correct
        states.training.num_samples = total_samples
        states.training.loss_sum += loss.item()

    train_acc = total_correct / total_samples
    train_loss = total_loss / total_samples

    print(
        f"Epoch {epoch}: "
        f"Train Accuracy: {100. * train_acc:.3f}, "
        f"Loss: {train_loss:.3f}"
    )


def test(
    network,
    testloader,
    epoch,
    states,
    network_config,
    layers_config,
    early_stopping,
    syn_a,
):
    network.eval()

    n_steps = network_config["n_steps"]
    n_class = network_config["n_class"]
    loss_type = network_config["loss"]
    device = next(network.parameters()).device

    total_correct = 0
    total_samples = 0
    y_pred, y_true = [], []

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = _expand_time_dimension(inputs, n_steps).float().to(device)
            labels = labels.to(device)

            outputs = network.forward(inputs, epoch, False, syn_a)

            predictions = _compute_spike_prediction(outputs)
            labels_np = labels.cpu().numpy()

            preds = predictions.argmax(axis=1)
            y_pred.append(preds)
            y_true.append(labels_np)

            total_correct += (preds == labels_np).sum()
            total_samples += labels_np.shape[0]

            states.testing.correct_samples += (preds == labels_np).sum()
            states.testing.num_samples = total_samples

    test_acc = total_correct / total_samples
    print(f"Epoch {epoch}: Test Accuracy: {100. * test_acc:.3f}")

    # -------------------------------------------------------
    # Early stopping
    # -------------------------------------------------------
    early_stopping(100.0 * test_acc, network, epoch)

    # -------------------------------------------------------
    # Best model confusion matrix
    # -------------------------------------------------------
    if test_acc > getattr(test, f"best_acc_{loss_type}", 0):
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)

        cf = confusion_matrix(y_true, y_pred, labels=np.arange(n_class))
        df_cm = pd.DataFrame(
            cf,
            index=[str(i) for i in range(n_class)],
            columns=[str(i) for i in range(n_class)],
        )

        plt.figure()
        sn.heatmap(df_cm, annot=True, cmap="viridis")
        plt.savefig(f"confusion_matrix_{loss_type}.png")
        plt.close()

        setattr(test, f"best_acc_{loss_type}", test_acc)