import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


class MultiTaskNetOpt(nn.Module):
    """
    A multi-task learning neural network with shared hidden layers and task-specific output layers.

    Args:
        input_size (int): Number of input features.
        hidden_sizes (list of int): List defining the number of neurons in each shared hidden layer.
        num_tasks (int): Number of tasks (outputs).
        dropout_rate (float): Dropout probability for regularization.
    """

    def __init__(self, input_size, hidden_sizes, num_tasks, dropout_rate):
        super(MultiTaskNetOpt, self).__init__()

        # Define shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]), nn.LeakyReLU()
        )

        for i in range(1, len(hidden_sizes)):
            self.shared_layers.extend(
                [
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]),
                    nn.LeakyReLU(),
                ]
            )

        # Task-specific output layers
        self.task_layers = nn.ModuleList(
            [nn.Linear(hidden_sizes[-1], 1) for _ in range(num_tasks)]
        )

    def forward(self, x):
        """Forward pass through the shared layers and task-specific layers."""
        x = self.shared_layers(x)
        return [task_layer(x) for task_layer in self.task_layers]


class MultiTaskSpec(nn.Module):
    """
    A multi-task learning network with both shared and task-specific layers.

    Args:
        input_size (int): Number of input features.
        hidden_sizes (list of int): List of neuron counts for shared hidden layers.
        specific_sizes (list of int): List defining neuron counts for task-specific hidden layers.
        num_tasks (int): Number of tasks (outputs).
        dropout_rate (float): Dropout probability for regularization.
    """

    def __init__(
        self, input_size, hidden_sizes, specific_sizes, num_tasks, dropout_rate
    ):
        super(MultiTaskSpec, self).__init__()

        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]), nn.LeakyReLU()
        )

        for i in range(1, len(hidden_sizes)):
            self.shared_layers.extend(
                [
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]),
                    nn.LeakyReLU(),
                ]
            )

        # Task-specific layers
        self.task_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_sizes[-1], specific_sizes[0]),
                    nn.ReLU(),
                    nn.Linear(specific_sizes[0], specific_sizes[1]),
                    nn.ReLU(),
                )
                for _ in range(num_tasks)
            ]
        )

        # Final output layers
        self.final_output_layers = nn.ModuleList(
            [nn.Linear(specific_sizes[1], 1) for _ in range(num_tasks)]
        )

    def forward(self, x):
        """Forward pass through shared and task-specific layers."""
        x = self.shared_layers(x)
        return [
            final_layer(task_layer(x))
            for task_layer, final_layer in zip(
                self.task_layers, self.final_output_layers
            )
        ]


def check_target_content(y_train):
    all_valid = True
    for col_name in y_train.columns:
        if len(y_train[col_name].dropna().tolist()) <= 0:
            print(col_name, " target not valid for training with this split")
            all_valid = False
    if all_valid:
        print(
            "Every target has at least 1 datapoint in the training set - Split valid!"
        )
        return "Fold valid"
    else:
        return "Fold not valid"


# Loss function that ignores NaN values
def custom_loss(output, target):
    mask = ~torch.isnan(target)
    return nn.MSELoss()(output[mask], target[mask])


# R-squared metric that ignores NaN values
def r_squared(output, target):
    mask = ~torch.isnan(target)
    mean_observed = torch.mean(target[mask])
    total_ss = torch.sum((target[mask] - mean_observed) ** 2)
    residual_ss = torch.sum((target[mask] - output[mask]) ** 2)
    return 1 - (residual_ss / total_ss)


# Compute R^2 score for valid test columns
def get_performance_scores(y_test, y_pred):
    eval_dict = {}
    valid_columns = [col for col in y_test.columns if len(y_test[col].dropna()) >= 5]

    if not valid_columns:
        print("No columns have more than 4 values.")
        return {}

    for col in valid_columns:
        y_pred_col = y_pred[col].dropna()
        y_test_col = y_test[col].dropna()
        if len(y_pred_col) > 0 and len(y_test_col) > 0:
            eval_dict[col] = r2_score(y_test_col, y_pred_col)

    return eval_dict


# Root Mean Squared Error (RMSE)
def rmse(y_test, y_pred):
    return np.sqrt(np.mean((y_test - y_pred) ** 2))


# Mean Absolute Error (MAE)
def mae(actual_values, predicted_values):
    return np.mean(np.abs(actual_values - predicted_values))


# Mean Squared Logarithmic Error (MSLE)
def msle(actual_values, predicted_values):
    return np.mean((np.log1p(actual_values) - np.log1p(predicted_values)) ** 2)


# Compute mean, standard deviation, and standard error of the mean (SEM)
def metrics_cv(list_of_vals):
    std_dev = np.std(list_of_vals, ddof=1)  # Sample standard deviation
    return [np.mean(list_of_vals), std_dev, std_dev / np.sqrt(len(list_of_vals))]


def set_seeds(seed: int):
    """
    Set random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to ensure deterministic behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(data: dict, b_params: dict, seed: int, epochs=300, device="cuda"):
    """
    Trains a multi-task neural network model using the provided data and hyperparameters.

    Args:
        data (dict): Dictionary containing training and validation tensors.
        b_params (dict): Dictionary containing model hyperparameters.
        seed (int): Random seed for reproducibility.
        device (torch.device): Device on which the model will be trained.

    Returns:
        tuple: The trained model, training loss history, validation loss history, and validation accuracy history.
    """
    print(f"Model training seed: {seed}")
    set_seeds(seed)

    # Extract training and validation data
    X_train, Y_train = data["tensors"]["train"]["X"], data["tensors"]["train"]["y"]
    X_val, Y_val = data["tensors"]["val"]["X"], data["tensors"]["val"]["y"]

    # Create DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=b_params["batch_size"],
        shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=b_params["batch_size"],
        shuffle=False,
    )

    # Define model architecture
    input_size, output_size = X_train.shape[1], Y_train.shape[1]
    hidden_sizes = [b_params["hidden_size1"], b_params["hidden_size2"]]
    if b_params["nr_of_shared_layers"] == 3:
        hidden_sizes.append(b_params["hidden_size3"])

    model = MultiTaskNetOpt(
        input_size, hidden_sizes, output_size, b_params["dropout_rate"]
    ).to(device)
    print(
        f"Model architecture: {hidden_sizes}, dropout rate: {b_params['dropout_rate']}"
    )

    optimizer = optim.Adam(model.parameters(), lr=b_params["learning_rate"])
    # scheduler = ReduceLROnPlateau(
    #     optimizer, mode="min", patience=5, factor=0.1, verbose=True
    # )

    # Training setup
    best_val_loss = float("inf")
    patience_counter = 0

    train_losses_history = []
    val_losses_history = []
    val_accuracies_history = []

    print(f"Initial weights of first layer: {model.shared_layers[0].weight.data[:5]}")

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = custom_loss(torch.cat(outputs, dim=1).view(-1), batch_Y.view(-1))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss_avg = np.mean(train_losses)
        train_losses_history.append(train_loss_avg)

        # Validation loop
        model.eval()
        val_losses = []
        val_accuracies = []
        with torch.no_grad():
            for val_batch_X, val_batch_Y in val_loader:
                val_batch_X, val_batch_Y = (
                    val_batch_X.to(device),
                    val_batch_Y.to(device),
                )
                val_outputs = model(val_batch_X)
                val_loss = custom_loss(
                    torch.cat(val_outputs, dim=1).view(-1), val_batch_Y.view(-1)
                )
                val_losses.append(val_loss.item())
                val_accuracies.append(
                    r_squared(
                        torch.cat(val_outputs, dim=1).view(-1), val_batch_Y.view(-1)
                    ).item()
                )

        val_loss_avg = np.mean(val_losses)
        val_losses_history.append(val_loss_avg)
        val_acc_avg = np.mean(val_accuracies)
        val_accuracies_history.append(val_acc_avg)

        print(
            f"Epoch [{epoch + 1}/{epochs}], Training Loss: {train_loss_avg}, Validation Loss: {val_loss_avg}, Validation R^2: {val_acc_avg}"
        )

        # Early stopping logic
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= b_params["patience"]:
                print(
                    f"Early stopping at epoch {epoch + 1} as validation loss did not improve."
                )
                break

    # Compute final validation loss
    with torch.no_grad():
        y_pred_val_cat = torch.cat(model(X_val), dim=1)
    final_val_loss = custom_loss(y_pred_val_cat, Y_val).item()
    print(final_val_loss)

    return model, train_losses_history, val_losses_history, val_accuracies_history


def cross_val_internal(clean_data, num_of_targets, param_d, device, seed):
    """
    Performs 5-fold cross-validation on the given dataset.

    Args:
        clean_data (pd.DataFrame): The preprocessed dataset.
        num_of_targets (int): The number of target columns in the dataset.
        param_d (dict): Dictionary of model parameters.
        device (torch.device): The device (CPU or GPU) to perform computations on.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - cv_dictio (dict): Mean, standard deviation, and standard error for each metric.
            - losses_dict_cv (dict): Training and validation losses for each fold.
            - just_eval_dict_list (list): Performance scores for each validation set.
    """

    print(f"Cross-validation with seed: {seed}")

    # Drop non-feature columns
    combined_df_copy = clean_data.drop(columns=["SMILES_stand"])

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Initialize metrics dictionary
    metrics_dict = {
        "train_loss": [],
        "test_loss": [],
        "val_loss": [],
        "train_r2": [],
        "test_r2": [],
        "val_r2": [],
        "rmse": [],
        "mae": [],
        "msle": [],
    }

    losses_dict_cv = {"train_losses_list": [], "val_losses_list": []}

    just_eval_dict_list = []

    # Cross-validation loop
    for count, (train_index, val_index) in enumerate(
        skf.split(combined_df_copy, combined_df_copy["new_target_label"])
    ):
        train_kf = combined_df_copy.iloc[train_index]
        val_kf = combined_df_copy.iloc[val_index]

        # Split into features (X) and targets (y)
        X_train_kf = train_kf.drop(columns=["new_target_label"]).iloc[
            :, :-num_of_targets
        ]
        X_val_kf = val_kf.drop(columns=["new_target_label"]).iloc[:, :-num_of_targets]

        y_train_kf = train_kf.drop(columns=["new_target_label"]).iloc[
            :, -num_of_targets:
        ]
        y_val_kf = val_kf.drop(columns=["new_target_label"]).iloc[:, -num_of_targets:]

        check_target_content(y_train_kf)

        # Convert to tensors
        X_train_kf_tensor = torch.tensor(X_train_kf.values, dtype=torch.float32).to(
            device
        )
        X_val_kf_tensor = torch.tensor(X_val_kf.values, dtype=torch.float32).to(device)

        y_train_kf_tensor = torch.tensor(y_train_kf.values, dtype=torch.float32).to(
            device
        )
        y_val_kf_tensor = torch.tensor(y_val_kf.values, dtype=torch.float32).to(device)

        # Prepare data dictionary
        cv_tensors_dict = {
            "train": {"X": X_train_kf_tensor, "y": y_train_kf_tensor},
            "val": {"X": X_val_kf_tensor, "y": y_val_kf_tensor},
        }

        cv_data_dict = {"tensors": cv_tensors_dict}
        print(cv_data_dict["tensors"]["train"]["X"])
        # Train model
        model_kf, train_mean_losses, val_mean_losses, val_mean_accuracies = train_model(
            cv_data_dict, param_d, epochs=300, seed=seed, device=device
        )

        # Store losses
        losses_dict_cv["train_losses_list"].append(train_mean_losses)
        losses_dict_cv["val_losses_list"].append(val_mean_losses)

        # Model evaluation
        with torch.no_grad():
            predicted_values_train_kf = model_kf(X_train_kf_tensor)
            predicted_values_val_kf = model_kf(X_val_kf_tensor)

        # Concatenate predicted outputs
        y_pred_cuda_train_kf = torch.cat(predicted_values_train_kf, dim=1)
        y_pred_cuda_val_kf = torch.cat(predicted_values_val_kf, dim=1)

        # Compute metrics
        metrics_dict["train_loss"].append(
            custom_loss(y_pred_cuda_train_kf, y_train_kf_tensor).item()
        )
        metrics_dict["val_loss"].append(
            custom_loss(y_pred_cuda_val_kf, y_val_kf_tensor).item()
        )
        metrics_dict["train_r2"].append(
            r_squared(y_pred_cuda_train_kf, y_train_kf_tensor).item()
        )
        metrics_dict["val_r2"].append(
            r_squared(y_pred_cuda_val_kf, y_val_kf_tensor).item()
        )

        # Convert predictions to DataFrame and apply mas
        df_predicted_values_val_kf = pd.DataFrame(
            y_pred_cuda_val_kf.cpu().numpy(), columns=y_val_kf.columns
        ).set_index(y_val_kf.index)
        mask_kf = ~np.isnan(y_val_kf)
        y_val_filtered_masklocs_kf = df_predicted_values_val_kf[mask_kf]  # .round(2)

        # Additional performance metrics
        metrics_dict["rmse"].append(
            rmse(y_val_kf, y_val_filtered_masklocs_kf)
        )  # alt df_predicted_values_test_kf
        metrics_dict["mae"].append(mae(y_val_kf, y_val_filtered_masklocs_kf))
        metrics_dict["msle"].append(msle(y_val_kf, y_val_filtered_masklocs_kf))

        # Store evaluation results
        print(f"Test R2: {r_squared(y_pred_cuda_val_kf, y_val_kf_tensor).item()}")

        just_eval_dict = get_performance_scores(y_val_kf, y_val_filtered_masklocs_kf)
        just_eval_dict_list.append(just_eval_dict)

        print(f"K-{count} done")

    # Compute summary statistics for cross-validation metric
    cv_dictio = {
        metric: [
            np.mean(values),
            np.std(values),
            np.std(values) / np.sqrt(len(values)),
        ]
        for metric, values in metrics_dict.items()
    }

    return cv_dictio, losses_dict_cv, just_eval_dict_list
