import random
import numpy as np
import pandas as pd

import optuna

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

def objective_rf(trial, X_train, y_train, X_val, y_val):
    """
    Defines the objective function for Optuna hyperparameter optimization of a RandomForestRegressor.

    Parameters:
    trial (optuna.trial.Trial): Current Optuna trial object.
    X_train (pd.DataFrame): Training feature matrix.
    y_train (pd.Series): Training target values.
    X_val (pd.DataFrame): Validation feature matrix.
    y_val (pd.Series): Validation target values.

    Returns:
    float: Mean squared error (MSE) of the model on the validation set.
    """
    n_estimators = trial.suggest_int('n_estimators', 50, 300, step=50)
    max_depth = trial.suggest_int('max_depth', 5, 50, step=5)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20, step=2)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10, step=1)
    max_features = trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    max_samples = trial.suggest_float('max_samples', 0.5, 1.0) if bootstrap else None

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        max_samples=max_samples,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return mean_squared_error(y_val, y_pred)





def single_task_train_with_summary(df, targets, n_trials=300, cv_folds=5):
    """
    Performs target-wise hyperparameter optimization, training, and cross-validation 
    using RandomForestRegressor models for multiple targets.

    Parameters:
    df (pd.DataFrame): Dataset containing molecular descriptors and target columns.
    targets (list): List of target column names to process.
    n_trials (int): Number of Optuna trials for hyperparameter search. Default is 300.
    cv_folds (int): Number of cross-validation folds. Default is 5.

    Returns:
    tuple:
        dict: Dictionary of trained models, parameters, and performance metrics per target.
        pd.DataFrame: Summary DataFrame with compounds count, mean CV Q², and MSE per target.
    """
    results = {}
    summary_rows = []

    for target in targets:
        print(f"\nProcessing target: {target}")

        target_df = (
            df.dropna(subset=[target])
              .drop(columns='new_target_label')
              .sample(frac=1, random_state=42)
              .reset_index(drop=True)
        )

        total_compounds = target_df.shape[0]
        X_df = target_df.iloc[:, 1:513].copy()
        X_df[target] = target_df[target]

        # Split data
        train, test_temp = train_test_split(X_df, test_size=1/3, random_state=42)
        val, test = train_test_split(test_temp, test_size=0.5, random_state=42)

        X_train, y_train = train.drop(columns=[target]), train[target]
        X_val, y_val = val.drop(columns=[target]), val[target]
        X_test, y_test = test.drop(columns=[target]), test[target]

        # Deterministic Optuna setup
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(lambda trial: objective_rf(trial, X_train, y_train, X_val, y_val),
                       n_trials=n_trials)

        best_params = study.best_params
        print(f"Best hyperparameters for {target}: {best_params}")

        # Train final model
        train_val = pd.concat([train, val], ignore_index=True, sort=False)
        X_train_val, y_train_val = train_val.drop(columns=[target]), train_val[target]

        model = RandomForestRegressor(**best_params, random_state=42)
        model.fit(X_train_val, y_train_val)

        # Manual CV for positive MSE
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        mse_scores, r2_scores = [], []
        for train_idx, val_idx in kf.split(X_train_val):
            X_tr, X_va = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
            y_tr, y_va = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            mse_scores.append(mean_squared_error(y_va, y_pred))
            r2_scores.append(r2_score(y_va, y_pred))

        mse_mean = np.mean(mse_scores)
        mse_std = np.std(mse_scores)
        r2_mean = np.mean(r2_scores)
        r2_std = np.std(r2_scores)

        print(f"{target} CV MSE: {mse_mean:.4f} ± {mse_std:.4f}")
        print(f"{target} CV R2 (q²): {r2_mean:.4f} ± {r2_std:.4f}")

        # Store summary and results
        summary_rows.append({
            'Target': target,
            'Total Compounds': total_compounds,
            'CV Q² ± SD': f"{r2_mean:.4f} ± {r2_std:.4f}",
            'CV MSE ± SD': f"{mse_mean:.4f} ± {mse_std:.4f}"
        })

        results[target] = {
            'model': model,
            'best_params': best_params,
            'cv_r2_mean': r2_mean,
            'cv_r2_std': r2_std,
            'splits': {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'train_val_df': train_val
            }
        }

    summary_df = pd.DataFrame(summary_rows)
    return results, summary_df


class SingleTaskNetOpt(nn.Module):
    """
    Feedforward neural network for single-target regression with configurable hidden layers and dropout.

    Parameters:
    input_size (int): Number of input features.
    hidden_sizes (list): List of hidden layer sizes.
    dropout_rate (float): Dropout probability applied between layers.

    Methods:
    forward(x): Performs a forward pass through the network.
    """
    def __init__(self, input_size, hidden_sizes, dropout_rate):
        super(SingleTaskNetOpt, self).__init__()
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.LeakyReLU()]
        for i in range(1, len(hidden_sizes)):
            layers.extend([
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]),
                nn.LeakyReLU()
            ])
        self.shared_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        x = self.shared_layers(x)
        return self.output_layer(x).squeeze(-1)



def train_model_stnn(model, X_train, y_train, X_val, y_val, lr, batch_size, patience, max_epochs=300):
    """
    Trains a single-target neural network with early stopping based on validation loss.

    Parameters:
    model (nn.Module): Neural network model to train.
    X_train (pd.DataFrame): Training feature matrix.
    y_train (pd.Series): Training target values.
    X_val (pd.DataFrame): Validation feature matrix.
    y_val (pd.Series): Validation target values.
    lr (float): Learning rate for the optimizer.
    batch_size (int): Mini-batch size for training.
    patience (int): Number of epochs with no improvement before stopping early.
    max_epochs (int): Maximum number of training epochs. Default is 300.

    Returns:
    float: Best validation loss (MSE) achieved during training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Ensure numeric inputs
    X_train_t = torch.tensor(X_train.astype(np.float32).values, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.astype(np.float32).values, dtype=torch.float32).to(device)
    X_val_t   = torch.tensor(X_val.astype(np.float32).values, dtype=torch.float32).to(device)
    y_val_t   = torch.tensor(y_val.astype(np.float32).values, dtype=torch.float32).to(device)

    best_val_loss = np.inf
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))
        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            preds_val = model(X_val_t)
            val_loss = criterion(preds_val, y_val_t).item()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_loss


def objective_stnn(trial, X_train, y_train, X_val, y_val):
    """
    Defines the Optuna objective function for tuning neural network hyperparameters.

    Parameters:
    trial (optuna.trial.Trial): Optuna trial object.
    X_train (pd.DataFrame): Training feature matrix.
    y_train (pd.Series): Training target values.
    X_val (pd.DataFrame): Validation feature matrix.
    y_val (pd.Series): Validation target values.

    Returns:
    float: Validation mean squared error (MSE) for the trial configuration.
    """
def objective_stnn(trial, X_train, y_train, X_val, y_val):
    nr_of_shared_layers = trial.suggest_int("nr_of_shared_layers", 2, 3)
    hidden_sizes = []

    hidden_size_1 = trial.suggest_int("hidden_size_1", 100, 1200, step=10)
    hidden_sizes.append(hidden_size_1)
    hidden_size_2 = trial.suggest_int("hidden_size_2", 5, 500, step=10)
    hidden_sizes.append(hidden_size_2)
    if nr_of_shared_layers == 3:
        hidden_size_3 = trial.suggest_int("hidden_size_3", 5, 200, step=10)
        hidden_sizes.append(hidden_size_3)

    dropout = trial.suggest_float("dropout", 0.07, 0.5, step=0.001)
    learning_rate = trial.suggest_float("learning_rate", 0.000035, 0.000170, step=0.000005)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    patience = trial.suggest_int("patience", 7, 10)

    model = SingleTaskNetOpt(
        input_size=X_train.shape[1],
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout
    )

    val_mse = train_model_stnn(
        model, X_train, y_train, X_val, y_val,
        lr=learning_rate,
        batch_size=batch_size,
        patience=patience
    )

    return val_mse



def single_target_cv_stnn(train_val_con, target, best_params, cv_folds=5, epochs=300):
    """
    Performs k-fold cross-validation for a single target using the optimized neural network.

    Parameters:
    train_val_con (pd.DataFrame): DataFrame containing descriptors and target column.
    target (str): Target column name.
    best_params (dict): Best hyperparameters from Optuna optimization.
    cv_folds (int): Number of cross-validation folds. Default is 5.
    epochs (int): Maximum number of training epochs per fold. Default is 300.

    Returns:
    tuple: Mean and standard deviation of R² and MSE across folds.
    """
    X = train_val_con.iloc[:, :512] # better to drop target 
    y = train_val_con[target]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    mse_scores, r2_scores = [], []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = SingleTaskNetOpt(
            input_size=X_train.shape[1],
            hidden_sizes=best_params["hidden_sizes"],
            dropout_rate=best_params["dropout_rate"]
        ).to(device)

        train_model_stnn(
            model, X_train, y_train, X_val, y_val,
            lr=best_params["learning_rate"],
            batch_size=best_params["batch_size"],
            patience=best_params["patience"],
            max_epochs=epochs
        )

        with torch.no_grad():
            X_val_t = torch.tensor(X_val.astype(np.float32).values, dtype=torch.float32).to(device)
            preds = model(X_val_t).cpu().numpy()

        mse_scores.append(mean_squared_error(y_val, preds))
        r2_scores.append(r2_score(y_val, preds))

    print(f"{target} CV MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
    print(f"{target} CV R2: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    return np.mean(r2_scores), np.std(r2_scores), np.mean(mse_scores), np.std(mse_scores)


def multi_task_train_with_summary_stnn(df, targets, n_trials=300, cv_folds=5, epochs=300):
    """
    Trains and optimizes single-target neural networks for multiple targets with Optuna,
    followed by cross-validation and summary generation.

    Parameters:
    df (pd.DataFrame): Dataset with molecular descriptors and target columns.
    targets (list): List of target column names to train models for.
    n_trials (int): Number of Optuna trials per target. Default is 300.
    cv_folds (int): Number of cross-validation folds. Default is 5.
    epochs (int): Maximum training epochs per fold. Default is 300.

    Returns:
    tuple:
        dict: Target-wise model parameters and CV performance metrics.
        pd.DataFrame: Summary of Q² and MSE statistics across targets.
    """
    results = {}
    summary_rows = []

    for target in targets:
        print(f"\nProcessing target: {target}")

        target_df = (
            df.dropna(subset=[target])
              .drop(columns='new_target_label', errors='ignore')
              .sample(frac=1, random_state=42)
              .reset_index(drop=True)
        )

        total_compounds = target_df.shape[0]
        X_df = target_df.iloc[:, 1:513].copy()
        X_df[target] = target_df[target]

        # Split data
        train, test_temp = train_test_split(X_df, test_size=1/3, random_state=42)
        val, test = train_test_split(test_temp, test_size=0.5, random_state=42)

        X_train, y_train = train.drop(columns=[target]), train[target]
        X_val, y_val = val.drop(columns=[target]), val[target]
        X_test, y_test = test.drop(columns=[target]), test[target]

        # Optuna optimization
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(lambda trial: objective_stnn(trial, X_train, y_train, X_val, y_val),
                       n_trials=n_trials)

        best_trial = study.best_trial
        best_params = best_trial.params

        # Extract hidden_sizes dynamically
        hidden_sizes = []
        for i in range(1, 4):
            key = f"hidden_size_{i}"
            if key in best_params:
                hidden_sizes.append(best_params.pop(key))
        best_params["hidden_sizes"] = hidden_sizes

        # Map other keys
        best_params["dropout_rate"] = best_params.pop("dropout")
        best_params["learning_rate"] = best_params.pop("learning_rate")
        best_params["batch_size"] = best_params.pop("batch_size")
        best_params["patience"] = best_params.pop("patience")

        print(f"Best hyperparameters for {target}: {best_params}")

        # Final CV
        #train_x_val_x = pd.concat([df1, df2], ignore_index=True)
        train_val  = pd.concat([train, val], ignore_index=True, sort=False)
        r2_mean, r2_std, mse_mean, mse_std = single_target_cv_stnn(train_val, target, best_params, cv_folds=cv_folds, epochs=epochs)

        summary_rows.append({
            "Target": target,
            "Total Compounds": total_compounds,
            'CV Q² ± SD': f"{r2_mean:.4f} ± {r2_std:.4f}",
            'CV MSE ± SD': f"{mse_mean:.4f} ± {mse_std:.4f}"
        })

        results[target] = {"best_params": best_params, "cv_r2_mean": r2_mean}

    summary_df = pd.DataFrame(summary_rows)
    return results, summary_df