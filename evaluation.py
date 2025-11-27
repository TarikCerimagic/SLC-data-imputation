import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from fancyimpute import SoftImpute, IterativeSVD, BiScaler, NuclearNormMinimization, MatrixFactorization, KNN
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.model_selection import ParameterGrid



def custom_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the Mean Squared Error (MSE) loss, ignoring NaN values in the target.

    Args:
        output (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground truth values (may contain NaNs).

    Returns:
        torch.Tensor: Computed MSE loss.
    """
    mask = ~torch.isnan(target)  # Mask to ignore NaN values
    loss = nn.MSELoss(reduction="mean")(output[mask], target[mask])
    return loss


def r_squared(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the R-squared (coefficient of determination) metric, ignoring NaN values in the target.

    Args:
        output (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground truth values (may contain NaNs).

    Returns:
        torch.Tensor: R-squared value.
    """
    mask = ~torch.isnan(target)
    mean_observed = torch.mean(target[mask])
    total_sum_of_squares = torch.sum((target[mask] - mean_observed) ** 2)
    residual_sum_of_squares = torch.sum((target[mask] - output[mask]) ** 2)
    return 1 - (residual_sum_of_squares / total_sum_of_squares)


def rmse(actual_values: torch.Tensor, predicted_values: torch.Tensor) -> torch.Tensor:
    """
    Computes the Root Mean Squared Error (RMSE), ignoring NaN values.

    Args:
        actual_values (torch.Tensor): Ground truth values (may contain NaNs).
        predicted_values (torch.Tensor): Predicted values.

    Returns:
        torch.Tensor: RMSE value.
    """
    mask = ~torch.isnan(actual_values)
    squared_errors = (actual_values[mask] - predicted_values[mask]) ** 2
    mean_squared_error = torch.mean(squared_errors)
    return torch.sqrt(mean_squared_error)


def calculate_mae(
    actual_values: torch.Tensor, predicted_values: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Mean Absolute Error (MAE), ignoring NaN values.

    Args:
        actual_values (torch.Tensor): Ground truth values (may contain NaNs).
        predicted_values (torch.Tensor): Predicted values.

    Returns:
        torch.Tensor: MAE value.
    """
    mask = ~torch.isnan(actual_values)
    absolute_errors = torch.abs(actual_values[mask] - predicted_values[mask])
    return torch.mean(absolute_errors)


def print_prediction_metrics(
    model_ec, X_test_tensor, Y_test_tensor, X_train_tensor, Y_train_tensor, device
):
    """
    Evaluates the model's predictions and computes relevant metrics.

    Args:
        model_ec: Trained PyTorch model.
        X_test_tensor (torch.Tensor): Test input features.
        Y_test_tensor (torch.Tensor): Test target values.
        X_train_tensor (torch.Tensor): Training input features.
        Y_train_tensor (torch.Tensor): Training target values.
        device: Computation device (CPU or GPU).

    Returns:
        Tuple: (Predicted test values, test loss, results dictionary)
    """
    with torch.no_grad():
        predicted_values_test = model_ec(X_test_tensor.to(device))
        predicted_values_train = model_ec(X_train_tensor.to(device))

    # Concatenate predictions along the second dimension
    y_pred_test = torch.cat(predicted_values_test, dim=1)
    y_pred_train = torch.cat(predicted_values_train, dim=1)

    # Compute evaluation metrics
    test_loss = custom_loss(y_pred_test, Y_test_tensor.to(device)).item()
    r2_train = r_squared(y_pred_train, Y_train_tensor).item()
    r2_test = r_squared(y_pred_test, Y_test_tensor).item()
    mae_test = calculate_mae(Y_test_tensor.to(device), y_pred_test).item()
    rmse_test = rmse(Y_test_tensor.to(device), y_pred_test).item()

    res_dict = {
        "MTS": [r2_train],  # Mean Training Score (R² on train set)
        "R²": [r2_test],  # R² on test set
        "MSE(t)": [test_loss],  # Test loss (MSE)
        "MAE(t)": [mae_test],  # Test MAE
        "RMSE(t)": [rmse_test],  # Test RMSE
    }

    # Print evaluation metrics
    print("R² Test:")
    print("Loss:", test_loss)
    print("R²:", r2_test)
    print("\nR² Train:")
    print("Loss:", custom_loss(y_pred_train, Y_train_tensor).item())
    print("R²:", r2_train)

    return y_pred_test, test_loss, res_dict


def pad_lists(data):
    """
    Pads a list of lists to the same length by:
      - Prepending NaN to each list.
      - Removing the first five elements.
      - Padding the remaining elements with NaNs to match the longest list.

    Parameters:
    data (list of lists): List of numerical lists of varying lengths.

    Returns:
    list of lists: Padded lists with equal length.
    """
    max_length = max(len(lst) for lst in data)  # Find the longest list length
    padded_data = []

    for lst in data:
        padded_lst = [
            np.nan
        ]  # Prepend a NaN (previously 2 were mentioned but only 1 was used)
        padded_lst.extend(lst[5:])  # Append values starting from index 5 onward
        padded_lst += [np.nan] * (
            max_length - len(padded_lst)
        )  # Pad with NaNs to match max length
        padded_data.append(padded_lst)

    return padded_data


def cv_plots(cddd_losses, ecfp_losses, eval_df, save_fig=False):
    """
    Generates cross-validation loss plots with mean curves and confidence intervals
    for both CDDD and ECFP6 models.

    Parameters:
    cddd_losses (dict): Dictionary containing 'train_losses_list' and 'val_losses_list' for CDDD model.
    ecfp_losses (dict): Dictionary containing 'train_losses_list' and 'val_losses_list' for ECFP6 model.
    eval_df (pd.DataFrame): DataFrame containing MSE values for training and validation.

    Returns:
    None (displays a plot)
    """
    # Process CDDD training losses
    padded_data1 = pad_lists(cddd_losses["train_losses_list"])
    mean_values1 = np.nanmean(padded_data1, axis=0)
    std_dev1 = np.nanstd(padded_data1, axis=0)

    # Process CDDD validation losses
    padded_data2 = pad_lists(cddd_losses["val_losses_list"])
    mean_values2 = np.nanmean(padded_data2, axis=0)
    std_dev2 = np.nanstd(padded_data2, axis=0)

    # Process ECFP6 training losses
    ecfp_padded_data1 = pad_lists(ecfp_losses["train_losses_list"])
    ecfp_mean_values1 = np.nanmean(ecfp_padded_data1, axis=0)
    ecfp_std_dev1 = np.nanstd(ecfp_padded_data1, axis=0)

    # Process ECFP6 validation losses
    ecfp_padded_data2 = pad_lists(ecfp_losses["val_losses_list"])
    ecfp_mean_values2 = np.nanmean(ecfp_padded_data2, axis=0)
    ecfp_std_dev2 = np.nanstd(ecfp_padded_data2, axis=0)

    # Define x-values for plotting (epochs)
    x_values = np.arange(len(mean_values1))
    ecfp_x_values = np.arange(len(ecfp_mean_values1))

    # Initialize the figure
    plt.figure(figsize=(14, 16), dpi=600)

    # Plot CDDD training losses
    plt.plot(
        x_values,
        mean_values1,
        label=f"CDDD: MSE(training) = {eval_df.loc[0, 'MSE(train)']}",
        color="blue",
        linewidth=0.5,
    )
    plt.fill_between(
        x_values,
        mean_values1 - std_dev1,
        mean_values1 + std_dev1,
        alpha=0.2,
        color="blue",
    )

    # Plot CDDD validation losses
    plt.plot(
        x_values,
        mean_values2,
        label=f"CDDD: MSE(validation) = {eval_df.loc[0, 'MSE(cv)']}",
        color="cyan",
    )
    plt.fill_between(
        x_values,
        mean_values2 - std_dev2,
        mean_values2 + std_dev2,
        alpha=0.2,
        color="cyan",
    )

    # Plot ECFP6 training losses
    plt.plot(
        ecfp_x_values,
        ecfp_mean_values1,
        label=f"ECFP6: MSE(training) = {eval_df.loc[1, 'MSE(train)']}",
        color="red",
        linewidth=0.5,
    )
    plt.fill_between(
        ecfp_x_values,
        ecfp_mean_values1 - ecfp_std_dev1,
        ecfp_mean_values1 + ecfp_std_dev1,
        alpha=0.2,
        color="red",
    )

    # Plot ECFP6 validation losses
    plt.plot(
        ecfp_x_values,
        ecfp_mean_values2,
        label=f"ECFP6: MSE(validation) = {eval_df.loc[1, 'MSE(cv)']}",
        color="orange",
    )
    plt.fill_between(
        ecfp_x_values,
        ecfp_mean_values2 - ecfp_std_dev2,
        ecfp_mean_values2 + ecfp_std_dev2,
        alpha=0.2,
        color="orange",
    )

    # Customize the plot
    plt.xlabel("Epochs", fontsize=30)
    plt.ylabel("MSE/Loss", fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=30)
    plt.grid(True)

    # Save fiure
    if save_fig == True:
        plt.savefig(r"results/learning_dynamics_og.png", format="png", dpi=1000)

    # Display the plot
    plt.show()


def evaluation_metrics_f1(res_dict_cd, cv_dictio_cddd):
    """
    Computes evaluation metrics for model performance and returns a reordered DataFrame.
    
    Parameters:
    res_dict_cd (dict): Dictionary containing base evaluation results.
    cv_dictio_cddd (dict): Dictionary containing cross-validation and training metrics.
    
    Returns:
    pd.DataFrame: DataFrame with computed metrics, reordered for clarity.
    """
    res_dict_cd_copy = res_dict_cd.copy()

    res_dict_cd_copy['MTS(cv)'] = [[cv_dictio_cddd['train_r2'][0].round(3), cv_dictio_cddd['train_r2'][1].round(3)]]
    res_dict_cd_copy['MSE(train)'] = [[cv_dictio_cddd['train_loss'][0].round(3), cv_dictio_cddd['train_loss'][1].round(3)]]
    res_dict_cd_copy['Q²'] = [[cv_dictio_cddd['val_r2'][0].round(3), cv_dictio_cddd['val_r2'][1].round(3)]]
    res_dict_cd_copy['MSE(cv)'] = [[cv_dictio_cddd['val_loss'][0].round(3), cv_dictio_cddd['val_loss'][1].round(3)]]
    res_dict_cd_copy['MAE(cv)'] = [[cv_dictio_cddd['mae'][0].round(3), cv_dictio_cddd['mae'][1].round(3)]]
    res_dict_cd_copy['RMSE(cv)'] = [[cv_dictio_cddd['rmse'][0].round(3), cv_dictio_cddd['rmse'][1].round(3)]]

    cd_metrics_pd = pd.DataFrame(res_dict_cd_copy).round(3)#.drop(columns = 'MTS(cv)')
    
    # Reorder columns, ensuring new metrics appear first, and drop 'MTS'
    df_reorder = cd_metrics_pd[cd_metrics_pd.columns[-6:].tolist() + cd_metrics_pd.columns[:-6].tolist()].drop(columns='MTS')
    
    return df_reorder


def concatanate_train_val_data(prediction_data_dict, imputation_data_dict):
    """
    Combines training and validation data for prediction and imputation workflows,
    and prints dataset size summaries.
    
    Parameters:
    prediction_data_dict (dict): Dictionary with train/val/test DataFrames for prediction (X, y).
    imputation_data_dict (dict): Dictionary with train/val/test DataFrames for imputation (X, y).
    
    Returns:
    tuple:
        pd.DataFrame: Combined prediction train+val DataFrame.
        pd.DataFrame: Merged imputation DataFrame with non-null values prioritized.
                      Merge combines train and validation matrices; test fields remain empty.
    """
    
    # Concatenate X and y prediction DataFrames
    pred_train_df = pd.concat(
        [prediction_data_dict['dfs']['train']['X'], prediction_data_dict['dfs']['train']['y']], axis=1
    )
    pred_val_df = pd.concat(
        [prediction_data_dict['dfs']['val']['X'], prediction_data_dict['dfs']['val']['y']], axis=1
    )
    pred_train_val_df = pd.concat([pred_train_df, pred_val_df], axis=0)

    # Concatenate X and y imputation DataFrames
    imp_train_df = pd.concat(
        [imputation_data_dict['dfs']['train']['X'], imputation_data_dict['dfs']['train']['y']], axis=1
    )
    imp_val_df = pd.concat(
        [imputation_data_dict['dfs']['val']['X'], imputation_data_dict['dfs']['val']['y']], axis=1
    )

    # Merge on 'SMILES_stand'
    imp_merged = pd.merge(
        imp_train_df, imp_val_df, on='SMILES_stand', how='outer', suffixes=('_df1', '_df2')
    )

    # Column combination using vectorized operations
    cols_to_combine = [col for col in imp_val_df.columns if col != 'SMILES_stand']

    combined_cols = {
        col: imp_merged[f'{col}_df1'].combine_first(imp_merged[f'{col}_df2'])
        for col in cols_to_combine
    }

    # Build final DataFrame
    imp_merged = pd.concat(
        [imp_merged[['SMILES_stand']], pd.DataFrame(combined_cols, index=imp_merged.index)],
        axis=1
    )

    # Summary statistics for prediction data
    print('Post-merging data sizes\n'
          '\n - Prediction data\n'
          '   - Number of compounds in train + val data:', pred_train_val_df.iloc[:, 513:].shape[0],
          '\n   - Number of targets in train + val data:', pred_train_val_df.iloc[:, 513:].shape[1],
          '\n   - Number of filled outputs in train + val matrix:', pred_train_val_df.iloc[:, 513:].notna().sum().sum(),
          '\n   - Number of filled outputs in test matrix:', prediction_data_dict['dfs']['test']['y'].notna().sum().sum(),
          '\n   - Total number of interactions:',
          pred_train_val_df.iloc[:, 513:].notna().sum().sum() + prediction_data_dict['dfs']['test']['y'].notna().sum().sum())

    # Summary statistics for imputation data
    print('\n - Imputation data\n'
          '   - Number of compounds in train + val data:', imp_merged.iloc[:, 513:].shape[0],
          '\n   - Number of targets in train + val data:', imp_merged.iloc[:, 513:].shape[1],
          '\n   - Number of filled outputs in train + val matrix:', imp_merged.iloc[:, 513:].notna().sum().sum(),
          '\n   - Number of filled outputs in test matrix:', imputation_data_dict['dfs']['test']['y'].notna().sum().sum(),
          '\n   - Total number of interactions:',
          imp_merged.iloc[:, 513:].notna().sum().sum() + imputation_data_dict['dfs']['test']['y'].notna().sum().sum())

    return pred_train_val_df, imp_merged


def imputation_data_editor(filled_matrix, data_dict, dataset, device='cuda'):
    """
    Converts an imputed NumPy matrix into a torch tensor aligned with target compounds
    (val/test), and evaluates performance using R² and MSE.

    Parameters:
        filled_matrix (np.ndarray): Imputed matrix from train fit_transform().
        data_dict (dict): Dataset containing dfs and tensors.
        dataset (str): One of {'val', 'test'}.
        device (str): Device for torch tensors.

    Returns:
        tuple: (torch.Tensor, r2, mse)
    """

    # Reference training data structure for columns
    imp_train_df = pd.concat(
        [data_dict['dfs']['train']['X'], data_dict['dfs']['train']['y']], axis=1
    )

    # Build DataFrame with imputed values and proper column names
    filled_df = pd.DataFrame(filled_matrix, columns=imp_train_df.columns[1:])
    filled_df['SMILES_stand'] = imp_train_df['SMILES_stand']
    filled_df = filled_df[['SMILES_stand'] + list(filled_df.columns[:-1])]

    # Keep only Y (target) columns
    filled_targets_df = filled_df.drop(columns=filled_df.columns[1:513])

    # Align imputed matrix to validation/test SMILES
    test_y = data_dict['dfs'][dataset]['y'].copy()
    test_y['SMILES_stand'] = data_dict['dfs'][dataset]['X']['SMILES_stand']
    test_y_sorted = test_y[['SMILES_stand'] + list(test_y.columns[:-1])]

    filled_aligned = filled_targets_df[
        filled_targets_df['SMILES_stand'].isin(test_y_sorted['SMILES_stand'])
    ].set_index('SMILES_stand').loc[test_y_sorted['SMILES_stand']].reset_index(drop=True)

    # Convert to tensor
    X_filled_tensor = torch.tensor(filled_aligned.values, dtype=torch.float32).to(device)

    # Evaluate performance
    r2 = r_squared(X_filled_tensor, data_dict['tensors'][dataset]['y']).item()
    mse = custom_loss(X_filled_tensor, data_dict['tensors'][dataset]['y'].to(device)).item()

    return X_filled_tensor, r2, mse



def fancy_impute_optimized(data_dict, method='SoftImpute', device='cuda', param_grids=None):
    """
    Performs imputation using fancyimpute methods with validation-based hyperparameter optimization.
    Trains on the training set, selects best hyperparameters using the validation set (lowest MSE),
    and evaluates the final imputer on the test set.

    Returns:
        dict with validation/test R², test MSE, and best parameters.
    """

    imputers = {
        'SoftImpute': lambda **kw: SoftImpute(**kw),
        'NuclearNormMinimization': lambda **kw: NuclearNormMinimization(**kw),
        'BiScaler': lambda **kw: BiScaler(**kw),
        'MatrixFactorization': lambda **kw: MatrixFactorization(**kw),
        'IterativeSVD': lambda **kw: IterativeSVD(**kw),
        'KNN': lambda **kw: KNN(**kw)
    }

    if method not in imputers:
        raise ValueError(f"Invalid imputer: {method}")

    # Select parameter grid for the chosen imputer
    if isinstance(param_grids, dict) and method in param_grids:
        param_grid = param_grids[method]
    else:
        param_grid = param_grids  # could be None or a single dict

    # --- Training matrix (X + Y) ---
    X_train = torch.cat(
        (data_dict['tensors']['train']['X'], data_dict['tensors']['train']['y']),
        dim=1
    ).cpu().numpy()

    # --- If no grid is provided ---
    if param_grid is None:
        imputer = imputers[method]()
        imputed_train = imputer.fit_transform(X_train)
        print(f'Param grid not provided for {method}. Using default parameters.')
        # Evaluate directly on val/test
        _, r2_val, mse_val = imputation_data_editor(imputed_train, data_dict, 'val', device=device)
        _, r2_test, mse_test = imputation_data_editor(imputed_train, data_dict, 'test', device=device)
        best_params = {}
    else:
        best_mse_val = np.inf
        best_params = None
        best_r2_val = None

        for params in ParameterGrid(param_grid):
            try:
                imputer = imputers[method](**params)
                imputed_train = imputer.fit_transform(X_train)
                _, r2_val, mse_val = imputation_data_editor(imputed_train, data_dict, 'val', device=device)

                # Optimization criterion: minimize MSE (closer to zero)
                if mse_val < best_mse_val:
                    best_mse_val = mse_val
                    best_r2_val = r2_val
                    best_params = params

            except Exception as e:
                print(f"Skipping params {params} due to error: {e}")
                continue

        print(f"\nBest {method} parameters: {best_params} | Val MSE = {best_mse_val:.6f}, Val R² = {best_r2_val:.4f}")
        imputer = imputers[method](**best_params)
        imputed_train = imputer.fit_transform(X_train)
        _, r2_test, mse_test = imputation_data_editor(imputed_train, data_dict, 'test', device=device)
        mse_val = best_mse_val
        r2_val = best_r2_val

    print(f"{method}: Val MSE = {mse_val:.6f}, Val R² = {r2_val:.4f}, Test R² = {r2_test:.4f}, Test MSE = {mse_test:.6f}")

    return {
        "method": method,
        "best_params": best_params,
        "val_r2": r2_val,
        "test_r2": r2_test,
        "val_mse": mse_val,
        "test_mse": mse_test
    }



def scikit_impute_optimized(data_dict, method='KNNImputer', device='cuda', param_grids=None):
    """
    Performs imputation using scikit-learn imputers with optional validation-based hyperparameter optimization.
    Trains on the training set, selects best hyperparameters using the validation set,
    and evaluates the final imputer on the test set.

    Returns:
        dict with validation/test R², test MSE, and best parameters.
    """

    imputers = {
        'SimpleImputer': lambda **kw: SimpleImputer(**kw),
        'KNNImputer': lambda **kw: KNNImputer(**kw),
        'IterativeImputer': lambda **kw: IterativeImputer(**kw, random_state=0)
    }

    if method not in imputers:
        raise ValueError(f"Invalid imputer: {method}")

    # Select parameter grid for the chosen imputer
    if isinstance(param_grids, dict) and method in param_grids:
        param_grid = param_grids[method]
    else:
        param_grid = param_grids  # could be None or a single dict

    # Training matrix preparation
    X_train = pd.concat([data_dict['dfs']['train']['X'], data_dict['dfs']['train']['y']], axis=1).drop(columns='SMILES_stand').to_numpy()


    # Validation and testing matricase for prediction
    # Validation
    val_place_holder_tens = data_dict['tensors']['val']['y'].clone().float()
    val_place_holder_tens.fill_(float('nan'))
    val_array_empty_tens = torch.cat((data_dict['tensors']['val']['X'], val_place_holder_tens), dim=1)
    X_Y_empty_val = val_array_empty_tens.cpu().numpy()


    # Test
    place_holder_tens = data_dict['tensors']['test']['y'].clone().float()
    place_holder_tens.fill_(float('nan'))
    test_array_tens = torch.cat((data_dict['tensors']['test']['X'], place_holder_tens), dim=1)
    X_Y_empty_test = test_array_tens.cpu().numpy()

    # If no grid is provided: use default imputer
    if param_grid is None:
        imputer = imputers[method]()
        imputer.fit(X_train)
    else:
        # Hyperparameter optimization on validation set
        best_r2_val, best_params = -np.inf, None

        for params in ParameterGrid(param_grid):
            imputer = imputers[method](**params)
            imputer.fit(X_train)

            imputed_val = imputer.transform(X_Y_empty_val)
            y_val_pred = torch.from_numpy(imputed_val[:, -data_dict['tensors']['val']['y'].shape[1]:]).float().to(device)

            r2_val = r_squared(y_val_pred, data_dict['tensors']['val']['y']).item()
            if r2_val > best_r2_val:
                best_r2_val = r2_val
                best_params = params

        print(f"\nBest {method} parameters: {best_params} | Val R² = {best_r2_val:.4f}")

        # Retrain on train+val with best params before testing
        imputer = imputers[method](**best_params)
        imputer.fit(X_train)

    # Evaluate on validation and test
    imputed_val = imputer.transform(X_Y_empty_val)
    imputed_test = imputer.transform(X_Y_empty_test)

    y_val_pred = torch.from_numpy(imputed_val[:, -data_dict['tensors']['val']['y'].shape[1]:]).float().to(device)
    y_test_pred = torch.from_numpy(imputed_test[:, -data_dict['tensors']['test']['y'].shape[1]:]).float().to(device)

    r2_val = r_squared(y_val_pred, data_dict['tensors']['val']['y']).item()
    r2_test = r_squared(y_test_pred, data_dict['tensors']['test']['y']).item()
    mse_val = custom_loss(y_val_pred, data_dict['tensors']['val']['y']).item()
    mse_test = custom_loss(y_test_pred, data_dict['tensors']['test']['y']).item()

    print(f"{method}: Test R² = {r2_test:.4f}, MSE = {mse_test:.4f}")

    return {
        "method": method,
        "best_params": None if param_grid is None else best_params,
        "val_r2": r2_val,
        "test_r2": r2_test,
        "val_mse": mse_val,
        "test_mse": mse_test
    }




def mask_none_predictions(true_y, pred_y_tens):
    """
    Masks out NaN values in true_y from predictions.

    Parameters:
    true_y (pd.DataFrame): True target values.
    pred_y_tens (torch.Tensor): Model predictions in tensor format.

    Returns:
    pd.DataFrame: Predictions with only non-NaN values retained.
    """
    # Convert tensor predictions to DataFrame with matching indices and column names
    Y_test_pred = pd.DataFrame(
        pred_y_tens.cpu().numpy(), columns=true_y.columns, index=true_y.index
    )

    # Apply mask to retain only non-NaN values from true_y
    y_mask = true_y.notna()
    y_test_nona = Y_test_pred[y_mask]

    return y_test_nona


def error_indices(Y_test, Y_test_pred_nona, outlier_tr, inlier_tr):
    """
    Identifies rows with high and low prediction errors based on thresholds.

    Parameters:
    Y_test (pd.DataFrame): True target values.
    Y_test_pred_nona (pd.DataFrame): Non-NaN predicted values.
    outlier_tr (float): Threshold for defining high errors.
    inlier_tr (float): Threshold for defining low errors.

    Returns:
    tuple: (Rows with high errors, Rows with low errors)
    """
    # Compute absolute error between true and predicted values
    error_df = (Y_test - Y_test_pred_nona).abs()

    # Identify high and low error rows based on thresholds
    rows_with_high_error = error_df[error_df.gt(outlier_tr).any(axis=1)].dropna(
        axis=1, how="all"
    )
    rows_with_low_error = error_df[error_df.lt(inlier_tr).any(axis=1)].dropna(
        axis=1, how="all"
    )

    return rows_with_high_error, rows_with_low_error


def locate_outlier_inlier(analyzed_target, Y_test_pred_nona, data, hev_rows, lev_rows):
    """
    Locates and retrieves data for the highest and lowest error instances.

    Parameters:
    analyzed_target (str): Target variable being analyzed.
    Y_test_pred_nona (pd.DataFrame): Non-NaN predicted values.
    data (dict): Dictionary containing test dataset information.
    hev_rows (pd.DataFrame): Rows with high errors.
    lev_rows (pd.DataFrame): Rows with low errors.

    Returns:
    tuple: (Data for highest error instance, Data for lowest error instance, Predicted values for each)
    """
    # Identify highest and lowest error values for 'SLC6A4'
    highest_er = hev_rows["SLC6A4"].max()
    lowest_er = lev_rows["SLC6A4"].min()

    highest_er_id = hev_rows["SLC6A4"].idxmax()
    lowest_er_id = lev_rows["SLC6A4"].idxmin()

    # Retrieve predicted values for the highest and lowest error instances
    hev_pred_pc = Y_test_pred_nona.loc[highest_er_id, analyzed_target]
    lev_pred_pc = Y_test_pred_nona.loc[lowest_er_id, analyzed_target]

    # Locate corresponding SMILES string in the dataset
    hev_smiles = data["dfs"]["test"]["X"].loc[highest_er_id, "SMILES_stand"]
    lev_smiles = data["dfs"]["test"]["X"].loc[lowest_er_id, "SMILES_stand"]

    # Retrieve full rows matching the identified SMILES strings
    hev_row = data["dfs"]["test"]["X"][
        data["dfs"]["test"]["X"]["SMILES_stand"] == hev_smiles
    ]
    lev_row = data["dfs"]["test"]["X"][
        data["dfs"]["test"]["X"]["SMILES_stand"] == lev_smiles
    ]

    # Print details about high and low error instances
    print(
        "High error value (HEV):",
        highest_er,
        "| HEV index:",
        highest_er_id,
        "| HEV actual value:",
        data["dfs"]["test"]["y"].loc[highest_er_id, analyzed_target],
        "| HEV predicted value:",
        hev_pred_pc,
    )
    print(
        "Low error value (LEV):",
        lowest_er,
        "| LEV index:",
        lowest_er_id,
        "| LEV actual value:",
        data["dfs"]["test"]["y"].loc[lowest_er_id, analyzed_target],
        "| LEV predicted value:",
        lev_pred_pc,
    )

    return hev_row, lev_row, hev_pred_pc, lev_pred_pc


def find_nearest_neighbors(data, row_ind, n_neighbors=4):
    """
    Find the nearest neighbors of a given data point in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    point_index (int): The index of the data point for which to find the nearest neighbors.
    n_neighbors (int): The number of nearest neighbors to find (default is 4).

    Returns:
    pd.DataFrame: A DataFrame containing the nearest neighbors.
    """
    X_train_y_train = pd.concat(
        [data["dfs"]["train"]["X"], data["dfs"]["train"]["y"]], axis=1
    )
    neighbour_search_df = pd.concat([X_train_y_train, row_ind], ignore_index=True)

    # Initialize and fit the NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(
        neighbour_search_df.iloc[:, 1:513]
    )

    # Find the nearest neighbors
    distances, indices = nbrs.kneighbors(neighbour_search_df.iloc[:, 1:513])

    # Get the indices and distances of the nearest neighbors for the specified data point
    point_index = neighbour_search_df.iloc[-1].name
    nearest_indices = indices[point_index]
    nearest_distances = distances[point_index]

    # Return the nearest neighbors DataFrame and distances
    nearest_neighbors_df = neighbour_search_df.iloc[
        nearest_indices
    ]  # .reset_index(drop=True)

    return nearest_neighbors_df


def nn_editor(
    nearest_neighbors_df, final_data_sem, analyzed_target, hev_pred_pc, cddd_matrix
):
    nn_smiles = nearest_neighbors_df["SMILES_stand"].tolist()
    nn_dict = {"SMILES_stand": [], "compound_chembl_id": [], "pchembl": []}
    for smi in nn_smiles:
        nn_dict["SMILES_stand"].append(smi)
        nn_dict["compound_chembl_id"].append(
            final_data_sem[final_data_sem["SMILES_stand"] == smi][
                "compound_chembl_id"
            ].iloc[0]
        )
        nn_dict["pchembl"].append(
            cddd_matrix[cddd_matrix["SMILES_stand"] == smi][analyzed_target].iloc[0]
        )

    nn_dict["pred_pchembl"] = [hev_pred_pc, np.nan, np.nan, np.nan]

    return nn_dict


from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdMolTransforms
import os


def plot_molecule_data(data_dict, ev, dpi=600, save=False):
    """
    Plot molecules with their respective data in subplots.

    Parameters:
        data_dict (dict): Dictionary with keys 'SMILES_stand', 'compound_chembl_id',
                          'pchembl', and 'pred_pchembl'. Values should be lists of equal length.
        dpi (int): Dots per inch for the molecule images.
    """
    # Extract data from the dictionary
    smiles_list = data_dict.get("SMILES_stand", [])
    chembl_ids = data_dict.get("compound_chembl_id", [])
    pchembl_values = data_dict.get("pchembl", [])
    pred_pchembl_values = data_dict.get("pred_pchembl", [])

    if not all(
        len(lst) == len(smiles_list)
        for lst in [chembl_ids, pchembl_values, pred_pchembl_values]
    ):
        raise ValueError("All lists in the dictionary must have the same length.")

    num_molecules = len(smiles_list)
    n_cols = 3  # Number of columns for the subplots
    n_rows = (
        num_molecules + n_cols - 1
    ) // n_cols  # Calculate the required number of rows

    n_cols = 1  # Number of columns for the subplots
    n_rows = num_molecules  # Calculate the required number of rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()  # Flatten the axes for easy indexing

    for i, smiles in enumerate(smiles_list):
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES string at index {i}: {smiles}")
            continue

        # Generate 2D coordinates and apply rotation
        rdDepictor.Compute2DCoords(mol, canonOrient=True)
        rot_y = np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        rdMolTransforms.TransformConformer(mol.GetConformer(0), rot_y)

        # Draw molecule
        img = Draw.MolToImage(mol, size=(1000, 500), dpi=dpi)
        if ev == "HEV" and i == 0:
            file_path = os.path.join("results", f"hev_structure.png")
            # plt.savefig(file_path, format='tiff', dpi=600)
        elif ev == "HEV" and i != 0:
            file_path = os.path.join("results", f"hev_neigbor_{i + 1}.png")
            # plt.savefig(file_path, format='tiff', dpi=600)
        elif ev == "LEV" and i == 0:
            file_path = os.path.join("results", f"lev_structure.png")
            # plt.savefig(file_path, format='tiff', dpi=600)
        elif ev == "LEV" and i != 0:
            file_path = os.path.join("results", f"lev_neigbor_{i + 1}.png")
            # plt.savefig(file_path, format='tiff', dpi=600)

        img.save(file_path, format="png", dpi=(600, 600))

        # Plot the image
        ax = axes[i]
        ax.imshow(img)
        ax.axis("off")  # Remove axis

        # Add labels below the image
        label = (
            f"ChEMBL ID: {chembl_ids[i]}\n"
            f"pChEMBL: {pchembl_values[i]}\n"
            f"Predicted: {pred_pchembl_values[i]}"
        )
        ax.set_title(label, fontsize=20)

    # Remove any unused subplots
    for j in range(num_molecules, len(axes)):
        axes[j].axis("off")

    # Adjust layout
    plt.tight_layout()

    if ev == "HEV" and save==True:
        plt.savefig(r"results/HEV_compound.png", format="png", dpi=1000)
    elif ev == "LEV" and save==True:
        plt.savefig(r"results/LEV_compound.png", format="png", dpi=1000)
    else:
        print("Method for saving 'save' was not set and no image was saved.")

    plt.show()


from functools import reduce
import random
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def cv_evaluation_processing(list_of_dicts):
    """
    Processes a list of dictionaries into a list of DataFrames, each containing cross-validation results.

    Args:
        list_of_dicts (list): List of dictionaries with cross-validation results.

    Returns:
        list: List of DataFrames where each DataFrame corresponds to a fold.
    """
    cv_r2_df_list = []
    for idx, data_dict in enumerate(list_of_dicts, start=1):
        df = pd.DataFrame(data_dict, index=[0]).T.reset_index()
        df.columns = ["SLC_name", f"k-{idx}"]
        cv_r2_df_list.append(df)

    return cv_r2_df_list


def plot_confidence_per_target(cv_r2_df_list, excluded_targets):
    """
    Merges cross-validation results and generates a box plot of Q² values per target.
    
    Args:
        cv_r2_df_list (list): List of DataFrames with cross-validation results.
        
    Returns:
        pd.DataFrame: Processed DataFrame with transposed values for visualization.
    """
    cv_r2_merged_df = reduce(lambda left, right: pd.merge(left, right, on='SLC_name', how='inner'), cv_r2_df_list)
    cv_r2_merged_df.set_index('SLC_name', inplace=True)
    df_values = cv_r2_merged_df.T
    
    # Exclude problematic target
    df_filtered = df_values.drop(columns=excluded_targets, errors='ignore')

    # Determine y-axis limits dynamically
    ymin = np.floor(df_filtered.min().min() - 0.1)
    ymax = np.ceil(1.05)
    
    # Box plot configuration
    # Dynamic y-limits
    ymin = np.floor(df_filtered.min().min() - 0.1)
    ymax = 1.05  # ensures 1.0 shows

    # Create tick positions in steps of 0.25
    yticks = np.arange(ymin, ymax + 0.01, 0.25)

    # Plot
    plt.figure(figsize=(25, 15), dpi=600)
    df_filtered.boxplot()
    plt.xlabel('Targets', fontsize=30)
    plt.ylabel('Q² values', fontsize=30)
    plt.xticks(rotation=45, fontsize=25)
    plt.yticks(yticks, fontsize=15)
    plt.ylim(ymin, ymax)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df_values


def data_content(tabular_data, nuft):
    """
    Processes tabular data to count target and family occurrences.

    Args:
        tabular_data (pd.DataFrame): DataFrame containing 'target_name' and 'family' columns.
        nuft (pd.DataFrame): DataFrame containing 'target_name' and 'family' columns.

    Returns:
        pd.DataFrame: DataFrame with target and family compound counts.
    """
    target_counts_df = tabular_data["target_name"].value_counts().reset_index()
    family_counts_df = tabular_data["family"].value_counts().reset_index()

    target_counts_df_ucnf = target_counts_df.merge(
        nuft[["target_name", "family"]], on="target_name", how="inner"
    )
    result_df = target_counts_df_ucnf.merge(family_counts_df, on="family", how="inner")
    result_df.columns = ["target_name", "comp_count_targ", "family", "comp_count_fam"]

    return result_df


def check_target_content(targ_fam_counts_df, target_list):
    """
    Prints compound and family statistics for given targets.

    Args:
        targ_fam_counts_df (pd.DataFrame): DataFrame containing target and family compound counts.
        target_list (list): List of target names to check.
    """
    for target in target_list:
        target_data = targ_fam_counts_df[targ_fam_counts_df["target_name"] == target]
        if not target_data.empty:
            comp_count = target_data["comp_count_targ"].values[0]
            family = target_data["family"].values[0]
            family_data = targ_fam_counts_df[targ_fam_counts_df["family"] == family]
            family_comp_count = family_data["comp_count_fam"].iloc[0]
            print(
                f"{target} has {comp_count} compounds | Belongs to {family} family | Family has {family_data.shape[0]} members total | Family has {family_comp_count} compounds"
            )


def shuffle_list_with_seed(input_list, seed):
    """
    Shuffles a list in a reproducible manner using a given seed.

    Args:
        input_list (list): List to be shuffled.
        seed (int): Seed for reproducibility.

    Returns:
        list: Shuffled list.
    """
    random.seed(seed)
    shuffled = input_list[:]
    random.shuffle(shuffled)
    return shuffled


def take_compound_sample_targ(tabular_data, target_list, seed):
    """
    Selects a random sample of 5 compounds per target.

    Args:
        tabular_data (pd.DataFrame): DataFrame with 'target_name' and 'SMILES_stand'.
        target_list (list): List of targets.
        seed (int): Seed for reproducibility.

    Returns:
        dict: Dictionary of targets with sampled compound lists.
    """
    return {
        targ: shuffle_list_with_seed(
            tabular_data[tabular_data["target_name"] == targ]["SMILES_stand"].tolist(),
            seed,
        )[:5]
        for targ in target_list
    }


def draw_molecule_from_smiles(smiles, dpi=1000):
    """
    Draws a molecular structure from a SMILES string.

    Args:
        smiles (str): SMILES representation of a molecule.
        dpi (int): Image resolution.

    Raises:
        ValueError: If the SMILES string is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    rdDepictor.Compute2DCoords(mol, canonOrient=True)
    rdMolTransforms.TransformConformer(
        mol.GetConformer(0),
        np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    img = Draw.MolToImage(mol, size=(500, 250), dpi=dpi)
    display(img)


def gen_array(table_df, cddd_matrix):
    """
    Extracts descriptors and SMILES from a DataFrame.

    Args:
        table_df (pd.DataFrame): Unused but retained for consistency.
        cddd_matrix (pd.DataFrame): DataFrame with descriptors.

    Returns:
        tuple: (SMILES series, descriptor numpy array)
    """
    return cddd_matrix.iloc[:, 0], cddd_matrix.iloc[:, 1:513].to_numpy()


# Function for dimensionality reduction
def dimension_reduction(array, model_type):
    """
    Perform dimensionality reduction using UMAP, PCA, or t-SNE.

    Parameters:
    - array (np.ndarray): Input array for dimensionality reduction
    - model_type (str): Type of model to use ('umap', 'pca', 'tsne')

    Returns:
    - np.ndarray: Transformed embeddings
    - model: Trained dimensionality reduction model
    """
    if model_type == "umap":
        umap_model = umap.UMAP(
            metric="euclidean",
            n_neighbors=200,  # (1-200)
            n_components=2,
            low_memory=False,
            min_dist=0.99,  # (0-1)
            random_state=42,
            n_jobs=1,
            learning_rate=0.2,
        )

        embeddings = umap_model.fit_transform(array)
        return embeddings, umap_model

    elif model_type == "pca":
        pca = PCA(n_components=2)
        embeddings = pca.fit_transform(array)
        return embeddings, pca

    elif model_type == "tsne":
        tsne = TSNE(n_components=2, random_state=42)
        embeddings = tsne.fit_transform(array)
        return embeddings, tsne

    else:
        raise ValueError("Invalid model type. Choose from 'umap', 'pca', or 'tsne'.")


def add_embeddings(emb, just_smiles, tabular_df_family):
    smi_emb = pd.DataFrame(just_smiles.copy())
    smi_emb["emb_x"] = emb[:, 0]
    smi_emb["emb_y"] = emb[:, 1]

    nfp_smi_emb = pd.merge(tabular_df_family, smi_emb, on="SMILES_stand", how="inner")
    return nfp_smi_emb


def chem_space_scatter_plot(df: pd.DataFrame, targets_ls) -> None:
    """
    Plots a scatter plot of embeddings for two groups based on the 'Cholestasis' column.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the following columns:
            - 'Cholestasis': Binary indicator (0 or 1).
            - 'x_embeddings': x-coordinates for plotting.
            - 'y_embeddings': y-coordinates for plotting.

    Returns:
        None: Displays a scatter plot.
    """
    # Separate the data based on 'Cholestasis' values
    targ_1 = df[df["target_name"] == targets_ls[0]]
    targ_2 = df[df["target_name"] == targets_ls[1]]
    targ_3 = df[df["target_name"] == targets_ls[2]]

    other = df[
        (df["target_name"] != targets_ls[0])
        & (df["target_name"] != targets_ls[1])
        & (df["target_name"] != targets_ls[2])
    ]

    # Plot parameters
    dot_size = 40
    colors = {"1": "cyan", "2": "red", "3": "orange", "4": "grey"}

    plt.figure(figsize=(8, 9), dpi=600)

    other_s = plt.scatter(
        other["emb_y"],
        other["emb_x"],
        c=colors["4"],
        s=dot_size,
        edgecolors="black",
        zorder=3,
        alpha=0.8,
        label="other",
    )

    targ_1_s = plt.scatter(
        targ_1["emb_y"],
        targ_1["emb_x"],
        c=colors["1"],
        s=dot_size,
        edgecolors="black",
        zorder=3,
        alpha=0.8,
        label=targets_ls[0],
    )
    targ_2_s = plt.scatter(
        targ_2["emb_y"],
        targ_2["emb_x"],
        c=colors["2"],
        s=dot_size,
        edgecolors="black",
        zorder=3,
        alpha=0.8,
        label=targets_ls[1],
    )

    targ_3_s = plt.scatter(
        targ_3["emb_y"],
        targ_3["emb_x"],
        c=colors["3"],
        s=dot_size,
        edgecolors="black",
        zorder=3,
        alpha=0.95,
        label=targets_ls[2],
    )

    # Axis labels and legend
    plt.xlabel("x embeddings", fontsize=25)
    plt.ylabel("y embeddings", fontsize=25)
    plt.legend(loc="lower left", fontsize=19)

    # Get current ticks
    current_xticks = plt.gca().get_xticks()
    current_yticks = plt.gca().get_yticks()

    # Shift ticks by subtracting 1
    plt.xticks(current_xticks - 2, fontsize=20)
    plt.yticks(current_yticks, fontsize=20)

    plt.savefig(r"results/chemical_space.png", format="png", dpi=600)

    # Show the plot
    plt.tight_layout()
    plt.show()
