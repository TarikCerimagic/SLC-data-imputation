import pandas as pd
import torch
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.model_selection import train_test_split

import numpy as np
import random
import torch
import pandas as pd

__all__ = ["MatrixSplitter"]


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


def get_scaffold(smi_row):
    """
    Get the scaffold SMILES string for a given SMILES row.

    Parameters:
    smi_row: The SMILES row.

    Returns:
    The scaffold SMILES string.
    """
    scf = MurckoScaffoldSmiles(smi_row)
    return scf


def count_scaffolds(df):
    """
    Count the occurrences of each scaffold in the dataframe.

    Parameters:
    df: The dataframe containing scaffold information.

    Returns:
    A dataframe with scaffold counts.
    """
    count_dict = {}

    for scf in df["scaffold"]:
        if scf not in count_dict:
            count_dict[scf] = 1
        else:
            count_dict[scf] += 1

    count_df = pd.DataFrame(
        list(count_dict.items()), columns=["scaffold", "scf_count"]
    ).sort_values(by="scf_count", ascending=False)
    return count_df


def add_scf_columns(df):
    df_copy = df.copy()
    df_copy["scaffold"] = df_copy["SMILES_stand"].apply(get_scaffold)
    scaffold_count_df = count_scaffolds(df_copy)

    combined_df = pd.merge(scaffold_count_df, df_copy, on="scaffold", how="inner")
    combined_df_copy = combined_df.copy()
    combined_df_copy.loc[combined_df_copy["scf_count"] <= 4, "scaffold"] = (
        "below 5 members"
    )

    return combined_df_copy


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    #if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


class MatrixSplitter:
    """
    Initialize a MatrixSplitter.

    Parameters:
    device: The device to move tensors to.
    """

    def __init__(self, device="cuda"):
        self.device = device

    def target_based_split(self, cleaned_dataset, num_of_clean_targets):
        """
        Perform a stratified split of the dataset based on the target label and convert to tensors.

        Parameters:
        cleaned_dataset: pandas DataFrame
            The dataset to be split.
        num_of_clean_targets: int
            The number of target columns in the dataset.

        Returns:
        dict:
            A nested dictionary containing DataFrames and tensors structured by dataset type (train, test, val).
        """
        if cleaned_dataset is None or num_of_clean_targets is None:
            return None

        # Perform stratified split based on 'new_SLC_label'
        train_targ_df, test_targ_df_temp = train_test_split(
            cleaned_dataset,
            test_size=0.2,
            stratify=cleaned_dataset["new_target_label"],
            random_state=42,
        )
        test_targ_df, val_targ_df = train_test_split(
            test_targ_df_temp,
            test_size=0.5,
            stratify=test_targ_df_temp["new_target_label"],
            random_state=42,
        )

        datasets = {"train": train_targ_df, "test": test_targ_df, "val": val_targ_df}
        data_dict = {"dfs": {}, "tensors": {}}

        for split, df in datasets.items():
            df_dropped = df.drop(columns=["SMILES_stand", "new_target_label"])
            x_df = df_dropped.iloc[:, :-num_of_clean_targets]
            y_df = df_dropped.iloc[:, -num_of_clean_targets:]

            if split == "train":
                check_target_content(y_df)

            x_tensor = torch.tensor(
                x_df.values, dtype=torch.float32, device=self.device
            )
            y_tensor = torch.tensor(
                y_df.values, dtype=torch.float32, device=self.device
            )

            df_dropped_label = df.drop(columns=["new_target_label"])
            x_df_smi = df_dropped_label.iloc[:, :-num_of_clean_targets]

            data_dict["dfs"][split] = {"X": x_df_smi, "y": y_df}
            data_dict["tensors"][split] = {"X": x_tensor, "y": y_tensor}

        return data_dict

    def transductive_split(self, df, descs, trh=2, remove_empty_targets=False):
        """
        Perform a transductive split of the dataset and return a nested dictionary for structured data access.

        Parameters:
        df: pandas DataFrame
            The dataset to be split.
        descs: pandas DataFrame
            A DataFrame containing descriptors for the compounds, indexed by 'SMILES_stand'.
        trh: int, default=2
            The minimum occurrence threshold for a compound (based on 'SMILES_stand') to be included in the split.
        remove_empty_targets: Whether to remove target columns that contain only NaNs.


        Returns:
        dict:
            A nested dictionary with structured access to both pandas DataFrames and tensors.
        """

        # Count occurrences of each unique 'SMILES_stand' value
        counts = df["SMILES_stand"].value_counts()

        # Select compounds that appear at least 'trh' times
        counted_comps_df = counts[counts >= trh].reset_index()

        # Merge back to the original dataframe to filter relevant compounds
        original_comp_count_df = df.merge(
            counted_comps_df, on="SMILES_stand", how="inner"
        ).sort_values(by="SMILES_stand")

        # Select one representative entry for each compound to be used in the test set
        duplicates_for_test = original_comp_count_df[
            ~original_comp_count_df.duplicated(subset=["SMILES_stand"], keep="last")
        ]
        duplicates_for_test_no_count = duplicates_for_test.drop(columns="count")

        # Merge with the original dataset and keep only the compounds not selected for the test set
        merged_template = df.merge(
            duplicates_for_test_no_count,
            on=["SMILES_stand", "target_name", "pchembl"],
            how="left",
            indicator=True,
        )
        df_filtered_transductive = merged_template[
            merged_template["_merge"] == "left_only"
        ].drop(columns=["_merge"])

        # Combine the remaining dataset (train candidates) and test set into a single structure
        trans_combined = pd.concat(
            [
                df_filtered_transductive.pivot(
                    index="SMILES_stand", columns="target_name", values="pchembl"
                ).reset_index(),
                duplicates_for_test_no_count.pivot(
                    index="SMILES_stand", columns="target_name", values="pchembl"
                ).reset_index(),
            ],
            ignore_index=True,
            sort=False,
        )

        # Separate training data
        trans_combined_train = trans_combined.iloc[
            : -duplicates_for_test_no_count.shape[0]
        ]

        # Shuffle and split the remaining data into validation and test sets
        trans_combined_val_test = (
            trans_combined.iloc[-duplicates_for_test_no_count.shape[0] :]
            .sample(frac=1, random_state=40)
            .reset_index(drop=True)
        )
        trans_combined_val = trans_combined_val_test.iloc[
            : len(trans_combined_val_test) // 2
        ]
        trans_combined_test = trans_combined_val_test.iloc[
            len(trans_combined_val_test) // 2 :
        ]

        # Organize data into train, validation, and test splits
        dataset_splits = {
            "train": trans_combined_train,
            "val": trans_combined_val,
            "test": trans_combined_test,
        }
        data_dict = {}

        # Get the number of descriptor columns
        desc_length = len(descs.columns)

        # Merge each split with descriptor data and create final structured output
        for split, split_df in dataset_splits.items():
            merged_df = (
                descs.merge(split_df, on="SMILES_stand", how="inner")
                .sample(frac=1, random_state=40)
                .reset_index(drop=True)
            )
            data_dict[split] = {
                "X": merged_df.iloc[:, :desc_length],  # Extract descriptor features
                "y": merged_df.iloc[:, desc_length:],  # Extract target values
            }

        return self._imputation_post_processing(data_dict, remove_empty_targets)

    def _imputation_post_processing(self, data_dict, remove_empty_targets):
        X_train, Y_train = data_dict["train"]["X"], data_dict["train"]["y"]
        X_val, Y_val = data_dict["val"]["X"], data_dict["val"]["y"]
        X_test, Y_test = data_dict["test"]["X"], data_dict["test"]["y"]

        # Identify target columns from validation and test sets
        if remove_empty_targets:
            targets_val = Y_val.dropna(axis=1, how="all").columns.tolist()
            targets_test = Y_test.dropna(axis=1, how="all").columns.tolist()
        else:
            targets_val = Y_val.columns.tolist()
            targets_test = Y_test.columns.tolist()

        # Unique list of target columns to retain in training
        target_columns = list(dict.fromkeys(targets_val + targets_test))

        # Filter train dataset based on identified target columns
        Y_train_filtered = Y_train.filter(items=target_columns, axis=1)
        X_Y_train = pd.concat([X_train, Y_train_filtered], axis=1)

        # Remove rows with all NaNs in the target columns
        X_Y_train_cleaned = X_Y_train.dropna(subset=Y_train_filtered.columns, how="all")

        # Convert to tensors
        def to_tensor(df):
            return torch.tensor(df.values, dtype=torch.float32).to(self.device)

        desc_length = len(data_dict["train"]["X"].columns)

        imp_df_dict = {
            "train": {
                "X": X_Y_train_cleaned.iloc[:, :desc_length],
                "y": X_Y_train_cleaned.iloc[:, desc_length:],
            },
            "val": {"X": X_val, "y": Y_val[target_columns]},
            "test": {"X": X_test, "y": Y_test[target_columns]},
        }

        imp_tens_dict = {
            "train": {
                "X": to_tensor(X_Y_train_cleaned.iloc[:, 1:desc_length]),
                "y": to_tensor(X_Y_train_cleaned.iloc[:, desc_length:]),
            },
            "val": {
                "X": to_tensor(X_val.iloc[:, 1:desc_length]),
                "y": to_tensor(Y_val[target_columns]),
            },
            "test": {
                "X": to_tensor(X_test.iloc[:, 1:desc_length]),
                "y": to_tensor(Y_test[target_columns]),
            },
        }

        return {"dfs": imp_df_dict, "tensors": imp_tens_dict}

