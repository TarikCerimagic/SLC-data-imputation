import numpy as np
import pandas as pd
from chembl_webresource_client import new_client
from scipy.stats import sem


class DataFetcher:
    """
    A class to fetch, clean, and merge data from different sources,
    and retrieve ChEMBL data based on UniProt IDs.
    """

    def __init__(self):
        self.client = new_client

    @staticmethod
    def extract_value(target_components):
        """
        Extracts the 'accession' value from a dictionary in a list.
        """
        if isinstance(target_components, list) and target_components:
            return target_components[0].get("accession", "Value not found")
        return "Value not found"

    def get_targets_by_uniprot(self, cleaned_merged_df):
        """
        Fetches ChEMBL target data based on UniProt IDs from the cleaned dataframe.
        """
        if cleaned_merged_df is None:
            raise ValueError(
                "Cleaned dataframe is not available. Call `clean_merge` first."
            )

        result_df = pd.DataFrame()
        error_ids = []

        for i, uni_id in enumerate(cleaned_merged_df["UniProt_ID"]):
            try:
                res = self.client.target.filter(target_components__accession=uni_id)
                target_df = pd.DataFrame(res)
                result_df = pd.concat([result_df, target_df], ignore_index=True)
            except Exception as e:
                error_ids.append(uni_id)
                print(f"Error with UniProt ID {uni_id}: {e}")

            print(f"Processed {i + 1}/{len(cleaned_merged_df)} UniProt IDs")

        result_df["UniProt_ID"] = result_df["target_components"].apply(
            DataFetcher.extract_value
        )
        return result_df, error_ids


def merge_resolute_with_uniprot(resolute_df, uniprot_df):
    """Cleans and merges resolute and UniProt data on 'HGNC' column."""
    # Clean UniProt 'HGNC' column and filter reviewed entries
    uniprot_df["HGNC"] = uniprot_df["HGNC"].str.split(";").str[0]
    uni_df_reviewed = uniprot_df[uniprot_df["Reviewed"] == "reviewed"]

    # Merge resolute and UniProt data
    merged_df = pd.merge(resolute_df, uni_df_reviewed, on="HGNC", how="inner")
    merged_df = merged_df.drop_duplicates(subset="HGNC", keep="first")
    merged_df = merged_df.rename(columns={"Entry": "UniProt_ID"})

    # self.cleaned_merged_df = merged_df
    return merged_df


def filter_mutant_targets(filtered_data, substrings_to_search):
    """
    Filters out mutant targets from the filtered assay data.
    """
    assay_descriptions = filtered_data["assay_description"].unique()
    matched_strings = []

    for value in assay_descriptions:
        for substring in substrings_to_search:
            if substring in value:
                matched_strings.append(value)

    matched_strings = list(set(matched_strings))
    print(
        "Assay descriptions containing 'muta' or 'recombi' as a substring: ",
        len(matched_strings),
    )

    filtered_mutants = filtered_data[
        ~filtered_data["assay_description"].str.contains("|".join(matched_strings))
    ]
    print(
        "DataFrame shape without the mutant target samples: ",
        filtered_mutants.shape,
    )

    return filtered_mutants


def threshold_and_average_activities(data, threshold, method):
    """Replaces duplicates with average values.

    Values that are spread wider than a certain threshold are instead
    removed from the dataset.

    """

    def calculate_deviation(data):
        return (
            sem(data, ddof=0)
            if (method == "sem" and len(data) > 1)
            else np.std(data, ddof=0)
            if method == "std"
            else 0
        )

    df_cleaned = (
        data.assign(
            pchembl=data.groupby(["compound_chembl_id", "target_chembl_id"])[
                "pchembl"
            ].transform(
                lambda g: np.nan if calculate_deviation(g) >= threshold else np.mean(g)
            ),
        )
        .dropna(subset=["pchembl"])
        .drop_duplicates(["SMILES_stand", "target_chembl_id"])
    )

    df_cleaned["chemblid_count"] = df_cleaned["compound_chembl_id"].map(
        df_cleaned["compound_chembl_id"].value_counts().to_dict()
    )

    most_common_compound_ids = df_cleaned.sort_values(
        by=["chemblid_count"], ascending=False
    ).drop_duplicates(["SMILES_stand"], keep="first")["compound_chembl_id"]

    return df_cleaned[
        df_cleaned["compound_chembl_id"].isin(most_common_compound_ids)
    ].reset_index(drop=True)


def sample_matrix(data, min_target_threshold=None, random_state=42):
    """Constructs the data matrix.

    Optionally filters targets that are rarer than a given threshold.

    """
    sampled_matrix = (
        pd.merge(
            data.drop_duplicates(["SMILES_stand"], keep="first")[["SMILES_stand"]],
            data.pivot_table(
                index="SMILES_stand",
                columns="target_name",
                values="pchembl",
                fill_value=None,
            ).reset_index(),
            on=["SMILES_stand"],
        )
        .sample(frac=1, random_state=random_state)
        .set_index(["SMILES_stand"])
    )

    if min_target_threshold is not None:
        sampled_matrix = sampled_matrix.loc[
            :, (~sampled_matrix.isna()).sum(axis=0) > min_target_threshold
        ]

    return sampled_matrix.dropna(how="all").reset_index()


def find_best_matrix_label(matrix, min_label_count):
    """Selects labels based on the output profile of compounds.

    In cases where multiple targets are available, the one with the
    least number of compounds in the dataset is selected.

    """
    target_counts = (~matrix.set_index("SMILES_stand").isna()).sum()

    def get_rarest_target_name(target_names):
        best_target_name = None

        for target_name in target_names:
            if (
                best_target_name is None
                or target_counts[best_target_name] >= target_counts[target_name]
            ):
                best_target_name = target_name

        return best_target_name

    target_labels = (
        matrix.set_index(["SMILES_stand"])
        .apply(lambda row: list(row[~row.isna()].index), axis=1)
        .map(get_rarest_target_name)
        .reset_index()
        .rename(columns={0: "new_target_label"})
    )

    if min_label_count is not None:
        target_label_counts = target_labels["new_target_label"].value_counts()

        target_labels["new_target_label"] = target_labels["new_target_label"].map(
            lambda label: label
            if target_label_counts[label] > min_label_count
            else "low label count"
        )

    return target_labels


def print_count_info(data, min_num_comp):
    """Counts occurrences of each target and prints summary statistics."""

    shared_counter_series = data["target_name"].value_counts()
    shared_counter_df = shared_counter_series.reset_index()
    shared_counter_df.columns = ["target_name", "compound_count_per_target"]

    num_of_targs_above_thr = len(
        shared_counter_df[
            shared_counter_df["compound_count_per_target"] > min_num_comp
        ]["target_name"].to_list()
    )
    num_of_comp_total = shared_counter_df[
        shared_counter_df["compound_count_per_target"] > min_num_comp
    ]["compound_count_per_target"].sum()

    print(
        f"Number of targets with more than {min_num_comp} compounds is {num_of_targs_above_thr}"
    )
    print(f"Total number of interactions is {num_of_comp_total}")
