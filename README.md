# Multi-Task Learning Approach for Data Imputation of Compound Bioactivity Values for SLC Transporter Superfamily

This repository contains a Jupyter Notebook that integrates multiple functions from various Python scripts to import, process, standardize, optimize, train, and evaluate a multi-task deep neural network (MTDNN) for data imputation of pChEMBL values across SLC protein targets.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Prerequisites](#prerequisites)
5. [Usage](#usage)
6. [Testing](#testing)
7. [Acknowledgements](#acknowledgements)
8. [License](#license)

---

## Overview

The repository includes a main Jupyter Notebook, [SLC_final_mtdnn_imputation.ipynb](./SLC_final_mtdnn_imputation.ipynb), along with supporting Python scripts that provide functional tools for developing and evaluating a data imputation model using multi-task learning. The notebook presents a streamlined workflow with chained functions for a clear overview of the project. Users can interact with input and output variables without delving into low-level code details. Alternatively, individual functions can be explored in their respective scripts for deeper analysis.

---

## Features

### **Data Retrieval and Merging**
- This step can be skipped as the data is already saved locally in the [datasets](./datasets) folder.
- Imports and merges Resolute and UniProt datasets to associate SLC names with corresponding UniProt-ID values.
- Fetches Target-ChEMBL-IDs via UniProt-IDs.
- Filters data based on `target_type` and organism of interest.

### **Processing and Standardization**
- Imports a dataset with calculated ECFP6 descriptors.
- Filters data based on `standard_type`, `standard_relation`, `standard_unit`, `target_organism`, recombinant, and mutagenized entries.
- Prioritizes columns, removes duplicates, and rounds extreme affinity values.
- Standardizes target-naming conventions.

### **Multi-Task Formatting of Data**
- Structures tabular data into a matrix suitable for a multi-task model.
- Removes targets with fewer compounds than a defined threshold.
- Generates row labels and merges descriptors.
- Implements two types of data splitting:
  - **Prediction split** (row-based splitting of compounds).
  - **Imputation split** (column-based splitting of outputs).
- Performs imputation post-processing to remove unrepresented targets in test/validation sets (optional step).
- Converts data to PyTorch tensors.
- Creates structured data dictionaries for easy access.
- Prepares structured data splits for cross-validation (CV), ensuring consistent evaluation across different subsets.

### **Model Development**
- Performs hyperparameter optimization.
- Trains the multi-task model.

### **Evaluation**
- Assesses overall performance using scoring metrics.
- Conducts cross-validation on prediction model splits.
- Formats evaluation tables.
- Plots learning dynamics using loss gradients.
- Identifies High Error Value (HEV) and Low Error Value (LEV) predictions.
- Extracts outlier compounds and corresponding targets.
- Identifies nearest neighbors of HEV and LEV compounds.
- **Target-Based Evaluation**:
  - Multi-task performance analysis.
  - Chemical space analysis.
  - Single-task performance analysis:
    - Creates single-task splits for selected target subsets.
    - Optimizes hyperparameters.
    - Trains single-task models.
    - Evaluates model performance.

---

## Installation

To install the required dependencies, use the appropriate YAML file based on your operating system:

- **Windows:** [mtdnn_env_win.yaml](./mtdnn_env_win.yaml)
- **Ubuntu/Linux:** [mtdnn_env_lin.yaml](./mtdnn_env_lin.yaml)

Run the following command to create a conda environment:

```sh
# For Windows
conda env create -f mtdnn_env_env.yaml  

# For Ubuntu/Linux
conda env create -f mtdnn_env_lin.yaml  
```

---

## Prerequisites

Ensure you have the following Python libraries with corresponding versions installed:
- `umap-learn` (0.5.5)
- `optuna` (3.6.1)
- `pytorch` (2.3.0)
- `chembl-webresource-client` (0.10.9)
- `scikit-learn` (1.5.1)
- `scipy` (1.13.0)
- `matplotlib` (3.9.1)
- `rdkit` (2024.03.5)
- `pandas` (2.2.2)
- `numpy` (1.24.4)

---

## Usage

### **Getting Started**

- Download the files from the repository.
- Install the required libraries manually or via the YAML file.
- Open [mtdnn_imputation.ipynb](./mtdnn_imputation.ipynb) and select the installed environment.

### **Using the Jupyter Notebook**

- Execute the cells in sequence.
- If data should be retrieved from ChEMBL rather than loaded, uncomment the indicated lines of code.
- Hyperparameter optimization is optional, as optimal values have already been selected. Uncomment the indicated lines of code to run Bayesian search if needed.
- Results are displayed in the Jupyter Notebook. Images are stored as TIFF files in the results folder for publication purposes. This can be changed in the evaluation source script manually.

For reproducibility, use the provided datasets, variables, and parameters from the repository.

---

## Testing

**Disclaimer:** The project was developed on Windows 11 Home version 24H2, Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz 2.59 GHz, 64-bit operating system, NVIDIA geForce RTX 2070 GPU, and Cuda compilation tools release 12.4, V12.4.131. If ran on Linux operating systems, outputs might show slight variations.

Figures of results are provided in the [results](./results) folder for comparative purposes.

The Jupyter Notebook can be adapted to train multi-task learning models on different targets and datasets. The corresponding processing, optimization, training, and evaluation steps should be adjusted based on the data format.

---

## Acknowledgements

Leo Gaskin assisted during the development of Jupyter Notebook Python scripts.

ChatGPT [https://chat.openai.com/] was used to aid during code and README file documentation.

## License

Repository license: [LICENSE](./LICENSE)
