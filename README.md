# DeCrypt: Bitcoin Fraud Detection

DeCrypt is a Python-based tool for analyzing Bitcoin transaction data to detect potential fraud and anomalous behavior. It utilizes machine learning techniques (anomaly detection and clustering) and network analysis to identify suspicious addresses and generate comprehensive risk reports.

## Features

- **Transaction Analysis:** Loads and processes Bitcoin transaction CSV data.
- **Feature Extraction:** Generates graph-based and statistical features using `networkx` and `pandas`.
- **Machine Learning:** 
  - Anomaly detection to find suspicious addresses based on transaction patterns.
  - K-Means clustering to group addresses by behavioral similarities.
- **Reporting:** Automatically generates visual plots (using `matplotlib` and `seaborn`) and detailed PDF reports (using `reportlab`).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AllRock666/DeCrypt.git
   cd DeCrypt
   ```
2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main analysis script from the command line:

```bash
python -m decrypt.main --data <path_to_csv> [options]
```

### Arguments

- `--data` (required): Path to the input CSV file containing Bitcoin transaction data.
- `--outdir` (optional): Directory to save the generated reports and plots. Defaults to the current directory (`.`).
- `--contamination` (optional): Expected proportion of anomalies for the detection model (default: `0.1`).
- `--clusters` (optional): Maximum number of clusters for KMeans (default: `10`).

## Example

```bash
python -m decrypt.main --data data/transactions.csv --outdir output --contamination 0.05 --clusters 5
```

This will analyze `transactions.csv`, output the results to the `output` directory, expect a 5% anomaly rate, and group addresses into up to 5 clusters.