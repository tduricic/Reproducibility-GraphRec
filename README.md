# Social Recommendations with GraphRec: A Reproducibility Study and the Accuracy-Diversity Trade-off

This is a repository to support the paper "Social Recommendations with GraphRec: A Reproducibility Study and the Accuracy-Diversity Trade-off". 
The goal of the study was to reproduce the original results in the original paper by Fan et al. for rating prediction, but also to modify GraphRec for top-K recommendation using different loss functions.
To evaluate GraphRec in a rating prediction setting, we use a modified version of the repository provided by the authors: [Original GraphRec repo](https://github.com/wenqifan03/GraphRec-WWW19).
To evaluate GraphRec in a top-K recommendation setting, we modify the original GraphRec code and implement it as an external approach in Elliot [2], a comprehensive and rigorous framework for reproducible recommender systems evaluation: [Elliot recommender repo](https://github.com/sisinflab/elliot).

## Prerequisites

- Python 3.6+
- PyTorch 0.2+
- TensorFlow 2.3.2+

## Installation

1. Clone the repository to your local machine:

```
git clone https://github.com/<username>/Reproducibility-GraphRec.git && cd Reproducibility-GraphRec
```

2. Clone the Elliot repository and install Elliot to your local machine:

### CONDA

```
git clone https://github.com//sisinflab/elliot.git && cd elliot
conda create --name elliot_env python=3.8
conda activate elliot_env
pip install --upgrade pip
pip install -e . --verbose
```

### VIRTUALENV

```
git clone https://github.com//sisinflab/elliot.git && cd elliot
virtualenv -p /usr/bin/python3.6 venv # your python location and version
source venv/bin/activate
pip install --upgrade pip
pip install -e . --verbose
```

3. Create an external model in Elliot for GraphRec:

Run the following script:

```
python setup_graphrec_elliot.py
```

## Datasets

And finally, you need to download and preprocess the datasets. In this paper, we work with three publicly available datasets, namely [Epinions](https://www.cse.msu.edu/~tangjili/datasetcode/epinions.zip), [Ciao](https://www.cse.msu.edu/~tangjili/datasetcode/ciao.zip), and [Last.fm](https://github.com/tommantonela/umap2022-mrecuri).
To download and preprocess the data you need to run the following script:

```
python download_and_preprocess_data.py
```


## Project Structure

After cloning and installing the necessary repositories and downloading the datasets, you should have the following project structure:

```
.
├── data
│   ├── epinions
│       ├── raw
│       └── processed
│   ├── ciao
│   └── lastfm
├── notebooks
│   └── data_exploration.ipynb
├── graphrec
│      ├── ...
│      └── run_graphrec_example.py 
├── elliot
│      ├── config_files
│      ├── data
│      ├── docs
│      ├── elliot
│      ├── external
│      ├── img
│      ├── log
│      ├── results
│      ├── ...
│      └── run_graphrec_example.py
├── .gitignore
├── download_and_preprocess_data.py
├── README.md
└── setup_graphrec_elliot.py

```

## Usage

To start the rating prediction experiments for GraphRec, you can run the following:

```
cd graphrec
python run_graphrec_example.py
```

If you want to run it for a different datasets or different parameters, check the arguments accepted by the script in `graphrec/run_graphrec_example.py`.

To start the top-K evaluation experiments for GraphRec, you can run the following:

```
cd elliot
python run_graphrec_example.py
```

If you want to configure the top-K experiments differently, please refer to the [Elliot documentation](https://elliot.readthedocs.io/en/latest/guide/config.html). 

## Acknowledgements

**TBD**

## References
[1] Fan, W., Ma, Y., Li, Q., He, Y., Zhao, E., Tang, J., & Yin, D. (2019, May). Graph neural networks for social recommendation. In The world wide web conference (pp. 417-426).

[2] Anelli, V. W., Bellogín, A., Ferrara, A., Malitesta, D., Merra, F. A., Pomo, C., ... & Di Noia, T. (2021, July). Elliot: a comprehensive and rigorous framework for reproducible recommender systems evaluation. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval (pp. 2405-2414).