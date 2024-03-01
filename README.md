# RAMP starting-kit on classification of michelin star restaurants
classification of michelin star restaurants

Authors : Charles Cuvillier, Lucas Selini, Quentin Garsault, Eric Patarin

## Getting started

### Download Support

To download the necessary files, you have to clone the repository:
```bash
git clone https://github.com/charlescuvillier/datacamp_michelin.git
```


### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar
usage.

### Download Data
To download the data you need a kaggle account. You need to have your account name and the api key of your account. It will be asked by the terminal to download the necessary data.

Then, you have to run the download_data.py file directly in the terminal, not in a notebook because you cannot write your account information in a notebook.

### Challenge description

Get started with the [dedicated notebook](michelin_starting_kit.ipynb)


### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)

