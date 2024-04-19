# Overview
 Code for the paper:
 > "Co-Training Realized Volatility Prediction Model with Neural Distributional Transformation (2023)" [[url]](https://dl.acm.org/doi/abs/10.1145/3604237.3626870)


# Acknowledgment
This repository used code of the paper `Neural Ordinary Differential Equations`, `Augmented ODEs`, `Latent ODEs for Irregularly-Sampled Time Series`, `Free-form Jacobian of Reversible Dynamics`.


# Dependency
* Python >= 3.9
* arch == 6.1.0

* Install required Python packages:
    ```bash
    >> pip install -r requirements.txt
    ```

Python 3.9 is used by the arch package, and is thus necessary.
As for other packages such as numpy and scipy, we recommend you to
install the packages of the exact versions written in the requirement*.txt files.



# Data


The data used in the paper is derived from non-public high-frequency data and therefore cannot be shared. As an alternative, there is randomly generated sample data available in the data directory



# Produce the results of RMSE 

## Train our model from scratch

We don't offer pre-trained models because our datasets are not publicly available. However, the model is designed with few parameters and is simple to train. We've developed several scripts for sample data, and their hyperparameters are in line with those used in the paper.

- `bash script/run_har_node.sh`.
- `bash script/run_har_tanh.sh`.
- `bash script/run_har_yeojohnson.sh`.
- `bash script/run_har_wallace.sh`.
- `bash script/run_har_identity.sh`.

The script at the top corresponds to the proposed model outlined in the paper. Below it, you'll find the baseline models as described in the paper. Additionally, there's a parameter called "phi" in the code, but it's not relevant to this paper. Therefore, we set "phi_dim=0" to nullify its effect.



Upon executing the script, you'll receive example results as follows, where A, B, and C representing example company names:
```
INFO   ----------------------------------------------------------------   
INFO   Testing on the test set...
INFO   ----------------------------------------------------------------
INFO   Best evaluation iteration = 200.Validset RMSE = 1.08293,Validset NLL = 0.88681
INFO   Evaluating ... eval_start_step=360, eval_steps=120
INFO   test NLL mean: 0.782269
INFO   test RMSE: 1.05801
INFO   test MAE: 0.602154
INFO   A  1.012326
       B  1.161171                 
       C  1.000539
       dtype: float32
```

## Hyperparameter searching
We utilize the model's Negative Log-Likelihood (NLL) on the evaluation set as the criterion for hyperparameter search. The model undergoes training for 200 iterations and is evaluated after every 5 iterations. During each evaluation, a snapshot is saved, resulting in 20 snapshots over the 200 iterations of training. The best snapshot is then used for testing.

For managing our experiment records, we've employed the `wandb` tool. To synchronize the experiment records with the wandb cloud, simply run the `run.py` script and specify `--use_wandb`. Before running the script, ensure you've configured the `wandb_setting.py`. Here are the necessary steps:

- register an account on wandb website
- replace the `WANDB_ENTITY` and `WANDB_API_KEY` with your account name and API key, respectively.
- run `wandb_sweep.py` for hyperparameter searching.
- run `agents` of a `sweep` task to start hyperparameter tuning.


