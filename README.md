# Domino Experiment Manager Examples

This repo will make comparisons between Weights and Biases and Domino's Experiment Management capability.


## Pre-requisites

Create a user environment variable `OPENAI_API_KEY`. Do not echo this anywhere in your notebooks.
For WANDB we will be using anonymous access.

Create two conda environments and the corresponding Jupyter Kernel

```shell
./create-ray-conda.sh  #Creates the "ray" kernel
./create-tensorflow-conda.sh #Creates the "tensorboard" kernel
```

There will be several notebooks in this repo to compare the features of these Experiment Tracking Products. 

## First the basic Experiment Management Flow

There are three basic notebooks included in this repo to demonstrate the use Experiment Manager for:
1. Running [basic](./notebooks/basic/mlflow-basic.ipynb) experiments - This notebook will demonstrate how to create basic experiments, runs and model versions. It will also demonstrate how to download artifacts for a specific Model Version via the experiment run id attached to the model version. This is useful to deploy models based on the artifacts contained in the model registry
2. Running [basic hyperparameter search using Spark](./notebooks/basic/pyspark_hyperparameter_search.ipynb)
3. Running [Ray Tune based hyperparameter search using ](./notebooks/basic/ray_tune_hyperparameter_search.ipynb). This demonstrates how you can create nested runs to better organize your experiment runs. **SELECT THE KERNEL- "ray"** to run this notebook

Execute these three notebooks to get a feel for the basics of the Domino Experiment Manager

## Domino Experiment Manager alongside Weights & Biases  


### Workbooks

1. Basic [Notebook](./notebooks/llm-dl/llm-dl/01_intro_starter.ipynb) demonstrating the use of WANDB and Experiment Manager to track the same experiment 

This notebook demonstrates tracking the DL Model Training using MLFLOW and WANDB. The example function which runs a simultaneous WANDB and MLFLOW experiment is `train_model_mlflow_wandb`

W&B vs. Mlflow comparisons based on this notebook are - 

Criteria | W&B | Domino Experiment Manager |
| ------------- | ------------- |-------------|
Track Parameters | `wandb.init` function receives the entire config of type `SimpleNamespace` | `mlflow.log_params(vars(config))` is needed in Domino. A helper function `mlflow_utils.init(project_name,config)` performs the same function as `wandb.init` | 
Track CPU/GPU and other system metrics| `wandb` provides this out of the box| `mlflow.init` uses `nvidia_smi` and `psutil` packages track these metrics via a running thread which has the same lifecycle as the experiment run. We only track a few CPU/GPU metrics currently. But this function can be expanded to track any system metrics of you choice.



Now invoke it for any metric you wish to graph
```python
metrics = train_model_mlflow(config)
df_train_loss = pd.DataFrame(data=metrics)
generate_plot(df_train_loss)
```
And this generates the graph ![Track Metrics using DCA](images/01_track_metrics_dca.png) 
Mlflow allows you to track metrics interactively or you could simple download this image and save to an artifact.

2. [Fine tune LLM Model](./notebooks/llm-dl/llm-dl/02_train_llm_starter-wandb.ipynb) - Track using WANDB
3. [Fine tune LLM Model](./notebooks/llm-dl/llm-dl/02_train_llm_starter-mlflow.ipynb) - Track using MLFLOW
4. [Tensorflow Example](./notebooks/llm-dl/llm-dl/03_tensorboard_example.ipynb) - This notebook demonstrates the autologging feature of MLFLOW for tensorboard. The tensorboard logs are automatically stored in the MLFLOW artifacts. The notebook demonstrates how these logs can be downloaded for any experiment run for which they have been logged to render in a local tensorboard instance.**SELECT THE KERNEL- "tensorboard"** to run this notebook


### NOTE

To clear all notebooks before commit run the following command
```
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb
```


### Environments

Use the following two environments-
1. [Ray Compute](./RayCompute-WORKSPACE.txt)
2. [Ray Cluster](./RayCluster.txt)
