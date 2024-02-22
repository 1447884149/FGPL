# Mitigating Data Imbalance and Generating Better Prototypes in Heterogeneous Federated Graph Learning
This is the official code of the paper: Mitigating Data Imbalance and Generating Better Prototypes in Heterogeneous Federated Graph Learning



## Acknowledgment

All code implementations are based on the FederatedScope V0.3.0: https://github.com/alibaba/FederatedScope 

We are grateful for their outstanding work.




## Models & Dataset & Methods

### Model setting

We consider three heterogeneous GNN backbones, i.e., GCN, GAT, and GPR-GNN . Each of the backbones has very different message propagation mechanisms. For each client, we assign its local model by sampling from the three backbones. We further modify each local model's number of layers and hidden state dimensions to enhance heterogeneity.

For details of the model architecture, please refer to [model setting files](federatedscope/model_heterogeneity/model_settings) and [model definition files](federatedscope/gfl/model)


### Dataset

Currently, we conduct experiments on three benchmark datasets: Cora, CiteSeer, and Computers.

### Methods

Our method FGPL and other baseline methods can be found in [Trainer](federatedscope/contrib/trainer) and [Worker](federatedscope/contrib/worker)

## Quickly Start

### Step 1. Install FederatedScope

Users need to clone the source code and install FederatedScope (we suggest python version >= 3.9).

- clone the source code

```python
git clone https://github.com/1447884149/FGPL.git
cd FGPL
```

- install the required packages:

```python
conda create -n fs python=3.9
conda activate fs

# install pytorch
conda install -y pytorch=1.10.1 torchvision=0.11.2 torchaudio=0.10.1 torchtext=0.11.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# install some extra dependencies
conda install -y pyg -c pyg
conda install -y nltk
pip install rdkit
pip install ipdb
pip install kornia
pip install timm
```


- Next, after the required packages is installed, you can install FederatedScope from `source`:

```python
pip install -e .[dev]
```


### Step 2. Run Algorithm

- Enter the "federatedscope" folder

```python
cd federatedscope
```

- Run the script

```python
# python main.py --cfg [the path of main cfg file] --client_cfg [the path of model cfg file]
# Cora for 10 clients
python main.py --cfg model_heterogeneity/SFL_methods/FGPL/FGPL_on_cora.yaml --client_cfg model_heterogeneity/model_settings/10_Heterogeneous_GNNs.yaml data.local_eval_whole_test_dataset True
```

## Run Baselines  (Take running FedProto as an example)

You can run our reproduced baseline methods in the same way as running FedPPN.

- Enter the "federatedscope" folder

```python
cd federatedscope
```


- Run the script

```
python main.py --cfg model_heterogeneity/SFL_methods/FedProto/FedProto_on_cora.yaml --client_cfg model_heterogeneity/model_settings/10_Heterogeneous_GNNs.yaml data.local_eval_whole_test_dataset True
```
