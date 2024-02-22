cd ../../../
python main.py \
--cfg model_heterogeneity/SFL_methods/FedGH/FedGH_on_cora.yaml \
--client_cfg model_heterogeneity/model_settings/5_Heterogeneous_GNNs.yaml
data.local_eval_whole_test_dataset True