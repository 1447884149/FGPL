cd ../../../
python main.py \
--cfg model_heterogeneity/SFL_methods/FedProto/FedProto_on_citeseer.yaml \
--client_cfg model_heterogeneity/model_settings/10_Heterogeneous_GNNs.yaml
data.local_eval_whole_test_dataset True