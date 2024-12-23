python -m vidur.main  \
--replica_config_device b200 \
--replica_config_model_name meta-llama/Llama-2-70b-hf \
--cluster_config_num_replicas 1 \
--replica_config_tensor_parallel_size 2 \
--replica_config_num_pipeline_stages 1 \
--request_generator_config_type trace_replay \
--length_generator_config_type trace \
--trace_request_length_generator_config_max_tokens 16384 \
--trace_request_length_generator_config_trace_file ./data/processed_traces/Azure_conv.csv \
--interval_generator_config_type poisson \
--poisson_request_interval_generator_config_qps 6.45 \
--replica_scheduler_config_type sarathi  \
--sarathi_scheduler_config_batch_size_cap 512  \
--sarathi_scheduler_config_chunk_size 512 \
--execution_time_predictor_config_type analytical \
--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 16384 \
--random_forrest_execution_time_predictor_config_prediction_max_batch_size 512 \
--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 16384