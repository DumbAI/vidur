import pandas as pd

experiment_name = "2024-11-22_13-00-13-472846"
request_metrics_file = f"simulator_output/{experiment_name}/request_metrics.csv"

request_metrics_pd = pd.read_csv(request_metrics_file)

token_per_sec = request_metrics_pd['request_num_tokens'] / request_metrics_pd['request_model_execution_time']

print('Token per second: ', token_per_sec.mean())
