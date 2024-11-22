# Reads in H100 profiling data and scale it to B200, B300

import pandas as pd

def scale_compute_profiling_data(profiling_data: pd.DataFrame, scale_factor: float) -> pd.DataFrame:
    for col in profiling_data.columns:
        if col.startswith("time_stats"):
            profiling_data[col] = profiling_data[col] * scale_factor
    return profiling_data


if __name__ == "__main__":
    profiling_data = pd.read_csv("data/profiling/compute/h100/meta-llama/Llama-2-70b-hf/attention.csv")
    scale_factor = 979.0 / 2250.0
    scaled_profiling_data = scale_compute_profiling_data(profiling_data, scale_factor)
    scaled_profiling_data.to_csv("data/profiling/compute/b200/meta-llama/Llama-2-70b-hf/attention.csv", index=False)

    profiling_data = pd.read_csv("data/profiling/compute/h100/meta-llama/Llama-2-70b-hf/mlp.csv")
    scale_factor = 979.0 / 2250.0
    scaled_profiling_data = scale_compute_profiling_data(profiling_data, scale_factor)
    scaled_profiling_data.to_csv("data/profiling/compute/b200/meta-llama/Llama-2-70b-hf/mlp.csv", index=False)
