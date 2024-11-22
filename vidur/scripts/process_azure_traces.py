from datetime import datetime


# read csv file and transform each line
raw_trace_file = "data/traces/AzureLLMInferenceTrace_conv.csv"
processed_trace_file = "data/processed_traces/Azure_conv.csv"

with open(raw_trace_file, "r") as file:
    start_timestamp = None
    line_num = 0
    parsed_lines = []
    for line in file.readlines():
        line_num += 1
        if line_num == 1:
            continue
        timestamp_str, prefill_tokens, decode_tokens = line.strip().split(",")
        timestamp_sec, timestamp_micro = timestamp_str.split(".")
        timestamp = datetime.strptime(timestamp_sec, "%Y-%m-%d %H:%M:%S").timestamp() + float(f'0.{timestamp_micro}')
        if start_timestamp is None:
            start_timestamp = timestamp

        parsed_lines.append(f'{timestamp - start_timestamp},{prefill_tokens},{decode_tokens}')
    
    with open(processed_trace_file, "w") as file:
        file.write("arrived_at,num_prefill_tokens,num_decode_tokens\n")
        file.write("\n".join(parsed_lines))