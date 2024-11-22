from vidur.execution_time_predictor.base_execution_time_predictor import (
    BaseExecutionTimePredictor,
)

from vidur.entities.execution_time import APIExecutionTime, ExecutionTime

class APIExecutionTimePredictor(BaseExecutionTimePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefill_tok_per_sec = 1000
        self._output_tok_per_sec = 1000

    def get_execution_time(self, batch: Batch, pipeline_stage: int) -> APIExecutionTime:
        prefill_time = batch.num_prefill_tokens / self._prefill_tok_per_sec
        output_time = batch.num_output_tokens / self._output_tok_per_sec
        return APIExecutionTime(prefill_time, output_time)
        

