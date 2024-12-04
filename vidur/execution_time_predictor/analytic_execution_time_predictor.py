# extimate execution time using analytic formula
from vidur.execution_time_predictor.base_execution_time_predictor import (
    BaseExecutionTimePredictor,
)

from vidur.entities import Batch, ExecutionTime
from vidur.config.model_config import BaseModelConfig
from vidur.config.device_sku_config import BaseDeviceSKUConfig

from vidur.config import (
    BaseReplicaSchedulerConfig,
    BaseExecutionTimePredictorConfig,
    MetricsConfig,
    ReplicaConfig,
)

def roof_line_execution_time(io_bandwidth_bytes_per_sec: float, data_size_bytes: float, tflops_per_sec: float, total_tflops: float) -> float:
    return max(
        (data_size_bytes / io_bandwidth_bytes_per_sec), 
        (total_tflops / tflops_per_sec)
    )

def get_dtype_bytes(dtype: str) -> float:
    if dtype == 'fp32':
        return 4
    elif dtype == 'fp16':
        return 2
    elif dtype == 'fp8':
        return 1
    elif dtype == 'fp4':
        return 0.5
    raise ValueError(f"Unknown dtype: {dtype}")


class AnalyticExecutionTimePredictor(BaseExecutionTimePredictor):
    def __init__(
        self,
        predictor_config: BaseExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
    ) -> None:
        super().__init__(
            predictor_config=predictor_config,
            replica_config=replica_config,
            replica_scheduler_config=replica_scheduler_config,
            metrics_config=metrics_config,
        )

        self._model_config = replica_config.model_config
        self._device_config = replica_config.device_config


    def _get_execution_time(self, data_size_bytes: float, total_flops: float) -> float:
        total_tflops = float(total_flops)/1e12
        return roof_line_execution_time(
            self._device_config.memory_bandwidth_gb_per_sec * 1e9,
            data_size_bytes,
            self._device_config.fp16_tflops,
            total_tflops
        )

    def _get_matmul_execution_time(self, m: int, k:int, n: int, dtype: str) -> float:
        data_size_bytes = get_dtype_bytes(dtype) * (m * k + k * n + m * n)
        total_flops = 2 * m * k * n
        return self._get_execution_time(
            data_size_bytes=data_size_bytes,
            total_flops=total_flops
        )

    def _get_attention_rope_execution_time(self, batch: Batch) -> float:
        total_num_tokens = batch._total_num_tokens
        embedding_dim = self._model_config.embedding_dim
        
        # Additiona of embeddings and position embeddings
        total_data_size = 2 * (total_num_tokens * embedding_dim * get_dtype_bytes('fp16'))
        total_flops = total_num_tokens * embedding_dim
        total_time = self._get_execution_time(
            data_size_bytes=total_data_size, 
            total_flops=total_flops
        )
        return total_time


    """ 
    Attention
    """

    def _get_attention_layer_pre_proj_execution_time(self, batch: Batch) -> float:
        # QKV projection
        total_num_tokens = batch._total_num_tokens
        embedding_dim = self._model_config.embedding_dim
        
        # Q projection
        q_proj_flops = 2 * (total_num_tokens * embedding_dim * (embedding_dim//self._model_config.num_q_heads))
        q_proj_data_size =  get_dtype_bytes('fp16') * ((total_num_tokens * embedding_dim) + (embedding_dim * embedding_dim) + (total_num_tokens * embedding_dim))
        q_proj_time = self._get_execution_time(
            data_size_bytes=q_proj_data_size, 
            total_flops=q_proj_flops
        ) 
        
        # KV projection
        kv_head_concat_size = (embedding_dim // self._model_config.num_q_heads) * self._model_config.num_kv_heads
        k_proj_flops = 2 * (total_num_tokens * embedding_dim * kv_head_concat_size)
        v_proj_flops = k_proj_flops
        kv_proj_flops = k_proj_flops + v_proj_flops
        kv_proj_data_size =  get_dtype_bytes('fp16') * ((total_num_tokens * embedding_dim) + (embedding_dim * kv_head_concat_size) + (total_num_tokens * kv_head_concat_size))
        kv_proj_time = self._get_execution_time(
            data_size_bytes=kv_proj_data_size, 
            total_flops=kv_proj_flops
        )

        return (q_proj_time + kv_proj_time)


    def _get_attention_kv_cache_save_execution_time(self, batch: Batch) -> float:
        return 0.0


    def _get_attention_prefill_execution_time(self, batch: Batch) -> float:
        # a simplified model for flash attention, only consider the matrix multiplication part
        embedding_dim = self._model_config.embedding_dim
        
        total_time = 0.0
        for request in batch.requests:
            prefill_tokens = request.num_prefill_tokens
            total_time += self._get_matmul_execution_time(
                m=prefill_tokens,
                k=embedding_dim,
                n=prefill_tokens,
                dtype='fp16'
            )
            
            total_time += self._get_matmul_execution_time(
                m=prefill_tokens,
                k=prefill_tokens,
                n=embedding_dim,
                dtype='fp16'
            )

        return total_time
    
    def _get_attention_decode_execution_time(self, batch: Batch) -> float:
        embedding_dim = self._model_config.embedding_dim
        batch_size = len(batch.requests)
        max_prefill_len = max([request.num_prefill_tokens for request in batch.requests])
        max_decode_len = max([request.num_decode_tokens for request in batch.requests])
        time_sec_per_batch = self._get_matmul_execution_time(
            m=batch_size,
            k=embedding_dim,
            n=max_prefill_len,
            dtype='fp16'
        ) + self._get_matmul_execution_time(
            m=batch_size,
            k=max_prefill_len,
            n=embedding_dim,
            dtype='fp16'
        )
        time_sec_per_token =  self._get_execution_time(
            data_size_bytes=data_size_bytes, 
            total_flops=total_flops
        ) 

        return time_sec_per_token * max_decode_len
    
    def _get_attention_layer_post_proj_execution_time(self, batch: Batch) -> float:
        # projection time during decoding phase
        # FIXME: what of projection time during prefill phase ???
        embedding_dim = self._model_config.embedding_dim
        
        total_time = 0.0
        
        # prefill time
        for request in batch.requests:
            total_flops = 2*request.num_prefill_tokens*embedding_dim*embedding_dim
            data_size_bytes = get_dtype_bytes('fp16') * (request.num_prefill_tokens*embedding_dim + embedding_dim*embedding_dim + request.num_prefill_tokens*embedding_dim)
            total_time += self._get_execution_time(
                data_size_bytes=data_size_bytes, 
                total_flops=total_flops
            )

        # decoding time
        for request in batch.requests:
            total_flops = 2*1*embedding_dim*embedding_dim
            data_size_bytes = get_dtype_bytes('fp16') * (1*embedding_dim + embedding_dim*embedding_dim + 1*embedding_dim)
            
            total_time_per_token = self._get_execution_time(
                data_size_bytes=data_size_bytes, 
                total_flops=total_flops
            )

            total_time += total_time_per_token * request.num_decode_tokens

        return total_time
    
    
    def _get_attn_norm_layer_act_execution_time(self, batch: Batch) -> float:
        # elementwise operation
        embedding_dim = self._model_config.embedding_dim
        total_num_tokens = batch.total_num_tokens
        data_size_bytes = get_dtype_bytes('fp16') * (total_num_tokens * embedding_dim)
        total_flops = total_num_tokens * embedding_dim
        total_time = self._get_execution_time(
            data_size_bytes=data_size_bytes, 
            total_flops=total_flops
        )

        return total_time
        
    
    """ 
    MLP
    """

    def _get_mlp_layer_up_proj_execution_time(self, batch: Batch) -> float:
        embedding_dim = self._model_config.embedding_dim
        mlp_hidden_dim = self._model_config.mlp_hidden_dim
        total_num_tokens = batch.total_num_tokens

        total_flops = 2*total_num_tokens*embedding_dim*mlp_hidden_dim
        data_size_bytes = get_dtype_bytes('fp16') * (total_num_tokens*embedding_dim + embedding_dim*mlp_hidden_dim + total_num_tokens*mlp_hidden_dim)
        total_time = self._get_execution_time(
            data_size_bytes=data_size_bytes, 
            total_flops=total_flops
        )

        return total_time
        
        

    def _get_mlp_layer_down_proj_execution_time(self, batch: Batch) -> float:
        embedding_dim = self._model_config.embedding_dim
        mlp_hidden_dim = self._model_config.mlp_hidden_dim
        total_num_tokens = batch.total_num_tokens

        # prefill time
        total_flops = 2*total_num_tokens*mlp_hidden_dim*embedding_dim
        data_size_bytes = get_dtype_bytes('fp16') * (total_num_tokens*mlp_hidden_dim + mlp_hidden_dim*embedding_dim + total_num_tokens*embedding_dim)
        total_time = self._get_execution_time(
            data_size_bytes=data_size_bytes, 
            total_flops=total_flops
        )

        return total_time


    def _get_mlp_layer_act_execution_time(self, batch: Batch) -> float:
        # elementwise operation
        embedding_dim = self._model_config.embedding_dim
        mlp_hidden_dim = self._model_config.mlp_hidden_dim
        total_num_tokens = batch.total_num_tokens
        data_size_bytes = get_dtype_bytes('fp16') * (total_num_tokens * mlp_hidden_dim)
        total_flops = total_num_tokens * mlp_hidden_dim
        total_time = self._get_execution_time(
            data_size_bytes=data_size_bytes, 
            total_flops=total_flops
        )

        return total_time

    def _get_mlp_norm_layer_act_execution_time(self, batch: Batch) -> float:
        embedding_dim = self._model_config.embedding_dim
        total_num_tokens = batch.total_num_tokens
        data_size_bytes = get_dtype_bytes('fp16') * (total_num_tokens * embedding_dim)
        total_flops = total_num_tokens * embedding_dim
        total_time = self._get_execution_time(
            data_size_bytes=data_size_bytes, 
            total_flops=total_flops
        )

        return total_time

    def _get_add_layer_act_execution_time(self, batch: Batch) -> float:
        embedding_dim = self._model_config.embedding_dim
        total_num_tokens = batch.total_num_tokens
        data_size_bytes = get_dtype_bytes('fp16') * (total_num_tokens * embedding_dim) * 2
        total_flops = total_num_tokens * embedding_dim
        
        total_time = self._get_execution_time(
            data_size_bytes=data_size_bytes, 
            total_flops=total_flops
        )

        return total_time
    
    """
    GPU Communication
    """

    def _get_tensor_parallel_communication_time(self, batch: Batch) -> float:
        return 0.0

    def _get_pipeline_parallel_communication_time(self, batch: Batch) -> float:
        return 0.0

    """
    CPU overhead
    """

    def _get_schedule_time(self, batch: Batch) -> float:
        # CPU overhead
        return 0.0

    def _get_sampler_e2e_time(self, batch: Batch) -> float:
        # CPU overhead
        return 0.0 

    def _get_prepare_inputs_e2e_time(self, batch: Batch) -> float:
        # CPU overhead
        return 0.0

    def _get_process_model_outputs_time(self, batch: Batch) -> float:
        # CPU overhead
        return 0.0

    def _get_ray_comm_time(self, batch: Batch) -> float:
        # CPU overhead
        return 0.0 
    