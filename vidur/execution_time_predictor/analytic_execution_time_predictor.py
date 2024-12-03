# extimate execution time using analytic formula
from vidur.execution_time_predictor.base_execution_time_predictor import (
    BaseExecutionTimePredictor,
)

from vidur.entities import Batch, ExecutionTime

def roof_line_execution_time(io_bandwidth_bytes_per_sec: float, data_size_bytes: float, tflops_per_sec: float, total_tflops: float) -> float:
    return max((data_size_bytes / io_bandwidth_bytes_per_sec), (total_tflops / tflops_per_sec))

def _get_dtype_bytes(dtype: str) -> float:
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
    def __init__(self, model_config: BaseModelConfig, device_config: BaseDeviceSKUConfig):
        self._model_config = model_config
        self._device_config = device_config

    def _get_execution_time(self, data_size_bytes: float, total_tflops: float) -> float:
        return roof_line_execution_time(
            self._device_config.memory_bandwidth_gb_per_sec * 1e9,
            data_size_bytes,
            self._device_config.fp16_tflops,
            total_tflops
        )
        

    def _get_attention_rope_execution_time(self, batch: Batch) -> float:
        total_num_tokens = batch._total_num_tokens
        embedding_dim = self._model_config.embedding_dim
        
        # Additiona of embeddings and position embeddings
        total_data_size = 2 * (total_num_tokens * embedding_dim * get_dtype_bytes('fp16'))
        total_tflops = (total_num_tokens * embedding_dim) / 1e12
        total_time = self._get_execution_time(
            data_size_bytes=total_data_size, 
            total_tflops=total_tflops
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
        q_proj_tflops = q_proj_flops / 1e12
        q_proj_data_size +=  get_dtype_bytes('fp16') * ((total_num_tokens * embedding_dim) + (embedding_dim * embedding_dim) + (total_num_tokens * embedding_dim))
        q_proj_time = self._get_execution_time(
            data_size_bytes=q_proj_data_size, 
            total_tflops=q_proj_tflops
        ) 
        
        # KV projection
        kv_head_concat_size = (embedding_dim // self._model_config.num_q_heads) * self._model_config.num_kv_heads
        k_proj_flops = 2 * (total_num_tokens * embedding_dim * kv_head_concat_size)
        v_proj_flops = k_proj_flops
        kv_proj_flops = k_proj_flops + v_proj_flops
        kv_proj_tflops = kv_proj_flops / 1e12
        kv_proj_data_size =  get_dtype_bytes('fp16') * ((total_num_tokens * embedding_dim) + (embedding_dim * kv_head_concat_size) + (total_num_tokens * kv_head_concat_size))
        kv_proj_time = self._get_execution_time(
            data_size_bytes=kv_proj_data_size, 
            total_tflops=kv_proj_tflops
        )

        return (q_proj_time + kv_proj_time) * self._model_config.num_layers

    def _get_attention_kv_cache_save_execution_time(self, batch: Batch) -> float:
        return 0.0

    def _get_attention_prefill_execution_time(self, batch: Batch) -> float:
        # a simplified model for flash attention, only consider the matrix multiplication part
        embedding_dim = self._model_config.embedding_dim
        
        total_time = 0.0
        for request in batch.requests:
            prefill_tokens = request.num_prefill_tokens
            total_flops += 2*prefill_tokens*embedding_dim*prefill_tokens + 2*prefill_tokens*prefill_tokens*embedding_dim
            total_tflops = total_flops / 1e12
            data_size_bytes = get_dtype_bytes('fp16') * ((prefill_tokens*embedding_dim + embedding_dim*prefill_tokens + prefill_tokens*prefill_tokens) + (prefill_tokens*prefill_tokens + prefill_tokens*embedding_dim + prefill_tokens*embedding_dim))
            
            total_time += self._get_execution_time(
                data_size_bytes=data_size_bytes, 
                total_tflops=total_tflops
            ) 

        return total_time * self._model_config.num_layers
    
    def _get_attention_decode_execution_time(self, batch: Batch) -> float:
        embedding_dim = self._model_config.embedding_dim
        
        total_time = 0.0
        for request in batch.requests:
            total_flops = 2*1*embedding_dim*n + 2*1*n*embedding_dim
            total_tflops = total_flops / 1e12
            data_size_bytes = get_dtype_bytes('fp16') * ((1*embedding_dim + embedding_dim*n + 1*n) + (1*n + n*embedding_dim + 1*embedding_dim))
            
            time_sec_per_token =  self._get_execution_time(
                data_size_bytes=data_size_bytes, 
                total_tflops=total_tflops
            ) 
            output_tokens = request.num_decode_tokens
            total_time += time_sec_per_token * output_tokens

        return total_time * self._model_config.num_layers
    
    def _get_attention_layer_post_proj_execution_time(self, batch: Batch) -> float:
        # projection time during decoding phase
        # FIXME: what of projection time during prefill phase ???
        embedding_dim = self._model_config.embedding_dim
        
        total_time = 0.0
        
        # prefill time
        for request in batch.requests:
            total_flops = 2*request.num_prefill_tokens*embedding_dim*embedding_dim
            total_tflops = total_flops / 1e12
            data_size_bytes = get_dtype_bytes('fp16') * (request.num_prefill_tokens*embedding_dim + embedding_dim*embedding_dim + request.num_prefill_tokens*embedding_dim)
            total_time += self._get_execution_time(
                data_size_bytes=data_size_bytes, 
                total_tflops=total_tflops
            )

        # decoding time
        for request in batch.requests:
            total_flops = 2*1*embedding_dim*embedding_dim
            total_tflops = total_flops / 1e12
            data_size_bytes = get_dtype_bytes('fp16') * (1*embedding_dim + embedding_dim*embedding_dim + 1*embedding_dim)
            
            total_time_per_token += self._get_execution_time(
                data_size_bytes=data_size_bytes, 
                total_tflops=total_tflops
            )

            total_time += total_time_per_token * request.num_decode_tokens

        return total_time * self._model_config.num_layers
    
    
    def _get_attn_norm_layer_act_execution_time(self, batch: Batch) -> float:
        # elementwise operation
        embedding_dim = self._model_config.embedding_dim
        total_num_tokens = batch.total_num_tokens
        data_size_bytes = get_dtype_bytes('fp16') * (total_num_tokens * embedding_dim)
        total_tflops = (total_num_tokens * embedding_dim) / 1e12 
        
        total_time += self._get_execution_time(
            data_size_bytes=data_size_bytes, 
            total_tflops=total_tflops
        )

        return total_time * self._model_config.num_layers 
        
    
    """ 
    MLP
    """

    def _get_mlp_layer_up_proj_execution_time(self, batch: Batch) -> float:
        pass

    def _get_mlp_layer_down_proj_execution_time(self, batch: Batch) -> float:
        pass

    def _get_mlp_layer_act_execution_time(self, batch: Batch) -> float:
        pass
    
    def _get_mlp_norm_layer_act_execution_time(self, batch: Batch) -> float:
        pass

    def _get_add_layer_act_execution_time(self, batch: Batch) -> float:
        pass  

    def _get_tensor_parallel_communication_time(self, batch: Batch) -> float:
        pass

    def _get_pipeline_parallel_communication_time(self, batch: Batch) -> float:
        pass

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
    