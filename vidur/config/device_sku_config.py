from dataclasses import dataclass, field

from vidur.config.base_fixed_config import BaseFixedConfig
from vidur.logger import init_logger
from vidur.types import DeviceSKUType

logger = init_logger(__name__)


@dataclass
class BaseDeviceSKUConfig(BaseFixedConfig):
    fp16_tflops: int
    total_memory_gb: int
    memory_bandwidth_gb_per_sec: int
    

@dataclass
class A40DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 150
    total_memory_gb: int = 45
    memory_bandwidth_gb_per_sec: int = 1555

    @staticmethod
    def get_type():
        return DeviceSKUType.A40


@dataclass
class A100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 312
    total_memory_gb: int = 80
    memory_bandwidth_gb_per_sec: int = 1555

    @staticmethod
    def get_type():
        return DeviceSKUType.A100


@dataclass
class H100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 1000
    total_memory_gb: int = 80
    memory_bandwidth_gb_per_sec: int = 3.35*1024

    @staticmethod
    def get_type():
        return DeviceSKUType.H100

@dataclass
class B200DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 2250
    total_memory_gb: int = 192
    memory_bandwidth_gb_per_sec: int = 7.7*1024

    @staticmethod
    def get_type():
        return DeviceSKUType.B200

@dataclass
class B300DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 2250
    total_memory_gb: int = 288
    memory_bandwidth_gb_per_sec: int = 8*1024

    @staticmethod
    def get_type():
        return DeviceSKUType.B300
