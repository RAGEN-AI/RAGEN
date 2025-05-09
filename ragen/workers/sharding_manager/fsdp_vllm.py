# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging
import os

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.torch_functional import check_cuda_is_available
from verl.utils.vllm_utils import patch_vllm_moe_model_weight_loader

from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager as VerlFSDPVLLMShardingManager
from peft import PeftModel
from collections import OrderedDict

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FSDPVLLMShardingManager(VerlFSDPVLLMShardingManager):
    @check_cuda_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
    ):
        super().__init__(module=module, inference_engine=inference_engine, model_config=model_config, device_mesh=device_mesh, offload_param=offload_param)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __enter__(self):
        # NOTE: Basically, we only need `torch.cuda.empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        torch.cuda.empty_cache()

        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        if self.offload_param:
            load_fsdp_model_to_gpu(self.module)

        if isinstance(self.module._fsdp_wrapped_module, PeftModel):
            # the model to sync weights to is a vLLM model (not a peft model), so we need to merge the adapters
            lora_params = OrderedDict()
            with FSDP.summon_full_params(self.module):
                self.module.merge_adapter()
                base_model = self.module._fsdp_wrapped_module.base_model.model
                # don't use model.state_dict() to avoid OOM
                for name, param in base_model.named_parameters():
                    if ".lora_" in name:
                        continue
                    
                    clean_name = name.replace(".base_layer.", ".").replace("._fsdp_wrapped_module.",".")
                    if hasattr(param, 'full_tensor'):
                        tensor = param.full_tensor().detach().cpu()
                    else:
                        tensor = param.detach().cpu().clone()
                    lora_params[clean_name] = tensor

            params = lora_params  
        else:
            params = self.module.state_dict()

        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)
        # Copy, not share memory
        load_format = "hf" if self.full_params else "dtensor"

        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            self.inference_engine.sync_model_weights(params, load_format=load_format)
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            del params
        else:
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["weights"])
            else:
                self.inference_engine.wake_up()

            # update model params
            self.update_params(params)
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            del params
            if self.offload_param:
                offload_fsdp_model_to_cpu(self.module)
            torch.cuda.empty_cache()

            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["kv_cache"])

        if isinstance(self.module._fsdp_wrapped_module, PeftModel):
            with FSDP.summon_full_params(self.module):
                self.module.unmerge_adapter()

        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)


    def update_params(self, updated_params):
        model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
        if model.config.architectures[0] in ['DeepseekV2ForCausalLM', 'DeepseekV3ForCausalLM']:
            loaded_params = patched_ds_v3_load_weights(
                model, ((name, param.full_tensor() if hasattr(param, 'full_tensor') else param)
                        for name, param in updated_params.items()))
        else:
            loaded_params = model.load_weights(
                ((name, param.full_tensor() if hasattr(param, 'full_tensor') else param) for name, param in updated_params.items()))
        logger.info(f"vLLM load weights, loaded_params: {len(loaded_params)}")