from typing import Callable, Optional

import pytorch_lightning as pl
import torch
from megatron.core.distributed import DistributedDataParallelConfig
from pytorch_lightning.callbacks.callback import Callback

from nemo import lightning as nl
from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.gpt.model.llama import Llama3Config70B, LlamaModel
from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.recipes.log.default import default_log, default_resume, tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed_plugin
from nemo.collections.llm.utils import Config, Partial
from nemo.utils.exp_manager import TimingCallback

NAME = "llama3_70b"


def model() -> Config[pl.LightningModule]:
    return Config(LlamaModel, config=Config(Llama3Config70B))


def trainer(
    tensor_parallelism: int,
    pipeline_parallelism: int,
    pipeline_parallelism_type: Optional[torch.dtype],
    virtual_pipeline_parallelism: Optional[int],
    context_parallelism: int,
    sequence_parallelism: bool,
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    max_steps: int = 1168251,
    callbacks: Optional[list[Config[Callback]]] = None,
) -> Config[nl.Trainer]:
    strategy = Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_type,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
    )

    trainer = Config(
        nl.Trainer,
        accelerator="gpu",
        accumulate_grad_batches=1,
        callbacks=callbacks,
        devices=num_gpus_per_node,
        gradient_clip_val=1.0,
        limit_test_batches=50,
        limit_val_batches=32,
        log_every_n_steps=10,
        max_steps=max_steps,
        num_nodes=num_nodes,
        plugins=bf16_mixed_plugin(),
        strategy=strategy,
        use_distributed_sampler=False,
        val_check_interval=2000,
    )

    return trainer


def pretrain_recipe(
    name: str, ckpt_dir: str, num_nodes: int, num_gpus_per_node: int, fn: Callable = pretrain
) -> Partial:
    return Partial(
        fn,
        model=model(),
        trainer=trainer(
            tensor_parallelism=4,
            pipeline_parallelism=4,
            pipeline_parallelism_type=torch.bfloat16,
            virtual_pipeline_parallelism=5,
            context_parallelism=2,
            sequence_parallelism=True,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            callbacks=[Config(TimingCallback)],
        ),
        data=Config(MockDataModule, seq_length=8192, global_batch_size=512, micro_batch_size=1),
        log=default_log(ckpt_dir=ckpt_dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=3e-4),
        resume=default_resume(),
    )


def hf_resume() -> Config[nl.AutoResume]:
    return Config(nl.AutoResume, import_path="hf://meta-llama/Meta-Llama-3.1-70B")


def finetune_recipe(name: str, ckpt_dir: str, num_nodes: int, num_gpus_per_node: int) -> Partial:
    recipe = pretrain_recipe(
        name=name, ckpt_dir=ckpt_dir, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node, fn=finetune
    )
    recipe.resume = hf_resume()
    recipe.peft = Config(LoRA)
    recipe.data = Config(SquadDataModule, seq_length=8192, global_batch_size=512, micro_batch_size=1)
    return recipe
