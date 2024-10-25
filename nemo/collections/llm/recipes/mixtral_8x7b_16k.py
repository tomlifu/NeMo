from typing import Callable

import torch

from nemo.collections.llm.api import pretrain
from nemo.collections.llm.recipes import mixtral_8x7b
from nemo.collections.llm.utils import Partial

NAME = "mixtral_8x7b_16k"


def pretrain_recipe(
    name: str, ckpt_dir: str, num_nodes: int, num_gpus_per_node: int, fn: Callable = pretrain
) -> Partial:
    recipe = mixtral_8x7b.pretrain_recipe(
        name=name, ckpt_dir=ckpt_dir, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node, fn=fn
    )

    trainer = mixtral_8x7b.trainer(
        tensor_parallelism=2,
        pipeline_parallelism=4,
        pipeline_parallelism_type=torch.bfloat16,
        virtual_pipeline_parallelism=8,
        context_parallelism=4,
        sequence_parallelism=True,
        expert_parallelism=8,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
    )
    model = mixtral_8x7b.model()
    model.config.seq_length = 16384

    recipe.model = model
    recipe.trainer = trainer

    return recipe


def finetune_recipe(name: str, ckpt_dir: str, num_nodes: int, num_gpus_per_node: int) -> Partial:
    recipe = mixtral_8x7b.finetune_recipe(
        name=name, ckpt_dir=ckpt_dir, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node
    )

    trainer = mixtral_8x7b.trainer(
        tensor_parallelism=2,
        pipeline_parallelism=2,
        pipeline_parallelism_type=torch.bfloat16,
        virtual_pipeline_parallelism=8,
        context_parallelism=2,
        sequence_parallelism=True,
        expert_parallelism=8,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
    )
    model = mixtral_8x7b.model()
    model.config.seq_length = 16384

    recipe.model = model
    recipe.trainer = trainer

    return recipe
