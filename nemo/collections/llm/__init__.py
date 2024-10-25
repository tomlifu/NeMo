# This is here to import it once, which improves the speed of launch when in debug-mode
try:
    import transformer_engine  # noqa
except ImportError:
    pass

from nemo.collections.llm import peft, tokenizer
from nemo.collections.llm.api import export_ckpt, finetune, import_ckpt, pretrain, train, validate
from nemo.collections.llm.gpt.data import (
    DollyDataModule,
    FineTuningDataModule,
    MockDataModule,
    PreTrainingDataModule,
    SquadDataModule,
)
from nemo.collections.llm.gpt.data.api import dolly, mock, squad
from nemo.collections.llm.gpt.model import (
    Baichuan2Config,
    Baichuan2Config7B,
    Baichuan2Model,
    ChatGLM2Config6B,
    ChatGLM3Config6B,
    ChatGLMConfig,
    ChatGLMModel,
    CodeGemmaConfig2B,
    CodeGemmaConfig7B,
    CodeLlamaConfig7B,
    CodeLlamaConfig13B,
    CodeLlamaConfig34B,
    CodeLlamaConfig70B,
    GemmaConfig,
    GemmaConfig2B,
    GemmaConfig7B,
    GemmaModel,
    GPTConfig,
    GPTModel,
    Llama2Config7B,
    Llama2Config13B,
    Llama2Config70B,
    Llama3Config8B,
    Llama3Config70B,
    LlamaConfig,
    LlamaModel,
    MaskedTokenLossReduction,
    MistralConfig7B,
    MistralModel,
    MixtralConfig8x3B,
    MixtralConfig8x7B,
    MixtralConfig8x22B,
    MixtralModel,
    Nemotron3Config4B,
    Nemotron3Config8B,
    Nemotron4Config15B,
    Nemotron4Config22B,
    Nemotron4Config340B,
    NemotronConfig,
    NemotronModel,
    gpt_data_step,
    gpt_forward_step,
)
from nemo.collections.llm.recipes import *  # noqa
from nemo.utils import logging

try:
    from nemo.collections.llm.api import deploy
except ImportError as error:
    deploy = None
    logging.warning(f"The deploy module could not be imported: {error}")

__all__ = [
    "MockDataModule",
    "GPTModel",
    "GPTConfig",
    "gpt_data_step",
    "gpt_forward_step",
    "MaskedTokenLossReduction",
    "MistralConfig7B",
    "MistralModel",
    "MixtralConfig8x3B",
    "MixtralConfig8x7B",
    "MixtralConfig8x22B",
    "MixtralModel",
    "NemotronModel",
    "Nemotron3Config4B",
    "Nemotron3Config8B",
    "Nemotron4Config15B",
    "Nemotron4Config22B",
    "Nemotron4Config340B",
    "NemotronConfig",
    "LlamaConfig",
    "Llama2Config7B",
    "Llama2Config13B",
    "Llama2Config70B",
    "Llama3Config8B",
    "Llama3Config70B",
    "CodeLlamaConfig7B",
    "CodeLlamaConfig13B",
    "CodeLlamaConfig34B",
    "CodeLlamaConfig70B",
    "LlamaModel",
    "GemmaConfig",
    "GemmaConfig2B",
    "GemmaConfig7B",
    "CodeGemmaConfig2B",
    "CodeGemmaConfig7B",
    "GemmaModel",
    "Baichuan2Config",
    "Baichuan2Config7B",
    "Baichuan2Model",
    "ChatGLMConfig",
    "ChatGLM2Config6B",
    "ChatGLM3Config6B",
    "ChatGLMModel",
    "PreTrainingDataModule",
    "FineTuningDataModule",
    "SquadDataModule",
    "DollyDataModule",
    "train",
    "import_ckpt",
    "export_ckpt",
    "pretrain",
    "validate",
    "finetune",
    "tokenizer",
    "mock",
    "squad",
    "dolly",
    "peft",
]

# add 'deploy' to __all__ if it was successfully imported
if deploy is not None:
    __all__.append("deploy")
