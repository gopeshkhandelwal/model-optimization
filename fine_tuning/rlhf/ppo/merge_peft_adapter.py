# Adapted from https://github.com/huggingface/optimum-habana/tree/v1.16.0/examples/trl
# Customized and enabled for Gaudi3
from dataclasses import dataclass, field
import logging
from typing import Optional

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser


@dataclass
class ScriptArguments:
    """
    The input names representing the Adapter and Base model fine-tuned with PEFT, and the output name representing the
    merged model.
    """

    adapter_model_name: Optional[str] = field(default=None, metadata={"help": "the adapter name"})
    base_model_name: Optional[str] = field(default=None, metadata={"help": "the base model name"})
    output_name: Optional[str] = field(default=None, metadata={"help": "the merged model name"})


from logging_utils import setup_logging
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
setup_logging()
logger = logging.getLogger(__name__)
logger.info(f"ScriptArguments: {script_args}")
assert script_args.adapter_model_name is not None, "please provide the name of the Adapter you would like to merge"
assert script_args.base_model_name is not None, "please provide the name of the Base model"
assert script_args.output_name is not None, "please provide the output name of the merged model"

peft_config = PeftConfig.from_pretrained(script_args.adapter_model_name)
logger.info(f"[Adapter] Loaded PEFT config from {script_args.adapter_model_name} (task_type={peft_config.task_type})")
if peft_config.task_type == "SEQ_CLS":
    logger.info("[IF] task_type == SEQ_CLS -> loading AutoModelForSequenceClassification (reward model style)")
    # The sequence classification task is used for the reward model in PPO
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.base_model_name, num_labels=1, torch_dtype=torch.bfloat16
    )
else:
    logger.info("[IF] task_type != SEQ_CLS -> loading AutoModelForCausalLM")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name, return_dict=True, torch_dtype=torch.bfloat16
    )

tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name)

# Load the PEFT model
model = PeftModel.from_pretrained(model, script_args.adapter_model_name)
logger.info(f"[Merge] Adapter weights loaded from {script_args.adapter_model_name}")
model.eval()

model = model.merge_and_unload()
logger.info("[Merge] merge_and_unload complete -> adapter merged into base model")

model.save_pretrained(f"{script_args.output_name}")
tokenizer.save_pretrained(f"{script_args.output_name}")
logger.info(f"[Save] Merged model + tokenizer saved to {script_args.output_name}")
# model.push_to_hub(f"{script_args.output_name}", use_temp_dir=False)
