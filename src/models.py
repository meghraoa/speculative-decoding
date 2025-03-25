
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def load_models(target_model_name, drafter_model_name):
    """
    Load drafter and target models with 8-bit quantization.

    Args:
        target_model_name: Name of the target model.
        drafter_model_name: Name of the drafter model.

    Returns:
        - target model
        - drafter model
    """
    quant_config = BitsAndBytesConfig(load_in_8bit=True)

    target = AutoModelForCausalLM.from_pretrained(target_model_name, quantization_config=quant_config)
    drafter = AutoModelForCausalLM.from_pretrained(drafter_model_name, quantization_config=quant_config)

    return target, drafter
