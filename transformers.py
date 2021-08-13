# Util functions for Huggingface Transformers

from typing import Dict
from transformers import AutoTokenizer

PREDEFINED_SPECIAL_TOKENS = ['bos_token', 'eos_token', 'unk_token', 'sep_token', 'pad_token', 'cls_token', 'mask_token']


def load_tokenizer(
    pretrained_model_name_or_path: str, 
    special_token_dict: Dict
):
    """Load transfomers tokenizer
    
    Args:
        pretrained_model_name_or_path: the `model id` of huggingface hub or 'model path' to the directory where the pretrained tokenizer saved
        special_token_dict: dictionary which key is speicial token name, and value is speical token
        
    Returns:
        tokenizer: transformers tokenizer
        
    Examples:
        special_token_dict = {
            'bos_token': '<s>',
            'eos_token': '</s>',
            'title_token': '<title>', # additional special token
        }
        tokenizer = load_tokenizer('skt/kogpt2-base-v2', special_token_dict)
        print(tokenizer.title_token) # '<title>'
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    predefined_special_token_dict = {}
    additional_special_token_dict = {}

    for k, v in special_token_dict.items():
        if k in PREDEFINED_SPECIAL_TOKENS:
            predefined_special_token_dict[k] = v
        else:
            additional_special_token_dict[k] = v

    predefined_special_token_dict['additional_special_tokens'] = list(additional_special_token_dict.values())
    tokenizer.add_special_tokens(predefined_special_token_dict)

    for k, v in additional_special_token_dict.items():
        setattr(tokenizer, f'{k}', v)
        setattr(tokenizer, f'{k}_id', tokenizer.convert_tokens_to_ids(v))

    return tokenizer
