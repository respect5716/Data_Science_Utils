
from typing import Union, Optional, List


def pad_sequences(
    seqs: List[List[Union[str, int]]], 
    pad_val: Union[str, int], 
    padding: str = 'post', 
    trunc: str = 'pre', 
    maxlen: Optional[int] = None
) -> List[List[Union[str, int]]]: 
    """pad sequences to the max length
    
    Args:
        seqs: List of sequences
        pad_val: padding value
        padding: where to put the padding ['pre', 'post']
        trunc: where truncate the sequence ['pre', 'post']
        maxlen: max length of sequence (only valid when it is shorter than max length of sequences)
        
    Returns:
        padded sequences
    """
    assert maxlen > 0, 'maxlen should be larger than 0'
    
    _maxlen = max([len(s) for s in seqs])
    maxlen = min(maxlen, _maxlen) if maxlen else _maxlen 
    
    padded_seqs = []
    for seq in seqs:
        seq = seq[-maxlen:] if trunc == 'pre' else seq[:maxlen]
        pads = [pad_val] * (maxlen - len(seq))
        seq = pads + seq if padding == 'pre' else seq + pads
        padded_seqs.append(seq)

    return padded_seqs
