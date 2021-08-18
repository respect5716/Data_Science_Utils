
import re
import kss
from typing import Union, Optional, List


def pad_sequences(
    seqs: List[List[Union[str, int]]], 
    pad_val: Union[str, int], 
    padding: str = 'post', 
    trunc: str = 'pre', 
    maxlen: Optional[int] = None
) -> List[List[Union[str, int]]]: 
    """Pad sequences to the max length
    
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


class KoreanSentenceSplitter(object):
    endings = ['다', '래', '아', '야', '해', '가', '까', '지', '네', '어', '요', '소', '세']
    puncs = ['.', '?', '!', '"']
    pattern = f"[{''.join(endings)}][{''.join(puncs)}]+\s+"
    
    # TODO: process internal sentence
    # TODO: other brackets, single quote
    def match_quote(self, sentence):
        if sentence[0] == '"' and sentence[-1] != '"':
            sentence = sentence + '"'
        elif sentence[-1] == '"' and sentence[0] != '"':
            sentence = '"' + sentence
        return sentence
        
    
    def split_sentences(self, text):
        start, sentences = 0, []
        for match in re.finditer(self.pattern, text):
            end = match.end()
            sent = text[start:end].strip()
            sent = self.match_quote(sent)
            sentences.append(sent)
            start = end
        
        # TODO: process last sentence
        if start < len(text) - 1:
            sentences.append(text[start:].strip())
        
        return sentences
    
    
    def split_sentences_kss(self, text):
        return kss.split_sentences(text)
    
    
    def split_chunks(self, 
        sentences: Union[str, List[str]], 
        chunk_size: int, 
        num_sents: Optional[int] = None, 
        sep: str = '\n', 
        drop_last: bool = True
        ):
        
        if type(sentences) == str:
            sentences = self.split_sentences(sentences)

        if num_sents:
            quotient, remainder = divmod(len(sentences), num_sents)
            num_chunks = quotient if drop_last else quotient + int(bool(remainder))
            chunks = [f'{sep}'.join(sentences[i*num_sents : (i+1)*num_sents]) for i in range(num_chunks)]
        
        else:
            chunks = []
            tmp, tmplen = [], 0
            for sent in sentences:
                tmp.append(sent)
                tmplen += len(sent)

                if tmplen > chunk_size:
                    chunks.append(f'{sep}'.join(tmp))
                    tmp, tmplen = [], 0

            if not drop_last and tmp:
                chunks.append(f'{sep}'.join(tmp))
        
        return chunks
