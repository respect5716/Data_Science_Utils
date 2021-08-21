
import re
# import kss
from typing import Union, Optional, List
from .count import topp


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

class PatternSentenceSplitter(object):
    def __init__(self, endchars: Optional[List[str]] = None):
        """
        Args:
            endchar: list of characters at the end of a sentence.
        """
        
        self.endchars = endchars if endchars else DEFAULT_ENDCHARS
        self.endpuncs = DEFAULT_ENDPUNCS
        self.brackets = DEFAULT_BRACKETS   
        self.quotes = DEFAULT_QUOTES
        self.options = {**self.brackets, **self.quotes}
        self.reversed_options = {v:k for k,v in self.options.items()}
        self.ignores = '|'.join(DEFAULT_IGNORE)
        self.specials = self.endpuncs + list(self.options.keys()) + list(self.options.values()) \
            + self.ignores.split('|')
    
    @property
    def pattern(self):
        endchars = ''.join(self.endchars)
        endpuncs = ''.join(self.endpuncs)
        endoptions = ''.join(self.options.values())
        return f'\S*[{endchars}][{endpuncs}]+[{endoptions}]*\s+'
    
    def is_punc(self, s):
        return s in self.endpuncs
    
    def is_open_bracket(self, s):
        return s in self.brackets.keys()
    
    def is_close_bracket(self, s):
        return s in self.brackets.values()
    
    def is_bracket(self, s):
        return self.is_open_bracket(s) or self.is_close_bracket(s)
    
    def is_open_quote(self, s):
        return s in self.quotes.keys()
    
    def is_close_quite(self, s):
        return s in self.quotes.values()
    
    def is_quote(self, s):
        return self.is_open_quote(s) or self.is_close_quote(s)
    
    def is_open_option(self, s):
        return s in self.options.keys()
    
    def is_close_option(self, s):
        return s in self.options.values()
    
    def to_close(self, s):
        return self.options[s]
    
    def to_open(self, s):
        return self.reversed_options[s]
    
    def ignore_sentence(self, s):
        return re.sub(self.ignores, '', s)
    
    
    def _split_sentences(self, text: str):
        start, sentences = 0, []
        for match in re.finditer(self.pattern, text):
            end = match.end()
            sent = text[start:end].strip()
            sentences.append(sent)
            start = end
            
        if start < len(text) - 1:
            sentences.append(text[start:].strip())
    
        return sentences
    
    def clean_sentences(self, sentences):
        results = []
        for sent in sentences:
            sent = re.sub('\s+', ' ', sent)
            results.append(sent)
        return results
    
    def split_option(self, sentences:List[str]):        
        results = []
        opt_ing = False
        
        for sent in sentences:
            real_sent = self.ignore_sentence(sent)
            s, e = real_sent[0], real_sent[-1]
            
            # 첫 글자가 option
            if self.is_open_option(s):
                
                # 마지막 글자와 호응
                if self.to_close(s) == e:
                    results.append(sent)
                    
                # 중간 글자와 호응
                # TODO: 이중 괄호 처리
                elif self.to_close(s) in real_sent[1:-1]:
                    results.append(sent)
                    
                # 호응 없음
                else:    
                    sent += self.to_close(s)
                    results.append(sent)
                    opt_ing = True
            
            # 마지막 글자가 option
            elif self.is_close_option(e):
                sent = self.to_open(e) + sent
                results.append(sent)
                opt_ing = False
            
            # 문장에 option이 없음
            else:
                
                # option 진행 중
                if opt_ing:
                    prev_sent = self.ignore_sentence(results[-1])
                    sent = prev_sent[0] + sent + prev_sent[-1]
                    results.append(sent)
                
                # option 진행 중 아님
                else:
                    results.append(sent)
                    
        return results
    
    def merge_option(self, sentences: List[str]):
        results = []
        buffer, opt_ing = [], False
        
        while sentences:
            sent = sentences.pop(0)
            real_sent = self.ignore_sentence(sent)
            s, e = real_sent[0], real_sent[-1]
            
            # option 진행 중
            if opt_ing:
                buffer.append(sent)
                opened_opt = self.ignore_sentence(buffer[0])[0]
                
                # 진행 중인 option의 close가 문장에 포함 (완벽하게 분리되지 않을 경우를 대비)
                if self.to_close(opened_opt) in sent:
                    results.append(' '.join(buffer))
                    buffer, opt_ing = [], False
                
                else:
                    pass
                
            # option 진행 중 아님
            else:
                
                # 첫 글자가 옵션
                if self.is_open_option(s):
                    
                    # 마지막 글자와 호응
                    if self.to_close(s) == e:
                        results.append(sent)

                    # 중간 글자와 호응
                    # TODO: 이중 괄호 처리
                    elif self.to_close(s) in real_sent[1:-1]:
                        results.append(sent)
                    
                    # 호응 없음
                    else:
                        buffer.append(sent)
                        opt_ing = True
                
                # option 아님
                else:
                    results.append(sent)
            
            
        if buffer:
            opened_opt = self.ignore_sentence(buffer[0])[0]
            remains = ' '.join(buffer) + self.to_close(opened_opt)
            results.append(remains)
        
        return results
                
        
    def split_sentences(self, text: str, option_method: str = 'merge'):
        sentences = self._split_sentences(text)
        sentences = self.clean_sentences(sentences)
 
        if option_method == 'split':
            sentences = self.split_option(sentences)
        elif option_method == 'merge':
            sentences = self.merge_option(sentences)
            
        return sentences
    
    def split_chunks(self, 
        sentences: Union[str, List[str]], 
        chunk_size: int, 
        num_sents: Optional[int] = None, 
        sep: str = '\n', 
        drop_last: bool = True
        ):
        
        if type(sentences) == str:
            sentences = self.split_sentences(sentences, option_method='merge')

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
    
    def forward_matches(self, texts: List[str], endchar: str):
        pattern = f'{endchar}.'
        res = [re.findall(pattern, t) for t in texts]
        res = list(itertools.chain(*res))
        res = [i.replace(endchar, '') for i in res]
        
        counts = collections.Counter(res).most_common()
        chars_cnt, puncs_cnt = 0, 0
        for k, v in counts:
            if k in self.endpuncs:
                puncs_cnt += v
            else:
                chars_cnt += v
        
        return puncs_cnt / (puncs_cnt + chars_cnt + 1e-5)

    def forward(self, texts: List[str], endchars: List[str], p: float):
        # ~ 10 minutes
        probs = [self.forward_matches(texts, char) for char in endchars]
        res = [char for char, prob in zip(endchars, probs) if prob >= p]
        return res
    
    def backward_matches(self, text: str, endchars: List[str], charlen: int):
        endchars = ''.join(endchars)
        endpuncs = ''.join(self.endpuncs)
        endoptions = ''.join(self.options.values())
        pattern = f'[\w\s]+[{endchars}][{endpuncs}]+[{endoptions}]*\s+'
        
        matches = re.findall(pattern, text)
        endpuncs_sub = '|'.join([f'\\{i}' for i in self.specials])
        res = [''.join(re.split(endpuncs_split, i))[-charlen:] for i in matches]
        return res
    
    def backward(self, texts: List[str], endchars: List[str], p: float, charlen: int = 2):
        # ~ 1 minute
        """구두점"""
        res = [self.backward_matches(i, endchars=endchars, charlen=charlen) for i in texts]
        res = list(itertools.chain(*res))
        counts = collections.Counter(res).most_common()
        topp = count_topp(counts, p=p)
        return topp
    

    
    def fit(self, texts: List[str], forward_p: float = 0.2, backward_p: float = 0.95):
        endchar = self.backward(texts, backword_p)
        res = self.forward(texts, endchar, foward_p)
        return res
