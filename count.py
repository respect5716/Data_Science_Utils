from typing import List, Tuple

def topp(counts: List[Tuple[str, int]], p=float):
    total_cnt = sum([i[1] for i in counts])
    target_cnt = int(total_cnt * p)
    
    res = []
    curr_cnt = 0
    for k, v in counts:
        res.append(k)
        curr_cnt += v
        
        if curr_cnt >= target_cnt:
            break
    return res
