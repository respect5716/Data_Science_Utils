from typing import List

def jaccard_similarity(lst1: List, lst2: List):
    """Jaccard similarity for guaging the similarity and diversity of sample sets
    """
    set1 = set(lst1)
    set2 = set(lst2)
    return len(set1 & set2) / len(set1 | set2)
