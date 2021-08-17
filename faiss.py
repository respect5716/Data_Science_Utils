from typing import Optional

import faiss
import numpy as np
import pandas as pd


class PandasFaissEngine(object):
    """Search engine combining Pandas Dataframe and Faiss Index
    """
    def __init__(self, database: pd.DataFrame, index: faiss.Index, multiplier: int):
        """
        Args:
            database: Pandas Dataframe
            index: Faiss Index
            multiplier: k (number of nearest neighbors) is multiplied by this 
                in case applying additional condition to result
        """
        assert len(database) == index.ntotal, 'dataframe and index should have same length.'
        self.database = database
        self.index = index
        self.multiplier = multiplier
    
    def search(self, vector: np.array, condition: Optional[str], k: int, ignore_first: bool = True):
        """Search nearest neighbors using vector and additional condition
        
        Args:
            vector: query vector
            condition: additional condition (pandas query string)
            k: number of nearest neighbors
            ignore_first: ignore the nearest neighbor (this means that the nearest neighbor is query data itself)
        """
        vector = vector[None, :] if len(vector.shape) == 1 else vector
        search_k = self.multiplier * k + ignore_first if query else k
        D, I = self.index.search(vector, search_k)
        I = I[0][ignore_first:]

        result = self.database.iloc[I]
        if query:
            result = result.query(condition)
        result = result.iloc[:k]
        return result
