import pandas as pd
from pymongo import MongoClient
from typing import Union, Optional, List, Dict


def pandas_to_mongo(
    client: MongoClient,
    dataframe: pd.DataFrame,
    database_name: str,
    collection_name: str,
    columns: Optional[Union[List, Dict]] = None,
    specify_id: bool = True,
) -> None:
    """Insert pandas dataframe to mongodb collection
  
    Args:
        client: mongodb client
        dataframe: dataframe to insert
        database_name
        collection_name
        columns: columns to be included 
          - if None, all columns will be included
          - if List, columns in the list will be included
          - if Dict, columns in the dict keys will be included as the dict values

    Examples:
        client = MongoClient(host='localhost', port=27017)
        data = pd.read_csv('data.csv')
        columns = {'col1':'new_col1', 'col3':'new_col3'}
        pandas_to_mongo(client, data, 'database', 'collection', columns)
    """
    database = client[database_name]
    collection = database[collection_name]
    cnt = collection.count()
    
    if columns:
        if type(columns) == dict:
            dataframe = dataframe[list(columns.keys())]
            dataframe.columns = list(columns.values())
        
        elif type(columns) == list:
            dataframe = dataframe[columns]
        
        if specify_id:
            dataframe['_id'] = range(cnt, cnt+len(dataframe))

    documents = dataframe.to_dict('records')
    _ = collection.insert_many(documents)
