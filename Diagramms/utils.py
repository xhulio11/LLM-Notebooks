import json 
import plotly as plt
import plotly.figure_factory as ff
import pandas as pd 
import plotly.figure_factory as ff
import numpy as np

def load_file(path): 
    """
    This function takes a path of a json file 
    File Structure: 
            [
                [      # 1st group
                    [    # 1st item in group
                    {"label": "LABEL_0", "score": 0.0074805}, 
                    {"label": "LABEL_1", "score": 0.9922749},
                    {"label": "LABEL_2", "score": 0.0002445}
                    ],
                    [    # 2nd item in group
                    {"label": "LABEL_0", "score": ...},
                    {"label": "LABEL_1", "score": ...},
                    ...
                    ],
                    ...
                ],
                [      # 2nd group
                    ...
                ],
                ...
            ]
    """    

    # Load the File 
    with open(path, 'r') as file: 
        data = json.load(file)
    
    # rows of the DataFrame
    rows = []

    for group_index, group in enumerate(data):
        # group is a list of items
        for item in group:
            # item is a list of dicts, e.g.:
            # [
            #   {"label": "LABEL_0", "score": 0.0074805},
            #   {"label": "LABEL_1", "score": 0.9922749},
            #   {"label": "LABEL_2", "score": 0.0002445}
            # ]
            row = {}
            for entry in item:
                # entry is a dict with "label" and "score"
                row[entry["label"]] = entry["score"]
            
            # Optionally, if you want to track which group the row belongs to:
            # row["group_id"] = group_index
            
            rows.append(row)

    # Create a DataFrame, which will have columns LABEL_0, LABEL_1, and LABEL_2
    df = pd.DataFrame(rows)
    
    # Rename Columns to Left, Center, Right 
    df.rename(columns={"LABEL_0":"Left", "LABEL_1":"Center", "LABEL_2":"Right"}, inplace=True)

    # Return Dataframe 
    return df 
    

def show_distribution(dataframes, columns=["Left", "Center", "Right"]):
    
    """
    Check that `data` is a dictionary with the structure:
    {
        "some_string": pd.DataFrame,
        "another_string": pd.DataFrame,
        ...
    }
    """
    # 1. Check that data is actually a dict
    if not isinstance(dataframes, dict):
        raise TypeError(f"`dataframes` must be a dictionary, got {type(dataframes)} instead.")

    # 2. Check each key/value
    for key, value in dataframes.items():
        # Key should be a string
        if not isinstance(key, str):
            raise TypeError(f"Dictionary key '{key}' is not a string (type={type(key)}).")

        # Value should be a pandas DataFrame
        if not isinstance(value, pd.DataFrame):
            raise TypeError(
                f"Value for key '{key}' is not a pandas DataFrame (type={type(value)})."
            )


    x = {}

    # Looping through selected columns    
    for i, column in enumerate(columns):
        x[column] = {}
        # Looping through the given DataFrames
        for name,df in dataframes.items():
            x[column][name] = df[column].to_numpy()

    figures = []
    for key, value in x.items(): 
        
        hist_data = []
        group_labels = []
        
        for name, array in value.items(): 
            hist_data.append(array)
            group_labels.append(name)

        # Create distplot with custom bin_size
        fig = ff.create_distplot(hist_data, group_labels,show_hist=False, bin_size=[.01 for _ in hist_data])

        # Add Title 
        fig.update_layout(title_text=f"PDF of {key} classified content")

        # Add annotations
        annotations = [
            dict(
                x=0.01,  # adjust as needed
                y=0.27,  # adjust as needed
                xref="paper",
                yref="paper",
                text=f"Less {key} bias ",
                showarrow=False,
                font=dict(size=14, color="blue")
            ),
            dict(
                x=0.99,  # adjust as needed
                y=0.27,  # adjust as needed
                xref="paper",
                yref="paper",
                text=f"More {key} bias",
                showarrow=False,
                font=dict(size=14, color="red")
            )
        ]
        fig.update_layout(annotations=annotations)

        # Add figure to the list 
        figures.append(fig)

    return figures 


def logit_transform(x, epsilon=1e-6):
    """
    Applies the logit transformation to a numpy array or pandas Series.
    
    Parameters:
    - x: Input data (numpy array or pandas Series) with values in [0, 1].
    - epsilon: Small constant to avoid logit issues with 0 and 1.
    
    Returns:
    - Transformed data.
    """
    # Clip the values to avoid logit issues
    x_clipped = np.clip(x, epsilon, 1 - epsilon)
    
    # Apply the logit transformation
    return np.log(x_clipped / (1 - x_clipped))
