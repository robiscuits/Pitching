import pandas as pd
import numpy as np

def recommend_pitch(context_row, pitch_types, clf, numeric_cols, categorical_cols, fill_value_numeric=0, fill_value_categorical='Unknown'):
    """
    Recommend the pitch with highest predicted success probability.
    
    Parameters:
    - context_row: dict or pd.Series of the pitch context (balls, strikes, plate_x, etc.)
    - pitch_types: list of pitch types to evaluate (e.g., ['FF', 'SL', 'CU'])
    - clf: trained sklearn pipeline (preprocessor + model)
    - numeric_cols: list of numeric column names
    - categorical_cols: list of categorical column names
    - fill_value_numeric: value to fill missing numeric columns
    - fill_value_categorical: value to fill missing categorical columns
    
    Returns:
    - best_pitch: pitch type with highest predicted success probability
    - probs: dict mapping pitch type → predicted success probability
    """
    
    probs = {}
    
    # Iterate over candidate pitch types
    for pitch in pitch_types:
        row = context_row.copy()
        row['pitch_type'] = pitch  # set candidate pitch type
        
        # Convert to single-row DataFrame
        if isinstance(row, dict):
            row_df = pd.DataFrame([row])
        else:  # Series
            row_df = row.to_frame().T

        # After constructing df_context with all candidate pitch types
        # Force all categorical columns + pitch_type to string and fill NaN
        for col in categorical_cols + ['pitch_type']:
            if col in df_context.columns:
                df_context[col] = df_context[col].fillna('Unknown').astype(str)

        # Ensure numeric columns are numeric and fill missing
        for col in numeric_cols:
            if col in df_context.columns:
                df_context[col] = pd.to_numeric(df_context[col], errors='coerce').fillna(0)

        # Predict probability of success
        p = clf.predict_proba(row_df)[0, 1]  # assuming binary classification
        probs[pitch] = p
    
    # Select pitch with highest probability
    best_pitch = max(probs, key=probs.get)
    
    return best_pitch, probs

def recommend_pitch_vectorized(context_row, pitch_types, clf, numeric_cols, categorical_cols,
                               fill_value_numeric=0, fill_value_categorical='Unknown'):
    """
    Vectorized pitch recommendation over multiple candidate pitch types.
    
    Parameters:
    - context_row: dict or pd.Series of pitch context (balls, strikes, plate_x, etc.)
    - pitch_types: list of pitch types to evaluate
    - clf: trained sklearn classifier pipeline
    - numeric_cols: list of numeric columns
    - categorical_cols: list of categorical columns
    - fill_value_numeric: value to fill missing numeric columns
    - fill_value_categorical: value to fill missing categorical columns
    
    Returns:
    - best_pitch: pitch type with highest predicted success probability
    - probs: dict mapping pitch type → predicted success probability
    """
    
    # Create a DataFrame with one row per pitch type
    if isinstance(context_row, dict):
        df_context = pd.DataFrame([context_row] * len(pitch_types))
    else:
        df_context = pd.DataFrame([context_row.to_dict()] * len(pitch_types))
    
    df_context['pitch_type'] = pitch_types
    
    # After constructing df_context with all candidate pitch types
    # Force all categorical columns + pitch_type to string and fill NaN
    for col in categorical_cols + ['pitch_type']:
        if col in df_context.columns:
            df_context[col] = df_context[col].fillna('Unknown').astype(str)

    # Ensure numeric columns are numeric and fill missing
    for col in numeric_cols:
        if col in df_context.columns:
            df_context[col] = pd.to_numeric(df_context[col], errors='coerce').fillna(0)

    # Now safe for predict_proba
    proba = clf.predict_proba(df_context)[:, 1]
    
    # Map pitch type → probability
    probs = dict(zip(pitch_types, proba))
    
    # Select the pitch with the highest probability
    best_pitch = max(probs, key=probs.get)
    
    return best_pitch, probs