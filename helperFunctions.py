import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def run_etl(start_dt, end_dt)
    pitchData = pb.statcast(start_dt=start_dt, end_dt=end_dt)
    mainDF = pitchData[['stand','game_date','player_name','events','description','p_throws','hit_location','bb_type','balls','strikes','pitch_type', 'release_speed', 'pfx_x', 'pfx_z', 'release_spin_rate','plate_x','plate_z','on_3b','on_2b','on_1b','outs_when_up','inning','inning_topbot','release_pos_y','at_bat_number','pitch_number','pitch_name','bat_score','fld_score','post_bat_score','post_fld_score', 'delta_home_win_exp','delta_run_exp','bat_speed','swing_length','pitcher_days_since_prev_game','n_thruorder_pitcher', 'arm_angle', 'batter','pitcher']]
    
    ver = mainDF[mainDF['player_name'] == 'Verlander, Justin'].copy()
    
    ver["pitch_success"] = np.where(
    (
        ver["events"].isin([
            'field_out', 'strikeout', 'field_error', 'fielders_choice',
            'double_play', 'grounded_into_double_play', 'force_out'
        ])
    ) |
    (
        ver["description"].isin(['called_strike', 'swinging_strike'])
    ) |
    (
        ver["description"].isin(['foul']) & (ver['strikes'] != 2)
    ),
    True,
    False
    )
    conditions = [
        ver["events"].isin(['double_play', 'grounded_into_double_play']),
        ver["events"].isin(['strikeout']), 
        ver["events"].isin(['field_out']),
        ver["description"].isin(['swinging_strike', 'called_strike']),
        ver["description"].isin(['foul', 'foul_tip']),
        ver["description"].isin(['ball']),
        ver["description"].isin(['hit_by_pitch']),
        ver["events"].isin(['single']),  # bad outcomes
        ver["events"].isin(['double']),  # bad outcomes
        ver["events"].isin(['triple']),  # bad outcomes
        ver["events"].isin(['home_run']),  # bad outcomes
    ]

    values = [1.00, 0.95, 0.90, 0.75, 0.70, 0.50, 0.25, 0.25, 0.20, 0.15, 0]

    ver["pitch_outcome_score"] = np.select(conditions, values, default=np.nan)
    
    ver["on_3b"] = ver["on_3b"].fillna(0)
    ver["on_2b"] = ver["on_2b"].fillna(0)
    ver["on_1b"] = ver["on_1b"].fillna(0)
    
    verClean = ver[['p_throws','stand','balls','strikes','pitch_type', 'release_speed', 'pfx_x', 'pfx_z', 'release_spin_rate','plate_x','plate_z','on_3b','on_2b','on_1b','outs_when_up','inning','release_pos_y','at_bat_number','pitch_number','pitch_name','bat_score','fld_score','pitcher_days_since_prev_game','n_thruorder_pitcher', 'arm_angle', 'batter', 'pitch_success','pitch_outcome_score']]
    verClean = verClean.dropna()
    return verClean
    
def train_model(split_stratifier, xcols, ycol, categorical_cols, use_PCA = False, n_components = 20)
    train, test = train_test_split(verClean, stratify=verClean[split_stratifier])

    if USE_PCA:
        n_components = 20 # you can tune this
        p = ColumnTransformer([
            ('scaler', StandardScaler(), numeric_cols),
            ("cat",OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=int), categorical_cols),
            ('pca', PCA(n_components=n_components))
        ])
    else:
        p = ColumnTransformer([
            ('scaler', StandardScaler(), numeric_cols),
            ("cat",OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=int), categorical_cols)
        ])

    model = GradientBoostingClassifier(n_estimators=200, max_depth=10, random_state=42)
    #model = LogisticRegression(max_iter = 10000, class_weight = {False:0.5,True:0.5})

    clf = Pipeline(steps=[
        ('preprocessor', p),
        ('model', model)
    ])
    
    clf.fit(train[xcols], train[ycol])

    return pitchData

def recommend_pitch_vectorized_safe(context_row, pitch_types, clf, numeric_cols, categorical_cols,
                                    fill_value_numeric=0, fill_value_categorical='Unknown'):

    # ---- Normalize pitch_types ----
    if isinstance(pitch_types, (pd.Series, np.ndarray)):
        pitch_types = list(pitch_types)

    # Cast all candidates to strings
    pitch_types = [str(p) for p in pitch_types]

    # ---- Determine known pitch types from the trained OneHotEncoder ----
    preprocessor = clf.named_steps['preprocessor']
    cat_transformers = preprocessor.named_transformers_['cat']

    # Find the index of 'pitch_type' in categorical_cols
    if 'pitch_type' in categorical_cols:
        pitch_type_index = categorical_cols.index('pitch_type')
    else:
        raise ValueError("'pitch_type' must be in categorical_cols.")

    known_pitch_types = cat_transformers.categories_[pitch_type_index]

    # Keep only candidates seen during training
    pitch_types = [p for p in pitch_types if p in known_pitch_types]
    if len(pitch_types) == 0:
        raise ValueError("No candidate pitch_types are recognized by the trained encoder.")

    # ---- Build replicated context rows ----
    if isinstance(context_row, dict):
        df_context = pd.DataFrame([context_row] * len(pitch_types))
    else:
        df_context = pd.DataFrame([context_row.to_dict()] * len(pitch_types))

    # Overwrite pitch_type column with candidate labels
    df_context['pitch_type'] = pitch_types

    # ---- Ensure all expected columns exist ----
    for col in numeric_cols:
        if col not in df_context.columns:
            df_context[col] = fill_value_numeric
    for col in categorical_cols:
        if col not in df_context.columns:
            df_context[col] = fill_value_categorical

    # ---- Hygiene: fill missing values and type enforcement ----
    for col in numeric_cols:
        df_context[col] = pd.to_numeric(df_context[col], errors='coerce').fillna(fill_value_numeric)
    for col in categorical_cols + ['pitch_type']:
        df_context[col] = df_context[col].fillna(fill_value_categorical).astype(str)

    # ---- Predict success probabilities using the pipeline ----
    proba = clf.predict_proba(df_context)[:, 1]   # Probability of success

    # ---- Build sorted mapping pitch_type -> probability ----
    probs = {p: float(prob) for p, prob in zip(pitch_types, proba)}
    probs_sorted = dict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))

    # Best pitch
    best_pitch = next(iter(probs_sorted))

    return best_pitch, probs_sorted