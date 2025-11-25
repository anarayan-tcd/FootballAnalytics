import pandas as pd

#-----------------CURRENT SEASON MATCHES (existing block)----------------#
# Load current season matches (Matches25)
schedule_df = pd.read_excel(
    "Chelsea_Match_Schedule.xlsx",
    sheet_name="Matches25",
    usecols=['MatchDate','Time','Round','Day','Venue','Result','GF','GA','Opponent']
)

schedule_df['MatchDate'] = pd.to_datetime(schedule_df['MatchDate'], dayfirst=True)
schedule_df['year'] = schedule_df['MatchDate'].dt.year.astype(int)
schedule_df['month'] = schedule_df['MatchDate'].dt.month.astype(int)

# Load ELO forecasts
elo_df = pd.read_csv("IMP/team_elo_forecast.csv")
elo_df['date'] = pd.to_datetime(elo_df['date'], dayfirst=True)
elo_df['year'] = elo_df['date'].dt.year.astype(int)
elo_df['month'] = elo_df['date'].dt.month.astype(int)

# Merge Chelsea ELO
chelsea_elo = elo_df[elo_df['team'] == 'Chelsea'][['year','month','avg_elo_forecast']]
schedule_df = schedule_df.merge(
    chelsea_elo,
    on=['year','month'],
    how='left'
).rename(columns={'avg_elo_forecast': 'Chelsea_ELO'})

# Merge Opponent ELO
opp_elo = elo_df.rename(columns={'team':'Opponent'})[['Opponent','year','month','avg_elo_forecast']]
schedule_df = schedule_df.merge(
    opp_elo,
    on=['Opponent','year','month'],
    how='left'
).rename(columns={'avg_elo_forecast':'Opponent_ELO'})

# Drop helper columns
schedule_df.drop(['year','month'], axis=1, inplace=True)
print(schedule_df)

#-----------------LOAD ALL TEAMS ELO BY MONTH---------------------------#
all_teams_df = pd.read_excel("IMP/All_Teams_ELO_By_Month.xlsx")

# The file already has an actual date column, e.g. "01-08-2021"
# Parse it directly
all_teams_df['date'] = pd.to_datetime(all_teams_df['date'], dayfirst=True)

# Extract year + month for merging with match data
all_teams_df['year'] = all_teams_df['date'].dt.year.astype(int)
all_teams_df['month'] = all_teams_df['date'].dt.month.astype(int)

print(all_teams_df)


#-----------------LOAD HISTORICAL MATCHES------------------------------#
history_df = pd.read_excel(
    "Chelsea_Match_Schedule.xlsx",
    sheet_name="ChelseaMatches",
    usecols=['MatchDate','Time','Round','Day','Venue','Result','GF','GA','Opponent']
)
history_df['MatchDate'] = pd.to_datetime(history_df['MatchDate'], dayfirst=True)
history_df['year'] = history_df['MatchDate'].dt.year.astype(int)
history_df['month'] = history_df['MatchDate'].dt.month.astype(int)
# Ensure numeric columns are numeric and replace only true NaNs with 0
numeric_cols_history = ['GF', 'GA']

for col in numeric_cols_history:
    # Convert to numeric, coercing invalid entries to NaN
    history_df[col] = pd.to_numeric(history_df[col], errors='coerce')
    # Replace NaNs with 0, leave existing 0s intact
    history_df[col] = history_df[col].fillna(0)

# Optional: if you have other numeric columns like xG/xGA in history_df, add them to the list

# Ensure GF and GA are numeric
history_df['GF'] = pd.to_numeric(history_df['GF'], errors='coerce')
history_df['GA'] = pd.to_numeric(history_df['GA'], errors='coerce')

# Filter historical matches: from August 2021 onward
# Filter historical matches: from August 2021 onward and rounds starting with "Matchday"
# Filter historical matches: from August 2021 onward AND Round contains "Matchday"
history_df = history_df.loc[
    ((history_df['year'] > 2021) | ((history_df['year'] == 2021) & (history_df['month'] >= 8))) &
    (history_df['Round'].str.contains("Matchweek", na=False))
]



# Merge Chelsea ELO for historical matches
chelsea_hist_elo = all_teams_df[all_teams_df['team'] == 'Chelsea'][['year','month','avg_elo']]
history_df = history_df.merge(
    chelsea_hist_elo,
    on=['year','month'],
    how='left'
).rename(columns={'avg_elo': 'Chelsea_ELO'})

# Merge Opponent ELO for historical matches
opp_hist_elo = all_teams_df.rename(columns={'team':'Opponent'})[['Opponent','year','month','avg_elo']]
history_df = history_df.merge(
    opp_hist_elo,
    on=['Opponent','year','month'],
    how='left'
).rename(columns={'avg_elo':'Opponent_ELO'})

# Drop helper columns
history_df.drop(['year','month'], axis=1, inplace=True)

#-----------------CHECK OUTPUT----------------------------------------#
print("Current season matches with forecasted ELOs:")
# print(schedule_df.head())
print("\nHistorical matches with ELOs:")
# print(history_df.head())

#-----------------WEIGHT FUNCTION BASED ON HISTORICAL RESULTS----------------#

def compute_match_weight(row):
    """
    Compute weight of a single match based on Chelsea vs Opponent ELO, result, and venue.
    Positive weight → advantage for Chelsea against that team
    Negative weight → disadvantage for Chelsea against that team
    Draw (GD=0) → weight 0
    """
    gd = row['GF'] - row['GA']
    elo_diff = row['Chelsea_ELO'] - row['Opponent_ELO']

    # Base weight from result and ELO
    if gd == 0:
        base_weight = 0.0
    elif elo_diff > 0:
        if gd < 0:
            base_weight = -1.0 if abs(gd) >= 3 else -0.5
        else:
            base_weight = 0.0
    elif elo_diff < 0:
        if gd > 0:
            base_weight = 1.0 if gd >= 3 else 0.5
        else:
            base_weight = 0.0
    else:
        base_weight = 0.5 if gd > 0 else -0.5

    # Venue adjustment
    venue_weight = 0
    if row['Venue'].lower() == 'home':
        venue_weight = 0.3
    elif row['Venue'].lower() == 'away':
        venue_weight = -0.2
    # Add venue factor to base weight
    return base_weight + venue_weight

# Calculate weight for each match
history_df['Match_Weight'] = history_df.apply(compute_match_weight, axis=1)

# Add recency weighting: more recent matches have higher impact
history_df['Recency_Factor'] = (history_df['MatchDate'] - history_df['MatchDate'].min()) / \
                               (history_df['MatchDate'].max() - history_df['MatchDate'].min())
history_df['Weighted_Match_Weight'] = history_df['Match_Weight'] * (0.5 + 0.5 * history_df['Recency_Factor'])

# Aggregate weights per opponent
team_weights_df = history_df.groupby('Opponent')['Weighted_Match_Weight'].sum().reset_index()
team_weights_df.rename(columns={'Weighted_Match_Weight': 'Weight', 'Opponent': 'Team'}, inplace=True)
team_weights_df = team_weights_df.sort_values('Weight', ascending=False).reset_index(drop=True)

#-----------------PREDICT CURRENT SEASON RESULTS USING ELO + WEIGHTS----------------#

# Merge team weights into current season schedule
schedule_df = schedule_df.merge(
    team_weights_df,
    left_on='Opponent',
    right_on='Team',
    how='left'
).drop(columns=['Team'])

# Fill missing weights
schedule_df['Weight'] = schedule_df['Weight'].fillna(0.0)

# ELO win probability function
def elo_win_prob(chelsea_elo, opponent_elo):
    return 1 / (1 + 10 ** ((opponent_elo - chelsea_elo) / 400))
def weight_return(row):
    venue = row['Venue']
    if venue == 'Home':
        v_w = 0.5
    elif venue == 'Away':
        v_w = -0.2
    return v_w
    

# Adjusted probability incorporating historical weight (including venue)
def adjusted_win_prob(row):
    venue=weight_return(row)
    base_prob = elo_win_prob(row['Chelsea_ELO'], row['Opponent_ELO'])
    adjusted_prob = base_prob + 0.05 * row['Weight']+venue # Weight now includes venue
    return min(max(adjusted_prob, 0), 1)

schedule_df['Chelsea_Win_Prob'] = schedule_df.apply(adjusted_win_prob, axis=1)
# Subtract the weight_return per row from existing 'Weight' column
schedule_df['Weight'] = schedule_df.apply(lambda row: row['Weight'] - weight_return(row), axis=1)


# Predict result
def predict_result(p):
    if p > 0.55:
        return 'W'
    elif p < 0.45:
        return 'L'
    else:
        return 'D'

schedule_df['Predicted_Result'] = schedule_df['Chelsea_Win_Prob'].apply(predict_result)

# Predict goal difference
def predict_gd(row):
    if row['Predicted_Result'] == 'W':
        return 1 + max(0, row['Weight']) * 0.5
    elif row['Predicted_Result'] == 'L':
        return -1 - max(0, -row['Weight']) * 0.5
    else:
        return 0

schedule_df['Predicted_GD'] = schedule_df.apply(predict_gd, axis=1)


#-----------------CHECK OUTPUT----------------------------------------#
print("\nCurrent season matches with predicted results and goal difference:")
print(schedule_df[['MatchDate', 'Round','Venue', 'Opponent','Chelsea_ELO','Opponent_ELO','Weight','Predicted_Result','Predicted_GD']])

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

#-----------------HISTORICAL SCORES DATA--------------------------------#
# Create list of unique opponents in current schedule
opponent_list = schedule_df['Opponent'].unique().tolist()

# Load historical scores
historicscores_df = pd.read_excel(
    "Chelsea_Match_Schedule.xlsx",
    sheet_name="ChelseaMatches",
    usecols=['MatchDate','Time','Round','Day','Venue','Result','GF','GA','Opponent','xG','xGA']
)

# Filter only relevant opponents
historicscores_df = historicscores_df[historicscores_df['Opponent'].isin(opponent_list)]

# Ensure xGD is calculated
historicscores_df['xGD'] = historicscores_df['xG'] - historicscores_df['xGA']
# Ensure numeric columns are numeric and replace only true NaNs with 0
numeric_cols = ['GF','GA','xG','xGA','xGD']

for col in numeric_cols:
    # Convert to numeric, coercing invalid entries to NaN
    historicscores_df[col] = pd.to_numeric(historicscores_df[col], errors='coerce')
    # Replace NaNs with 0, leave existing 0s intact
    historicscores_df[col] = historicscores_df[col].fillna(0)


#-----------------LINEAR REGRESSION MODEL--------------------------------#
# Features: Round, Day, Venue, Opponent, xGD
features = ['Round', 'Day', 'Venue', 'Opponent', 'xGD']

# Target variables: GF and GA
target_GF = 'GF'
target_GA = 'GA'

# Prepare column transformer for categorical variables
categorical_features = ['Round','Day','Venue','Opponent']
numeric_features = ['xGD']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ]
)

# Create pipelines
pipeline_GF = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', LinearRegression())
])

pipeline_GA = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', LinearRegression())
])

# Train models
pipeline_GF.fit(historicscores_df[features], historicscores_df[target_GF])
pipeline_GA.fit(historicscores_df[features], historicscores_df[target_GA])

#-----------------PREDICT FOR CURRENT SEASON--------------------------------#
# Use predicted GD as xGD
schedule_df['xGD'] = schedule_df['Predicted_GD']

# Predict GF and GA
schedule_df['Predicted_GF'] = pipeline_GF.predict(schedule_df[features])
schedule_df['Predicted_GA'] = pipeline_GA.predict(schedule_df[features])

#-----------------ADJUST NEGATIVE GF/GA--------------------------------#
# If Predicted_GF is negative, add 1 to both GF and GA
schedule_df.loc[schedule_df['Predicted_GF'] < 0, ['Predicted_GF','Predicted_GA']] += 1
schedule_df.loc[schedule_df['Predicted_GA'] < 0, ['Predicted_GF','Predicted_GA']] += 1
# Optional: ensure all values are integers
schedule_df['Predicted_GF'] = schedule_df['Predicted_GF'].round().astype(int)
schedule_df['Predicted_GA'] = schedule_df['Predicted_GA'].round().astype(int)

#-----------------CHECK OUTPUT----------------------------------------#
schedule_df['xGD'] = schedule_df['xGD'].round().astype(int)
print("\nAdjusted predicted GF and GA (no negative values):")
print(schedule_df[['MatchDate','Opponent','Predicted_GF','Predicted_GA']])
schedule_df[['MatchDate', 'Round','Venue', 'Opponent','Chelsea_ELO','Opponent_ELO','Weight','Predicted_Result','Predicted_GF', 'Predicted_GA', 'xGD']].to_excel("Chelsea_Weighted_Predicted_Results_Final_SARIMA.xlsx", sheet_name="PredictedResults")

