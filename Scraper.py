import soccerdata as sd
from soccerdata import FBref
from soccerdata import ClubElo
import datetime as dt
import pandas as pd
import numpy as np

season_bounds = {
    "2021/22": (pd.Timestamp("2021-08-14"), pd.Timestamp("2022-05-22")),
    "2022/23": (pd.Timestamp("2022-08-05"), pd.Timestamp("2023-05-28")),
    "2023/24": (pd.Timestamp("2023-08-11"), pd.Timestamp("2024-05-19")),
    "2024/25": (pd.Timestamp("2024-08-16"), pd.Timestamp("2025-05-25")),
}

def assign_season_quarter(date):
    """
    Given a timestamp, return (season_str, quarter_int)
    quarter: 1=first quarter of season, …, 4=last quarter
    """
    for season, (start, end) in season_bounds.items():
        if start <= date <= end:
            # compute relative position in season as fraction (0..1)
            total = (end - start).days
            offset = (date - start).days
            frac = offset / total
            # quarters based on frac
            if frac < 0.25:
                q = 1
            elif frac < 0.5:
                q = 2
            elif frac < 0.75:
                q = 3
            else:
                q = 4
            return pd.Series({"season": season, "quarter": q})
    # if it doesn’t fall into any season
    return pd.Series({"season": None, "quarter": None})

def assign_season_month(date):
    """
    Given a timestamp, return (season_str, month_in_season)
    month_in_season: 1 = first month of season, etc.
    """
    for season, (start, end) in season_bounds.items():
        if start <= date <= end:
            # compute number of months since season start
            year_diff = date.year - start.year
            month_diff = date.month - start.month
            month_in_season = year_diff * 12 + month_diff + 1  # +1 to start from 1
            return pd.Series({"season": season, "month_in_season": month_in_season})
    # if it doesn’t fall into any season
    return pd.Series({"season": None, "month_in_season": None})


elo = ClubElo()
s25_cutoff = dt.datetime(2025, 7, 1)
start_date = dt.datetime(2021, 7,1)
#get chelsea's ELO from 2021/22 till now
# chelsea_elo = elo.read_team_history("Chelsea")
# # print(chelsea_elo.dtypes)
# # print(chelsea_elo.head())
# #chelsea's elo for this season
# chelsea_elo_25 = chelsea_elo[chelsea_elo['to'] >= s25_cutoff]
# chelsea_elo_25.to_excel("Chelsea_ELO_25.xlsx", sheet_name= "Chelsea Elo 25", index=False)
# print("Chelsea's ELO for the 25/26 Season:")
# print(chelsea_elo_25)
# #chelsea's elo for prior seasons
# chelsea_elo = chelsea_elo[chelsea_elo['to'] >= start_date]
# chelsea_elo = chelsea_elo[chelsea_elo['to'] <= s25_cutoff]
# print("Chelsea's ELO for seasons 21/22 to 24/25:")
# print(chelsea_elo)
# #chelsea_elo.to_excel("Chelsea_ELO_21to25.xlsx", sheet_name= "Chelsea Elo 21-25", index=False)
# #get Chelsea's match schedule
fbref = FBref(leagues='ENG-Premier League', seasons=[20,21,22,23,24])
# # chelsea_match_stats = fbref.read_team_match_stats(stat_type='schedule', team='Chelsea')
# # print(chelsea_match_stats)
player_stats = fbref.read_player_season_stats(stat_type = 'defense')
# print(player_stats.dtypes)
# print(player_stats.head())
# # print(player_stats['player'].to_string())
player_stats.to_excel("EPL_Player_Stats_DEF.xlsx", sheet_name= "PlayerStats")

#chelsea_match_stats.to_excel("Chelsea_Match_Schedule.xlsx", sheet_name= "Chelsea Matches", index=False)
# opponents_list = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Burnley',
#  'Crystal Palace', 'Everton', 'Fulham', 'Leeds', 'Liverpool',
#  'Man City', 'Man United', 'Newcastle', "Forest",
#  'Sunderland', 'Tottenham', 'West Ham', 'Wolves']

# all_avg_elo = pd.DataFrame(columns=["team", "season", "quarter", "avg_elo"])

# i = elo.read_team_history("Chelsea")
# i = i[i['to'] >= start_date]
# i = i[i['to'] <= s25_cutoff]
# i = i.copy()  # to avoid modifying in-place if needed
# i[['season', 'quarter']] = i['to'].apply(assign_season_quarter)

# # Drop anything outside the seasons (optional but often useful)
# i = i.dropna(subset=['season', 'quarter'])

# # Now group by team (if you want per-opponent), season, and quarter
# avg_elo = (i
#         .groupby(['team', 'season', 'quarter'])
#         .agg(avg_elo=('elo', 'mean'))
#         .reset_index())
# all_avg_elo = pd.concat([all_avg_elo, avg_elo], ignore_index=True)

# print(all_avg_elo.head())
# all_avg_elo.to_excel("Chelsea_ELO_Quarters.xlsx", sheet_name="ELO By Quarter", index=False)

# import pandas as pd

# # List of all teams
# opponents_list = ['Chelsea', 'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Burnley',
#                   'Crystal Palace', 'Everton', 'Fulham', 'Leeds', 'Liverpool',
#                   'Man City', 'Man United', 'Newcastle', "Forest",
#                   'Sunderland', 'Tottenham', 'West Ham', 'Wolves']

# # Empty DataFrame to store all results
# all_avg_elo = pd.DataFrame(columns=["team", "season", "month_in_season", "avg_elo"])

# for team_name in opponents_list:
#     # Read team history
#     i = elo.read_team_history(team_name)
    
#     # Filter for relevant dates
#     i = i[i['to'] >= start_date]
#     i = i[i['to'] <= s25_cutoff]
#     i = i.copy()  # avoid modifying in-place
    
#     # Assign season and month in season
#     i[['season', 'month_in_season']] = i['to'].apply(assign_season_month)
    
#     # Drop rows outside defined seasons
#     i = i.dropna(subset=['season', 'month_in_season'])
    
#     # Compute average ELO per month in each season
#     avg_elo = (
#         i.groupby(['team', 'season', 'month_in_season'])
#         .agg(avg_elo=('elo', 'mean'))  # mean over matchday ELOs in that month
#         .reset_index()
#     )
    
#     # Append to master DataFrame
#     all_avg_elo = pd.concat([all_avg_elo, avg_elo], ignore_index=True)

# # View top rows
# print(all_avg_elo.head(20))

# # Save all teams' monthly ELO to Excel
# all_avg_elo.to_excel("All_Teams_ELO_By_Month.xlsx", sheet_name="ELO By Month", index=False)




