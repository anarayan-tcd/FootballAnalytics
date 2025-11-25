import pandas as pd
def season_to_year(season, month_in_season):
    """
    Convert Premier League season + month_in_season to actual date.
    month_in_season: 1 → 10, starting from August of season start year
    """
    start_year = int(season.split("/")[0])
    
    # Mapping month_in_season (1-10) → actual month
    month_map = {
        1: 8,   # August
        2: 9,   # September
        3: 10,  # October
        4: 11,  # November
        5: 12,  # December
        6: 1,   # January (next year)
        7: 2,   # February
        8: 3,   # March
        9: 4,   # April
        10: 5   # May
    }
    
    actual_month = month_map[month_in_season]
    actual_year = start_year if actual_month >= 8 else start_year + 1
    
    return pd.Timestamp(year=actual_year, month=actual_month, day=1)

df = pd.read_excel("IMP/All_Teams_ELO_By_Month.xlsx")
df['date'] = df.apply(lambda x: season_to_year(x['season'], x['month_in_season']), axis=1)

# Keep only needed columns
df = df[['team','date','avg_elo']]

# Overwrite the original Excel file
df.to_excel("All_Teams_ELO_By_Month.xlsx", index=False)
