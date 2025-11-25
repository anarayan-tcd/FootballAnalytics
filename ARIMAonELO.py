# import pandas as pd
# from statsmodels.tsa.arima.model import ARIMA

# #-----------------FUNCTIONS----------------------------------------------
# def season_to_year(season, month_in_season):
#     """
#     Convert Premier League season + month_in_season to actual date.
#     month_in_season: 1 → 10, starting from August of season start year
#     """
#     start_year = int(season.split("/")[0])
    
#     # Mapping month_in_season (1-10) → actual month
#     month_map = {
#         1: 8,   # August
#         2: 9,   # September
#         3: 10,  # October
#         4: 11,  # November
#         5: 12,  # December
#         6: 1,   # January (next year)
#         7: 2,   # February
#         8: 3,   # March
#         9: 4,   # April
#         10: 5   # May
#     }
    
#     actual_month = month_map[month_in_season]
#     actual_year = start_year if actual_month >= 8 else start_year + 1
    
#     return pd.Timestamp(year=actual_year, month=actual_month, day=1)
# #------------------------------------------------------------------------

# # Load data
# df = pd.read_excel("All_Teams_ELO_By_Month.xlsx")
# df['date'] = df.apply(lambda x: season_to_year(x['season'], x['month_in_season']), axis=1)

# # Keep only needed columns
# df = df[['team','date','avg_elo']]

# # List of teams
# teams_list = ['Chelsea', 'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Burnley',
#               'Crystal Palace', 'Everton', 'Fulham', 'Leeds', 'Liverpool',
#               'Man City', 'Man United', 'Newcastle', "Forest",
#               'Sunderland', 'Tottenham', 'West Ham', 'Wolves']

# team_forecasts = {}

# for team in teams_list:
#     # Filter team data
#     team_df = df[df['team'] == team].sort_values('date')
    
#     # Set datetime index with monthly frequency to avoid statsmodels warning
#     team_df = team_df.set_index('date').asfreq('MS')
    
#     # Fit ARIMA model
#     model = ARIMA(team_df['avg_elo'], order=(1,1,1))
#     model_fit = model.fit()
    
#     # Forecast next 10 months
#     forecast = model_fit.forecast(steps=10)
    
#     # Create forecast dates starting from next month (should be August after last month)
#     last_date = team_df.index.max()
    
#     # Adjust so forecast always starts from August after last season
#     if last_date.month < 8:
#         # last_date in Jan-May → start from August same year
#         start_year = last_date.year
#     else:
#         # last_date in Aug-Dec → start from August next year
#         start_year = last_date.year + 1
    
#     forecast_dates = pd.date_range(start=pd.Timestamp(year=start_year, month=8, day=1), periods=10, freq='MS')
    
#     # Create forecast dataframe
#     forecast_df = pd.DataFrame({
#         'team': team,
#         'date': forecast_dates,
#         'avg_elo_forecast': forecast
#     })
    
#     # Save to dictionary
#     team_forecasts[team] = forecast_df

# # Combine all team forecasts
# all_forecasts = pd.concat(team_forecasts.values())
# all_forecasts_reset = all_forecasts.reset_index(drop=True)

# # Save to CSV (Power BI compatible)
# # all_forecasts_reset.to_csv("team_elo_forecast.csv", index=False)

# print("Forecast completed and saved to team_elo_forecast.csv")

import pandas as pd
import numpy as np
import warnings
import statsmodels.api as sm

# Silence ALL warnings globally (SARIMAX convergence, overflow, etc.)
warnings.filterwarnings("ignore")

# Fourier generator
def fourier_terms(index, period, K):
    t = np.arange(len(index))
    X = pd.DataFrame(
        {f"sin_{period}_{k}": np.sin(2*np.pi*k*t/period) for k in range(1, K+1)},
        index=index
    )
    for k in range(1, K+1):
        X[f"cos_{period}_{k}"] = np.cos(2*np.pi*k*t/period)
    return X

# Load data
df = pd.read_excel("IMP/All_Teams_ELO_By_Month.xlsx")
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df = df[['team', 'date', 'avg_elo']]

teams_list = [
    'Chelsea','Arsenal','Aston Villa','Bournemouth','Brentford',
    'Brighton','Burnley','Crystal Palace','Everton','Fulham',
    'Leeds','Liverpool','Man City','Man United','Newcastle',
    'Forest','Sunderland','Tottenham','West Ham','Wolves'
]

team_forecasts = {}

# Season lengths
season_period  = 10
quarter_period = 3

# Fourier components
K_season  = 2
K_quarter = 1

# Reduced & stable ARIMA search grid
p = d = q = [0, 1]
param_grid = [(pi, di, qi) for pi in p for di in d for qi in q]

for team in teams_list:

    team_df = df[df['team'] == team].sort_values('date')
    team_df = team_df.set_index('date').asfreq('MS')
    y = team_df['avg_elo']

    # Fourier regressors
    X = pd.concat([
        fourier_terms(y.index, season_period,  K_season),
        fourier_terms(y.index, quarter_period, K_quarter)
    ], axis=1)

    best_aic = np.inf
    best_model = None

    for (pi, di, qi) in param_grid:
        try:
            model = sm.tsa.statespace.SARIMAX(
                y,
                exog=X,
                order=(pi, di, qi),
                seasonal_order=(0,0,0,0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            # Fit with bounded iterations
            result = model.fit(disp=False, maxiter=50)

            # Skip models with crazy likelihood values
            if np.isnan(result.aic) or np.isinf(result.aic):
                continue

            if result.aic < best_aic:
                best_aic = result.aic
                best_model = result

        except Exception:
            # Skip models that fail in fitting
            continue

    # Forecast future regressors
    future_idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=10, freq='MS')
    X_future = pd.concat([
        fourier_terms(future_idx, season_period, K_season),
        fourier_terms(future_idx, quarter_period, K_quarter)
    ], axis=1)

    forecast_values = best_model.get_forecast(steps=10, exog=X_future).predicted_mean

    # Align with your August rule
    last_date = y.index.max()
    start_year = last_date.year if last_date.month < 8 else last_date.year + 1
    forecast_dates = pd.date_range(start=pd.Timestamp(start_year, 8, 1), periods=10, freq='MS')

    forecast_df = pd.DataFrame({
        'team': team,
        'date': forecast_dates,
        'avg_elo_forecast': forecast_values.values
    })

    team_forecasts[team] = forecast_df

# Combine outputs
all_forecasts = pd.concat(team_forecasts.values()).reset_index(drop=True)
all_forecasts.to_excel("team_elo_forecast_stable_SARIMAX.xlsx", index=False)

print("Finished clean SARIMAX forecasting with Fourier seasonality.")
