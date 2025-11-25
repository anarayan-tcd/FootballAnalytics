import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# -----------------------------
# STEP 1: Load EPL and Chelsea Player Stats
# -----------------------------
epl_df = pd.read_excel("EPL_Player_Stats.xlsx", sheet_name="PlayerStats")
chelsea_df = pd.read_excel("Chelsea_Player_Stats.xlsx")

columns_to_keep = [
    "season","team","player","nation","pos","born","MP","Starts","Min","90s",
    "Gls","Ast","G+A","G-PK","PK","PKatt","CrdY","CrdR","PrgC","PrgP","PrgR",
    "Gls90","Ast90","G+A90","G-PK90","G+A-PK90"
]

epl_chelsea_df = epl_df[epl_df['team']=="Chelsea"][columns_to_keep].reset_index(drop=True)

# -----------------------------
# STEP 2: Add Multirole column
# -----------------------------
epl_chelsea_df['Multirole'] = epl_chelsea_df['pos'].apply(lambda x: 'yes' if ',' in str(x) else 'no')

# -----------------------------
# STEP 3: Split into FW, MF, DF, GK
# -----------------------------
def split_positions(df):
    fw_df, mf_df, df_df, gk_df = [pd.DataFrame(columns=df.columns) for _ in range(4)]
    for _, row in df.iterrows():
        positions = [p.strip() for p in row['pos'].split(',')]
        if 'FW' in positions:
            fw_df = pd.concat([fw_df, pd.DataFrame([row])], ignore_index=True)
        if 'MF' in positions:
            mf_df = pd.concat([mf_df, pd.DataFrame([row])], ignore_index=True)
        if 'DF' in positions:
            df_df = pd.concat([df_df, pd.DataFrame([row])], ignore_index=True)
        if 'GK' in positions:
            gk_df = pd.concat([gk_df, pd.DataFrame([row])], ignore_index=True)
    # Drop pos column
    for df_temp in [fw_df, mf_df, df_df, gk_df]:
        if 'pos' in df_temp.columns:
            df_temp.drop(columns=['pos'], inplace=True)
    return fw_df, mf_df, df_df, gk_df

fw_df, mf_df, df_df, gk_df = split_positions(epl_chelsea_df)

# -----------------------------
# STEP 4: Load Defensive Stats and update DF and MF DataFrames
# -----------------------------
def_df = pd.read_excel("EPL_Player_Stats_DEF.xlsx")
def_df = def_df[(def_df['team']=="Chelsea") & (def_df['pos'].str.contains("DF|MF", na=False))].reset_index(drop=True)
def_df['Multirole'] = def_df['pos'].apply(lambda x: 'yes' if ',' in str(x) else 'no')

# Update df_df with defensive stats where pos is DF
df_df_cols_keep = ['MP']
df_cols = ['season','team','player','nation','born','90s','Tkl','TklW','Tkl%','Blocks','Int','Tkl+Int','Clr','Err','Multirole']
for idx, row in df_df.iterrows():
    match = def_df[(def_df['player']==row['player']) & (def_df['season']==row['season']) & (def_df['pos'].str.contains("DF"))]
    if not match.empty:
        for col in def_df.columns:
            if col not in df_df_cols_keep and col in df_df.columns or col in ['90s','Tkl','TklW','Tkl%','Blocks','Int','Tkl+Int','Clr','Err','Multirole']:
                df_df.loc[idx, col] = match.iloc[0][col]

# Append defensive stats to MF where player and season match
mf_def_cols = ['Tkl','TklW','Tkl%','Blocks','Int','Tkl+Int','Clr','Err']
for col in mf_def_cols:
    mf_df[col] = np.nan
for idx, row in mf_df.iterrows():
    match = def_df[(def_df['player']==row['player']) & (def_df['season']==row['season'])]
    if not match.empty:
        for col in mf_def_cols:
            mf_df.loc[idx, col] = match.iloc[0][col]

# -----------------------------
# STEP 5: Compute weighted average stats for each player, per DF, FW, MF
# -----------------------------
def compute_weighted_average(df, avg_cols, season_col='season'):
    avg_rows = []
    for player in df['player'].unique():
        player_rows = df[df['player']==player]
        # Weight by season, higher season more weight
        weights = player_rows[season_col].astype(float)
        avg_data = {}
        for col in avg_cols:
            avg_data[col] = np.average(player_rows[col].astype(float), weights=weights)
        # Copy other columns from the first row
        first_row = player_rows.iloc[0].to_dict()
        first_row.update(avg_data)
        first_row['season'] = 2025
        first_row['Average'] = 'yes'
        avg_rows.append(first_row)
    # Add Average column = no for existing rows
    df['Average'] = 'no'
    df = pd.concat([df, pd.DataFrame(avg_rows)], ignore_index=True)
    return df

# Define columns for weighted averages
df_avg_cols = ['90s','Tkl','TklW','Tkl%','Blocks','Int','Tkl+Int','Clr','Err','MP']
fw_avg_cols = ['MP','Starts','Min','90s','Gls','Ast','G+A','G-PK','PK','PKatt','CrdY','CrdR',
               'PrgC','PrgP','PrgR','Gls90','Ast90','G+A90','G-PK90','G+A-PK90']
mf_avg_cols = fw_avg_cols + ['Tkl','TklW','Tkl%','Blocks','Int','Tkl+Int','Clr','Err']

df_df = compute_weighted_average(df_df, df_avg_cols)
fw_df = compute_weighted_average(fw_df, fw_avg_cols)
mf_df = compute_weighted_average(mf_df, mf_avg_cols)

# Add Average column for GK as no (we do not average)
gk_df['Average'] = 'no'

# -----------------------------
# STEP 6: Sort each DataFrame by player name
# -----------------------------
for df_temp in [fw_df, mf_df, df_df, gk_df]:
    df_temp.sort_values('player', inplace=True)
    df_temp.reset_index(drop=True, inplace=True)

# -----------------------------
# STEP 7: Print only 2025 rows
# -----------------------------
print("FW 2025 Rows:\n", fw_df[fw_df['season']==2025])
print("MF 2025 Rows:\n", mf_df[mf_df['season']==2025])
print("DF 2025 Rows:\n", df_df[df_df['season']==2025])
print("GK 2025 Rows:\n", gk_df[gk_df['season']==2025])
# -----------------------------
# Print 2025 rows
# -----------------------------
print("FW 2025 Rows:\n", fw_df)
print("MF 2025 Rows:\n", mf_df)
print("DF 2025 Rows:\n", df_df)
print("GK 2025 Rows:\n", gk_df)

# -----------------------------
# Save all 4 DataFrames into one Excel workbook
# -----------------------------
output_file = "Chelsea_Player_Stats_2025.xlsx"

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    fw_df.to_excel(writer, sheet_name='FW', index=False)
    mf_df.to_excel(writer, sheet_name='MF', index=False)
    df_df.to_excel(writer, sheet_name='DF', index=False)
    gk_df.to_excel(writer, sheet_name='GK', index=False)

print(f"All 4 DataFrames saved to {output_file}")

# # -----------------------------
# # STEP 8: Clustering on 2025 rows
# # -----------------------------
# def cluster_df(df, n_clusters=4):
#     df_2025 = df[df['season']==2025].copy()
#     numeric_cols = df_2025.select_dtypes(include=np.number).columns
#     X = df_2025[numeric_cols].fillna(0)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     df_2025['Cluster'] = kmeans.fit_predict(X_scaled)
#     return df_2025

# fw_clusters = cluster_df(fw_df)
# mf_clusters = cluster_df(mf_df)
# df_clusters = cluster_df(df_df)
# #gk_clusters = cluster_df(gk_df)

# # Preview clustered 2025 data
# print("FW Clusters:\n", fw_clusters.head())
# print("MF Clusters:\n", mf_clusters.head())
# print("DF Clusters:\n", df_clusters.head())
# #print("GK Clusters:\n", gk_clusters.head())
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from scipy.spatial import ConvexHull
# import numpy as np

# def plot_clusters_with_labels(df_clustered, title):
#     # Select numeric columns for PCA
#     numeric_cols = df_clustered.select_dtypes(include=np.number).columns.drop('Cluster')
#     X = df_clustered[numeric_cols].fillna(0)
    
#     # PCA to 2D
#     pca = PCA(n_components=2, random_state=42)
#     X_pca = pca.fit_transform(X)
    
#     df_clustered['PCA1'] = X_pca[:,0]
#     df_clustered['PCA2'] = X_pca[:,1]
    
#     plt.figure(figsize=(10,8))
    
#     # Plot each cluster with convex hull
#     for cluster in df_clustered['Cluster'].unique():
#         cluster_data = df_clustered[df_clustered['Cluster']==cluster]
#         plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}', s=100, alpha=0.6)
        
#         # Draw convex hull if there are enough points
#         if len(cluster_data) >= 3:
#             points = cluster_data[['PCA1','PCA2']].values
#             hull = ConvexHull(points)
#             hull_points = np.append(hull.vertices, hull.vertices[0])  # close the loop
#             plt.plot(points[hull_points,0], points[hull_points,1], linestyle='--', color='black')
    
#         # # Add player labels
#         # for _, row in cluster_data.iterrows():
#         #     plt.text(row['PCA1']+0.02, row['PCA2']+0.02, row['player'], fontsize=8)
    
#     plt.title(title, fontsize=14)
#     plt.xlabel('PCA 1')
#     plt.ylabel('PCA 2')
#     plt.grid(True)
#     plt.legend()
#     plt.show()

# # Plot each position with labels and outlines
# plot_clusters_with_labels(fw_clusters, "FW 2025 Clusters with Player Labels")
# plot_clusters_with_labels(mf_clusters, "MF 2025 Clusters with Player Labels")
# plot_clusters_with_labels(df_clusters, "DF 2025 Clusters with Player Labels")
# #plot_clusters_with_labels(gk_clusters, "GK 2025 Clusters with Player Labels")
# # Define output file name
# output_file = "Chelsea_Player_Stats_2025_Clusters.xlsx"

# # Create a dictionary mapping sheet names to DataFrames
# dfs_to_save = {
#     "FW": fw_clusters,
#     "MF": mf_clusters,
#     "DF": df_clusters,
#     #"GK": gk_clusters
# }

# # Write to Excel
# with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
#     for sheet_name, df in dfs_to_save.items():
#         df.to_excel(writer, sheet_name=sheet_name, index=False)

# print(f"All 4 DataFrames saved to {output_file}")
