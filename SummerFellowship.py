# ============================================
# Imports
# ============================================
from tslearn.utils import to_time_series, to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
from glob import glob
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import stumpy
import seaborn as sns
import numpy as np
import pickle
import os

# ============================================
# Load activity list
# ============================================
# Read the CSV file containing activity metadata for all players/games
activities = pd.read_csv("../../data/catapult_activity_list/2021.csv")

# Show the table in a notebook environment
display(activities)


# ============================================
# Function definitions
# ============================================

def grab_files(activity_name):
    """
    Load and prepare GPS files for a given activity (game) into a single DataFrame.

    Parameters
    ----------
    activity_name : str
        The activity/game name used to filter the `activities` table and locate
        player CSVs in '../../data/10hz/'.

    Returns
    -------
    df : pd.DataFrame
        Concatenated GPS data for all found players in the activity. Columns include
        at least ['x', 'y', 'time'], plus:
          - 'y_flip' : float
              Flipped y-coordinate (-1 * y).
          - 'time_rounded' : float
              Time rounded to 0.1s for grouping.
    """
    # ---- Filter source table for the activity ----
    gps_files = activities[activities['Activity_Name'] == activity_name]

    # ---- Build list of valid file paths for each player ----
    file_names = [
        "../../data/10hz/" + row['Activity'] + '_' + row['Player'] + '.csv'
        for i, row in gps_files.iterrows()
        if os.path.exists("../../data/10hz/" + row['Activity'] + '_' + row['Player'] + '.csv')
    ]

    # ---- Read and concatenate all player CSVs ----
    df = pd.concat([pd.read_csv(f) for f in tqdm(file_names)]).reset_index()

    # ---- Engineering convenience columns ----
    df['y_flip'] = -1 * df['y']              # Flip field direction
    df['time_rounded'] = df['time'].round(1) # 10 Hz → 0.1s bins

    return df


def play_times(play_data):
    """
    Build a table of candidate play periods from the video/annotation data.

    Parameters
    ----------
    play_data : pd.DataFrame
        Raw period-level CSV for a game containing at least:
        ['Period Name', 'Unix Start Time', 'Unix End Time'].

    Returns
    -------
    times : pd.DataFrame
        Filtered period table with columns:
          - 'Period Name'
          - 'Unix Start Time'
          - 'Unix End Time'
          - 'Duration' (seconds)
        Only periods with Duration <= 28 are retained.
    """
    # ---- Period names to exclude from template matching ----
    exclude_periods = [
        'O 1', 'D 1', 'O 2', 'O 3', 'D 3', 'O 4', 'D 4', 'D 2', 'D 5', 'O 5',
        'D 6', 'O 6', 'D 7', 'D8', 'O 7', 'D 9', 'O 8', 'D 8', '00D INDY', 'Free Form',
        'Session', '00A FLEX', '00A PRE FLEX', '00B FLEX', '00C INDY', '06 OT 1',
        '01 1ST QUARTER', '02 2ND QUARTER', '03 FLEX', '04 3RD QUARTER', 'O 14',
        '05 4TH QUARTER', 'D 10', 'D 11', 'O 9', 'O 10', 'O 11', 'D 12', 'O 12', 'D 13', 'O 13'
    ]

    # ---- Remove excluded periods ----
    filtered_df = play_data[~play_data['Period Name'].isin(exclude_periods)]

    # ---- Collapse repeats of the same period name to min start / max end ----
    times = filtered_df.groupby('Period Name').agg({
        'Unix Start Time': 'min',  # earliest start per period
        'Unix End Time': 'max'     # latest end per period
    }).reset_index()

    # ---- Compute duration and keep short bursts only ----
    times['Duration'] = times['Unix End Time'] - times['Unix Start Time']
    times = times[times['Duration'] <= 28]

    return times


def grouped_df(df, play_data):
    """
    Restrict GPS data to regulation game time and aggregate metabolic power by rounded time.

    Parameters
    ----------
    df : pd.DataFrame
        Raw GPS data with at least ['time', 'mp'] and optionally 'time_rounded'.
    play_data : pd.DataFrame
        Video/annotation table that includes quarter start/end rows with:
        ['Period Name', 'Unix Start Time', 'Unix End Time'] for:
          - '01 1ST QUARTER'
          - '02 2ND QUARTER'
          - '04 3RD QUARTER'
          - '05 4TH QUARTER'

    Returns
    -------
    grouped_data : pd.DataFrame
        Aggregation of metabolic power by rounded time with columns:
          - 'time_rounded'
          - 'mp' (mean over all athletes at that instant)
    """
    # ---- Extract quarter boundaries from play_data ----
    q1_start_time = play_data[play_data['Period Name'] == '01 1ST QUARTER'].iloc[0]['Unix Start Time']
    q1_end_time   = play_data[play_data['Period Name'] == '01 1ST QUARTER'].iloc[0]['Unix End Time']
    q2_start_time = play_data[play_data['Period Name'] == '02 2ND QUARTER'].iloc[0]['Unix Start Time']
    q2_end_time   = play_data[play_data['Period Name'] == '02 2ND QUARTER'].iloc[0]['Unix End Time']
    q3_start_time = play_data[play_data['Period Name'] == '04 3RD QUARTER'].iloc[0]['Unix Start Time']
    q3_end_time   = play_data[play_data['Period Name'] == '04 3RD QUARTER'].iloc[0]['Unix End Time']
    q4_start_time = play_data[play_data['Period Name'] == '05 4TH QUARTER'].iloc[0]['Unix Start Time']
    q4_end_time   = play_data[play_data['Period Name'] == '05 4TH QUARTER'].iloc[0]['Unix End Time']

    # ---- Keep game-time only (exclude gaps between quarters) ----
    df = df[
        (df['time'] >= q1_start_time) &
        ~((df['time'] > q1_end_time) & (df['time'] < q2_start_time)) &
        ~((df['time'] > q2_end_time) & (df['time'] < q3_start_time)) &
        ~((df['time'] > q3_end_time) & (df['time'] < q4_start_time))
    ]

    # ---- Round time and aggregate mp by 0.1s bins ----
    df['time_rounded'] = df['time'].round(1)
    grouped_data = df.groupby('time_rounded')['mp'].mean().reset_index()
    grouped_data.columns = ['time_rounded', 'mp']

    return grouped_data


def pattern_matches(templates, df, grouped_data, times=times):
    """
    Match metabolic power templates against grouped metabolic power data
    to identify play patterns and collect relevant match statistics.

    Parameters
    ----------
    templates : list of pd.DataFrame
        Each DataFrame is a metabolic power template with columns ['Time', 'mp'].
    df : pd.DataFrame
        Raw GPS data containing columns ['time', 'time_rounded', 'y_flip', 'athlete_id', 'x', 'y_flip', 'mp'].
    grouped_data : pd.DataFrame
        Aggregated GPS data grouped by rounded time with columns ['time_rounded', 'mp'].
    times : pd.DataFrame
        Play period information with columns ['Period Name', 'Unix Start Time', 'Unix End Time', 'Duration'].

    Returns
    -------
    all_matches_df : pd.DataFrame
        DataFrame containing all detected matches with their statistics and period labels.
    """
    # ---- Store all matches from all templates ----
    all_matches = []

    # ---- Pre-sort df for faster querying ----
    df = df.sort_values(by='time').reset_index(drop=True)

    # Convert relevant columns to NumPy arrays for faster access
    df_times = df['time'].values
    df_y_flip = df['y_flip'].values
    df_athlete_ids = df['athlete_id'].values

    grouped_data_times = grouped_data.iloc[:, 0].values
    grouped_data_mp = grouped_data.iloc[:, 1].values

    # ---- Iterate over each template ----
    for index, template in enumerate(templates):
        rows = []  # store matches for current template

        template_mp = template['mp'].values

        # Threshold values for each template
        thresholds = [1.7, 1.9, 8.2, 14.2, 6.1, 2.7]
        threshold = thresholds[index]

        # ---- Find matches using STUMPY ----
        matches = stumpy.match(template_mp, grouped_data_mp, max_distance=lambda D: threshold)

        # ---- Process each match found ----
        for match in matches:
            # Extract match details
            distance, start_index = match
            end_index = start_index + len(template_mp)
            start_time = grouped_data_times[start_index]
            end_time = grouped_data_times[end_index - 1]
            duration = end_time - start_time

            # If duration exceeds 17.2, shorten it
            if duration > 17.2:
                for _ in range(300):
                    end_index = end_index - 1
                    end_time = grouped_data_times[end_index - 1]
                    duration = end_time - start_time
                    if duration <= 17.2:
                        break

            # ---- Filter df rows within this match time range ----
            filtered_df = df[(df['time_rounded'] >= start_time) & (df['time_rounded'] <= end_time)]

            # Get unique athlete IDs in the match
            unique_athletes = filtered_df['athlete_id'].unique()
            combined_data = []

            # ---- For each athlete, calculate movement distances ----
            for athlete in unique_athletes:
                athlete_data = filtered_df[filtered_df['athlete_id'] == athlete]

                movement_data = athlete_data[['x', 'y_flip']].copy()
                movement_data['delta_x'] = movement_data['x'].diff().abs()
                movement_data['delta_y'] = movement_data['y_flip'].diff().abs()

                movement_data['Distance Between Points'] = np.sqrt(
                    (movement_data['delta_x']**2) + (movement_data['delta_y']**2)
                )

                movement_data['athlete_id'] = athlete
                combined_data.append(movement_data)

            # Combine all athletes' data (kept for potential downstream use)
            combined_data_df = pd.concat(combined_data, ignore_index=True)

            # ---- Compute key statistics ----
            mp_max = np.max(grouped_data_mp[start_index:end_index])

            # Locate data slice using binary search
            start_pos = np.searchsorted(df_times, start_time)
            end_pos = np.searchsorted(df_times, end_time, side='right')

            # Determine players "in play" (y_flip < 0)
            in_play_mask = df_y_flip[start_pos:end_pos] < 0
            in_play_data = df_athlete_ids[start_pos:end_pos][in_play_mask]
            athletes = np.unique(in_play_data).size

            # Calculate average number of players on/off field
            play = df.iloc[start_pos:end_pos].copy()
            play['On Field'] = (play['y_flip'] < 0).astype(int)
            play['Off Field'] = (play['y_flip'] >= 0).astype(int)
            athlete_summary = play.groupby('time_rounded').agg({'On Field': 'sum', 'Off Field': 'sum'})
            on_field_avg = athlete_summary['On Field'].mean()
            off_field_avg = athlete_summary['Off Field'].mean()

            # ---- Store match if criteria met ----
            if mp_max > 3.6 and 5 < athletes < 43:
                rows.append({
                    'Game': game_name,
                    'Template Id': index,
                    'Distance': distance,
                    'Start Index': start_index,
                    'End Index': end_index,
                    'Start Time': start_time,
                    'Duration': duration,
                    'End Time': end_time,
                    'Max MP': mp_max,
                    'On Field Avg': on_field_avg,
                    'Off Field Avg': off_field_avg
                })

        # ---- Save matches for this template ----
        rows_df = pd.DataFrame(rows)
        all_matches.append(rows_df)

    # ---- Combine all template matches ----
    all_matches_df = pd.concat(all_matches, ignore_index=True)
    all_matches_df = all_matches_df.sort_values(by='Start Time')

    # ---- Assign Period Names based on closest Unix Start Time (≤ 5s) ----
    for i, row in all_matches_df.iterrows():
        start_time = row['Start Time']

        for j, time in times.iterrows():
            period_name = time['Period Name']
            time_diff = np.abs(time['Unix Start Time'] - start_time)

            # Only include time differences less than or equal to 5 seconds
            if time_diff <= 5:
                all_matches_df.at[i, 'Period Name'] = period_name

    return all_matches_df


# Optional compatibility alias if you call `mp_matches(...)` elsewhere
mp_matches = pattern_matches  # no logic change; just an alias name


# ============================================
# Initial grouped_df call (left exactly as in your script)
# ============================================
# Note: df and play_data must already exist for this line to work as-is.
# Kept here unchanged per your original script.
grouped_data = grouped_df(df=df, play_data=play_data)


# ============================================
# Define game list and build [game_name, game_path] pairs
# ============================================
games = [['CAL'], ['COLORADO'], ['FRESNO STATE'],
         ['OSU'], ['STONY BROOK'], ['WSU']]

game_data = []
mp_play_times = []

for game in games:
    game_name = game[0]
    game_path = f"../../data/video/{game_name}/{game_name}.csv"
    game_data.append([game_name, game_path])


# ============================================
# Collect play windows and build mp_play_times
# ============================================
for game_name, game_path in game_data:
    # ---- Load GPS and play annotation data for this game ----
    df = grab_files(activity_name=game_name)

    # Some video CSVs have header rows; try skipping first 9 rows then fallback
    play_data = pd.read_csv(game_path, skiprows=range(9))
    try:
        play_data['Period Name']
    except KeyError:
        play_data = pd.read_csv(game_path)

    # ---- Aggregate MP by rounded time ----
    grouped_data = grouped_df(df=df, play_data=play_data)

    # ---- Build candidate play windows ----
    times = play_times(play_data=play_data)

    # ---- Expand each play window by ±3s and collect MP segments ----
    for i, time in times.iterrows():
        if time['Duration'] < 40:
            start_time_prelim = time['Unix Start Time']
            start_time = start_time_prelim - 3
            end_time_prelim = time['Unix End Time']
            end_time = end_time_prelim + 3

            # Filter the MP series between the expanded start/end
            data = grouped_data[
                (grouped_data['time_rounded'] > start_time) &
                (grouped_data['time_rounded'] < end_time)
            ]
            mp_data = data['mp']
            mp_play_times.append(mp_data)


# ============================================
# Convert to time series dataset and scale
# ============================================
X = to_time_series_dataset(mp_play_times)                      # ragged → padded 3D tensor
X = TimeSeriesScalerMeanVariance().fit_transform(X)            # z-score per series
X = np.nan_to_num(X, nan=0)                                    # replace any NaNs with 0


# ============================================
# Elbow method to choose k
# ============================================
ranges = range(1, 50)      # start at 1 cluster (k=0 is invalid)
inertia_values = []
seed = 0

for k in ranges:
    km = TimeSeriesKMeans(n_clusters=k, verbose=True, random_state=seed)
    y_pred = km.fit_predict(X)
    inertia_values.append(km.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(ranges, inertia_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()


# ============================================
# Fit KMeans with k=6 and save templates
# ============================================
km = TimeSeriesKMeans(n_clusters=6, verbose=True, random_state=seed)
y_pred = km.fit_predict(X)
inter = km.inertia_

# Save cluster centers for later reuse
np.save("template_results.npy", km.cluster_centers_)


# ============================================
# Plot clusters and centers
# ============================================
plt.figure(figsize=(6, 15))  # stack 6 subplots vertically

for yi in range(6):
    plt.subplot(6, 1, yi + 1)
    for x in X[y_pred == yi]:
        plt.plot(x.ravel(), "k-", alpha=.2)                     # individual series
    plt.plot(km.cluster_centers_[yi].ravel(), "r-", linewidth=2) # cluster center
    plt.xlim(0, X.shape[1])
    plt.ylim(-2.5, 2.5)
    plt.text(0.55, 0.85, f'Cluster {yi + 1}', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()


# ============================================
# Summarize typical length per cluster
# ============================================
# Note: your original code used range(994); preserved here.
x = range(994)
clusters_df = []

for i in x:
    cluster = y_pred[i]
    len_cluster = len(mp_play_times[i])
    clusters_df.append({'Cluster': cluster, 'Length of Cluster': len_cluster})

clusters_dff = pd.DataFrame(clusters_df)
clusters_df_grouped = clusters_dff.groupby('Cluster').mean().round()
clusters_df_grouped['Length of Cluster'] = clusters_df_grouped['Length of Cluster'].astype(int)

print(clusters_df_grouped)  # optional


# ============================================
# Build "centers" DataFrames and plot templates
# ============================================
cluster_centers = km.cluster_centers_
centers = []

for i, center in enumerate(cluster_centers):
    cluster_index = i
    center_length_prelim = clusters_df_grouped.loc[cluster_index, 'Length of Cluster']
    center_length = center_length_prelim + 30  # extend to show a bit more context

    # Create DataFrame for the cluster center
    center_df = pd.DataFrame(center, columns=['mp'])
    center_df = center_df.head(center_length)
    center_df['Time'] = center_df.index
    center_df = center_df[['Time', 'mp']]
    centers.append(center_df)

    # Plot the template curve
    plt.figure(figsize=(10, 5))
    plt.plot(center_df['Time'], center_df['mp'], marker='o', linestyle='-', color='b')
    plt.xlabel('Time')
    plt.ylabel('mp')
    plt.title(f'Cluster {cluster_index} - Metabolic Power Template')
    plt.grid(True)
    plt.show()


# ============================================
# Load clustered_desss from pickle
# ============================================
with open('clustered_desss.pkl', 'rb') as f:
    clustered_desss = pickle.load(f)


# ============================================
# Run matching per game
# ============================================
games = [['CAL'], ['COLORADO'], ['FRESNO STATE'], ['OSU'], ['STONY BROOK'], ['WSU']]

# List to store all individual match DataFrames
prelim_all_matches_combined = []
play_annotations = []

for game in games:
    game_name = game[0]
    game_path = f"../../data/video/{game_name}/{game_name}.csv"

    # ---- Load GPS and play annotation data ----
    df = grab_files(activity_name=game_name)

    play_data = pd.read_csv(game_path, skiprows=range(9))
    try:
        play_data['Period Name']
    except KeyError:
        play_data = pd.read_csv(game_path)

    # ---- Aggregate MP by rounded time and compute playable windows ----
    grouped_data = grouped_df(df=df, play_data=play_data)
    times = play_times(play_data=play_data)

    # ---- Match templates to grouped MP series ----
    # Your original script called mp_matches(...). We preserved that call.
    all_matches_df = mp_matches(templates=centers, df=df, grouped_data=grouped_data, times=times)

    print(all_matches_df)  # optional
    prelim_all_matches_combined.append(all_matches_df)
    play_annotations.append(times)

# ---- Combine all games' matches into one DataFrame ----
prelim_combined = pd.concat(prelim_all_matches_combined)

