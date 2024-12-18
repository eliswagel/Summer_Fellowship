from tslearn.utils import to_time_series
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
from glob import glob
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import stumpy
import seaborn as sns
import numpy as np

# Read the CSV file containing activity data
activities = pd.read_csv("../../data/catapult_activity_list/2021.csv")
# Display the DataFrame in the notebook
display(activities)



# Call the grouped_df function with the df and play_data to calculate the average metabolic power
grouped_data = grouped_df(df = df, play_data = play_data)


# Define game list
games = [['CAL'], ['COLORADO'], ['FRESNO STATE'],
         ['OSU'], ['STONY BROOK'], ['WSU']]

game_data = []
mp_play_times = []

# Iterate through the games list and create paths
for game in games:
    game_name = game[0]
    game_path = f"../../data/video/{game_name}/{game_name}.csv"
    game_data.append([game_name, game_path])

# Process each game
for game_name, game_path in game_data:
    df = grab_files(activity_name=game_name)
    play_data = pd.read_csv(game_path, skiprows=range(9))
    try:
        play_data['Period Name']
    except KeyError:
        play_data = pd.read_csv(game_path)

    grouped_data = grouped_df(df=df, play_data=play_data)
    times = play_times(play_data=play_data)
    
    for i, time in times.iterrows():
        if time['Duration'] < 40:
            start_time_prelim = time['Unix Start Time']
            start_time = start_time_prelim - 3
            end_time_prelim = time['Unix End Time']
            end_time = end_time_prelim + 3
            # Filter the acceleration data based on the rounded time
            data = grouped_data[(grouped_data['time_rounded'] > start_time) & (grouped_data['time_rounded'] < end_time)]
            mp_data = data['mp']
            mp_play_times.append(mp_data)


# Convert to a time series dataset
X = to_time_series_dataset(mp_play_times)
# Scale the data
X = TimeSeriesScalerMeanVariance().fit_transform(X)
# Replace NaN values with 0
X = np.nan_to_num(X, nan=0)


# Define the range of clusters and initialize an empty list to store inertia values
ranges = range(1, 50)  # Starts at 1 to avoid having 0 clusters
inertia_values = []
seed = 0

# Iterate over the range of clusters, fit the model, and collect the inertia values
for k in ranges:
    km = TimeSeriesKMeans(n_clusters=k, verbose=True, random_state=seed)
    y_pred = km.fit_predict(X)
    inertia_values.append(km.inertia_)

# Plot inertia against the number of clusters
plt.figure(figsize=(10, 6))
plt.plot(ranges, inertia_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()




# Perform clustering with 6 clusters
km = TimeSeriesKMeans(n_clusters=6, verbose=True, random_state=seed)
y_pred = km.fit_predict(X)
inter = km.inertia_

# Save the cluster centers to a .npy file
np.save("template_results.npy", km.cluster_centers_)

# Plot each cluster
plt.figure(figsize=(6, 15))  # Increase figure size to fit all subplots

for yi in range(6):  # Loop through each of the clusters
    plt.subplot(6, 1, yi + 1)  # Create a 6x1 grid of subplots
    for x in X[y_pred == yi]:  # Iterate over each time series in the cluster
        plt.plot(x.ravel(), "k-", alpha=.2)  # Plot each time series in the cluster
    plt.plot(km.cluster_centers_[yi].ravel(), "r-", linewidth=2)  # Plot cluster center
    plt.xlim(0, X.shape[1])  # Set x-axis limits based on time series length
    plt.ylim(-2.5, 2.5)  # Set y-axis limits (adjust based on your data)
    plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1), transform=plt.gca().transAxes)  # Cluster label

plt.tight_layout()  # Adjust layout to avoid overlapping subplots
plt.show()  # Show the entire figure with all subplots


x = range(994)
clusters_df = []

for i in x:
    cluster = y_pred[i]
    len_cluster = len(mp_play_times[i])
    clusters_df.append({'Cluster': cluster, 'Length of Cluster': len_cluster})

clusters_dff = pd.DataFrame(clusters_df)
clusters_df_grouped = clusters_dff.groupby('Cluster').mean().round()
clusters_df_grouped['Length of Cluster'] = clusters_df_grouped['Length of Cluster'].astype(int)

# Display grouped DataFrame (optional)
print(clusters_df_grouped)

# Second Code Block: Process cluster centers and plot metabolic power templates
cluster_centers = km.cluster_centers_
centers = []

for i, center in enumerate(cluster_centers):
    cluster_index = i
    center_length_prelim = clusters_df_grouped.loc[cluster_index, 'Length of Cluster']
    center_length = center_length_prelim + 30  # Adjust length

    # Create DataFrame for the cluster center
    center_df = pd.DataFrame(center, columns=['mp'])
    center_df = center_df.head(center_length)
    center_df['Time'] = center_df.index
    center_df = center_df[['Time', 'mp']]
    centers.append(center_df)

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(center_df['Time'], center_df['mp'], marker='o', linestyle='-', color='b')
    plt.xlabel('Time')
    plt.ylabel('mp')
    plt.title(f'Cluster {cluster_index} - Metabolic Power Template')
    plt.grid(True)
    plt.show()





# Open the pickle file in read-binary mode
with open('clustered_desss.pkl', 'rb') as f:
    # Load the data from the pickle file
    clustered_desss = pickle.load(f)





games = [['CAL'], ['COLORADO'], ['FRESNO STATE'], ['OSU'], ['STONY BROOK'], ['WSU']]
# List to store all individual all_matches_df DataFrames
prelim_all_matches_combined = []
play_annotations = []
# Process each game
for game in games:
    game_name = game[0]
    game_path = f"../../data/video/{game_name}/{game_name}.csv"
    
    # Read and process files
    df = grab_files(activity_name=game_name)
    
    play_data = pd.read_csv(game_path, skiprows=range(9))
    try:
        play_data['Period Name']
    except KeyError:
        play_data = pd.read_csv(game_path)
    
    grouped_data = grouped_df(df=df, play_data=play_data)
    times = play_times(play_data=play_data)
    all_matches_df = mp_matches(templates = centers, df = df, grouped_data = grouped_data, times = times)
    print(all_matches_df)
    prelim_all_matches_combined.append(all_matches_df)
    play_annotations.append(times)
    
prelim_combined = pd.concat(prelim_all_matches_combined)



def grab_files(activity_name):
    # Filter the activities DataFrame for the given activity_name
    gps_files = activities[activities['Activity_Name'] == activity_name]
    # Generate a list of valid file paths for the corresponding activity and player
    file_names = [
        "../../data/10hz/" + row['Activity'] + '_' + row['Player'] + '.csv' 
        for i, row in gps_files.iterrows() 
        if os.path.exists("../../data/10hz/" + row['Activity'] + '_' + row['Player'] + '.csv')
    ]
    # Read and concatenate all the valid CSV files into a single DataFrame
    df = pd.concat([pd.read_csv(f) for f in tqdm(file_names)]).reset_index()
    # Flip the 'y' values
    df['y_flip'] = -1 * df['y']
    # Round the 'time' values to one decimal place
    df['time_rounded'] = df['time'].round(1)
    # Return the processed DataFrame
    return df

def play_times(play_data):
    # List of period names to exclude from the analysis
    exclude_periods = [ 
        'O 1', 'D 1', 'O 2', 'O 3', 'D 3', 'O 4', 'D 4', 'D 2', 'D 5', 'O 5',
        'D 6', 'O 6', 'D 7', 'D8', 'O 7', 'D 9', 'O 8', 'D 8', '00D INDY', 'Free Form',
        'Session', '00A FLEX', '00A PRE FLEX', '00B FLEX', '00C INDY', '06 OT 1',
        '01 1ST QUARTER', '02 2ND QUARTER', '03 FLEX', '04 3RD QUARTER', 'O 14',
        '05 4TH QUARTER', 'D 10', 'D 11', 'O 9', 'O 10', 'O 11', 'D 12', 'O 12', 'D 13', 'O 13'
    ]
    # Create new dataframe excluding rows with period names listed in 'exclude_periods'
    filtered_df = play_data[~play_data['Period Name'].isin(exclude_periods)]
    # Compute the minimum 'Unix Start Time' and maximum 'Unix End Time' for each period
    times = filtered_df.groupby('Period Name').agg({
        'Unix Start Time': 'min',  # Get the earliest start time for each period
        'Unix End Time': 'max'     # Get the latest end time for each period
    }).reset_index()
    # Calculate the duration for each period by subtracting 'Unix Start Time' from 'Unix End Time'
    times['Duration'] = times['Unix End Time'] - times['Unix Start Time']
    # Filter out periods where the duration is greater than 28 (duration threshold)
    times = times[times['Duration'] <= 28]
    # Return the DataFrame with the filtered periods and their durations
    return times


def grouped_df(df, play_data):
    # Define the time ranges for each quarter using the play_data
    q1_start_time = play_data[play_data['Period Name'] == '01 1ST QUARTER'].iloc[0]['Unix Start Time']
    q1_end_time = play_data[play_data['Period Name'] == '01 1ST QUARTER'].iloc[0]['Unix End Time']
    q2_start_time = play_data[play_data['Period Name'] == '02 2ND QUARTER'].iloc[0]['Unix Start Time']
    q2_end_time = play_data[play_data['Period Name'] == '02 2ND QUARTER'].iloc[0]['Unix End Time']
    q3_start_time = play_data[play_data['Period Name'] == '04 3RD QUARTER'].iloc[0]['Unix Start Time']
    q3_end_time = play_data[play_data['Period Name'] == '04 3RD QUARTER'].iloc[0]['Unix End Time']
    q4_start_time = play_data[play_data['Period Name'] == '05 4TH QUARTER'].iloc[0]['Unix Start Time']
    q4_end_time = play_data[play_data['Period Name'] == '05 4TH QUARTER'].iloc[0]['Unix End Time']
    # Filter out data that falls before Q1 start time and in between quarters
    df = df[
        (df['time'] >= q1_start_time) & 
        ~((df['time'] > q1_end_time) & (df['time'] < q2_start_time)) &
        ~((df['time'] > q2_end_time) & (df['time'] < q3_start_time)) &
        ~((df['time'] > q3_end_time) & (df['time'] < q4_start_time))
    ]
    # Round the 'time' column to one decimal place for easier grouping
    df['time_rounded'] = df['time'].round(1)
    # Group the filtered data by the rounded time and calculate the mean of 'mp' (metabolic power)
    grouped_data = df.groupby('time_rounded')['mp'].mean().reset_index()
    # Rename the columns for clarity
    grouped_data.columns = ['time_rounded', 'mp']
    return grouped_data



def pattern_matches(templates, df, grouped_data, times=times):
    all_matches = []

    # Pre-sort df for faster querying
    df = df.sort_values(by='time').reset_index(drop=True)

    # Convert relevant columns to NumPy arrays for faster access
    df_times = df['time'].values
    df_y_flip = df['y_flip'].values
    df_athlete_ids = df['athlete_id'].values

    grouped_data_times = grouped_data.iloc[:, 0].values
    grouped_data_mp = grouped_data.iloc[:, 1].values

    # Iterate over each template
    for index, template in enumerate(templates):
        rows = []
        template_mp = template['mp'].values
        thresholds = [1.7, 1.9, 8.2, 14.2, 6.1, 2.7]
        threshold = thresholds[index]

        # Find pattern matches with the given threshold
        matches = stumpy.match(template_mp, grouped_data_mp, max_distance=lambda D: threshold)

        # Process each match
        for match in matches:
            # Extract match details
            distance, start_index = match
            end_index = start_index + len(template_mp)
            start_time = grouped_data_times[start_index]
            end_time = grouped_data_times[end_index - 1]
            duration = end_time - start_time
            
            if duration > 17.2:
                for _ in range(300):
                    end_index = end_index - 1
                    end_time = grouped_data_times[end_index - 1]
                    duration = end_time - start_time
                    if duration <= 17.2:
                        break
            # Filter df to get rows between start_time and end_time
            filtered_df = df[(df['time_rounded'] >= start_time) & (df['time_rounded'] <= end_time)]
            
            # Get unique athlete IDs in the filtered DataFrame
            unique_athletes = filtered_df['athlete_id'].unique()
            combined_data = []
            
            for athlete in unique_athletes:
                # Filter for specific athlete
                athlete_data = filtered_df[filtered_df['athlete_id'] == athlete]
                # Select only 'x' and 'y_flip' columns
                movement_data = athlete_data[['x', 'y_flip']].copy()
                # Calculate the delta (difference) for x and y columns
                movement_data['delta_x'] = movement_data['x'].diff().abs()
                movement_data['delta_y'] = movement_data['y_flip'].diff().abs()
                # Calculate the distance between points
                movement_data['Distance Between Points'] = np.sqrt((movement_data['delta_x']**2) + (movement_data['delta_y']**2))
                # Add a column for athlete_id
                movement_data['athlete_id'] = athlete
                # Append to combined_data
                combined_data.append(movement_data)
            
            # Combine data for all athletes in the current match
            combined_data_df = pd.concat(combined_data, ignore_index=True)
                        
            # Maximum metabolic power in the match
            mp_max = np.max(grouped_data_mp[start_index:end_index])

            # Use binary search to find start and end indexes in df
            start_pos = np.searchsorted(df_times, start_time)
            end_pos = np.searchsorted(df_times, end_time, side='right')

            # Filter in-play data
            in_play_mask = df_y_flip[start_pos:end_pos] < 0
            in_play_data = df_athlete_ids[start_pos:end_pos][in_play_mask]

            # Count unique athletes
            athletes = np.unique(in_play_data).size

            # Calculate on_field and off_field averages
            play = df.iloc[start_pos:end_pos].copy()
            play['On Field'] = (play['y_flip'] < 0).astype(int)
            play['Off Field'] = (play['y_flip'] >= 0).astype(int)

            athlete_summary = play.groupby('time_rounded').agg({'On Field': 'sum', 'Off Field': 'sum'})
            on_field_avg = athlete_summary['On Field'].mean()
            off_field_avg = athlete_summary['Off Field'].mean()

            # Append the match details to the rows list if criteria met
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

        # Convert rows list into a DataFrame and append to all_matches
        rows_df = pd.DataFrame(rows)
        all_matches.append(rows_df)

    # Concatenate all_matches DataFrames into one DataFrame
    all_matches_df = pd.concat(all_matches, ignore_index=True)
    all_matches_df = all_matches_df.sort_values(by='Start Time')

    # Iterate over each row in all_matches_df
    for i, row in all_matches_df.iterrows():
        start_time = row['Start Time']

        for j, time in times.iterrows():
            period_name = time['Period Name']
            time_diff = np.abs(time['Unix Start Time'] - start_time)

            # Only include time differences less than 5 seconds
            if time_diff <= 5:
                all_matches_df.at[i, 'Period Name'] = period_name

    return all_matches_df