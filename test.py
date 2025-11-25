import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns

def load_data(json_path, meta_path, state):

    df_json = pd.read_json(json_path, lines=True, compression='gzip')
    df_json['State'] = state

    df_json.drop(columns=['pics'], inplace=True, errors='ignore')

    df_json['datetime'] = pd.to_datetime(df_json['time'], unit='ms')

    df_json.sort_values(by=['user_id', 'datetime'], inplace=True)

    df_json['day_of_week'] = df_json['datetime'].dt.day_name()
    df_json['prev_time'] = df_json.groupby('user_id')['datetime'].shift(1)
    df_json['days_diff'] = (df_json['datetime'] - df_json['prev_time']).dt.days
    df_json['days_diff'].fillna(0, inplace=True)

    df_meta = pd.read_json(meta_path, lines=True, compression='gzip')
    wanted_cols = ['gmap_id', 'latitude', 'longitude', 'name', 'category']

    existing_cols = [c for c in wanted_cols if c in df_meta.columns]
    df_meta_clean = df_meta[existing_cols]

    full_data = pd.merge(df_json, df_meta_clean, on='gmap_id', how='left')

    return full_data

def business_density_plot(window_size, full_df, state):
    # We use the median to find the "center of gravity" of the reviews
    lat_center = full_df['latitude'].median() 
    lon_center = full_df['longitude'].median()

    # Set up figure
    fig, axes = plt.subplots(1, 1, figsize=(15, 6))

    axes[0].scatter(full_df['longitude'], full_df['latitude'], 
                    alpha=0.1, s=3, c='blue', label='Review')
    axes[0].set_title(f'{state}: Urban Density (0.25° Window)')

    # Force the Aspect Ratio (Crucial for maps)
    axes[0].set_aspect('equal')

    # Force the exact window size
    axes[0].set_xlim(lon_center - window_size, lon_center + window_size)
    axes[0].set_ylim(lat_center - window_size, lat_center + window_size)

    plt.savefig(f"business_density_by_geo_{state}.png", dpi=300)
    return

def get_top_categories(df, state_name):
    # 1. Explode the list of categories so ['Food', 'Pizza'] becomes two rows
    # (If the column is already just strings, this won't hurt anything)
    exploded = df['category'].explode()
    
    # 2. Count the values
    top_counts = exploded.value_counts().head(10)
    
    # 3. Convert to a DataFrame (Seaborn loves DataFrames, hates raw Series)
    # This creates columns: 'index' (category name) and 'category' (the count)
    df_plot = top_counts.reset_index()
    
    # Rename columns to be crystal clear
    # Note: Depending on your pandas version, the count column might be named 'category' or 'count'
    # We rename them explicitly to avoid confusion
    df_plot.columns = ['Category_Name', 'Review_Count']

    # --- Plotting Code ---
    plt.figure(figsize=(12, 6))

    sns.barplot(data=df_plot, x='Review_Count', y='Category_Name', palette='Blues_r')
    plt.title(f'Top Business Categories ({state_name})')
    plt.xlabel('Number of Reviews')
    plt.ylabel('')
    
    return 

def localize_time(df, region_timezone):
    # Ensure datetime is in UTC first (if not already)
    # The 'dt.tz_localize' might be needed if it's currently "naive" (no timezone info)
    # The 'dt.tz_convert' is used if it already knows it's UTC
    
    # Check if we already have timezone info; if not, assume UTC
    if df['datetime'].dt.tz is None:
        df['datetime'] = df['datetime'].dt.tz_localize('UTC')
    
    # Convert to the target timezone
    df['local_time'] = df['datetime'].dt.tz_convert(region_timezone)
    
    # Extract the 'local_hour' for plotting
    df['local_hour'] = df['local_time'].dt.hour
    return df

def activity_by_hour_plot(df, region_timezone, state):

    df = localize_time(df, region_timezone)

    df['fractional_hour'] = df['local_time'].dt.hour + (df['local_time'].dt.minute / 60)

    plt.figure(figsize=(12, 6))

    sns.kdeplot(df['fractional_hour'], label=state, fill=True, color='blue', bw_adjust=1.5)

    plt.title('User Activity by Time of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Density of Activity')
    plt.xlim(0, 24)
    plt.xticks(range(0, 25, 2)) # Tick mark every 2 hours
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("activity_by_hour_by_geo.png", dpi=300)
    
    return

def haversine_np(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c

    return km

def plot_distances(df, state):
    # 1. Sort by User and Time
    df.sort_values(['user_id', 'datetime'], inplace=True)

    # 2. Shift columns to get "Previous Location" on the same row
    df['prev_lat'] = df.groupby('user_id')['latitude'].shift(1)
    df['prev_lon'] = df.groupby('user_id')['longitude'].shift(1)

    # 3. Calculate Distance
    df['dist_km'] = haversine_np(
        df['prev_lon'], df['prev_lat'],
        df['longitude'], df['latitude']
    )

    state_dist = df[df['dist_km'].between(0.1, 100)]['dist_km']

    fig, axes = plt.subplots(1, 1, figsize=(16, 6))

    # --- PLOT 1: Density Curve (KDE) ---
    # common_norm=False ensures the area under EACH curve sums to 1
    # This fixes the "Scale" problem (DC having more data doesn't matter)
    sns.kdeplot(state_dist, ax=axes[0], fill=True, color='blue', label=state, clip=(0, 100))

    axes[0].set_title('Distribution of Travel Distances (Density)')
    axes[0].set_xlabel('Distance (km)')
    axes[0].set_xlim(0, 50) # Zoom in on the local activity
    axes[0].legend()

    # --- PLOT 2: Cumulative Distribution (CDF) ---
    # This answers: "What % of trips are shorter than X km?"
    sns.ecdfplot(state_dist, ax=axes[1], color='blue', label=state)

    axes[1].set_title('CDF: Probability of Visiting within X km')
    axes[1].set_xlabel('Distance (km)')
    axes[1].set_ylabel('Proportion of Trips')
    axes[1].set_xlim(0, 50)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("travel_distance_by_geo.png", dpi=300)
    return


def split_data(df):
    # 2. Filter: Only keep users with at least 2 reviews
    # (We need at least 1 to train and 1 to test)
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 2].index
    filtered_df = df[df['user_id'].isin(valid_users)].copy()

    # 3. Split
    # The 'tail(1)' is the Test set (Target)
    test_df = filtered_df.groupby('user_id').tail(1)

    # The rest is the Train set (History)
    train_df = filtered_df.drop(test_df.index)

    # 4. Create a set of "visited items" for every user
    # We need this to ensure we don't recommend things they already saw
    user_history = train_df.groupby('user_id')['gmap_id'].apply(set).to_dict()

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    return train_df, test_df, user_history

def predict_popularity(user_id, user_history, top_k=10):
    # Get items this user has already seen
    seen = user_history.get(user_id, set())
    
    recommendations = []
    for item in popular_items:
        if item not in seen:
            recommendations.append(item)
            if len(recommendations) >= top_k:
                break
    return recommendations

def haversine_distance(lat1, lon1, lat2, lon2):
    # Simplified calculation for speed
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

def predict_distance(user_id, current_lat, current_lon, user_history, top_k=10):
    seen = user_history.get(user_id, set())
    
    # Calculate distance to ALL items in train set (Brute force is okay for small subsets)
    # In production, you'd use a KD-Tree or Geo-Hash
    candidates = []
    
    for item, coords in item_locs.items():
        if item not in seen:
            dist = haversine_distance(current_lat, current_lon, coords['latitude'], coords['longitude'])
            candidates.append((item, dist))
    
    # Sort by distance (ASCENDING)
    candidates.sort(key=lambda x: x[1])
    
    return [x[0] for x in candidates[:top_k]]

def predict_jaccard(target_user, top_k=10):
    # If new user (cold start), fallback to popularity
    if target_user not in user_sets:
        return predict_popularity(target_user, top_k)
    
    target_items = user_sets[target_user]
    scores = []
    
    # Compare target_user to ALL other users
    # (Optimized: In reality, you'd use an Inverted Index here to skip users with 0 overlap)
    for other_user, other_items in user_sets.items():
        if other_user == target_user:
            continue
            
        # Calculate Intersection
        intersection = len(target_items.intersection(other_items))
        
        # Optimization: If no overlap, skip (Jaccard is 0)
        if intersection == 0:
            continue
            
        # Calculate Union
        union = len(target_items.union(other_items))
        
        # Jaccard Score
        jaccard = intersection / union
        scores.append((other_user, jaccard))
    
    # Get top 50 similar users (Neighbors)
    scores.sort(key=lambda x: x[1], reverse=True)
    neighbors = scores[:50]
    
    # Aggregate votes from neighbors
    candidate_scores = {}
    for neighbor, weight in neighbors:
        neighbor_items = user_sets[neighbor]
        for item in neighbor_items:
            # Don't recommend what target user already saw
            if item not in target_items:
                candidate_scores[item] = candidate_scores.get(item, 0) + weight
    
    # Sort candidates by accumulated weight
    sorted_cands = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [x[0] for x in sorted_cands[:top_k]]

popular_items = train_df['gmap_id'].value_counts().index.tolist()
item_locs = train_df.drop_duplicates('gmap_id').set_index('gmap_id')[['latitude', 'longitude']].to_dict('index')


dc_json_path = 'datasets/review-District_of_Columbia.json.gz' 
wy_json_path = 'datasets/review-Wyoming.json.gz'
dc_meta_path = 'datasets/meta-District_of_Columbia.json.gz'
wy_meta_path = 'datasets/meta-Wyoming.json.gz'

df_dc = load_data(dc_json_path, dc_meta_path, 'DC')
df_wy = load_data(wy_json_path, wy_meta_path, 'WY')

df_full = pd.concat([df_dc, df_wy], ignore_index=True)

# Save the combined dataframe to a CSV file
output_csv_path = 'datasets/combined_reviews.csv'
df_full.to_csv(output_csv_path, index=False, compression='gzip')

# 5 fold CV split based on user_id, stratified by state
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df_full['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(X=df_full, y=df_full['State'])):
    df_full.loc[val_idx, 'fold'] = fold
df_full.to_csv('datasets/combined_reviews_folds.csv', index=False, compression='gzip')





