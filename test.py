import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

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


### Plotting code for data exploration slides

def business_density_plot(window_size, full_df, state):
    # Find the geographical center of the reviews
    lat_center = full_df['latitude'].median()
    lon_center = full_df['longitude'].median()

    fig, axes = plt.subplots(1, 1, figsize=(15, 6))

    axes[0].scatter(full_df['longitude'], full_df['latitude'],
                    alpha=0.1, s=3, c='blue', label='Review')
    axes[0].set_title(f'{state}: Urban Density (0.25° Window)')

    axes[0].set_aspect('equal')

    axes[0].set_xlim(lon_center - window_size, lon_center + window_size)
    axes[0].set_ylim(lat_center - window_size, lat_center + window_size)

    plt.savefig(f"business_density_by_geo_{state}.png", dpi=300)
    return

def get_top_categories(df, state_name):

    exploded = df['category'].explode()
    top_counts = exploded.value_counts().head(10)
    df_plot = top_counts.reset_index()
    df_plot.columns = ['Category_Name', 'Review_Count']

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x='Review_Count', y='Category_Name', palette='Blues_r')
    plt.title(f'Top Business Categories ({state_name})')
    plt.xlabel('Number of Reviews')
    plt.ylabel('')

    return

def localize_time(df, region_timezone):
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
    plt.xticks(range(0, 25, 2))
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

    df['prev_lat'] = df.groupby('user_id')['latitude'].shift(1)
    df['prev_lon'] = df.groupby('user_id')['longitude'].shift(1)

    df['dist_km'] = haversine_np(
        df['prev_lon'], df['prev_lat'],
        df['longitude'], df['latitude']
    )

    state_dist = df[df['dist_km'].between(0.1, 100)]['dist_km']

    fig, axes = plt.subplots(1, 1, figsize=(16, 6))

    # Density curve
    sns.kdeplot(state_dist, ax=axes[0], fill=True, color='blue', label=state, clip=(0, 100))

    axes[0].set_title('Distribution of Travel Distances (Density)')
    axes[0].set_xlabel('Distance (km)')
    axes[0].set_xlim(0, 50)
    axes[0].legend()

    # Cumulative distribution (CDF)
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
    # Only keep users with at least 2 reviews
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 2].index
    filtered_df = df[df['user_id'].isin(valid_users)].copy()

    # Most recent value is the target, all others are training
    test_df = filtered_df.groupby('user_id').tail(1)
    train_df = filtered_df.drop(test_df.index)

    # Items visited by user
    user_history = train_df.groupby('user_id')['gmap_id'].apply(set).to_dict()

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    popular_places = train_df['gmap_id'].value_counts().index.tolist()
    user_sets = train_df.groupby('user_id')['gmap_id'].apply(set).to_dict()
    place_locs = train_df.drop_duplicates('gmap_id').set_index('gmap_id')[['latitude', 'longitude']].to_dict('index')

    return train_df, test_df, user_history, popular_places, user_sets, place_locs


def predict_popularity(user_id, user_history, popular_places, top_k=10):

    # Baseline: Just predict based on the most popular business globally

    seen = user_history.get(user_id, set())

    recommendations = []
    for item in popular_places:
        if item not in seen:
            recommendations.append(item)
            if len(recommendations) >= top_k:
                break

    return recommendations


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))


def predict_distance(user_id, current_lat, current_lon, user_history, place_locs, top_k=10):

    # Alternative baseline: just predict the businesses that are closest to the user's last known location

    seen = user_history.get(user_id, set())

    candidates = []

    for item, coords in place_locs.items():
        if item not in seen:
            dist = haversine_distance(current_lat, current_lon, coords['latitude'], coords['longitude'])
            candidates.append((item, dist))

    candidates.sort(key=lambda x: x[1])

    return [x[0] for x in candidates[:top_k]]


def predict_jaccard(target_user, user_sets, top_k=10):
    # If unseen user, fallback to popularity
    if target_user not in user_sets:
        return predict_popularity(target_user, top_k)

    target_places = user_sets[target_user]
    scores = []

    # Find similarity with all other users
    for other_user, other_places in user_sets.items():
        if other_user == target_user:
            continue

        intersection = len(target_places.intersection(other_places))
        if intersection == 0:
            continue

        union = len(target_places.union(other_places))

        jaccard = intersection / union
        scores.append((other_user, jaccard))

    scores.sort(key=lambda x: x[1], reverse=True)
    neighbors = scores[:50]

    # Aggregate votes from neighbors
    candidate_scores = {}
    for neighbor, weight in neighbors:
        neighbor_places = user_sets[neighbor]
        for place in neighbor_places:
            if place not in target_places:
                candidate_scores[place] = candidate_scores.get(place, 0) + weight

    sorted_cands = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

    return [x[0] for x in sorted_cands[:top_k]]


def predict_hybrid(user_id, current_lat, current_lon, place_locs, top_k=10):

    # Basic hybrid function using Jaccard and then scoring places based on location
    jaccard_candidates = predict_jaccard(user_id, top_k=50)

    if not jaccard_candidates:
        return predict_distance(user_id, current_lat, current_lon, top_k)

    hybrid_scores = []

    for place in jaccard_candidates:
        if place in place_locs:
            coords = place_locs[place]
            dist = haversine_distance(current_lat, current_lon, coords['latitude'], coords['longitude'])
            hybrid_scores.append((place, dist))

    hybrid_scores.sort(key=lambda x: x[1])

    return [x[0] for x in hybrid_scores[:top_k]]


def get_jaccard_scores(target_user, user_sets):

    # Just get Jaccard scores, don't actually predict places. To be used in new hybrid function
    if target_user not in user_sets:
        return []

    target_places = user_sets[target_user]
    scores = []

    for other_user, other_places in user_sets.items():
        if other_user == target_user: continue
        intersection = len(target_places.intersection(other_places))
        if intersection == 0: continue
        union = len(target_places.union(other_places))
        scores.append((other_user, intersection / union))

    scores.sort(key=lambda x: x[1], reverse=True)
    neighbors = scores[:50]

    candidate_scores = {}
    for neighbor, weight in neighbors:
        neighbor_items = user_sets[neighbor]
        for item in neighbor_items:
            if item not in target_places:
                candidate_scores[item] = candidate_scores.get(item, 0) + weight

    return candidate_scores


def predict_weighted_hybrid(user_id, current_lat, current_lon, place_locs, top_k=10):

    # New hybrid function that accounts for preferences more thoroughly

    jaccard_dict = get_jaccard_scores(user_id)

    if not jaccard_dict:
        return predict_distance(user_id, current_lat, current_lon, top_k)

    final_scores = []

    for item, j_score in jaccard_dict.items():
        if item in place_locs:
            coords = place_locs[item]
            dist = haversine_distance(current_lat, current_lon, coords['latitude'], coords['longitude'])

            distance_factor = 1.0 / (np.log1p(dist) + 0.5)
            final_score = j_score * distance_factor

            final_scores.append((item, final_score))

    final_scores.sort(key=lambda x: x[1], reverse=True)

    return [x[0] for x in final_scores[:top_k]]


def train_svd(train_df):

    train_users = train_df['user_id'].unique()
    train_places = train_df['gmap_id'].unique()

    user_to_idx = {u: i for i, u in enumerate(train_users)}
    place_to_idx = {i: j for j, i in enumerate(train_places)}

    idx_to_place = {j: i for i, j in place_to_idx.items()}

    rows = [user_to_idx[u] for u in train_df['user_id']]
    cols = [place_to_idx[i] for i in train_df['gmap_id']]
    data = np.ones(len(rows))

    sparse_matrix = csr_matrix((data, (rows, cols)),
                            shape=(len(train_users), len(train_places)))

    svd = TruncatedSVD(n_components=20, random_state=42)

    svd.fit(sparse_matrix)

    user_factors = svd.transform(sparse_matrix)
    place_factors = svd.components_

    return user_to_idx, place_to_idx, user_factors, place_factors


def predict_svd(user_id, user_history, user_to_idx, place_to_idx, idx_to_place, user_factors, place_factors, top_k=10):

    if user_id not in user_to_idx:
        return []

    u_idx = user_to_idx[user_id]

    user_vector = user_factors[u_idx]
    scores = np.dot(user_vector, place_factors)

    seen_items = user_history.get(user_id, set())
    seen_indices = [place_to_idx[i] for i in seen_items if i in place_to_idx]

    scores[seen_indices] = -np.inf

    top_indices = scores.argsort()[-top_k:][::-1]

    return [idx_to_place[i] for i in top_indices]



def evaluate_methods(train_df, test_df):
    hits_pop = 0
    hits_dist = 0
    hits_jaccard = 0
    hits_hybrid = 0
    hits_svd = 0
    total = 0

    sample_test_users = test_df['user_id'].unique()[:500]

    for user in sample_test_users:

        true_place = test_df[test_df['user_id'] == user]['gmap_id'].values[0]

        user_train_data = train_df[train_df['user_id'] == user]
        if user_train_data.empty:
            continue

        last_lat = user_train_data.iloc[-1]['latitude']
        last_lon = user_train_data.iloc[-1]['longitude']

        preds_pop = predict_popularity(user, top_k=10)
        preds_dist = predict_distance(user, last_lat, last_lon, top_k=10)
        preds_jaccard = predict_jaccard(user, top_k=10)
        preds_hybrid = predict_weighted_hybrid(user, last_lat, last_lon, top_k=10)
        preds_svd = predict_svd(user)

        if true_place in preds_pop: hits_pop += 1
        if true_place in preds_dist: hits_dist += 1
        if true_place in preds_jaccard: hits_jaccard += 1
        if true_place in preds_hybrid: hits_hybrid += 1
        if true_place in preds_svd: hits_svd += 1

        total += 1
        if total % 100 == 0: print(f"Processed {total} users...")

    print(f"\n--- Results (Hit Rate @ 10) ---")
    print(f"Popularity: {hits_pop / total:.4f}")
    print(f"Distance:   {hits_dist / total:.4f}")
    print(f"Jaccard:    {hits_jaccard / total:.4f}")
    print(f"Hybrid:    {hits_hybrid / total:.4f}")
    print(f"SVD:        {hits_svd / total:.4f}")


########


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
