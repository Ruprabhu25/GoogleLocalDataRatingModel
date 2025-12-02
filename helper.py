import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from scipy.sparse import csr_matrix, vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold

def load_data(json_path, meta_path, state, compression=False):
    if compression:
        df_json = pd.read_json(json_path, lines=True, compression='gzip')
    else:
        df_json = pd.read_json(json_path, lines=True)
    df_json['State'] = state
    df_json.drop(columns=['pics'], inplace=True, errors='ignore')
    df_json['datetime'] = pd.to_datetime(df_json['time'], unit='ms', errors='coerce')
    df_json.sort_values(by=['user_id', 'datetime'], inplace=True)
    df_json['day_of_week'] = df_json['datetime'].dt.day_name()
    df_json['prev_time'] = df_json.groupby('user_id')['datetime'].shift(1)
    df_json['days_diff'] = (df_json['datetime'] - df_json['prev_time']).dt.days
    df_json['days_diff'].fillna(0, inplace=True)
    if compression:
        df_meta = pd.read_json(meta_path, lines=True, compression='gzip')
    else:
        df_meta = pd.read_json(meta_path, lines=True)
    wanted_cols = ['gmap_id', 'latitude', 'longitude', 'name', 'category']
    existing_cols = [c for c in wanted_cols if c in df_meta.columns]
    df_meta_clean = df_meta[existing_cols]
    full_data = pd.merge(df_json, df_meta_clean, on='gmap_id', how='left')
    return full_data

def business_density_plot(window_size, full_df, state, out_path=None):
    lat_center = full_df['latitude'].median()
    lon_center = full_df['longitude'].median()
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.scatter(full_df['longitude'], full_df['latitude'], alpha=0.1, s=3, c='blue', label='Review')
    ax.set_title(f'{state}: Urban Density ({window_size}° Window)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(lon_center - window_size, lon_center + window_size)
    ax.set_ylim(lat_center - window_size, lat_center + window_size)
    ax.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300)
    return fig, ax

def get_top_categories(df, state_name, out_path=None):
    exploded = df['category'].dropna().explode()
    top_counts = exploded.value_counts().head(10)
    df_plot = top_counts.reset_index()
    df_plot.columns = ['Category_Name', 'Review_Count']
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_plot, x='Review_Count', y='Category_Name', palette='Blues_r', ax=ax)
    ax.set_title(f'Top Business Categories ({state_name})')
    ax.set_xlabel('Number of Reviews')
    ax.set_ylabel('')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300)
    return fig, ax

def localize_time(df, region_timezone):
    if df['datetime'].dt.tz is None:
        df['datetime'] = df['datetime'].dt.tz_localize('UTC')
    df['local_time'] = df['datetime'].dt.tz_convert(region_timezone)
    df['local_hour'] = df['local_time'].dt.hour
    return df

def activity_by_hour_plot(df, region_timezone, state, out_path=None):
    df = localize_time(df, region_timezone)
    df['fractional_hour'] = df['local_time'].dt.hour + (df['local_time'].dt.minute / 60.0)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.kdeplot(df['fractional_hour'].dropna(), label=state, fill=True, bw_adjust=1.5, ax=ax)
    ax.set_title('User Activity by Time of Day')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Density of Activity')
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300)
    return fig, ax

def haversine_vectorized(user_lat, user_lon, place_lats, place_lons):
    lat1 = np.radians(user_lat)
    lon1 = np.radians(user_lon)
    lat2 = np.radians(place_lats)
    lon2 = np.radians(place_lons)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    R = 6371.0
    return R * c

def plot_distances(df, state, out_path=None):
    df = df.copy()
    df['prev_lat'] = df.groupby('user_id')['latitude'].shift(1)
    df['prev_lon'] = df.groupby('user_id')['longitude'].shift(1)
    df['dist_km'] = haversine_vectorized(df['prev_lat'].values, df['prev_lon'].values,
                                        df['latitude'].values, df['longitude'].values)
    state_dist = df[df['dist_km'].between(0.1, 100)]['dist_km'].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    if not state_dist.empty:
        sns.kdeplot(state_dist, ax=axes[0], fill=True, label=state, clip=(0, 100))
    axes[0].set_title('Distribution of Travel Distances (Density)')
    axes[0].set_xlabel('Distance (km)')
    axes[0].set_xlim(0, 50)
    axes[0].legend()
    if not state_dist.empty:
        sns.ecdfplot(state_dist, ax=axes[1], label=state)
    axes[1].set_title('CDF: Probability of Visiting within X km')
    axes[1].set_xlabel('Distance (km)')
    axes[1].set_ylabel('Proportion of Trips')
    axes[1].set_xlim(0, 50)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300)
    return fig, axes

def split_data(df):
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 2].index
    filtered_df = df[df['user_id'].isin(valid_users)].copy()
    test_df = filtered_df.groupby('user_id').tail(1)
    train_df = filtered_df.drop(test_df.index)
    user_history = train_df.groupby('user_id')['gmap_id'].apply(set).to_dict()
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    popular_places = train_df['gmap_id'].value_counts().index.tolist()
    user_sets = train_df.groupby('user_id')['gmap_id'].apply(set).to_dict()
    place_locs = train_df.drop_duplicates('gmap_id').set_index('gmap_id')[['latitude', 'longitude']].to_dict('index')
    return train_df, test_df, user_history, popular_places, user_sets, place_locs

def build_interaction_matrix(train_df):
    users = train_df['user_id'].unique()
    places = train_df['gmap_id'].unique()
    user_to_idx = {u: i for i, u in enumerate(users)}
    place_to_idx = {p: j for j, p in enumerate(places)}
    idx_to_place = {j: p for p, j in place_to_idx.items()}

    rows = [user_to_idx[u] for u in train_df['user_id']]
    cols = [place_to_idx[p] for p in train_df['gmap_id']]
    data = np.ones(len(rows), dtype=np.int8)
    M = csr_matrix((data, (rows, cols)), shape=(len(users), len(places)), dtype=np.int8)

    user_degrees = np.ravel(M.sum(axis=1))
    place_degrees = np.ravel(M.sum(axis=0))

    place_ids = np.array(places)
    place_coords_df = train_df.drop_duplicates('gmap_id').set_index('gmap_id').loc[place_ids][['latitude', 'longitude']]
    place_lats = place_coords_df['latitude'].values
    place_lons = place_coords_df['longitude'].values

    return {
        'M': M,
        'users': users,
        'places': places,
        'user_to_idx': user_to_idx,
        'place_to_idx': place_to_idx,
        'idx_to_place': idx_to_place,
        'user_degrees': user_degrees,
        'place_degrees': place_degrees,
        'place_ids': place_ids,
        'place_lats': place_lats,
        'place_lons': place_lons
    }

def predict_popularity(user_id, user_history, popular_places, top_k=10):
    seen = user_history.get(user_id, set())
    recommendations = []
    for item in popular_places:
        if item not in seen:
            recommendations.append(item)
            if len(recommendations) >= top_k:
                break
    return recommendations

def predict_distance_vectorized(user_id, current_lat, current_lon, user_history, idx_struct, top_k=10):
    seen = user_history.get(user_id, set())
    place_lats = idx_struct['place_lats']
    place_lons = idx_struct['place_lons']
    dists = haversine_vectorized(current_lat, current_lon, place_lats, place_lons)
    if seen:
        seen_idx = [idx_struct['place_to_idx'][p] for p in seen if p in idx_struct['place_to_idx']]
        if seen_idx:
            dists[seen_idx] = np.inf
    k = min(top_k, len(dists))
    idx = np.argpartition(dists, k - 1)[:k]
    idx = idx[np.argsort(dists[idx])]
    return list(idx_struct['place_ids'][idx])

def predict_jaccard_vectorized(target_user, idx_struct, top_k=10, neighbor_limit=50):
    if target_user not in idx_struct['user_to_idx']:
        return []
    M = idx_struct['M']
    u_idx = idx_struct['user_to_idx'][target_user]
    target_row = M[u_idx]  # 1 x n_places
    intersections = np.ravel(M.dot(target_row.T).toarray())  # intersections per user
    intersections[u_idx] = 0
    user_deg = idx_struct['user_degrees']
    unions = user_deg + user_deg[u_idx] - intersections
    valid = unions > 0
    jaccard_scores = np.zeros_like(intersections, dtype=float)
    jaccard_scores[valid] = intersections[valid] / unions[valid]
    neighbor_idx = np.argpartition(-jaccard_scores, min(neighbor_limit, len(jaccard_scores)-1))[:neighbor_limit]
    neighbor_weights = jaccard_scores[neighbor_idx]
    neighbor_weights = neighbor_weights[neighbor_weights > 0]
    if neighbor_weights.size == 0:
        return []
    neighbor_idx = neighbor_idx[:len(neighbor_weights)]
    neighbor_rows = M[neighbor_idx]  # k x n_places
    candidate_scores = neighbor_rows.T.dot(neighbor_weights)  # n_places
    target_places_indices = target_row.indices
    if target_places_indices.size > 0:
        candidate_scores[target_places_indices] = 0
    k = min(top_k, candidate_scores.shape[0])
    top_idx = np.argpartition(-candidate_scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-candidate_scores[top_idx])]
    return list(idx_struct['place_ids'][top_idx])

def get_jaccard_scores_vectorized(target_user, idx_struct, neighbor_limit=50):
    if target_user not in idx_struct['user_to_idx']:
        return {}
    M = idx_struct['M']
    u_idx = idx_struct['user_to_idx'][target_user]
    target_row = M[u_idx]
    intersections = np.ravel(M.dot(target_row.T).toarray())
    intersections[u_idx] = 0
    user_deg = idx_struct['user_degrees']
    unions = user_deg + user_deg[u_idx] - intersections
    valid = unions > 0
    jaccard_scores = np.zeros_like(intersections, dtype=float)
    jaccard_scores[valid] = intersections[valid] / unions[valid]
    neighbor_idx = np.argpartition(-jaccard_scores, min(neighbor_limit, len(jaccard_scores)-1))[:neighbor_limit]
    neighbor_weights = jaccard_scores[neighbor_idx]
    mask = neighbor_weights > 0
    neighbor_idx = neighbor_idx[mask]
    neighbor_weights = neighbor_weights[mask]
    if neighbor_idx.size == 0:
        return {}
    neighbor_rows = M[neighbor_idx]
    candidate_scores = neighbor_rows.T.dot(neighbor_weights)
    target_places_indices = target_row.indices
    if target_places_indices.size > 0:
        candidate_scores[target_places_indices] = 0
    place_ids = idx_struct['place_ids']
    return {place_ids[i]: float(candidate_scores[i]) for i in np.nonzero(candidate_scores)[0]}

def predict_weighted_hybrid_vectorized(user_id, current_lat, current_lon, idx_struct, user_sets, top_k=10):
    jaccard_dict = get_jaccard_scores_vectorized(user_id, idx_struct)
    if not jaccard_dict:
        return predict_distance_vectorized(user_id, current_lat, current_lon, user_sets, idx_struct, top_k=top_k)
    place_ids = idx_struct['place_ids']
    place_index_map = idx_struct['place_to_idx']
    place_lats = idx_struct['place_lats']
    place_lons = idx_struct['place_lons']
    j_items = np.array(list(jaccard_dict.keys()))
    j_scores = np.array(list(jaccard_dict.values()))
    j_idx = np.array([place_index_map[p] for p in j_items if p in place_index_map])
    if j_idx.size == 0:
        return predict_distance_vectorized(user_id, current_lat, current_lon, user_sets, idx_struct, top_k=top_k)
    dists = haversine_vectorized(current_lat, current_lon, place_lats[j_idx], place_lons[j_idx])
    distance_factor = 1.0 / (np.log1p(dists) + 0.5)
    final_scores = j_scores[:len(j_idx)] * distance_factor
    k = min(top_k, final_scores.shape[0])
    top = np.argpartition(-final_scores, k - 1)[:k]
    top = top[np.argsort(-final_scores[top])]
    return list(j_items[top])

def train_svd_optimized(train_df, idx_struct, n_components=20, random_state=42):
    M = idx_struct['M']
    n_comp = min(n_components, M.shape[0] - 1 if M.shape[0] > 1 else 1, M.shape[1] - 1 if M.shape[1] > 1 else 1)
    if n_comp < 1:
        n_comp = 1
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    user_factors = svd.fit_transform(M)
    place_factors = svd.components_
    return user_factors, place_factors

def predict_svd(user_id, user_history, idx_struct, user_factors, place_factors, top_k=10):
    if user_id not in idx_struct['user_to_idx']:
        return []
    user_to_idx = idx_struct['user_to_idx']
    place_ids = idx_struct['place_ids']
    place_to_idx = idx_struct['place_to_idx']
    u_idx = user_to_idx[user_id]
    user_vector = user_factors[u_idx]
    scores = np.dot(user_vector, place_factors)
    seen_items = user_history.get(user_id, set())
    if seen_items:
        seen_indices = [place_to_idx[i] for i in seen_items if i in place_to_idx]
        if seen_indices:
            scores[seen_indices] = -np.inf
    k = min(top_k, scores.shape[0])
    top_indices = np.argpartition(-scores, k - 1)[:k]
    top_indices = top_indices[np.argsort(-scores[top_indices])]
    return list(place_ids[top_indices])

def evaluate_methods_optimized(train_df, test_df, user_history, popular_places, user_sets, place_locs, sample_limit=100000):
    idx_struct = build_interaction_matrix(train_df)
    user_factors, place_factors = train_svd_optimized(train_df, idx_struct, n_components=20)
    hits = {'pop': 0, 'dist': 0, 'jaccard': 0, 'hybrid': 0, 'svd': 0}
    total = 0
    sample_test_users = list(test_df['user_id'].unique())[:sample_limit]
    for user in sample_test_users:
        true_place_series = test_df[test_df['user_id'] == user]['gmap_id'].values
        if len(true_place_series) == 0:
            continue
        true_place = true_place_series[0]
        user_train_data = train_df[train_df['user_id'] == user]
        if user_train_data.empty:
            continue
        last_lat = user_train_data.iloc[-1]['latitude']
        last_lon = user_train_data.iloc[-1]['longitude']
        preds_pop = predict_popularity(user, user_history, popular_places, top_k=10)
        preds_dist = predict_distance_vectorized(user, last_lat, last_lon, user_history, idx_struct, top_k=10)
        preds_jaccard = predict_jaccard_vectorized(user, idx_struct, top_k=10)
        preds_hybrid = predict_weighted_hybrid_vectorized(user, last_lat, last_lon, idx_struct, user_sets, top_k=10)
        preds_svd = predict_svd(user, user_history, idx_struct, user_factors, place_factors, top_k=10)
        if true_place in preds_pop: hits['pop'] += 1
        if true_place in preds_dist: hits['dist'] += 1
        if true_place in preds_jaccard: hits['jaccard'] += 1
        if true_place in preds_hybrid: hits['hybrid'] += 1
        if true_place in preds_svd: hits['svd'] += 1
        total += 1
        if total % 100 == 0:
            print(f"Processed {total} users...")
    if total == 0:
        print("No test users processed. Check split_data or data sufficiency.")
        return
    print(f"\n--- Results (Hit Rate @ 10) ---")
    print(f"Popularity: {hits['pop'] / total:.4f}")
    print(f"Distance:   {hits['dist'] / total:.4f}")
    print(f"Jaccard:    {hits['jaccard'] / total:.4f}")
    print(f"Hybrid:     {hits['hybrid'] / total:.4f}")
    print(f"SVD:        {hits['svd'] / total:.4f}")

if __name__ == "__main__":
    dc_json_path = './datasets/review-District_of_Columbia.json'
    wy_json_path = './datasets/review-Wyoming.json'
    dc_meta_path = './datasets/meta-District_of_Columbia.json'
    wy_meta_path = './datasets/meta-Wyoming.json'
    df_dc = load_data(dc_json_path, dc_meta_path, 'DC')
    df_wy = load_data(wy_json_path, wy_meta_path, 'WY')
    df_full = pd.concat([df_dc, df_wy], ignore_index=True)
    # output_csv_path = 'datasets/combined_reviews.csv'
    # os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    # df_full.to_csv(output_csv_path, index=False, compression='gzip')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df_full['fold'] = -1
    for fold, (_, val_idx) in enumerate(skf.split(X=df_full, y=df_full['State'])):
        df_full.loc[val_idx, 'fold'] = fold
    # df_full.to_csv('datasets/combined_reviews_folds.csv', index=False, compression='gzip')
    train_df, test_df, user_history, popular_places, user_sets, place_locs = split_data(df_full)
    evaluate_methods_optimized(train_df, test_df, user_history, popular_places, user_sets, place_locs)


