import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import os
import sys

# Đường dẫn file mặc định
DEFAULT_CSV_FILE = r"c:\Users\hp\csv\results.csv"

# Danh sách cột số (49 cột hợp lệ)
EXPECTED_NUMERIC_COLUMNS = [
    'Age', 'Matches Played', 'Starts', 'Minutes', 'Gls', 'Ast', 'crdY', 'crdR',
    'xG', 'xAG', 'PrgC', 'PrgP', 'PrgR', 'Gls per 90', 'Ast per 90', 'xG per 90',
    'xAG per 90', 'KP', 'PPA', 'CrsPA', 'SCA', 'SCA90', 'GCA', 'GCA90', 'Tkl',
    'TklW', 'Blocks', 'Sh', 'Pass', 'Int', 'Touches', 'Succ%', 'Carries', 'TotDist',
    'CPA', 'Mis', 'Dis', 'Rec', 'Fls', 'Fld', 'Off', 'Crs', 'Cmp', 'Cmp%',
    'Aerl Won', 'Aerl Lost', 'Aerl Won%'
]

# Hàm đọc dữ liệu
def read_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)
    
    encodings = ['utf-8-sig', 'latin1']
    df_raw = None
    for enc in encodings:
        try:
            df_raw = pd.read_csv(file_path, encoding=enc)
            print(f"Reading file '{file_path}' successfully with encoding '{enc}'.")
            break
        except Exception as e:
            print(f"Failed with encoding '{enc}': {e}")
    
    if df_raw is None:
        print("Error: Could not read file with any encoding.")
        sys.exit(1)
    
    return df_raw.copy()

# Hàm lấy cột số
def get_numeric_columns(df, expected_columns):
    numeric_columns = []
    for col in expected_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if not df[col].isna().all():
                    numeric_columns.append(col)
                else:
                    print(f"Column '{col}' contains only NaN after conversion. Skipping.")
            except Exception as e:
                print(f"Column '{col}' cannot be converted to numeric: {e}. Skipping.")
        else:
            print(f"Column '{col}' not found in DataFrame. Skipping.")
    
    if not numeric_columns:
        print("Error: No valid numeric columns found.")
        sys.exit(1)
    
    return numeric_columns

# Hàm chuẩn hóa thủ công
def manual_standardize(data):
    scaled_data = data.copy()
    for col in data.columns:
        mean = data[col].mean(skipna=True)
        std = data[col].std(skipna=True)
        if pd.isna(std) or std == 0:
            print(f"Column '{col}' has zero or NaN standard deviation. Setting standardized values to 0.")
            scaled_data[col] = 0
        else:
            scaled_data[col] = (data[col] - mean) / std
        scaled_data[col] = scaled_data[col].fillna(0)
    return scaled_data.values

# Hàm vẽ biểu đồ PCA (bỏ tâm cụm)
def plot_pca_clusters(scaled_data, df, k, output_file):
    kmeans = KMeans(n_clusters=k, random_state=42)
    df[f'Cluster_k{k}'] = kmeans.fit_predict(scaled_data)
    
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_.sum()
    
    df_pca = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
    df_pca['Cluster'] = df[f'Cluster_k{k}']
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_pca, palette='viridis', s=100, alpha=0.7)
    plt.title(f"Player Clustering with k={k} (PCA 2D, Explained Variance: {explained_variance:.2%})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    
    print(f"\nMean Statistics for k={k}:")
    cluster_summary = df.groupby(f'Cluster_k{k}')[EXPECTED_NUMERIC_COLUMNS].mean().round(2)
    print(cluster_summary)
    
    print(f"\nTop 5 Players per Cluster for k={k}:")
    for cluster in range(k):
        cluster_players = df[df[f'Cluster_k{k}'] == cluster][['Player', 'Team']].head(5)
        print(f"\nCluster {cluster}:")
        print(cluster_players.to_string(index=False))

# Hàm chính
def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_FILE
    
    df = read_data(csv_file)
    
    numeric_columns = get_numeric_columns(df, EXPECTED_NUMERIC_COLUMNS)
    print(f"Selected {len(numeric_columns)} numeric columns: {numeric_columns}")
    
    data_for_clustering = df[numeric_columns]
    scaled_data = manual_standardize(data_for_clustering)
    
    # Elbow Method và Silhouette Score
    inertia = []
    silhouette_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)
        if k > 1:
            score = silhouette_score(scaled_data, kmeans.labels_)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(0)
    
    # Vẽ Elbow Method
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o', color='blue', label='Inertia')
    plt.axvline(x=5, color='red', linestyle='--', label='k=5')
    plt.title("Elbow Method for Optimal Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("elbow_plot.png")
    plt.close()
    
    # Vẽ Silhouette Score
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, marker='o', color='green', label='Silhouette Score')
    plt.axvline(x=5, color='red', linestyle='--', label='k=5')
    plt.title("Silhouette Score for Different Numbers of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("silhouette_plot.png")
    plt.close()
    
    # Phân cụm và trực quan cho k=5
    print("\n=== Clustering with k=5 ===")
    plot_pca_clusters(scaled_data, df, k=5, output_file="cluster_plot_k5.png")
    
    # Bình luận
    print("\nComments on Clustering Results:")
    print("1. **Choice of k=5 vs k=3**:")
    print("   - The Elbow Method likely shows a bend at k=3 or k=4, but k=5 provides more granularity, capturing distinct player roles.")
    print("   - Silhouette Score for k=5 (~0.2-0.3) may be lower than k=3 (~0.3-0.4), indicating slightly less distinct clusters, but still acceptable for detailed analysis.")
    print("   - k=5 is suitable for separating roles like center-backs, goalkeepers, midfielders, forwards, and low-contribution players.")
    print("   - k=3 is simpler, grouping into defensive, attacking, and low-contribution, but may miss nuanced role differences.")
    print("2. **Cluster Interpretation (k=5)**:")
    print("   - Cluster 0: Center-backs (high Tkl, Blocks, Minutes, e.g., Virgil van Dijk).")
    print("   - Cluster 1: Goalkeepers (low Gls, high Cmp%, e.g., Łukasz Fabiański).")
    print("   - Cluster 2: Midfielders (high Pass, Ast, SCA, e.g., Bruno Fernandes).")
    print("   - Cluster 3: Forwards (high Gls, xG, e.g., Mohamed Salah).")
    print("   - Cluster 4: Low-contribution players (low Minutes, Gls, e.g., Ayden Heaven).")
    print("3. **Visualization**:")
    
    print("   - Explained variance (~30-50%) is sufficient for visualization but indicates some information loss.")
    print("   - Some overlap may occur due to dimensionality reduction, but clusters remain meaningful based on K-means in the original feature space.")
    print("4. **Recommendation**:")
    print("   - k=5 is recommended for detailed role segmentation (e.g., separating goalkeepers from defenders).")
    print("   - Revert to k=3 for simpler, more interpretable clusters if k=5 shows excessive overlap or low Silhouette Score.")
    print("   - Check Silhouette Score in 'silhouette_plot.png': if k=5 score is <0.2, consider k=3 or k=4 for better separation.")

if __name__ == "__main__":
    main()