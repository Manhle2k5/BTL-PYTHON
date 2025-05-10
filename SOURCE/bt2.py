import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from io import StringIO

# Hàm tạo thư mục nếu chưa tồn tại
def create_directory(path):
    os.makedirs(path, exist_ok=True)

# Hàm đọc và xử lý dữ liệu từ file CSV
def load_data(csv_path, exclude_columns):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File {csv_path} không tồn tại. Vui lòng chạy script trích xuất dữ liệu trước.")
    
    df = pd.read_csv(csv_path, na_values=["N/A"])
    df_calc = df.copy()
    
    # Lấy các cột số
    numeric_columns = [col for col in df_calc.columns if col not in exclude_columns]
    
    # Chuyển đổi sang numeric, điền NaN thành 0
    for col in numeric_columns:
        df_calc[col] = pd.to_numeric(df_calc[col], errors="coerce").fillna(0)
    
    return df_calc, numeric_columns

# Yêu cầu 1: Xác định 3 cầu thủ có điểm cao nhất và thấp nhất cho mỗi thống kê
def save_top_3_players(df, numeric_columns, output_path):
    rankings = {}
    for col in numeric_columns:
        # Top 3 cao nhất
        top_3_high = df[["Player", "Team", col]].sort_values(by=col, ascending=False).head(3)
        top_3_high = top_3_high.rename(columns={col: "Value"})
        top_3_high["Rank"] = ["1st", "2nd", "3rd"]
        
        # Top 3 thấp nhất (bỏ NaN)
        top_3_low = df[["Player", "Team", col]].sort_values(by=col, ascending=True).dropna(subset=[col]).head(3)
        top_3_low = top_3_low.rename(columns={col: "Value"})
        top_3_low["Rank"] = ["1st", "2nd", "3rd"]
        
        rankings[col] = {"Highest": top_3_high, "Lowest": top_3_low}
    
    print(f"✅ Saving top 3 rankings to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for stat, data in rankings.items():
            f.write("=" * 50 + "\n")
            f.write(f"Statistic: {stat}\n\n")
            f.write("Top 3 Lowest:\n")
            # Sử dụng StringIO và to_csv để định dạng bảng
            buffer = StringIO()
            data["Lowest"][["Rank", "Player", "Team", "Value"]].to_csv(buffer, index=False, sep="|")
            f.write(buffer.getvalue())
            f.write("\nTop 3 Highest:\n")
            buffer = StringIO()
            data["Highest"][["Rank", "Player", "Team", "Value"]].to_csv(buffer, index=False, sep="|")
            f.write(buffer.getvalue())

# Yêu cầu 2: Tính median, mean, std cho mỗi thống kê (toàn giải và từng đội)
def save_stats_summary(df, numeric_columns, output_path):
    rows = []
    
    # Tính cho toàn giải
    all_stats = {"": "all"}
    for col in numeric_columns:
        all_stats[f"Median of {col}"] = df[col].median()
        all_stats[f"Mean of {col}"] = df[col].mean()
        all_stats[f"Std of {col}"] = df[col].std()
    rows.append(all_stats)
    
    # Tính cho từng đội
    teams = sorted(df["Team"].unique())
    for team in teams:
        team_df = df[df["Team"] == team]
        team_stats = {"": team}
        for col in numeric_columns:
            team_stats[f"Median of {col}"] = team_df[col].median()
            team_stats[f"Mean of {col}"] = team_df[col].mean()
            team_stats[f"Std of {col}"] = team_df[col].std()
        rows.append(team_stats)
    
    results_df = pd.DataFrame(rows).rename(columns={"": ""})
    for col in results_df.columns:
        if col != "":
            results_df[col] = results_df[col].round(2)
    
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved statistics to {output_path} with {results_df.shape[0]} rows and {results_df.shape[1]} columns.")

# Yêu cầu 3: Vẽ histogram phân phối thống kê
def plot_histograms(df, numeric_columns, base_dir, teams):
    all_players_dir = os.path.join(base_dir, "histograms", "all_players")
    by_team_dir = os.path.join(base_dir, "histograms", "teams")
    
    create_directory(all_players_dir)
    create_directory(by_team_dir)
    
    for stat in numeric_columns:
        if stat not in df.columns:
            print(f"Statistic {stat} not found in DataFrame. Skipping...")
            continue
        
        # Histogram toàn giải
        plt.figure(figsize=(10, 6))
        stat_file = stat.replace(" ", "_")
        plt.savefig(os.path.join(all_players_dir, f"{stat_file}_league.png"), bbox_inches="tight")
        plt.hist(df[stat], bins=20, color="skyblue", edgecolor="black")
        plt.grid(True, alpha=0.3)
        plt.ylabel("Number of Players")
        plt.xlabel(stat)
        plt.title(f"League-Wide Distribution of {stat}")
        plt.close()
        print(f"Generated league histogram for {stat}")
        
        # Histogram từng đội
        for team in teams:
            team_subset = df[df["Team"] == team]
            plt.figure(figsize=(8, 6))
            color = "lightgreen" if stat in ["GA90", "TklW", "Blocks"] else "skyblue"
            plt.hist(team_subset[stat], bins=10, color=color, edgecolor="black", alpha=0.7)
            stat_file = stat.replace(" ", "_")
            plt.savefig(os.path.join(by_team_dir, f"{team}_{stat_file}.png"), bbox_inches="tight")
            plt.grid(True, alpha=0.3)
            plt.ylabel("Number of Players")
            plt.xlabel(stat)
            plt.title(f"{team} - Distribution of {stat}")
            plt.close()
            print(f"Generated team histogram for {team} - {stat}")
    
    print("✅ Completed histogram generation for all statistics.")

# Yêu cầu 4: Xác định đội có điểm trung bình cao nhất cho mỗi thống kê
def save_highest_team_stats(df, numeric_columns, output_path):
    team_means = df.groupby("Team")[numeric_columns].mean().reset_index()
    highest_teams = []
    
    for stat in numeric_columns:
        if stat not in df.columns:
            print(f"Statistic {stat} not found in DataFrame. Skipping...")
            continue
        max_row = team_means.loc[team_means[stat].idxmax()]
        highest_teams.append({
            "Statistic": stat,
            "Team": max_row["Team"],
            "Mean Value": round(max_row[stat], 2)
        })
    
    highest_teams_df = pd.DataFrame(highest_teams)
    highest_teams_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved highest team stats to {output_path} with {highest_teams_df.shape[0]} rows.")

# Yêu cầu 5: Tìm đội hoạt động tốt nhất
def find_best_team(highest_team_stats_path, output_path):
    negative_stats = ["GA90", "CrdY", "CrdR", "Lost", "Mis", "Dis", "Fls", "Off", "Aerl Lost"]
    highest_teams_df = pd.read_csv(highest_team_stats_path)
    positive_stats_df = highest_teams_df[~highest_teams_df["Statistic"].isin(negative_stats)]
    
    team_wins = positive_stats_df["Team"].value_counts()
    best_team = team_wins.idxmax()
    win_count = team_wins.max()
    
    result_text = (
        f"The best-performing team in the 2024-2025 Premier League season is: {best_team}\n"
        f"They lead in {win_count} out of {len(positive_stats_df)} positive statistics."
    )
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result_text)
    
    print(f"✅ Best team result saved to {output_path}")

# Hàm chính
def main():
    # Đường dẫn cơ bản
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_dir = os.path.join(base_dir, "csv")
    txt_dir = os.path.join(base_dir, "txt")
    
    create_directory(csv_dir)
    create_directory(txt_dir)
    
    result_path = os.path.join(csv_dir, "results.csv")  # Đổi từ result.csv thành results.csv
    top_3_path = os.path.join(txt_dir, "top_3.txt")
    results2_path = os.path.join(csv_dir, "results2.csv")
    highest_team_stats_path = os.path.join(csv_dir, "highest_team_stats.csv")
    best_team_path = os.path.join(txt_dir, "The best-performing team.txt")
    
    exclude_columns = ["Player", "Nation", "Team", "Position"]
    
    try:
        # Đọc dữ liệu
        df, numeric_columns = load_data(result_path, exclude_columns)
        teams = sorted(df["Team"].unique())
        
        # Yêu cầu 1: Top 3 cầu thủ
        save_top_3_players(df, numeric_columns, top_3_path)
        
        # Yêu cầu 2: Median, mean, std
        save_stats_summary(df, numeric_columns, results2_path)
        
        # Yêu cầu 3: Histogram
        plot_histograms(df, numeric_columns, base_dir, teams)
        
        # Yêu cầu 4: Đội có điểm cao nhất
        save_highest_team_stats(df, numeric_columns, highest_team_stats_path)
        
        # Yêu cầu 5: Đội tốt nhất
        find_best_team(highest_team_stats_path, best_team_path)
        
    except Exception as e:
        print(f"Lỗi: {e}")
        print("❌ Không thể hoàn thành phân tích.")

if __name__ == "__main__":
    main()