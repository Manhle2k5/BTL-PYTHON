
# Import các thư viện cần thiết
import pandas as pd  # Xử lý dữ liệu CSV
import numpy as np  # Xử lý số học
import os  # Quản lý đường dẫn file
from fuzzywuzzy import fuzz, process  # So khớp tên cầu thủ
import re  # Xử lý chuỗi
from sklearn.model_selection import train_test_split  # Chia dữ liệu train/test
from sklearn.ensemble import GradientBoostingRegressor  # Mô hình Gradient Boosting
from sklearn.preprocessing import StandardScaler  # Chuẩn hóa dữ liệu số
from sklearn.metrics import mean_squared_error, r2_score  # Đánh giá mô hình
from category_encoders import TargetEncoder  # Mã hóa đặc trưng phân loại
from sklearn.pipeline import Pipeline  # Xây dựng pipeline xử lý

# Cấu hình các vị trí cầu thủ và đặc trưng
positions_config = {
    'GK': {
        'position_filter': 'GK',
        'features': ['Save%', 'CS%', 'GA90', 'PK Save%', 'Minutes', 'Age', 'Team', 'Nation']
    },
    'DF': {
        'position_filter': 'DF',
        'features': ['Tkl', 'Int', 'Blocks', 'Aerl Won%', 'Recov', 'Cmp%', 'PrgP', 'Minutes', 'Age', 'Team', 'Nation']
    },
    'MF': {
        'position_filter': 'MF',
        'features': ['Cmp%', 'KP', 'PPA', 'PrgP', 'SCA', 'xAG', 'Tkl', 'Ast', 'Minutes', 'Age', 'Team', 'Nation']
    },
    'FW': {
        'position_filter': 'FW',
        'features': ['Gls', 'Ast', 'Gls per 90', 'xG per 90', 'SCA90', 'GCA90', 'PrgC', 'Minutes', 'Age', 'Team', 'Nation']
    }
}

# Hàm lấy đường dẫn thư mục csv
def get_csv_dir():
    """Tạo đường dẫn đến thư mục csv."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(base_dir, "csv")

# Hàm chuyển đổi giá trị chuyển nhượng (VD: "€2.5M" -> 2500000)
def parse_etv(etv_text):
    """Chuyển đổi chuỗi giá trị chuyển nhượng thành số."""
    if pd.isna(etv_text) or etv_text in ["N/A", ""]:
        return np.nan
    try:
        etv_text = re.sub(r'[€£]', '', etv_text).strip().upper()
        multiplier = 1000000 if 'M' in etv_text else 1000 if 'K' in etv_text else 1
        value = float(re.sub(r'[MK]', '', etv_text)) * multiplier
        return value
    except (ValueError, TypeError):
        return np.nan

# Hàm so khớp tên cầu thủ
def fuzzy_match_name(name, choices, score_threshold=90):
    """Tìm tên gần giống nhất trong choices với ngưỡng tương đồng 90."""
    if not isinstance(name, str):
        return None, None
    shortened_name = shorten_name(name).lower()
    shortened_choices = [shorten_name(c).lower() for c in choices if isinstance(c, str)]
    match = process.extractOne(
        shortened_name,
        shortened_choices,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=score_threshold
    )
    if match:
        matched_idx = shortened_choices.index(match[0])
        return choices[matched_idx], match[1]
    return None, None

# Hàm rút gọn tên cầu thủ
def shorten_name(name):
    """Rút gọn tên thành 2 từ đầu tiên."""
    if not isinstance(name, str):
        return ""
    parts = name.strip().split()
    return " ".join(parts[:2]) if len(parts) >= 2 else name

# Hàm xử lý dữ liệu và huấn luyện mô hình cho mỗi vị trí
def process_position(position, config, results_path, etv_path):
    """
    Xử lý dữ liệu, huấn luyện mô hình, dự đoán ETV cho một vị trí.
    Input: vị trí (GK, DF, MF, FW), config, đường dẫn file results.csv và player_transfer_fee.csv
    Output: DataFrame kết quả, danh sách cầu thủ không khớp
    """
    # Đọc dữ liệu
    try:
        df_results = pd.read_csv(results_path)
        df_etv = pd.read_csv(etv_path)
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file cho {position} - {e}")
        return None, None

    # Lấy vị trí chính
    df_results['Primary_Position'] = df_results['Position'].astype(str).str.split(r'[,/]').str[0].str.strip()
    df_results = df_results[df_results['Primary_Position'].str.upper() == config['position_filter'].upper()].copy()

    # So khớp tên cầu thủ
    player_names = df_etv['Player'].dropna().tolist()
    df_results['Matched_Name'] = None
    df_results['Match_Score'] = None
    df_results['ETV'] = np.nan

    for idx, row in df_results.iterrows():
        matched_name, score = fuzzy_match_name(row['Player'], player_names)
        if matched_name:
            df_results.at[idx, 'Matched_Name'] = matched_name
            df_results.at[idx, 'Match_Score'] = score
            matched_row = df_etv[df_etv['Player'] == matched_name]
            if not matched_row.empty:
                df_results.at[idx, 'ETV'] = parse_etv(matched_row['Price'].iloc[0])

    # Lọc dữ liệu đã khớp
    df_filtered = df_results[df_results['Matched_Name'].notna()].drop_duplicates(subset='Matched_Name')
    unmatched = df_results[df_results['Matched_Name'].isna()]['Player'].dropna().tolist()
    if unmatched:
        print(f"Cầu thủ {position} không khớp: {len(unmatched)} cầu thủ.")
        print(unmatched)

    # Chuẩn bị đặc trưng và mục tiêu
    features = config['features']
    target = 'ETV'
    for col in features:
        if col in ['Team', 'Nation']:
            df_filtered[col] = df_filtered[col].fillna('Unknown')
        else:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
            df_filtered[col] = df_filtered[col].fillna(df_filtered[col].median() if not pd.isna(df_filtered[col].median()) else 0)
            df_filtered[col] = np.log1p(df_filtered[col].clip(lower=0))

    df_ml = df_filtered.dropna(subset=[target]).copy()
    if df_ml.empty:
        print(f"Lỗi: Không có dữ liệu ETV hợp lệ cho {position}.")
        return None, unmatched

    X = df_ml[features]
    y = df_ml[target]

    # Chia dữ liệu
    if len(df_ml) > 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        print(f"Cảnh báo: Không đủ dữ liệu cho {position} để chia train/test.")
        X_train, y_train = X, y
        X_test, y_test = X, y

    # Tạo pipeline
    numeric_features = [col for col in features if col not in ['Team', 'Nation']]
    categorical_features = [col for col in features if col in ['Team', 'Nation']]
    pipeline = Pipeline([
        ('encoder', TargetEncoder(cols=categorical_features)),  # Mã hóa Team, Nation
        ('scaler', StandardScaler()),  # Chuẩn hóa dữ liệu số
        ('model', GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42))
    ])

    # Huấn luyện mô hình
    pipeline.fit(X_train, y_train)

    # Đánh giá mô hình
    if len(X_test) > 0:
        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Đánh giá cho {position}: RMSE = {rmse:.2f}, R² = {r2:.2f}")

    # Dự đoán ETV
    df_filtered['Predicted_Transfer_Value'] = pipeline.predict(df_filtered[features])
    df_filtered['Predicted_Transfer_Value'] = df_filtered['Predicted_Transfer_Value'].clip(lower=100_000, upper=200_000_000)
    df_filtered['Predicted_Transfer_Value_M'] = (df_filtered['Predicted_Transfer_Value'] / 1_000_000).round(2)
    df_filtered['Actual_Transfer_Value_M'] = (df_filtered['ETV'] / 1_000_000).round(2)

    # Chuẩn bị đầu ra
    output_columns = ['Player', 'Team', 'Nation', 'Position', 'Actual_Transfer_Value_M', 'Predicted_Transfer_Value_M']
    df_filtered['Position'] = position
    result = df_filtered[output_columns].copy()

    return result, unmatched

# Hàm chính
def main():
    """Chạy quy trình ước lượng giá trị cầu thủ cho tất cả vị trí."""
    csv_dir = get_csv_dir()
    results_path = os.path.join(csv_dir, "results.csv")  # Thay result.csv bằng results.csv
    etv_path = os.path.join(csv_dir, "player_transfer_fee.csv")
    output_path = os.path.join(csv_dir, "ml_transfer_values_gradient.csv")

    all_results = []
    all_unmatched = []

    for position, config in positions_config.items():
        print(f"\nXử lý vị trí {position}...")
        result, unmatched = process_position(position, config, results_path, etv_path)
        if result is not None:
            all_results.append(result)
        if unmatched:
            all_unmatched.extend([(position, player) for player in unmatched])

    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results = combined_results.sort_values(by='Predicted_Transfer_Value_M', ascending=False)
        combined_results.to_csv(output_path, index=False)
        print(f"Kết quả đã được lưu vào: {output_path}")
    if all_unmatched:
        print(f"\nCầu thủ không khớp: {len(all_unmatched)}")
        for pos, player in all_unmatched:
            print(f"{pos}: {player}")

if __name__ == "__main__":
    main()