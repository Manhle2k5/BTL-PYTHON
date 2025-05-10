# Import các thư viện cần thiết
import pandas as pd  # Xử lý dữ liệu CSV
import os  # Làm việc với đường dẫn file
from fuzzywuzzy import fuzz, process  # So khớp tên cầu thủ
from selenium import webdriver  # Tự động hóa trình duyệt
from selenium.webdriver.chrome.service import Service  # Quản lý ChromeDriver
from selenium.webdriver.chrome.options import Options  # Cấu hình Chrome
from selenium.webdriver.common.by import By  # Tìm kiếm phần tử HTML
from selenium.webdriver.support.ui import WebDriverWait  # Chờ tải trang
from selenium.webdriver.support import expected_conditions as EC  # Điều kiện chờ
from webdriver_manager.chrome import ChromeDriverManager  # Tự động tải ChromeDriver
import time  # Dùng để tạm dừng khi thử lại
from selenium.common.exceptions import WebDriverException  # Xử lý lỗi Selenium

# Hàm lấy đường dẫn thư mục csv
def get_csv_dir():
    """Tạo đường dẫn đến thư mục csv, nằm ngoài thư mục chứa script."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(base_dir, "csv")

# Hàm lọc cầu thủ từ results.csv
def filter_players(input_path, output_path):
    """
    Đọc file results.csv, lọc cầu thủ có >900 phút, lưu vào file mới.
    Input: Đường dẫn file results.csv
    Output: Đường dẫn file players_over_900_minutes.csv
    """
    # Đọc file CSV, coi "N/A" là giá trị NaN
    df = pd.read_csv(input_path, na_values=["N/A"])
    # Lọc cầu thủ có thời gian thi đấu >900 phút
    filtered_df = df[df['Minutes'] > 900].copy()
    print(f"Số cầu thủ có trên 900 phút: {len(filtered_df)}")
    # Lưu file CSV mới
    filtered_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Đã lưu danh sách cầu thủ vào: {output_path}")
    return filtered_df

# Hàm rút gọn tên cầu thủ
def shorten_name(name):
    """Rút gọn tên cầu thủ thành 2 từ đầu tiên để so khớp."""
    parts = name.strip().split()
    return " ".join(parts[:2]) if len(parts) >= 2 else name

# Hàm crawl dữ liệu chuyển nhượng
def scrape_transfer_data(driver, urls, player_names):
    """
    Crawl dữ liệu chuyển nhượng từ danh sách URL, khớp với danh sách cầu thủ.
    Input: driver (ChromeDriver), danh sách URL, danh sách tên cầu thủ
    Output: Danh sách [tên cầu thủ, giá chuyển nhượng]
    """
    data = []
    seen_players = set()  # Lưu tên cầu thủ đã xử lý để tránh trùng lặp

    for url in urls:
        for attempt in range(3):  # Thử tối đa 3 lần nếu lỗi
            try:
                print(f"Đang crawl: {url}")
                driver.get(url)
                # Chờ bảng chuyển nhượng xuất hiện (tối đa 10 giây)
                table = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "transfer-table"))
                )
                # Lấy tất cả các hàng trong bảng
                rows = table.find_elements(By.TAG_NAME, "tr")

                # Duyệt từng hàng
                for row in rows:
                    cols = row.find_elements(By.TAG_NAME, "td")
                    if cols and len(cols) >= 3:  # Đảm bảo có đủ cột
                        # Lấy tên cầu thủ
                        player_name = cols[0].text.strip().split("\n")[0].strip()
                        # Rút gọn tên để so khớp
                        shortened_name = shorten_name(player_name)
                        # Lấy giá trị chuyển nhượng
                        transfer_value = cols[-1].text.strip() if cols[-1].text.strip() else "N/A"

                        # Chỉ xử lý nếu có giá trị chuyển nhượng hợp lệ
                        if transfer_value != "N/A":
                            # So khớp tên với danh sách cầu thủ
                            best_match = process.extractOne(
                                shortened_name, player_names, scorer=fuzz.token_sort_ratio
                            )
                            if best_match and best_match[1] >= 90:  # Ngưỡng so khớp 90%
                                if player_name not in seen_players:
                                    seen_players.add(player_name)
                                    data.append([player_name, transfer_value])
                                    print(f"Khớp: {player_name} (Giá: {transfer_value})")
                break  # Thoát vòng lặp retry nếu crawl thành công
            except WebDriverException as e:
                print(f"Lỗi khi crawl {url} (lần {attempt + 1}/3): {e}")
                time.sleep(2)  # Chờ 2 giây trước khi thử lại
                if attempt == 2:
                    print(f"Không thể crawl {url} sau 3 lần thử.")
    
    return data

# Hàm chính để chạy chương trình
def main():
    """Chạy toàn bộ quy trình: lọc cầu thủ, crawl dữ liệu, lưu kết quả."""
    # Lấy đường dẫn thư mục csv
    csv_dir = get_csv_dir()
    result_path = os.path.join(csv_dir, "results.csv")
    filtered_path = os.path.join(csv_dir, "players_over_900_minutes.csv")
    output_path = os.path.join(csv_dir, "player_transfer_fee.csv")

    # Lọc cầu thủ từ results.csv
    filtered_df = filter_players(result_path, filtered_path)

    # Tạo danh sách tên cầu thủ rút gọn
    player_names = [shorten_name(name) for name in filtered_df['Player'].str.strip()]

    # Cấu hình ChromeDriver
    options = Options()
    options.add_argument("--no-sandbox")  # Tắt sandbox để tránh lỗi
    options.add_argument("--disable-dev-shm-usage")  # Tắt bộ nhớ chia sẻ
    options.add_argument("--ignore-certificate-errors")  # Bỏ qua lỗi SSL
    options.add_argument("--allow-insecure-localhost")  # Cho phép kết nối không an toàn
    # Tạm thời tắt headless để kiểm tra lỗi SSL
    # options.add_argument("--headless=new")  # Có thể bật lại sau khi sửa lỗi

    # Khởi tạo ChromeDriver
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )

    # Tạo danh sách URL (crawl 10 trang để đảm bảo đủ dữ liệu)
    base_url = "https://www.footballtransfers.com/us/transfers/confirmed/2024-2025/uk-premier-league/"
    urls = [f"{base_url}{i}" for i in range(1, 11)]

    # Crawl dữ liệu chuyển nhượng
    data = scrape_transfer_data(driver, urls, player_names)

    # Đóng trình duyệt
    driver.quit()

    # Lưu kết quả vào file CSV
    if data:
        df_output = pd.DataFrame(data, columns=['Player', 'Price'])
        df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Kết quả đã được lưu vào: {output_path} với {len(df_output)} cầu thủ.")
    else:
        print("Không tìm thấy cầu thủ nào khớp.")

# Chạy chương trình
if __name__ == "__main__":
    main()
