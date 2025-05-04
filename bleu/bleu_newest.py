import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sys

def evaluate_bleu(caption_test_path, results_path):
    # Tải dữ liệu
    caption_test = pd.read_csv(caption_test_path, delimiter='|')
    results = pd.read_csv(results_path, delimiter='|')

    # Đảm bảo tên cột đúng
    caption_test.columns = ['image_name', 'comment_number', 'caption']
    results.columns = ['image_name', 'comment_number', 'caption']

    # Giữ lại 5 bình luận đầu tiên cho mỗi image_name
    results_unique = results.groupby('image_name').head(5)

    # Chuyển dữ liệu thành dạng Pivot để có nhiều caption cho mỗi ảnh
    results_pivot = results_unique.pivot(index='image_name', columns='comment_number', values='caption').reset_index()
    results_pivot.columns = ['image_name', 'caption_0', 'caption_1', 'caption_2', 'caption_3', 'caption_4']

    # Gộp dữ liệu theo image_name để khớp các caption
    merged_df = pd.merge(caption_test[['image_name', 'caption']], results_pivot, on='image_name')

    # Khởi tạo Smoothing Function
    smoothing_function = SmoothingFunction().method1

    # Tính toán điểm BLEU và độ chính xác cho mỗi cấp độ BLEU
    results_data = []
    for _, row in merged_df.iterrows():
        reference = row['caption'].split()  # Chia câu tham chiếu thành các từ
        candidates = [row[f'caption_{i}'].split() for i in range(5) if pd.notna(row[f'caption_{i}'])]  # Lọc bỏ caption trống

        bleu_scores_1 = [
            sentence_bleu([reference], candidate, weights=(1, 0, 0, 0), smoothing_function=smoothing_function) for candidate in candidates
        ]
        bleu_scores_2 = [
            sentence_bleu([reference], candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function) for candidate in candidates
        ]
        bleu_scores_3 = [
            sentence_bleu([reference], candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function) for candidate in candidates
        ]
        bleu_scores_4 = [
            sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function) for candidate in candidates
        ]

        avg_bleu_score = (sum(bleu_scores_1) / len(bleu_scores_1) if bleu_scores_1 else 0) * 100  # Tính điểm trung bình BLEU-1

        # Thêm điểm BLEU vào kết quả
        results_data.append({
            "image_name": row['image_name'],
            "bleu-1": bleu_scores_1,
            "bleu-2": bleu_scores_2,
            "bleu-3": bleu_scores_3,
            "bleu-4": bleu_scores_4,
            "average_bleu_score": avg_bleu_score
        })

    # Tạo DataFrame với kết quả
    results_df = pd.DataFrame(results_data)
    results_df_sorted = results_df.sort_values(by="average_bleu_score", ascending=False)

    # Tính điểm BLEU trung bình cho từng cấp độ
    bleu_1_avg = results_df_sorted["bleu-1"].apply(lambda x: sum(x) / len(x) if x else 0).mean()
    bleu_2_avg = results_df_sorted["bleu-2"].apply(lambda x: sum(x) / len(x) if x else 0).mean()
    bleu_3_avg = results_df_sorted["bleu-3"].apply(lambda x: sum(x) / len(x) if x else 0).mean()
    bleu_4_avg = results_df_sorted["bleu-4"].apply(lambda x: sum(x) / len(x) if x else 0).mean()

    # Tính điểm BLEU trung bình
    average_bleu_score = results_df_sorted["average_bleu_score"].mean()

    # Thêm điểm BLEU trung bình vào cuối file CSV
    results_df_sorted = results_df_sorted.append({
        "image_name": "Average",
        "bleu-1": bleu_1_avg,
        "bleu-2": bleu_2_avg,
        "bleu-3": bleu_3_avg,
        "bleu-4": bleu_4_avg,
        "average_bleu_score": average_bleu_score
    }, ignore_index=True)

    output_file = "bleu_score_results_with_average.csv"
    results_df_sorted.to_csv(output_file, index=False)

    print(f"Đánh giá BLEU hoàn thành. Kết quả đã được lưu vào {output_file}")
    print(f"Điểm BLEU trung bình: {average_bleu_score}")
    print(f"Điểm BLEU trung bình các cấp độ: BLEU-1: {bleu_1_avg}, BLEU-2: {bleu_2_avg}, BLEU-3: {bleu_3_avg}, BLEU-4: {bleu_4_avg}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Cách sử dụng: python bleu_evaluation.py <caption_test_path> <results_path>")
    else:
        caption_test_path = sys.argv[1]
        results_path = sys.argv[2]
        evaluate_bleu(caption_test_path, results_path)
