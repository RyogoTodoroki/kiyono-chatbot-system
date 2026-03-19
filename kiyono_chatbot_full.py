import os
import re
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

# =========================
# テキスト分析用辞書
# =========================

NEGATIVE_WORDS = [
    "できない", "印刷できない", "動かない", "反応しない", "ダメ",
    "使えない", "直らない", "分からない", "わからない", "困る",
    "困ってる", "困っています", "おかしい", "違う", "無理"
]

EMPHASIS_WORDS = [
    "全然", "まだ", "何回", "何度", "ずっと", "さっきから",
    "まったく", "全く"
]

QUESTION_PATTERNS_POLITE = [
    "ですか", "ますか"
]

QUESTION_PATTERNS_SOLUTION = [
    "どうすればいい", "どうしたらいい", "何を確認", "原因わかりますか"
]

QUESTION_PATTERNS_DIRECT = [
    "なんで", "なぜ", "どうなってる", "どうすればいい", "どうしたらいい"
]

DIRECT_EXPRESSIONS = [
    "ダメ", "無理", "おかしい", "違う", "使えない", "できない"
]

TEXT_WEIGHTS = {
    "negative_count": 1.5,
    "emphasis_count": 1.2,
    "question_direct": 1.8,
    "question_solution": 1.0,
    "question_polite": -0.5,
    "direct_expression_count": 1.3,
    "short_text": 1.2,
    "very_short_text": 1.8,
    "long_explanatory_text": -0.8,
}

TEXT_SCALE = 1.3

# =========================
# 行動分析用重み
# =========================

REPLY_SPEED_SCORES = {
    1: 2.0,   # 5分以内
    2: 1.0,   # 1時間以内
    3: 0.5,   # 1日以上
}

BEHAVIOR_WEIGHTS = {
    "burst": 0.8,
    "turns": 0.5,
    "repeat": 1.2,
    "option_retry": 0.9,
    "free_text": 1.0,
    "dropout": 2.5,
}

COMBO_WEIGHTS = {
    "fast_reply_and_dropout": 2.5,
    "option_failure_mild": 1.5,
    "option_failure_strong": 3.0,
}

BEHAVIOR_SCALE = 2.0

# =========================
# モデル保存先
# =========================

MODEL_FILE = "frustration_model.joblib"

# =========================
# テキスト分析関数
# =========================

def count_matches(text, word_list):
    count = 0
    for word in word_list:
        count += text.count(word)
    return count


def detect_question_type(text):
    for pattern in QUESTION_PATTERNS_DIRECT:
        if pattern in text:
            return "direct", TEXT_WEIGHTS["question_direct"]

    for pattern in QUESTION_PATTERNS_SOLUTION:
        if pattern in text:
            return "solution", TEXT_WEIGHTS["question_solution"]

    for pattern in QUESTION_PATTERNS_POLITE:
        if pattern in text:
            return "polite", TEXT_WEIGHTS["question_polite"]

    if "?" in text or "？" in text:
        return "generic", 0.3

    return "none", 0.0


def text_length_score(text):
    cleaned = re.sub(r"\s+", "", text)
    length = len(cleaned)

    if length <= 12:
        return length, TEXT_WEIGHTS["very_short_text"]
    elif length <= 22:
        return length, TEXT_WEIGHTS["short_text"]
    elif length >= 45:
        return length, TEXT_WEIGHTS["long_explanatory_text"]
    else:
        return length, 0.0


def analyze_text(text):
    negative_count = count_matches(text, NEGATIVE_WORDS)
    emphasis_count = count_matches(text, EMPHASIS_WORDS)
    direct_expression_count = count_matches(text, DIRECT_EXPRESSIONS)

    question_type, question_score = detect_question_type(text)
    text_length, length_score = text_length_score(text)

    raw_score = 0.0
    raw_score += negative_count * TEXT_WEIGHTS["negative_count"]
    raw_score += emphasis_count * TEXT_WEIGHTS["emphasis_count"]
    raw_score += direct_expression_count * TEXT_WEIGHTS["direct_expression_count"]
    raw_score += question_score
    raw_score += length_score

    frustration_score = round(max(0, min(10, raw_score / TEXT_SCALE)))

    return {
        "negative_count": negative_count,
        "emphasis_count": emphasis_count,
        "direct_expression_count": direct_expression_count,
        "question_type": question_type,
        "text_length": text_length,
        "text_raw_score": round(raw_score, 2),
        "text_score_0_to_10": frustration_score,
    }

# =========================
# 行動分析関数
# =========================

def analyze_behavior(reply_speed, burst, turns, repeat, option_retry, free_text, dropout):
    reply_score = REPLY_SPEED_SCORES.get(reply_speed, 0.0)

    raw_score = 0.0
    raw_score += reply_score
    raw_score += burst * BEHAVIOR_WEIGHTS["burst"]
    raw_score += turns * BEHAVIOR_WEIGHTS["turns"]
    raw_score += repeat * BEHAVIOR_WEIGHTS["repeat"]
    raw_score += option_retry * BEHAVIOR_WEIGHTS["option_retry"]
    raw_score += free_text * BEHAVIOR_WEIGHTS["free_text"]
    raw_score += dropout * BEHAVIOR_WEIGHTS["dropout"]

    fast_reply_and_dropout = 0
    if reply_speed == 1 and dropout == 1:
        fast_reply_and_dropout = 1
        raw_score += COMBO_WEIGHTS["fast_reply_and_dropout"]

    option_failure_flag = "none"
    if option_retry >= 3 and free_text >= 2:
        option_failure_flag = "strong"
        raw_score += COMBO_WEIGHTS["option_failure_strong"]
    elif option_retry >= 2 and free_text >= 1:
        option_failure_flag = "mild"
        raw_score += COMBO_WEIGHTS["option_failure_mild"]

    frustration_score = round(max(0, min(10, raw_score / BEHAVIOR_SCALE)))

    return {
        "reply_speed_score": reply_score,
        "fast_reply_and_dropout": fast_reply_and_dropout,
        "option_failure_flag": option_failure_flag,
        "behavior_raw_score": round(raw_score, 2),
        "behavior_score_0_to_10": frustration_score,
    }

# =========================
# 統合スコア
# =========================

def combine_scores(text_score, behavior_score, text_weight=0.5, behavior_weight=0.5):
    combined = text_score * text_weight + behavior_score * behavior_weight
    return round(max(0, min(10, combined)), 2)

# =========================
# 入力補助
# =========================

def input_integer(message, min_value=0, max_value=None):
    while True:
        try:
            value = int(input(message))
            if value < min_value:
                print(f"{min_value}以上の整数を入力してください。")
                continue
            if max_value is not None and value > max_value:
                print(f"{max_value}以下の整数を入力してください。")
                continue
            return value
        except ValueError:
            print("整数を入力してください。")

# =========================
# 学習用特徴量作成
# =========================

def build_feature_row(text, reply_speed, burst, turns, repeat, option_retry, free_text, dropout):
    text_result = analyze_text(text)
    behavior_result = analyze_behavior(reply_speed, burst, turns, repeat, option_retry, free_text, dropout)

    question_type = text_result["question_type"]
    question_direct = 1 if question_type == "direct" else 0
    question_solution = 1 if question_type == "solution" else 0
    question_polite = 1 if question_type == "polite" else 0

    option_failure_mild = 1 if behavior_result["option_failure_flag"] == "mild" else 0
    option_failure_strong = 1 if behavior_result["option_failure_flag"] == "strong" else 0

    return {
        "original_text": text,

        "negative_count": text_result["negative_count"],
        "emphasis_count": text_result["emphasis_count"],
        "direct_expression_count": text_result["direct_expression_count"],
        "text_length": text_result["text_length"],

        "question_direct": question_direct,
        "question_solution": question_solution,
        "question_polite": question_polite,

        "reply_speed": reply_speed,
        "burst": burst,
        "turns": turns,
        "repeat": repeat,
        "option_retry": option_retry,
        "free_text": free_text,
        "dropout": dropout,

        "fast_reply_and_dropout": behavior_result["fast_reply_and_dropout"],
        "option_failure_mild": option_failure_mild,
        "option_failure_strong": option_failure_strong,
    }
    text_result = analyze_text(text)
    behavior_result = analyze_behavior(reply_speed, burst, turns, repeat, option_retry, free_text, dropout)

    question_type = text_result["question_type"]
    question_direct = 1 if question_type == "direct" else 0
    question_solution = 1 if question_type == "solution" else 0
    question_polite = 1 if question_type == "polite" else 0

    option_failure_mild = 1 if behavior_result["option_failure_flag"] == "mild" else 0
    option_failure_strong = 1 if behavior_result["option_failure_flag"] == "strong" else 0

    return {
        "negative_count": text_result["negative_count"],
        "emphasis_count": text_result["emphasis_count"],
        "direct_expression_count": text_result["direct_expression_count"],
        "text_length": text_result["text_length"],
        "question_direct": question_direct,
        "question_solution": question_solution,
        "question_polite": question_polite,
        "reply_speed": reply_speed,
        "burst": burst,
        "turns": turns,
        "repeat": repeat,
        "option_retry": option_retry,
        "free_text": free_text,
        "dropout": dropout,
        "fast_reply_and_dropout": behavior_result["fast_reply_and_dropout"],
        "option_failure_mild": option_failure_mild,
        "option_failure_strong": option_failure_strong,
    }

# =========================
# 機械学習
# =========================

def train_model_from_csv(csv_file):
    df = pd.read_csv(csv_file)

    feature_columns = [
        "negative_count",
        "emphasis_count",
        "direct_expression_count",
        "text_length",
        "question_direct",
        "question_solution",
        "question_polite",
        "reply_speed",
        "burst",
        "turns",
        "repeat",
        "option_retry",
        "free_text",
        "dropout",
        "fast_reply_and_dropout",
        "option_failure_mild",
        "option_failure_strong",
    ]

    target_column = "frustration"

    X = df[feature_columns]
    y = df[target_column]

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump({
        "model": model,
        "feature_columns": feature_columns,
    }, MODEL_FILE)

    print("\n===== 学習結果 =====")
    print("モデルを保存しました:", MODEL_FILE)
    print("切片:", round(model.intercept_, 4))
    print("重み:")
    for name, coef in zip(feature_columns, model.coef_):
        print(f"  {name}: {round(coef, 4)}")


def predict_with_model(feature_row):
    if not os.path.exists(MODEL_FILE):
        return None

    saved = joblib.load(MODEL_FILE)
    model = saved["model"]
    feature_columns = saved["feature_columns"]

    X_input = pd.DataFrame([feature_row])[feature_columns]
    pred = model.predict(X_input)[0]

    pred = round(max(0, min(10, pred)), 2)
    return pred

# =========================
# 学習データ作成補助
# =========================

def append_training_data(csv_file, feature_row, frustration_label):
    row = feature_row.copy()
    row["frustration"] = frustration_label
    df_new = pd.DataFrame([row])

    if os.path.exists(csv_file):
        df_old = pd.read_csv(csv_file)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"\n学習データを保存しました: {csv_file}")

# =========================
# メイン
# =========================

def main():
    print("===== テキスト分析 + 行動分析 + 重み学習 =====\n")

    text = input("ユーザー入力文を入れてください: ").strip()

    print("\n返信速度を選択してください")
    print("1: 5分以内")
    print("2: 1時間以内")
    print("3: 1日以上")

    reply_speed = input_integer("返信速度 (1〜3): ", 1, 3)
    burst = input_integer("連続送信回数: ", 0)
    turns = input_integer("会話ターン数: ", 0)
    repeat = input_integer("再入力回数: ", 0)
    option_retry = input_integer("選択肢の選び直し回数: ", 0)
    free_text = input_integer("自由入力切替回数: ", 0)
    dropout = input_integer("離脱あり? (0:なし / 1:あり): ", 0, 1)

    text_result = analyze_text(text)
    behavior_result = analyze_behavior(reply_speed, burst, turns, repeat, option_retry, free_text, dropout)

    combined_score = combine_scores(
        text_result["text_score_0_to_10"],
        behavior_result["behavior_score_0_to_10"],
        text_weight=0.5,
        behavior_weight=0.5
    )

    print("\n===== テキスト分析結果 =====")
    print("否定語数:", text_result["negative_count"])
    print("強調語数:", text_result["emphasis_count"])
    print("直接表現数:", text_result["direct_expression_count"])
    print("疑問文タイプ:", text_result["question_type"])
    print("文字数:", text_result["text_length"])
    print("テキスト生スコア:", text_result["text_raw_score"])
    print("テキスト不満度(0〜10):", text_result["text_score_0_to_10"])

    print("\n===== 行動分析結果 =====")
    print("行動生スコア:", behavior_result["behavior_raw_score"])
    print("急返信離脱フラグ:", behavior_result["fast_reply_and_dropout"])
    print("選択肢失敗フラグ:", behavior_result["option_failure_flag"])
    print("行動不満度(0〜10):", behavior_result["behavior_score_0_to_10"])

    print("\n===== 統合結果（手動重み） =====")
    print("総合不満度(0〜10):", combined_score)

    feature_row = build_feature_row(
        text, reply_speed, burst, turns, repeat, option_retry, free_text, dropout
    )

    ai_prediction = predict_with_model(feature_row)
    if ai_prediction is not None:
        print("\n===== AI予測結果 =====")
        print("学習済みモデルによる予測不満度(0〜10):", ai_prediction)
    else:
        print("\n学習済みモデルはまだありません。")
        print("先に学習データCSVを作って学習させてください。")

    print("\n===== 追加メニュー =====")
    print("1: 今回のデータを学習用CSVに保存")
    print("2: CSVからモデルを学習")
    print("3: 何もしない")

    choice = input_integer("選択してください (1〜3): ", 1, 3)

    if choice == 1:
        csv_file = "training_data.csv"
        frustration_label = input_integer("このデータの正解不満度(0〜10)を入力してください: ", 0, 10)
        append_training_data(csv_file, feature_row, frustration_label)

    elif choice == 2:
        csv_file = input("学習に使うCSVファイル名を入力してください（例: training_data.csv）: ").strip()
        if os.path.exists(csv_file):
            train_model_from_csv(csv_file)
        else:
            print("CSVファイルが見つかりません。")

    else:
        print("終了します。")


if __name__ == "__main__":
    main()