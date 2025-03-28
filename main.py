# ==============================================================================
# ライブラリのインポート
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime # 日付計算のために必要

# ==============================================================================
# 設定
# ==============================================================================
# --- データファイルが置かれているディレクトリを指定 ---
data_dir = './olist_data' # ご自身の環境に合わせて変更してください

# --- 目的変数作成のための設定 ---
prediction_days = 90 # 予測対象期間の日数
target_category = 'computers_accessories' # 予測対象カテゴリ

# ==============================================================================
# データの読み込み
# ==============================================================================
print("--- データの読み込み開始 ---")
try:
    customers = pd.read_csv(os.path.join(data_dir, 'olist_customers_dataset.csv'))
    orders = pd.read_csv(os.path.join(data_dir, 'olist_orders_dataset.csv'))
    order_items = pd.read_csv(os.path.join(data_dir, 'olist_order_items_dataset.csv'))
    products = pd.read_csv(os.path.join(data_dir, 'olist_products_dataset.csv'))
    category_translation = pd.read_csv(os.path.join(data_dir, 'product_category_name_translation.csv'))
    # 必要に応じて他のファイルも読み込みます
    # payments = pd.read_csv(os.path.join(data_dir, 'olist_order_payments_dataset.csv'))
    # reviews = pd.read_csv(os.path.join(data_dir, 'olist_order_reviews_dataset.csv'))
    # sellers = pd.read_csv(os.path.join(data_dir, 'olist_sellers_dataset.csv'))
    # geolocation = pd.read_csv(os.path.join(data_dir, 'olist_geolocation_dataset.csv'))

    print("データの読み込みが完了しました。")

except FileNotFoundError:
    print(f"エラー: 指定されたディレクトリ '{data_dir}' にデータファイルが見つかりません。パスを確認してください。")
    exit() # 例: ファイルがない場合は終了
except Exception as e:
    print(f"データの読み込み中にエラーが発生しました: {e}")
    exit() # 例: 他のエラーでも終了

# ==============================================================================
# データ基本情報の確認 (必要な場合コメント解除)
# ==============================================================================
# print("\n--- Customers Info ---")
# customers.info()
# ... (他のテーブルも同様) ...

# ==============================================================================
# 商品カテゴリ名の英語化
# ==============================================================================
print("\n--- 商品カテゴリ名の英語化 ---")
products_english = pd.merge(products, category_translation, on='product_category_name', how='left')
print("products_english データフレームを作成しました。")

# ==============================================================================
# (参考) カテゴリ別集計 (必要な場合コメント解除)
# ==============================================================================
# print("\n--- 商品カテゴリごとの注文件数 (上位30) ---")
# if 'order_items' in locals() and 'products_english' in locals():
#     order_items_products_ref = pd.merge(order_items, products_english[['product_id', 'product_category_name_english']], on='product_id', how='left')
#     order_category_counts = order_items_products_ref['product_category_name_english'].value_counts()
#     print(order_category_counts.head(30))

# ==============================================================================
# デバッグ用 Print 文 (コメントアウト)
# ==============================================================================
# print("\n--- 目的変数作成前のローカル変数チェック ---")
# print(locals().keys())
# print("\n--- if文の評価チェック ---")
# required_dfs_for_y = ['orders', 'products_english', 'order_items', 'customers']
# results = {df_name: (df_name in locals()) for df_name in required_dfs_for_y}
# print(f"各データフレームの存在確認: {results}")
# print(f"all()の結果: {all(df_name in locals() for df_name in required_dfs_for_y)}")

# ==============================================================================
# 目的変数 Y の作成
# ==============================================================================
print(f"\n*** 目的変数 Y ({target_category} 購入フラグ) の作成を開始します ***")

# --- 前提: 必要なデータが読み込まれていることの確認 (修正済み) ---
if ('orders' in locals() and
    'products_english' in locals() and
    'order_items' in locals() and
    'customers' in locals()):

    # 1. 基準日と予測期間を定義
    if not pd.api.types.is_datetime64_any_dtype(orders['order_purchase_timestamp']):
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'], errors='coerce')

    orders = orders.dropna(subset=['order_purchase_timestamp'])
    max_date = orders['order_purchase_timestamp'].max()
    prediction_period_start_date = max_date - datetime.timedelta(days=prediction_days - 1)
    feature_period_end_date = prediction_period_start_date - datetime.timedelta(days=1)

    print(f"データ全体の最終日: {max_date}")
    print(f"特徴量作成期間の終了日 (基準日): {feature_period_end_date}")
    print(f"予測対象期間: {prediction_period_start_date} から {max_date}")

    # 2. 予測期間内の注文データを抽出
    orders_prediction_period = orders[orders['order_purchase_timestamp'] >= prediction_period_start_date].copy()
    print(f"\n予測期間内の全注文数: {len(orders_prediction_period)} 件")

    # 3. 予測期間内に「特定カテゴリ」を購入した注文を特定
    # order_items と products_english を結合 (未定義の場合のみ)
    if 'order_items_products' not in locals():
         order_items_products = pd.merge(order_items, products_english[['product_id', 'product_category_name_english']], on='product_id', how='left')

    prediction_order_ids = orders_prediction_period['order_id'].unique()

    target_items_prediction_period = order_items_products[
        (order_items_products['order_id'].isin(prediction_order_ids)) &
        (order_items_products['product_category_name_english'] == target_category)
    ].copy()
    print(f"予測期間内の '{target_category}' の注文件商品数: {len(target_items_prediction_period)} 件")

    # 4. 購入した顧客ID (`customer_id`) を取得
    target_order_ids_in_period = target_items_prediction_period['order_id'].unique()
    target_customers = orders_prediction_period[orders_prediction_period['order_id'].isin(target_order_ids_in_period)]['customer_id'].unique()
    print(f"予測期間内の '{target_category}' の購入顧客数 (customer_idベース): {len(target_customers)} 人")

    # 5. 全顧客リストと結合して目的変数Yを作成
    # customer_id と customer_unique_id を紐付ける (未定義の場合のみ)
    if 'customer_id_map' not in locals():
        customer_id_map = customers[['customer_id', 'customer_unique_id']].drop_duplicates()

    target_unique_customers_in_period = customer_id_map[customer_id_map['customer_id'].isin(target_customers)]['customer_unique_id'].unique()
    print(f"予測期間内の '{target_category}' の購入顧客数 (unique IDベース): {len(target_unique_customers_in_period)} 人")

    all_unique_customers = customers['customer_unique_id'].unique()
    df_target = pd.DataFrame({'customer_unique_id': all_unique_customers})

    df_target['Y_purchase_computers_accessories'] = df_target['customer_unique_id'].apply(
        lambda x: 1 if x in target_unique_customers_in_period else 0
    )

    print(f"\n目的変数Yデータフレーム (`df_target`) 作成完了 (全 {len(df_target)} 顧客)")
    print(f"購入者(Y=1)の割合: {df_target['Y_purchase_computers_accessories'].mean():.4f}")
    print(df_target['Y_purchase_computers_accessories'].value_counts())
    print("\n--- df_target の先頭5行 ---")
    print(df_target.head())

else:
    # このエラーメッセージは、 if 文の条件が修正されたことで、通常は表示されなくなるはずです。
    print("エラー: 目的変数作成に必要なデータフレームが準備できていません。")

# ==============================================================================
# --- ここまでが目的変数Yの作成 ---
# ==============================================================================