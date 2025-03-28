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
# ==============================================================================
# === ↓↓↓ ここから特徴量Xの作成コードを追加 ↓↓↓ ===
# ==============================================================================
print("\n*** 特徴量 X の作成を開始します ***")

# --- 特徴量作成期間のデータを準備 ---
# 基準日以前の注文データ
orders_feature_period = orders[orders['order_purchase_timestamp'] <= feature_period_end_date].copy()
print(f"特徴量作成期間の注文数: {len(orders_feature_period)} 件")

# 特徴量作成期間の注文に紐づく order_items データ
# (もし order_items_products が未定義 or 全期間のものならここで再作成/フィルタ)
if 'order_items_products' not in locals():
     order_items_products = pd.merge(order_items, products_english[['product_id', 'product_category_name_english']], on='product_id', how='left')
order_items_feature = order_items_products[order_items_products['order_id'].isin(orders_feature_period['order_id'])].copy()


# --- 1. 顧客属性特徴量：居住地 (州) ---
print("\n--- 特徴量1: 顧客の州 ---")
# customer_unique_id と customer_state を紐付け
customer_state = customers[['customer_unique_id', 'customer_state']].drop_duplicates(subset=['customer_unique_id'])

# df_target にマージ
df_features = pd.merge(df_target, customer_state, on='customer_unique_id', how='left')
print("customer_state を追加しました。")
print(f"追加後のdf shape: {df_features.shape}")
# print(df_features.head())


# --- 2. RFM指標の作成 ---
print("\n--- 特徴量2: RFM指標 ---")
# 顧客ID (`customer_unique_id`) を追加
orders_feature_customer = pd.merge(orders_feature_period[['order_id', 'customer_id', 'order_purchase_timestamp']],
                                   customer_id_map, on='customer_id', how='left')

# R (Recency): 最終購入日からの経過日数
# 顧客ごとの最終購入日を計算
df_recency = orders_feature_customer.groupby('customer_unique_id')['order_purchase_timestamp'].max().reset_index()
df_recency.columns = ['customer_unique_id', 'last_purchase_date']
# 基準日からの経過日数を計算
df_recency['Recency'] = (feature_period_end_date - df_recency['last_purchase_date']).dt.days
df_recency = df_recency[['customer_unique_id', 'Recency']]

# F (Frequency): 購入回数
df_frequency = orders_feature_customer.groupby('customer_unique_id')['order_id'].nunique().reset_index()
df_frequency.columns = ['customer_unique_id', 'Frequency']

# M (Monetary): 合計購入金額 (order_items の price を使う場合)
# 特徴量作成期間の注文商品データと顧客IDを結合
order_items_customer = pd.merge(order_items_feature[['order_id', 'price']], 
                                orders_feature_customer[['order_id', 'customer_unique_id']], 
                                on='order_id', how='left')
df_monetary = order_items_customer.groupby('customer_unique_id')['price'].sum().reset_index()
df_monetary.columns = ['customer_unique_id', 'Monetary']
# (支払情報 payments テーブルの payment_value を使う場合は別途計算)

# RFM特徴量を df_features にマージ
df_features = pd.merge(df_features, df_recency, on='customer_unique_id', how='left')
df_features = pd.merge(df_features, df_frequency, on='customer_unique_id', how='left')
df_features = pd.merge(df_features, df_monetary, on='customer_unique_id', how='left')

# 欠損値処理 (RFM計算期間内に購入がない顧客)
df_features['Recency'] = df_features['Recency'].fillna(9999) # 大きな値で埋める (例)
df_features['Frequency'] = df_features['Frequency'].fillna(0)
df_features['Monetary'] = df_features['Monetary'].fillna(0)

print("RFM指標 (Recency, Frequency, Monetary) を追加しました。")
print(f"追加後のdf shape: {df_features.shape}")
print("\n--- df_features の基本統計量 ---")
print(df_features[['Recency', 'Frequency', 'Monetary']].describe())
print("\n--- df_features の先頭5行 ---")
print(df_features.head())


# ==============================================================================
# --- ここまでが特徴量Xの作成 (第一弾) ---
# ==============================================================================

# (オプション) 不要になった中間変数を削除してメモリ解放
# import gc
# del orders_feature_period, order_items_feature, customer_state 
# del orders_feature_customer, df_recency, df_frequency, order_items_customer, df_monetary
# gc.collect()

# ==============================================================================
# === ↓↓↓ ここから特定カテゴリの特徴量作成コードを追加 ↓↓↓ ===
# ==============================================================================
print(f"\n--- 特徴量3: 特定カテゴリ ({target_category}) の過去購買行動 ---")

# --- 特徴量作成期間 & 特定カテゴリの注文商品データを抽出 ---
# (前のステップで order_items_feature, orders_feature_customer が作成されている前提)
if 'order_items_feature' in locals() and 'orders_feature_customer' in locals():
    
    target_items_feature = order_items_feature[order_items_feature['product_category_name_english'] == target_category].copy()
    
    # 特定カテゴリの注文に顧客ID (`customer_unique_id`) を紐付け
    target_items_customer = pd.merge(target_items_feature[['order_id', 'price']],
                                     orders_feature_customer[['order_id', 'customer_unique_id', 'order_purchase_timestamp']],
                                     on='order_id', how='left')

    # --- 特徴量の計算 ---
    # 顧客ごとに集計
    df_target_features = target_items_customer.groupby('customer_unique_id').agg(
        target_purchase_count=('order_id', 'nunique'),         # 購入回数 (注文IDのユニーク数)
        target_purchase_amount=('price', 'sum'),               # 購入金額合計
        target_last_purchase_date=('order_purchase_timestamp', 'max') # 最終購入日
    ).reset_index()

    # target_recency の計算
    df_target_features['target_recency'] = (feature_period_end_date - df_target_features['target_last_purchase_date']).dt.days
    df_target_features = df_target_features.drop(columns=['target_last_purchase_date']) # 元の日付カラムは削除

    # --- df_features へマージ ---
    df_features = pd.merge(df_features, df_target_features, on='customer_unique_id', how='left')

    # --- 欠損値処理 & has_purchased_target_before の作成 ---
    # 購入歴フラグを作成 (NaNでない = 購入したことがある)
    df_features['has_purchased_target_before'] = (~df_features['target_purchase_count'].isnull()).astype(int)
    
    # 各特徴量の欠損値を埋める
    df_features['target_purchase_count'] = df_features['target_purchase_count'].fillna(0)
    df_features['target_purchase_amount'] = df_features['target_purchase_amount'].fillna(0)
    df_features['target_recency'] = df_features['target_recency'].fillna(9999) # 大きな値で埋める

    print(f"特定カテゴリ ({target_category}) に関する特徴量を追加しました。")
    print(f"追加後のdf shape: {df_features.shape}")
    print("\n--- 追加された特徴量の基本統計量 ---")
    print(df_features[['has_purchased_target_before', 'target_purchase_count', 'target_purchase_amount', 'target_recency']].describe())
    print("\n--- df_features の先頭5行 (関連列) ---")
    print(df_features[['customer_unique_id', 'Y_purchase_computers_accessories', 'has_purchased_target_before', 'target_purchase_count', 'target_purchase_amount', 'target_recency']].head())

else:
    print("エラー: 特定カテゴリ特徴量の作成に必要なデータフレームが準備できていません。")

# ==============================================================================
# --- ここまでが特徴量Xの作成 (第二弾) ---
# ==============================================================================
# ==============================================================================
# === ↓↓↓ ここからデータ前処理コードを追加 ↓↓↓ ===
# ==============================================================================
print("\n*** データ前処理を開始します ***")

from sklearn.model_selection import train_test_split

if 'df_features' in locals():
    # --- 1. データ型の確認 ---
    print("\n--- 前処理前のデータ型 ---")
    print(df_features.info())

    # --- 2. カテゴリ変数の処理 (One-Hot Encoding) ---
    print("\n--- カテゴリ変数 (customer_state) の One-Hot Encoding ---")
    df_processed = pd.get_dummies(df_features, columns=['customer_state'], dummy_na=False) 
    # dummy_na=False: 欠損値があってもその列は作らない (今回は欠損ないはずだが念のため)
    print(f"One-Hot Encoding 後の df shape: {df_processed.shape}")
    # print(df_processed.head())

    # --- 3. 不要な列の削除 ---
    # customer_unique_id は後で使うかもしれないので、ここでは残しておくことも可能
    # df_processed = df_processed.drop(columns=['customer_unique_id']) 

    # --- 4. データの分割 (特徴量Xと目的変数y、学習データとテストデータ) ---
    print("\n--- 学習データとテストデータへの分割 ---")
    
    # 特徴量 X (目的変数YとID列を除く)
    X = df_processed.drop(columns=['customer_unique_id', 'Y_purchase_computers_accessories'])
    # 目的変数 y
    y = df_processed['Y_purchase_computers_accessories']

    # 学習データとテストデータに分割 (テストデータ20%, 不均衡考慮)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,    # テストデータの割合 (例: 20%)
        random_state=42,  # 再現性のための乱数シード
        stratify=y        # yのクラス比率を保ったまま分割
    )

    print(f"学習データ (X_train) の shape: {X_train.shape}")
    print(f"テストデータ (X_test) の shape: {X_test.shape}")
    print(f"学習データ (y_train) の Y=1 の割合: {y_train.mean():.4f}")
    print(f"テストデータ (y_test) の Y=1 の割合: {y_test.mean():.4f}") # y_train とほぼ同じになるはず

    print("\nデータ前処理と分割が完了しました。")
    # print("\n--- X_train の先頭5行 ---")
    # print(X_train.head())

else:
    print("エラー: 前処理に必要なデータフレーム df_features が準備できていません。")

# ==============================================================================
# --- ここまでがデータ前処理 ---
# ==============================================================================
# ==============================================================================
# === ↓↓↓ ここからモデル学習・評価コードを追加 ↓↓↓ ===
# ==============================================================================
print("\n*** モデル学習と評価を開始します ***")

import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, 
    recall_score, f1_score, roc_auc_score, roc_curve
)

# --- 前提: X_train, X_test, y_train, y_test が準備できていること ---
if 'X_train' in locals() and 'y_train' in locals() and 'X_test' in locals() and 'y_test' in locals():

    # --- 1. LightGBM モデルの学習 ---
    print("\n--- LightGBM モデル学習 ---")
    
    # 不均衡データ対策: scale_pos_weight を計算
    # (Y=0 の数) / (Y=1 の数)
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"scale_pos_weight (for imbalance): {scale_pos_weight:.4f}")

    # LightGBM分類器を初期化
    lgbm = lgb.LGBMClassifier(
        objective='binary',       # 二値分類
        metric='auc',             # 評価指標にAUCを指定
        scale_pos_weight=scale_pos_weight, # 不均衡データ対策
        random_state=42           # 再現性のため
        # 他のパラメータはまずデフォルトで試す
        # n_estimators=100, learning_rate=0.1, etc.
    )

    # モデルを学習 (学習中の評価も表示する例)
    lgbm.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             eval_metric='auc',
             callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True)]) # 10回連続でAUCが改善しなければ停止

    # --- 2. テストデータでの予測 ---
    print("\n--- テストデータで予測 ---")
    # クラス確率を予測 (Y=1 の確率を取得)
    y_pred_proba = lgbm.predict_proba(X_test)[:, 1]
    # クラスラベルを予測 (デフォルト閾値 0.5)
    y_pred = lgbm.predict(X_test)

    # --- 3. モデル評価 ---
    print("\n--- モデル評価結果 ---")
    # 混同行列
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    # [[TN, FP],
    #  [FN, TP]]

    # 主要な評価指標
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba) # AUCは確率で評価

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}") # Y=1と予測した中で実際にY=1だった割合
    print(f"Recall:    {recall:.4f}")    # 実際にY=1だった中でY=1と予測できた割合
    print(f"F1-Score:  {f1:.4f}")       # PrecisionとRecallの調和平均
    print(f"AUC:       {auc:.4f}")       # モデルの総合的な識別能力 (不均衡データで重要)

    # --- 4. 特徴量重要度の確認 ---
    print("\n--- 特徴量重要度 (Top 20) ---")
    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': lgbm.feature_importances_
    }).sort_values('importance', ascending=False)

    print(feature_importance_df.head(20))

    # (オプション) 特徴量重要度を可視化
    # plt.figure(figsize=(10, 8))
    # sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20))
    # plt.title('LightGBM Feature Importance (Top 20)')
    # plt.tight_layout()
    # plt.show() 
    # plt.savefig('feature_importance.png') # 画像として保存

    print("\nモデル学習と評価が完了しました。")

else:
    print("エラー: モデル学習に必要なデータ (X_train, y_train, X_test, y_test) が準備できていません。")

# ==============================================================================
# --- ここまでがモデル学習・評価 ---
# ==============================================================================