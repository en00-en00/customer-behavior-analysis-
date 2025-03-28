# ==============================================================================
# ライブラリのインポート
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, roc_auc_score
)

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
    # payments = pd.read_csv(os.path.join(data_dir, 'olist_order_payments_dataset.csv')) # 必要になったら読み込む
    # reviews = pd.read_csv(os.path.join(data_dir, 'olist_order_reviews_dataset.csv')) # 必要になったら読み込む

    print("データの読み込み完了。")

except FileNotFoundError:
    print(f"エラー: 指定されたディレクトリ '{data_dir}' にデータファイルが見つかりません。パスを確認してください。")
    exit()
except Exception as e:
    print(f"データの読み込み中にエラーが発生しました: {e}")
    exit()

# ==============================================================================
# データ準備 (カテゴリ英語化など)
# ==============================================================================
# 商品カテゴリ名を英語に変換
products_english = pd.merge(products, category_translation, on='product_category_name', how='left')

# order_items に商品情報 (カテゴリ名含む) を結合 (後で複数回使うため最初に結合)
order_items_products = pd.merge(order_items, products_english[['product_id', 'product_category_name_english']], on='product_id', how='left')

# orders の日付データを datetime 型に変換
if not pd.api.types.is_datetime64_any_dtype(orders['order_purchase_timestamp']):
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'], errors='coerce')
orders = orders.dropna(subset=['order_purchase_timestamp']) # 欠損除去

# customers から customer_id と unique_id のマップを作成
customer_id_map = customers[['customer_id', 'customer_unique_id']].drop_duplicates()

# ==============================================================================
# 目的変数 Y の作成
# ==============================================================================
print(f"\n--- 目的変数 Y ({target_category} 購入フラグ) の作成 ---")

# 1. 基準日と予測期間を定義
max_date = orders['order_purchase_timestamp'].max()
prediction_period_start_date = max_date - datetime.timedelta(days=prediction_days - 1)
feature_period_end_date = prediction_period_start_date - datetime.timedelta(days=1)
print(f"基準日: {feature_period_end_date}, 予測期間: {prediction_period_start_date} から {max_date}")

# 2. 予測期間内の注文データを抽出
orders_prediction_period = orders[orders['order_purchase_timestamp'] >= prediction_period_start_date].copy()

# 3. 予測期間内に特定カテゴリを購入した注文を特定
prediction_order_ids = orders_prediction_period['order_id'].unique()
target_items_prediction_period = order_items_products[
    (order_items_products['order_id'].isin(prediction_order_ids)) &
    (order_items_products['product_category_name_english'] == target_category)
].copy()

# 4. 購入した顧客IDを取得
target_order_ids_in_period = target_items_prediction_period['order_id'].unique()
target_customers = orders_prediction_period[orders_prediction_period['order_id'].isin(target_order_ids_in_period)]['customer_id'].unique()
target_unique_customers_in_period = customer_id_map[customer_id_map['customer_id'].isin(target_customers)]['customer_unique_id'].unique()
print(f"予測期間内の '{target_category}' 購入顧客数 (unique ID): {len(target_unique_customers_in_period)} 人")

# 5. 目的変数Yデータフレームを作成
all_unique_customers = customers['customer_unique_id'].unique()
df_target = pd.DataFrame({'customer_unique_id': all_unique_customers})
df_target['Y_purchase_computers_accessories'] = df_target['customer_unique_id'].apply(
    lambda x: 1 if x in target_unique_customers_in_period else 0
)
print(f"目的変数Y作成完了 (全 {len(df_target)} 顧客, Y=1 割合: {df_target['Y_purchase_computers_accessories'].mean():.4f})")

# ==============================================================================
# 特徴量 X の作成
# ==============================================================================
print("\n--- 特徴量 X の作成 ---")
df_features = df_target.copy() # 目的変数DFをコピーして特徴量DFのベースとする

# --- 特徴量作成期間のデータを準備 ---
orders_feature_period = orders[orders['order_purchase_timestamp'] <= feature_period_end_date].copy()
order_items_feature = order_items_products[order_items_products['order_id'].isin(orders_feature_period['order_id'])].copy()
orders_feature_customer = pd.merge(orders_feature_period[['order_id', 'customer_id', 'order_purchase_timestamp']],
                                   customer_id_map, on='customer_id', how='left')

# --- 1. 顧客属性特徴量：居住地 (州) ---
customer_state = customers[['customer_unique_id', 'customer_state']].drop_duplicates(subset=['customer_unique_id'])
df_features = pd.merge(df_features, customer_state, on='customer_unique_id', how='left')

# --- 2. RFM指標の作成 ---
# R (Recency)
df_recency = orders_feature_customer.groupby('customer_unique_id')['order_purchase_timestamp'].max().reset_index()
df_recency.columns = ['customer_unique_id', 'last_purchase_date']
df_recency['Recency'] = (feature_period_end_date - df_recency['last_purchase_date']).dt.days
df_features = pd.merge(df_features, df_recency[['customer_unique_id', 'Recency']], on='customer_unique_id', how='left')
df_features['Recency'] = df_features['Recency'].fillna(9999) # 欠損値処理

# F (Frequency)
df_frequency = orders_feature_customer.groupby('customer_unique_id')['order_id'].nunique().reset_index()
df_frequency.columns = ['customer_unique_id', 'Frequency']
df_features = pd.merge(df_features, df_frequency, on='customer_unique_id', how='left')
df_features['Frequency'] = df_features['Frequency'].fillna(0) # 欠損値処理

# M (Monetary)
order_items_customer_feature = pd.merge(order_items_feature[['order_id', 'price']],
                                        orders_feature_customer[['order_id', 'customer_unique_id']],
                                        on='order_id', how='left')
df_monetary = order_items_customer_feature.groupby('customer_unique_id')['price'].sum().reset_index()
df_monetary.columns = ['customer_unique_id', 'Monetary']
df_features = pd.merge(df_features, df_monetary, on='customer_unique_id', how='left')
df_features['Monetary'] = df_features['Monetary'].fillna(0) # 欠損値処理

# --- 3. 特定カテゴリの過去購買行動 ---
target_items_feature = order_items_feature[order_items_feature['product_category_name_english'] == target_category].copy()
target_items_customer = pd.merge(target_items_feature[['order_id', 'price']],
                                 orders_feature_customer[['order_id', 'customer_unique_id', 'order_purchase_timestamp']],
                                 on='order_id', how='left')

df_target_features = target_items_customer.groupby('customer_unique_id').agg(
    target_purchase_count=('order_id', 'nunique'),
    target_purchase_amount=('price', 'sum'),
    target_last_purchase_date=('order_purchase_timestamp', 'max')
).reset_index()

df_target_features['target_recency'] = (feature_period_end_date - df_target_features['target_last_purchase_date']).dt.days

df_features = pd.merge(df_features,
                       df_target_features[['customer_unique_id', 'target_purchase_count', 'target_purchase_amount', 'target_recency']],
                       on='customer_unique_id', how='left')

df_features['has_purchased_target_before'] = (~df_features['target_purchase_count'].isnull()).astype(int)
df_features['target_purchase_count'] = df_features['target_purchase_count'].fillna(0)
df_features['target_purchase_amount'] = df_features['target_purchase_amount'].fillna(0)
df_features['target_recency'] = df_features['target_recency'].fillna(9999)

print("特徴量X作成完了。")
# print(f"最終的な特徴量データ shape: {df_features.shape}")
# print(df_features.info())

# ==============================================================================
# データ前処理
# ==============================================================================
print("\n--- データ前処理 ---")

# --- カテゴリ変数の処理 (One-Hot Encoding) ---
df_processed = pd.get_dummies(df_features, columns=['customer_state'], dummy_na=False)
# print(f"One-Hot Encoding 後の df shape: {df_processed.shape}")

# --- データの分割 (特徴量Xと目的変数y、学習データとテストデータ) ---
X = df_processed.drop(columns=['customer_unique_id', 'Y_purchase_computers_accessories'])
y = df_processed['Y_purchase_computers_accessories']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # テストデータの割合
    random_state=42,  # 再現性のため
    stratify=y        # 不均衡データのためクラス比率を維持
)

print(f"学習データ shape: {X_train.shape}, テストデータ shape: {X_test.shape}")
print(f"学習データ Y=1 割合: {y_train.mean():.4f}, テストデータ Y=1 割合: {y_test.mean():.4f}")
print("データ前処理と分割完了。")

# ==============================================================================
# モデル学習と評価
# ==============================================================================
print("\n--- モデル学習と評価 ---")

# --- LightGBM モデルの学習 ---
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"LightGBM scale_pos_weight: {scale_pos_weight:.4f}")

lgbm = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

lgbm.fit(X_train, y_train,
         eval_set=[(X_test, y_test)],
         eval_metric='auc',
         callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]) # 学習ログ抑制のため verbose=False

print("モデル学習完了。 Best iteration:", lgbm.best_iteration_)

# --- テストデータでの予測 ---
y_pred_proba = lgbm.predict_proba(X_test)[:, 1]
y_pred = lgbm.predict(X_test) # デフォルト閾値 0.5

# --- モデル評価結果の表示 ---
print("\n--- モデル評価結果 (Test Data) ---")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")

# --- 特徴量重要度の表示 ---
print("\n--- 特徴量重要度 (Top 20) ---")
feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': lgbm.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance_df.head(20))

print("\nモデル学習と評価が完了しました。")

# ==============================================================================
# --- スクリプト終了 ---
# ==============================================================================
# ==============================================================================
# === ↓↓↓ main.py の末尾にこのコードブロックを追加 ↓↓↓ ===
# ==============================================================================
print("\n*** 予測閾値の調整と評価 ***")

from sklearn.metrics import precision_recall_curve # metrics から追加インポート

# --- 前提: y_test と y_pred_proba が計算されていること ---
if 'y_test' in locals() and 'y_pred_proba' in locals():

    # --- 1. Precision-Recall カーブの描画 ---
    print("\n--- Precision-Recall Curve ---")
    # precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba) 
    # Note: thresholds_prの要素数はprecision/recallより1つ少ない
    
    precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    # thresholds_prに対応する部分のみプロットするためスライス
    plt.plot(recall_curve[:-1], precision_curve[:-1], marker='.', label='LightGBM') 
    plt.xlabel('Recall (再現率)')
    plt.ylabel('Precision (適合率)')
    plt.title('Precision-Recall Curve for computers_accessories Purchase')
    plt.grid(True)
    plt.legend()
    try:
        plt.savefig('precision_recall_curve.png') # 画像として保存
        print("Precision-Recall Curve を 'precision_recall_curve.png' として保存しました。")
    except Exception as e:
        print(f"グラフの保存中にエラーが発生しました: {e}")
    # plt.show() # スクリプト実行の場合、表示が止まることがあるため保存を推奨

    # --- 2. 特定の閾値での評価指標を確認 ---
    print("\n--- 閾値ごとの評価指標 ---")
    thresholds_to_check = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print("Threshold | Precision | Recall    | F1-Score")
    print("--------------------------------------------")

    # デフォルト(0.5)の結果も表示 (前のステップで計算済みだが比較のため)
    y_pred_default = (y_pred_proba >= 0.5).astype(int)
    precision_default = precision_score(y_test, y_pred_default, zero_division=0)
    recall_default = recall_score(y_test, y_pred_default, zero_division=0)
    f1_default = f1_score(y_test, y_pred_default, zero_division=0)
    print(f"  0.5     |  {precision_default:.4f}   |  {recall_default:.4f}   |  {f1_default:.4f}  <- Default")

    for threshold in thresholds_to_check:
        if threshold == 0.5: continue 

        y_pred_new_threshold = (y_pred_proba >= threshold).astype(int)
        
        precision_new = precision_score(y_test, y_pred_new_threshold, zero_division=0)
        recall_new = recall_score(y_test, y_pred_new_threshold, zero_division=0)
        f1_new = f1_score(y_test, y_pred_new_threshold, zero_division=0)
        
        print(f"  {threshold:.1f}     |  {precision_new:.4f}   |  {recall_new:.4f}   |  {f1_new:.4f}")

else:
    print("エラー: 閾値調整に必要なデータ (y_test, y_pred_proba) が準備できていません。")

# ==============================================================================
# --- ここまでが閾値調整 ---
# ==============================================================================