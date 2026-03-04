import os
import yfinance as yf
from pykrx import stock
import numpy as np
import pandas as pd
import requests
import gc
import datetime
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Conv1D, MaxPooling1D, Attention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# 🛡️ [보안 적용] 깃허브 금고(Secrets)에서 내 서버 IP를 몰래 꺼내옵니다.
try:
    SERVER_IP = os.environ["MY_SERVER_IP"]
except KeyError:
    print("⚠️ 오류: 깃허브 Secrets에 'MY_SERVER_IP'가 설정되지 않았습니다!")
    SERVER_IP = "127.0.0.1" # 임시 방어 코드

SERVER_URL = f"http://{SERVER_IP}:8080/upload"

TICKERS = [
    "005930", "000660", "035420", "005380", "035720",
    "000270", "068270", "005490", "105560", "012330",
    "012450", "064350", "103140"
]
NUM_ENSEMBLE = 5 

@tf.keras.utils.register_keras_serializable()
def directional_loss(y_true, y_pred):
    huber = tf.keras.losses.huber(y_true, y_pred)
    penalty = tf.where(y_true * y_pred < 0, tf.abs(y_true - y_pred) * 3.0, 0.0)
    return huber + tf.reduce_mean(penalty)

def create_windows(X, y, window_size):
    X_win, y_win = [], []
    for i in range(window_size, len(X)):
        X_win.append(X[i-window_size:i, :])
        y_win.append(y[i, 0])
    return np.array(X_win), np.array(y_win)

def train_beast_mode_bot():
    final_output = {}
    
    end_date = datetime.datetime.now().strftime("%Y%m%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=365*5)).strftime("%Y%m%d")

    print("🌍 거시 경제 및 반도체 지수 다운로드 중...")
    market_df = yf.download(["^KS11", "^GSPC", "KRW=X", "^VIX", "^SOX"], period="5y", progress=False)['Close']
    market_df = market_df.rename(columns={"^KS11": "KOSPI", "^GSPC": "SP500", "KRW=X": "EXCHANGE_RATE", "^VIX": "VIX", "^SOX": "SOX_SEMI"})
    market_returns = market_df.pct_change().fillna(0)

    for t in TICKERS:
        try:
            print(f"\n[🔥 {t}] 수급 데이터 결합 및 딥러닝 분석 시작...")
            
            ticker_yf = t + ".KS"
            df_price = yf.download(ticker_yf, period="5y", progress=False)
            if isinstance(df_price.columns, pd.MultiIndex):
                df_price.columns = df_price.columns.get_level_values(0)

            print(f"   📊 {t} 외국인/기관 수급 분석 중...")
            df_investor = stock.get_market_net_purchases_of_equities_by_ticker(start_date, end_date, t)
            df_investor = df_investor[['외국인', '기관합계', '개인']]
            df_investor.columns = ['Foreign_Net', 'Inst_Net', 'Retail_Net']
            
            df = df_price.join(df_investor, how='inner')

            df['Target_Return'] = df['Close'].pct_change().shift(-1) * 100 
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).ewm(alpha=1/14).mean() / (-df['Close'].diff().clip(upper=0)).ewm(alpha=1/14).mean())))
            
            df['Foreign_Rate'] = df['Foreign_Net'] / df['Volume']
            df['Inst_Rate'] = df['Inst_Net'] / df['Volume']
            
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            df['VWAP_5'] = (typical_price * df['Volume']).rolling(window=5).sum() / df['Volume'].rolling(window=5).sum()
            df['VWAP_5_Ratio'] = df['Close'] / df['VWAP_5']
            df['Overnight_Gap'] = (df['Open'] / df['Close'].shift(1) - 1) * 100
            df['Day_of_Week'] = df.index.dayofweek

            df = df.join(market_returns, how='left').ffill()
            
            feature_cols = ['RSI', 'VWAP_5_Ratio', 'Overnight_Gap', 'Day_of_Week', 
                            'Foreign_Rate', 'Inst_Rate', 'KOSPI', 'SP500', 'VIX', 'SOX_SEMI']

            df_valid = df.dropna(subset=feature_cols + ['Target_Return'])
            X_raw = df_valid[feature_cols].values
            y_raw = df_valid[['Target_Return']].values

            scaler_X = RobustScaler()
            X_scaled = scaler_X.fit_transform(X_raw)

            window_size = 30
            X_win, y_win = create_windows(X_scaled, y_raw, window_size)

            split = int(len(X_win) * 0.8)
            X_train, X_val = X_win[:split], X_win[split:]
            y_train, y_val = y_win[:split], y_win[split:]

            ensemble_preds = []
            ensemble_losses = []

            for i in range(NUM_ENSEMBLE):
                inputs = Input(shape=(window_size, len(feature_cols)))
                x = Conv1D(64, 3, activation='relu')(inputs)
                x = MaxPooling1D(2)(x)
                lstm_out = Bidirectional(LSTM(64, return_sequences=True))(x)
                att_out = Attention()([lstm_out, lstm_out])
                x = GlobalAveragePooling1D()(att_out)
                x = Dense(32, activation='swish')(x)
                outputs = Dense(1)(x)
                
                model = Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer='adam', loss=directional_loss)
                
                history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                                    epochs=100, batch_size=32, verbose=0,
                                    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])
                
                last_input = X_scaled[-window_size:].reshape(1, window_size, len(feature_cols))
                ensemble_preds.append(model.predict(last_input, verbose=0)[0, 0])
                ensemble_losses.append(min(history.history['val_loss']))
                del model; gc.collect(); tf.keras.backend.clear_session()

            weights = 1.0 / (np.array(ensemble_losses) + 1e-6)
            weights /= weights.sum()
            final_pred_return = np.sum(np.array(ensemble_preds) * weights)
            
            last_price = df['Close'].iloc[-1]
            pred_price = int(last_price * (1 + final_pred_return/100))
            std_dev = np.std(ensemble_preds)
            conf = max(0, 100 - (std_dev * 50))

            final_output[t] = {
                "pred": f"{pred_price:,}",
                "expected_return": f"{final_pred_return:.2f}%",
                "confidence": f"{conf:.1f}점"
            }
            print(f"   ✅ {t} 분석 완료: 예상 {final_pred_return:.2f}% | 확신도 {conf:.1f}점")

        except Exception as e:
            print(f"   ❌ {t} 오류 발생: {e}")

    try:
        response = requests.post(SERVER_URL, json=final_output, timeout=20)
        if response.status_code == 200:
            print("\n🚀 [풀옵션 봇 완료] 수급 데이터가 반영된 최종 분석값이 서버로 전송되었습니다.")
        else:
            print(f"\n⚠️ 서버 전송 실패. 상태 코드: {response.status_code}")
    except Exception as e:
        print(f"\n❌ 서버 전송 중 오류 발생: {e}")

if __name__ == "__main__":
    train_beast_mode_bot()
