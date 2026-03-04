import os
import time
import yfinance as yf
import numpy as np
import pandas as pd
import requests
import gc
import datetime
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D

# 🛡️ [보안] 깃허브 금고(Secrets)에서 서버 IP 로드 (설정 안 되어 있으면 기본값 사용)
# 깃허브 Actions의 env 섹션에서 설정한 이름과 동일해야 합니다.
SERVER_IP = os.environ.get("MY_SERVER_IP") 

# 📊 KOSPI 시가총액 상위 50종목 (최신 순서 반영)
TICKERS = [
    "005930.KS", "000660.KS", "373220.KS", "207940.KS", "005380.KS", 
    "000270.KS", "068270.KS", "005490.KS", "035420.KS", "051910.KS", 
    "028260.KS", "006400.KS", "105560.KS", "055550.KS", "035720.KS", 
    "066570.KS", "012330.KS", "086790.KS", "015760.KS", "032830.KS", 
    "323410.KS", "033780.KS", "003670.KS", "011200.KS", "316140.KS", 
    "034730.KS", "010130.KS", "018260.KS", "096770.KS", "009150.KS", 
    "017670.KS", "010950.KS", "352820.KS", "030200.KS", "003550.KS", 
    "051900.KS", "036570.KS", "000810.KS", "090430.KS", "086280.KS", 
    "024110.KS", "011170.KS", "004020.KS", "259960.KS", "329180.KS", 
    "005830.KS", "022100.KS", "010140.KS", "138040.KS", "042660.KS"
]
NUM_ENSEMBLE = 5  # 정확도를 높이기 위한 앙상블 모델 수

def create_windows(X, y, window_size):
    X_win, y_win = [], []
    for i in range(window_size, len(X)):
        X_win.append(X[i-window_size:i, :])
        y_win.append(y[i, 0])
    return np.array(X_win), np.array(y_win)

def train_true_quant_bot():
    final_output = {}
    
    print(f"📡 접속 대상 서버 IP: {SERVER_IP if SERVER_IP else '설정 안됨 (로컬 테스트 모드)'}")
    
    print("🌍 글로벌 시장 지수 다운로드 중...")
    market_df = yf.download(["^KS11", "^GSPC", "KRW=X", "^VIX", "^SOX"], period="3y", progress=False)['Close']
    market_df = market_df.rename(columns={"^KS11": "KOSPI", "^GSPC": "SP500", "KRW=X": "EXCHANGE_RATE", "^VIX": "VIX", "^SOX": "SOX_SEMI"})
    market_df.index = market_df.index.tz_localize(None) 
    market_returns = market_df.pct_change().fillna(0)

    for idx, t in enumerate(TICKERS, 1):
        try:
            print(f"\n[🚀 {idx}/{len(TICKERS)} - {t}] 분석 시작...")
            
            df = yf.download(t, period="3y", progress=False) 
            time.sleep(1) # IP 차단 방지

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = df.index.tz_localize(None)

            if len(df) < 50: continue

            # 지표 계산
            df['Target_Return'] = df['Close'].pct_change().shift(-1) * 100 
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
            df['Momentum_5'] = df['Close'].pct_change(5) * 100
            df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).ewm(alpha=1/14).mean() / (-df['Close'].diff().clip(upper=0)).ewm(alpha=1/14).mean())))
            df['Overnight_Gap'] = (df['Open'] / df['Close'].shift(1) - 1) * 100
            df['Day_of_Week'] = df.index.dayofweek

            df = df.join(market_returns, how='left').ffill()
            
            feature_cols = ['RSI', 'Overnight_Gap', 'Day_of_Week', 'MACD', 'Momentum_5',
                            'KOSPI', 'SP500', 'VIX', 'SOX_SEMI']

            df_valid = df.dropna(subset=feature_cols + ['Target_Return'])
            X_raw = df_valid[feature_cols].values
            y_raw = df_valid[['Target_Return']].values

            scaler_X = StandardScaler(); X_scaled = scaler_X.fit_transform(X_raw)
            scaler_y = StandardScaler(); y_scaled = scaler_y.fit_transform(y_raw)

            window_size = 15 
            X_win, y_win = create_windows(X_scaled, y_scaled, window_size)

            split = int(len(X_win) * 0.8)
            X_train, y_train = X_win[:split], y_win[:split]

            ensemble_preds = []
            for i in range(NUM_ENSEMBLE):
                inputs = Input(shape=(window_size, len(feature_cols)))
                x = LSTM(64, return_sequences=True)(inputs)
                att_out = Attention()([x, x])
                x = GlobalAveragePooling1D()(att_out)
                x = Dense(32, activation='relu')(x)
                x = Dropout(0.2)(x)
                outputs = Dense(1, activation='linear')(x)
                
                model = Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss='mse')
                model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
                
                last_input = X_scaled[-window_size:].reshape(1, window_size, len(feature_cols))
                pred_scaled = model.predict(last_input, verbose=0)
                ensemble_preds.append(scaler_y.inverse_transform(pred_scaled)[0, 0])
                del model; gc.collect(); tf.keras.backend.clear_session()

            final_pred_return = np.mean(ensemble_preds)
            last_price = float(df['Close'].iloc[-1])
            pred_price = int(last_price * (1 + final_pred_return/100))
            std_dev = np.std(ensemble_preds)
            conf = max(50.0, min(99.0, 95.0 - (std_dev * 10.0)))
            
            symbol = t.split('.')[0]
            final_output[symbol] = {
                "pred": f"{pred_price:,}",
                "expected_return": f"{final_pred_return:.2f}%",
                "confidence": f"{conf:.1f}점"
            }
            print(f"   ✅ {symbol} 분석완료")

        except Exception as e:
            print(f"   ❌ {t} 오류: {e}")

    # 📤 최종 전송 단계
    if SERVER_IP:
        SERVER_URL = f"http://{SERVER_IP}:8080/upload"
        try:
            response = requests.post(SERVER_URL, json=final_output, timeout=30)
            if response.status_code == 200:
                print(f"\n🚀 전송 완료! 서버({SERVER_IP})에 데이터가 업데이트되었습니다.")
            else:
                print(f"\n⚠️ 서버 응답 에러: {response.status_code}")
        except Exception as e:
            print(f"\n❌ 서버 전송 실패: {e}")
    else:
        print("\n📢 MY_SERVER_IP 환경변수가 없어 데이터를 전송하지 않았습니다. (로컬 테스트)")

if __name__ == "__main__":
    train_true_quant_bot()
