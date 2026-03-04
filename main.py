import yfinance as yf
import numpy as np
import pandas as pd
import requests
import gc
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# [설정] 서버 IP 및 URL
SERVER_IP = "34.30.112.251"
SERVER_URL = f"http://{SERVER_IP}:8080/upload"

TICKERS = [
    "005930.KS", "000660.KS", "035420.KS", "005380.KS", "035720.KS",
    "000270.KS", "068270.KS", "005490.KS", "105560.KS", "012330.KS",
    "012450.KS", "064350.KS", "103140.KS"
]
NUM_ENSEMBLE = 5 

def create_windows(X, y, window_size):
    X_win, y_win = [], []
    for i in range(window_size, len(X)):
        X_win.append(X[i-window_size:i, :])
        y_win.append(y[i, 0])
    return np.array(X_win), np.array(y_win)

def train_return_based_extreme_model():
    final_output = {}

    # 🚀 [업그레이드 1] 거시 동향에 '원/달러 환율(KRW=X)' 추가
    print("🌍 시장 동향(KOSPI, S&P500, 환율) 데이터 다운로드 중...")
    market_df = yf.download(["^KS11", "^GSPC", "KRW=X"], period="10y", progress=False)['Close']
    market_df = market_df.rename(columns={"^KS11": "KOSPI", "^GSPC": "SP500", "KRW=X": "EXCHANGE_RATE"})
    market_returns = market_df.pct_change().fillna(0)

    for t in TICKERS:
        try:
            print(f"\n[🚀 {t}] 심층 분석 시작 (앙상블 {NUM_ENSEMBLE}회)...")
            df = yf.download(t, period="10y", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 스케일러 폭발 방지를 위해 수익률을 퍼센트(%) 단위로 변환
            df['Return'] = df['Close'].pct_change() * 100 

            # 기본 지표 계산
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_60'] = df['Close'].rolling(window=60).mean()
            df['EMA_120'] = df['Close'].ewm(span=120, adjust=False).mean()
            
            std20 = df['Close'].rolling(window=20).std()
            df['BBM'] = df['SMA_20']
            df['BBU'] = df['BBM'] + 2 * std20
            df['BBL'] = df['BBM'] - 2 * std20
            
            delta = df['Close'].diff()
            gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            
            L14 = df['Low'].rolling(window=14).min()
            H14 = df['High'].rolling(window=14).max()
            df['STOCH_k'] = (df['Close'] - L14) / (H14 - L14) * 100
            
            tr1 = df['High'] - df['Low']
            tr2 = abs(df['High'] - df['Close'].shift())
            tr3 = abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(window=14).mean()

            # 🚀 [업그레이드 2] 세력의 매집/이탈을 파악하는 OBV 지표 추가
            df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

            # 절대 가격을 비율로 변환
            df['SMA_20_Ratio'] = df['Close'] / df['SMA_20']
            df['SMA_60_Ratio'] = df['Close'] / df['SMA_60']
            df['EMA_120_Ratio'] = df['Close'] / df['EMA_120']
            df['BB_Upper_Ratio'] = df['BBU'] / df['Close']
            df['BB_Lower_Ratio'] = df['BBL'] / df['Close']

            # 거시 데이터 병합 및 결측치 제거
            df = df.join(market_returns, how='left').ffill().bfill()
            df_final = df.dropna().copy()
            target_col = 'Return'
            
            # 모델 혼란 방지를 위해 절대 가격/수치 컬럼 제외
            absolute_price_cols = ['Open', 'High', 'Low', 'Adj 지수', 'Adj Close', 'Close', 'SMA_20', 'SMA_60', 'EMA_120', 'BBL', 'BBM', 'BBU', target_col]
            feature_cols = [col for col in df_final.columns if col not in absolute_price_cols]
            
            X_data = df_final[feature_cols].values
            y_data = df_final[[target_col]].values

            split_idx = int(len(df_final) * 0.8)
            X_train_raw, X_val_raw = X_data[:split_idx], X_data[split_idx:]
            
            # 타겟(y) 데이터는 오염 방지를 위해 절대 스케일링하지 않음!
            y_train = y_data[:split_idx]
            y_val = y_data[split_idx:]

            scaler_X = RobustScaler()
            X_train_scaled = scaler_X.fit_transform(X_train_raw)
            X_val_scaled = scaler_X.transform(X_val_raw)

            window_size = 60
            X_train, y_train_win = create_windows(X_train_scaled, y_train, window_size)
            X_val, y_val_win = create_windows(X_val_scaled, y_val, window_size)

            X_all_scaled = scaler_X.transform(X_data)
            last_window = X_all_scaled[-window_size:].reshape(1, window_size, len(feature_cols))
            last_close_price = df_final['Close'].iloc[-1]

            ensemble_returns = []
            
            for i in range(NUM_ENSEMBLE):
                print(f"   -> 🧠 CNN-LSTM 하이브리드 모델 {i+1}/{NUM_ENSEMBLE} 정밀 학습 중...")
                
                # 🚀 [업그레이드 3] 노이즈를 깎아내는 CNN 층을 LSTM 앞에 장착
                model = Sequential([
                    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, len(feature_cols))),
                    MaxPooling1D(pool_size=2),
                    BatchNormalization(),
                    
                    Bidirectional(LSTM(128, return_sequences=True)),
                    BatchNormalization(),
                    Dropout(0.3),
                    Bidirectional(LSTM(64)),
                    BatchNormalization(),
                    Dropout(0.3),
                    
                    Dense(64, activation='swish', kernel_regularizer=l2(0.001)), 
                    Dense(1)
                ])
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='huber')

                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
                ]

                model.fit(X_train, y_train_win, validation_data=(X_val, y_val_win), epochs=200, batch_size=32, verbose=0, callbacks=callbacks)
                
                # 예측값은 스케일링 복원 없이 바로 순수 퍼센트(%) 값으로 나옴
                pred_real = model.predict(last_window, verbose=0)[0, 0]
                ensemble_returns.append(pred_real)
                
                del model; gc.collect(); tf.keras.backend.clear_session()

            avg_pred_return = np.mean(ensemble_returns)
            
            # 100으로 나눈 퍼센트를 적용하여 현실적인 최종 가격 계산
            final_price = int(last_close_price * (1 + (avg_pred_return / 100)))

            symbol = t.split('.')[0]
            final_output[symbol] = {"pred": f"{final_price:,}"}
            print(f" ✅ {symbol} 완료: 평균 예상 수익률 {avg_pred_return:.2f}% -> 예측가 {final_price:,}원")

        except Exception as e:
            print(f" ❌ {t} 오류: {e}")

    try:
        response = requests.post(SERVER_URL, json=final_output, timeout=20)
        if response.status_code == 200:
            print("\n🚀 [분석 완료] 고도화된 AI 예측값이 서버로 성공적으로 전송되었습니다.")
        else:
            print(f"\n⚠️ 서버 전송 실패. 상태 코드: {response.status_code}")
    except Exception as e:
        print(f"\n❌ 서버 전송 중 오류 발생: {e}")

if __name__ == "__main__":
    train_return_based_extreme_model()
