import os
import time
import yfinance as yf
import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import requests
import gc
import datetime
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D

# 🛡️ 텐서플로우 경고(잔소리) 및 과부하 완벽 차단 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

SERVER_IP = os.environ.get("MY_SERVER_IP") 
NUM_ENSEMBLE = 5  

def create_windows(X, y, window_size):
    X_win, y_win = [], []
    for i in range(window_size, len(X)):
        X_win.append(X[i-window_size:i, :])
        y_win.append(y[i, 0])
    return np.array(X_win), np.array(y_win)

def train_true_quant_bot():
    now_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    if now_kst.weekday() >= 5: 
        print(f"오늘은 주말 휴장일입니다. 푹 쉬고 월요일에 뵙겠습니다! 🏖️")
        return 

    final_output = {}
    print("🌟 [단타 스나이퍼 모드] AI Quant Prediction Bot 실행!")
    
    print("🌐 서버에서 VVIP 종목 리스트를 가져오는 중...")
    try:
        url = f"http://{SERVER_IP}:8080/api/tickers"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        raw_tickers = response.json()
        TICKERS = [f"{t}.KS" if t.isdigit() else t for t in raw_tickers]
        print(f"✅ 성공! 총 {len(TICKERS)}개의 종목 분석 시작.")
    except Exception as e:
        print("⚠️ 서버 접속 실패. 비상용 기본 종목으로 테스트합니다.")
        TICKERS = ["005930.KS", "000660.KS", "AAPL", "TSLA"]

    print("\n🌍 글로벌 시장 지수 다운로드 중...")
    try:
        market_df = yf.download(["^KS11", "^GSPC", "KRW=X", "^VIX", "^SOX"], period="3y", progress=False)['Close']
        if isinstance(market_df, pd.DataFrame):
            market_df = market_df.rename(columns={"^KS11": "KOSPI", "^GSPC": "SP500", "KRW=X": "EXCHANGE_RATE", "^VIX": "VIX", "^SOX": "SOX_SEMI"})
            market_df.index = pd.to_datetime(market_df.index).tz_localize(None) 
            market_returns = market_df.pct_change().fillna(0)
        else:
            market_returns = pd.DataFrame()
    except:
        market_returns = pd.DataFrame()

    for idx, t in enumerate(TICKERS, 1):
        try:
            print(f"\n[🚀 {idx}/{len(TICKERS)} - {t}] 데이터 수집 및 AI 학습 (단타 분류모델)...")
            
            symbol = t.replace('.KS', '')
            is_korean = symbol.isdigit()
            
            if is_korean:
                start_date = (datetime.datetime.now() - datetime.timedelta(days=3*365)).strftime('%Y-%m-%d')
                df = fdr.DataReader(symbol, start_date)
                time.sleep(0.5)
            else:
                df = yf.download(t, period="3y", progress=False) 
                time.sleep(1) 
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

            if df.empty or len(df) < 50: continue
            df.index = pd.to_datetime(df.index).tz_localize(None)

            # 💡 [핵심 변경] 단타용 타겟 라벨링 (시가 매수 -> 종가 매도)
            df['Next_Open'] = df['Open'].shift(-1)
            df['Next_Close'] = df['Close'].shift(-1)
            df['Intraday_Return'] = (df['Next_Close'] - df['Next_Open']) / df['Next_Open'] * 100
            
            # 💡 타겟 클래스: 오르면 1, 떨어지면 0 (완벽한 이분법)
            df['Target_Class'] = (df['Intraday_Return'] > 0).astype(int)

            # 기술적 지표 생성
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
            df['Momentum_5'] = df['Close'].pct_change(5) * 100
            
            delta = df['Close'].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ema_up = up.ewm(com=13, adjust=False).mean()
            ema_down = down.ewm(com=13, adjust=False).mean()
            rs = ema_up / ema_down
            df['RSI'] = 100 - (100 / (1 + rs))
            
            df['Overnight_Gap'] = (df['Open'] / df['Close'].shift(1) - 1) * 100
            df['Day_of_Week'] = df.index.dayofweek

            if not market_returns.empty:
                df = df.join(market_returns, how='left').ffill()
                feature_cols = ['RSI', 'Overnight_Gap', 'Day_of_Week', 'MACD', 'Momentum_5', 'KOSPI', 'SP500', 'VIX', 'SOX_SEMI']
            else:
                feature_cols = ['RSI', 'Overnight_Gap', 'Day_of_Week', 'MACD', 'Momentum_5']
                
            valid_features = [col for col in feature_cols if col in df.columns]
            # Next_Open이 NaN인 맨 마지막 줄(오늘)은 정답이 없으니 학습에서 제외
            df_valid = df.dropna(subset=valid_features + ['Intraday_Return', 'Target_Class'])
            
            if df_valid.empty: continue

            # 입력 피처(X)만 스케일링, 정답(y)은 0 또는 1이므로 그대로 둠
            X_raw = df_valid[valid_features].values
            y_raw = df_valid[['Target_Class']].values 

            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(X_raw)

            window_size = 15 
            if len(X_scaled) <= window_size: continue
                
            X_win, y_win = create_windows(X_scaled, y_raw, window_size)
            split = int(len(X_win) * 0.8)
            X_train, y_train = X_win[:split], y_win[:split]

            ensemble_preds = []
            for i in range(NUM_ENSEMBLE):
                inputs = Input(shape=(window_size, len(valid_features)))
                x = LSTM(64, return_sequences=True)(inputs)
                att_out = Attention()([x, x])
                x = GlobalAveragePooling1D()(att_out)
                x = Dense(32, activation='relu')(x)
                x = Dropout(0.2)(x)
                
                # 💡 [핵심 변경] 확률을 뱉는 Sigmoid 사용
                outputs = Dense(1, activation='sigmoid')(x)
                
                model = Model(inputs=inputs, outputs=outputs)
                # 💡 [핵심 변경] 회귀(MSE)가 아닌 이진 분류(binary_crossentropy) 사용
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss='binary_crossentropy', metrics=['accuracy'])
                model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
                
                # 오늘장 마감 후, 내일의 단타 수익 확률 예측!
                last_input = scaler_X.transform(df[valid_features].values[-window_size:]).reshape(1, window_size, len(valid_features))
                prob_up = model(last_input, training=False).numpy()[0, 0]
                ensemble_preds.append(prob_up)
                
                del model; gc.collect(); tf.keras.backend.clear_session()

            # 💡 [결과 가공] 앙상블 상승 확률 평균 (0.0 ~ 1.0)
            final_prob_up = np.mean(ensemble_preds)
            
            # 과거 평균 상승률/하락률 (현실적인 예상 수익률 제공용)
            avg_up_return = df_valid[df_valid['Target_Class'] == 1]['Intraday_Return'].mean()
            avg_down_return = df_valid[df_valid['Target_Class'] == 0]['Intraday_Return'].mean()
            if np.isnan(avg_up_return): avg_up_return = 1.0
            if np.isnan(avg_down_return): avg_down_return = -1.0

            # 상승 우세 vs 하락 우세 판단
            if final_prob_up >= 0.5:
                conf = final_prob_up * 100  # 상승 확률 자체가 확신도
                expected_return = avg_up_return * final_prob_up
                direction = "📈 상승"
            else:
                conf = (1 - final_prob_up) * 100  # 하락 확률 자체가 확신도
                expected_return = avg_down_return * (1 - final_prob_up)
                direction = "📉 하락"

            last_price = float(df['Close'].iloc[-1])
            pred_price = last_price * (1 + expected_return/100)
            
            pred_str = f"{int(pred_price):,}" if is_korean else f"${pred_price:.2f}"
            
            final_output[symbol] = {
                "pred": pred_str,
                "expected_return": f"{expected_return:.2f}%",
                "confidence": f"{conf:.1f}점"
            }
            print(f"   🎯 {direction} 예측! (확률/확신도: {conf:.1f}점, 가상 예측가: {pred_str})")

        except Exception as e:
            print(f"   ❌ {t} 오류: {e}")

    if SERVER_IP and final_output:
        SERVER_URL = f"http://{SERVER_IP}:8080/upload"
        try:
            response = requests.post(SERVER_URL, json=final_output, timeout=30)
            if response.status_code == 200:
                print(f"\n🚀 전송 완료! 단타 스나이퍼 데이터가 업데이트되었습니다.")
            else:
                print(f"\n⚠️ 서버 응답 에러 (상태코드: {response.status_code})")
        except Exception as e:
            print(f"\n❌ 서버 전송 실패: {e}")

if __name__ == "__main__":
    train_true_quant_bot()
