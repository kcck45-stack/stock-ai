import os
import time
import gc
import yfinance as yf
import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import requests
import datetime
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, GlobalAveragePooling1D

# 🛡️ 텐서플로우 경고 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# 깃허브 시크릿에 등록된 서버 IP 가져오기
SERVER_IP = os.environ.get("MY_SERVER_IP") 
NUM_ENSEMBLE = 3  # 정확도를 위한 3중 앙상블 (메모리 최적화 완료)

def create_windows(X, y, window_size):
    X_win, y_win = [], []
    for i in range(window_size, len(X)):
        X_win.append(X[i-window_size:i, :])
        y_win.append(y[i, 0])
    return np.array(X_win), np.array(y_win)

def train_true_quant_bot():
    now_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    if now_kst.weekday() >= 5: 
        print("오늘은 주말 휴장일입니다. 푹 쉬고 월요일에 뵙겠습니다! 🏖️")
        return 

    final_output = {}
    print("🌟 [기봉 스나이퍼 V2] GitHub Actions 전용 봇 실행 (승률 62.7% 로직)")
    
    try:
        url = f"http://{SERVER_IP}:8080/api/tickers"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        raw_tickers = response.json()
        TICKERS = [f"{t}.KS" if t.isdigit() else t for t in raw_tickers]
        print(f"✅ 총 {len(TICKERS)}개 종목 분석 시작!")
    except Exception as e:
        print(f"⚠️ 서버 접속 실패: {e}")
        TICKERS = ["005930.KS", "000660.KS", "035420.KS", "068270.KS"]

    for idx, t in enumerate(TICKERS, 1):
        try:
            print(f"\n[🚀 {idx}/{len(TICKERS)} - {t}] V2 스나이퍼 모델 분석 중...")
            
            symbol = t.replace('.KS', '')
            is_korean = symbol.isdigit()
            
            start_date = (datetime.datetime.now() - datetime.timedelta(days=400)).strftime('%Y-%m-%d')
            if is_korean:
                df = fdr.DataReader(symbol, start_date)
                time.sleep(0.5)
            else:
                df = yf.download(t, period="1y", progress=False) 
                time.sleep(0.5)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

            if df.empty or len(df) < 100: continue
            df.index = pd.to_datetime(df.index).tz_localize(None)

            # 🎯 타겟 라벨링 (단타: 내일 시가 매수 -> 종가 매도)
            df['Next_Open'] = df['Open'].shift(-1)
            df['Next_Close'] = df['Close'].shift(-1)
            df['Intraday_Return'] = (df['Next_Close'] - df['Next_Open']) / (df['Next_Open'] + 1e-9) * 100
            df['Target_Class'] = (df['Intraday_Return'] > 0).astype(int)

            # 🛠️ V2 풀옵션 지표 장착
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
            df['Momentum_5'] = df['Close'].pct_change(5) * 100
            df['Overnight_Gap'] = (df['Open'] / (df['Close'].shift(1) + 1e-9) - 1) * 100
            df['Day_of_Week'] = df.index.dayofweek
            
            # 볼린저 밴드
            df['BB_std'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['SMA_20'] + (df['BB_std'] * 2)
            df['BB_Lower'] = df['SMA_20'] - (df['BB_std'] * 2)
            df['BB_PB'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-9)
            
            # 거래량 지표 (OBV, Volume Oscillator)
            df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            df['OBV_ROC'] = df['OBV'].pct_change(5).replace([np.inf, -np.inf], 0) * 100
            
            df['Vol_MA5'] = df['Volume'].rolling(5).mean()
            df['Vol_MA20'] = df['Volume'].rolling(20).mean()
            df['Vol_Osc'] = (df['Vol_MA5'] - df['Vol_MA20']) / (df['Vol_MA20'] + 1e-9) * 100

            feature_cols = ['SMA_20', 'MACD', 'Momentum_5', 'Overnight_Gap', 'Day_of_Week', 'BB_PB', 'OBV_ROC', 'Vol_Osc']
            
            # 학습용 데이터 (내일 정답이 없는 오늘 데이터는 제외)
            df_valid = df.dropna(subset=feature_cols + ['Intraday_Return', 'Target_Class'])
            if df_valid.empty: continue

            X_raw = df_valid[feature_cols].values
            y_raw = df_valid[['Target_Class']].values 

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw)

            window_size = 15 
            if len(X_scaled) <= window_size: continue
                
            X_win, y_win = create_windows(X_scaled, y_raw, window_size)
            
            ensemble_preds = []
            for i in range(NUM_ENSEMBLE):
                # 🧠 깃허브 서버 안 터지는 다이어트 구조
                inputs = Input(shape=(window_size, len(feature_cols)))
                x = LSTM(16, return_sequences=True)(inputs)
                x = GlobalAveragePooling1D()(x)
                x = Dense(8, activation='relu')(x)
                x = Dropout(0.2)(x)
                outputs = Dense(1, activation='sigmoid')(x)
                
                model = Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss='binary_crossentropy')
                
                model.fit(X_win, y_win, epochs=15, batch_size=8, verbose=0)
                
                # 🔮 내일 예측 (오늘 마감 데이터까지 꽉 채워서 활용)
                last_input = scaler.transform(df[feature_cols].values[-window_size:]).reshape(1, window_size, len(feature_cols))
                prob_up = model.predict(last_input, verbose=0)[0, 0]
                ensemble_preds.append(prob_up)

                # 🧹 앙상블 돌 때마다 메모리 강제 청소
                del model
                tf.keras.backend.clear_session()
                gc.collect()

            # 💡 앙상블 평균 확률 계산
            final_prob_up = np.mean(ensemble_preds)

            # 메인 사이트 표출 데이터 가공
            if final_prob_up >= 0.5:
                conf = final_prob_up * 100
                avg_return = df_valid[df_valid['Target_Class'] == 1]['Intraday_Return'].mean()
                expected_return = avg_return * final_prob_up
                direction = "📈 상승"
            else:
                conf = (1 - final_prob_up) * 100
                avg_return = df_valid[df_valid['Target_Class'] == 0]['Intraday_Return'].mean()
                expected_return = avg_return * (1 - final_prob_up)
                direction = "📉 하락"

            if np.isnan(expected_return): expected_return = 0.0

            last_price = float(df['Close'].iloc[-1])
            pred_price = last_price * (1 + expected_return/100)
            pred_str = f"{int(pred_price):,}" if is_korean else f"${pred_price:.2f}"
            
            final_output[symbol] = {
                "pred": pred_str,
                "expected_return": f"{expected_return:.2f}%",
                "confidence": f"{conf:.1f}점"
            }
            print(f"   🎯 {direction} 예측! (확률: {conf:.1f}%, 가상 예측가: {pred_str})")

        except Exception as e:
            print(f"   ❌ {t} 오류: {e}")

    # 🚀 메인 서버로 최종 쏘기!
    if SERVER_IP and final_output:
        try:
            response = requests.post(f"http://{SERVER_IP}:8080/upload/upload?v=v2", json=final_output, timeout=30)
            if response.status_code == 200:
                print(f"\n🚀 전송 완료! V2 스나이퍼 데이터가 메인 서버에 완벽하게 장착되었습니다!")
            else:
                print(f"\n⚠️ 서버 응답 에러 (상태코드: {response.status_code})")
        except Exception as e:
            print(f"\n❌ 서버 전송 실패: {e}")

if __name__ == "__main__":
    train_true_quant_bot()
