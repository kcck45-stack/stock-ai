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
    # ==========================================
    # 💡 [추가됨] 주말 휴장일 자동 감지 및 퇴근 로직
    # ==========================================
    now_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    if now_kst.weekday() >= 5:  # 5는 토요일, 6은 일요일
        print(f"오늘은 주말 휴장일입니다. 푹 쉬고 월요일에 뵙겠습니다! 🏖️")
        return  # 분석을 돌리지 않고 여기서 즉시 봇을 종료합니다.

    final_output = {}
    print("🌟 AI Quant Prediction Bot 실행을 시작합니다!")
    print(f"📡 접속 대상 서버 IP: {SERVER_IP if SERVER_IP else '설정 안됨'}")
    
    # ==========================================
    # 💡 1. 폰 서버에서 '오늘 분석할 종목 리스트' 동기화
    # ==========================================
    print("🌐 서버에서 VVIP 종목 리스트를 가져오는 중...")
    try:
        url = f"http://{SERVER_IP}:8080/api/tickers"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        raw_tickers = response.json()
        
        # 야후 파이낸스 형식(.KS)으로 자동 변환
        TICKERS = [f"{t}.KS" if t.isdigit() else t for t in raw_tickers]
        
        print(f"✅ 성공! 총 {len(TICKERS)}개의 종목 분석을 시작합니다.")
        print(f"📋 타겟 종목: {TICKERS}")
    except Exception as e:
        print(f"❌ 서버 접속 실패: {e}")
        print("⚠️ 서버가 꺼져있을 수 있습니다. 비상용 기본 종목으로 테스트합니다.")
        TICKERS = ["005930.KS", "000660.KS", "AAPL", "TSLA"]

    # ==========================================
    # 💡 2. 글로벌 시장 지수 다운로드 (회원님 모델 핵심)
    # ==========================================
    print("\n🌍 글로벌 시장 지수 다운로드 중...")
    try:
        market_df = yf.download(["^KS11", "^GSPC", "KRW=X", "^VIX", "^SOX"], period="3y", progress=False)['Close']
        if isinstance(market_df, pd.DataFrame):
            market_df = market_df.rename(columns={"^KS11": "KOSPI", "^GSPC": "SP500", "KRW=X": "EXCHANGE_RATE", "^VIX": "VIX", "^SOX": "SOX_SEMI"})
            market_df.index = market_df.index.tz_localize(None) 
            market_returns = market_df.pct_change().fillna(0)
        else:
            market_returns = pd.DataFrame() # 에러 방지
    except:
        market_returns = pd.DataFrame()
        print("⚠️ 시장 지수 다운로드 실패. 지수 데이터 없이 진행합니다.")

    # ==========================================
    # 💡 3. AI 딥러닝 앙상블 분석 시작 (회원님 로직)
    # ==========================================
    for idx, t in enumerate(TICKERS, 1):
        try:
            print(f"\n[🚀 {idx}/{len(TICKERS)} - {t}] 데이터 수집 및 AI 학습 시작...")
            df = yf.download(t, period="3y", progress=False) 
            time.sleep(1) 

            if df.empty or len(df) < 50: 
                print(f"   ⚠️ {t} 데이터 부족으로 패스합니다.")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = df.index.tz_localize(None)

            # 기술적 지표 생성
            df['Target_Return'] = df['Close'].pct_change().shift(-1) * 100 
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
            df['Momentum_5'] = df['Close'].pct_change(5) * 100
            
            # RSI 계산 (예외 처리 추가)
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
                
            # 유효한 피처가 실제로 DataFrame에 있는지 확인
            valid_features = [col for col in feature_cols if col in df.columns]
            df_valid = df.dropna(subset=valid_features + ['Target_Return'])
            
            if df_valid.empty:
                print(f"   ⚠️ {t} 전처리 후 데이터가 없어 패스합니다.")
                continue

            X_raw = df_valid[valid_features].values
            y_raw = df_valid[['Target_Return']].values

            scaler_X = StandardScaler(); X_scaled = scaler_X.fit_transform(X_raw)
            scaler_y = StandardScaler(); y_scaled = scaler_y.fit_transform(y_raw)

            window_size = 15 
            if len(X_scaled) <= window_size:
                continue
                
            X_win, y_win = create_windows(X_scaled, y_scaled, window_size)
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
                outputs = Dense(1, activation='linear')(x)
                
                model = Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss='mse')
                model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
                
                last_input = X_scaled[-window_size:].reshape(1, window_size, len(valid_features))
                
                # 메모리 누수 방지 로직 (회원님 코드 유지)
                pred_scaled = model(last_input, training=False).numpy()
                ensemble_preds.append(scaler_y.inverse_transform(pred_scaled)[0, 0])
                del model; gc.collect(); tf.keras.backend.clear_session()

            final_pred_return = np.mean(ensemble_preds)
            last_price = float(df['Close'].iloc[-1])
            pred_price = last_price * (1 + final_pred_return/100)
            
            std_dev = np.std(ensemble_preds)
            conf = max(50.0, min(99.0, 95.0 - (std_dev * 10.0)))
            
            # 티커에서 .KS 제거 (한국/미국 주식 포맷팅)
            symbol = t.replace('.KS', '')
            
            # 미국 주식($ 소수점) vs 한국 주식(콤마 정수) 포맷 분기
            is_korean = symbol.isdigit()
            pred_str = f"{int(pred_price):,}" if is_korean else f"${pred_price:.2f}"
            
            final_output[symbol] = {
                "pred": pred_str,
                "expected_return": f"{final_pred_return:.2f}%",
                "confidence": f"{conf:.1f}점"
            }
            print(f"   ✅ {symbol} 분석완료 (예측가: {pred_str}, 확신도: {conf:.1f}점)")

        except Exception as e:
            print(f"   ❌ {t} 오류: {e}")

    # ==========================================
    # 💡 4. 폰 서버로 최종 결과물 발사!
    # ==========================================
    if SERVER_IP and final_output:
      SERVER_URL = f"http://{SERVER_IP}:8080/upload/upload?v=v1"
        try:
            response = requests.post(SERVER_URL, json=final_output, timeout=30)
            if response.status_code == 200:
                print(f"\n🚀 전송 완료! 서버({SERVER_IP})에 데이터가 완벽하게 업데이트되었습니다.")
            else:
                print(f"\n⚠️ 서버 응답 에러 (상태코드: {response.status_code})")
        except Exception as e:
            print(f"\n❌ 서버 전송 실패: {e}")
    else:
        print("\n📢 서버 IP가 없거나 분석된 데이터가 없어 전송하지 않았습니다.")

if __name__ == "__main__":
  train_true_quant_bot()
