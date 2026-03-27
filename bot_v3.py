import os, time, datetime, gc, requests
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

SERVER_IP = os.environ.get("MY_SERVER_IP")

def create_windows(X, y, window_size):
    X_win, y_win = [], []
    for i in range(window_size, len(X)):
        X_win.append(X[i-window_size:i, :])
        y_win.append(y[i, 0])
    return np.array(X_win), np.array(y_win)

def run_v3_master_bot():
    now_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    if now_kst.weekday() >= 5:
        print("휴장일이므로 V3 로봇은 가동을 멈춥니다.")
        return

    final_output = {}
    print("🌌 [V3 마스터 엔진] 거시경제 & 스마트머니 추적 시작!")
    
    try:
        url = f"http://{SERVER_IP}:8080/api/tickers"
        TICKERS = [f"{t}.KS" if t.isdigit() else t for t in requests.get(url, timeout=10).json()]
    except:
        TICKERS = ["005930.KS", "000660.KS", "AAPL", "TSLA"]

    # 1. 시장 트렌드 필터 (코스피 / 나스닥(S&P) 20일선 추세 확인)
    market_trend = {'KR': True, 'US': True}
    try:
        m_df = yf.download(["^KS11", "^GSPC"], period="3mo", progress=False)['Close']
        if not m_df.empty:
            ks11_20ma = m_df['^KS11'].rolling(20).mean().iloc[-1]
            gspc_20ma = m_df['^GSPC'].rolling(20).mean().iloc[-1]
            if m_df['^KS11'].iloc[-1] < ks11_20ma: market_trend['KR'] = False
            if m_df['^GSPC'].iloc[-1] < gspc_20ma: market_trend['US'] = False
    except: pass

    for t in TICKERS:
        try:
            df = yf.download(t, period="2y", progress=False)
            if df.empty or len(df) < 60: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.index = df.index.tz_localize(None)

            # 2. 스마트머니 (세력 거래량 폭발) 필터
            df['Vol_60MA'] = df['Volume'].rolling(60).mean()
            df['Vol_Spike'] = (df['Volume'] > df['Vol_60MA'] * 2.0).astype(int)
            recent_spike = df['Vol_Spike'].tail(15).sum() > 0 # 최근 15일 내에 거래량 폭발 이력이 있는가?
            
            if not recent_spike:
                continue # 시장의 소외주는 아예 분석하지 않음 (시간 절약 & 승률 상승)

            # 지표 계산
            df['Target'] = df['Close'].pct_change().shift(-1) * 100
            df['RSI'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).ewm(com=13).mean() / (df['Close'].diff().clip(upper=0).abs().ewm(com=13).mean() + 1e-9)))
            df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
            df['Dist_From_High'] = (df['Close'] / df['Close'].rolling(10).max() - 1) * 100 # 고점 대비 눌림목 깊이

            features = ['RSI', 'MACD', 'Dist_From_High', 'Vol_Spike']
            df_valid = df.dropna(subset=features + ['Target'])
            if len(df_valid) < 50: continue

            # AI 학습
            scaler_X = StandardScaler(); scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(df_valid[features].values)
            y_scaled = scaler_y.fit_transform(df_valid[['Target']].values)

            window = 10
            X_win, y_win = create_windows(X_scaled, y_scaled, window)
            
            inp = Input(shape=(window, len(features)))
            x = LSTM(32, return_sequences=False)(inp)
            x = Dropout(0.2)(x)
            out = Dense(1, activation='linear')(x)
            model = Model(inputs=inp, outputs=out)
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_win, y_win, epochs=15, batch_size=16, verbose=0)

            last_X = X_scaled[-window:].reshape(1, window, len(features))
            pred_scaled = model.predict(last_X, verbose=0)
            pred_return = scaler_y.inverse_transform(pred_scaled)[0,0]

            # 3. 시장 필터 적용: 하락장일 경우 확신도를 대폭 깎아서 매매 통제
            is_kr = t.isdigit() or '.KS' in t
            trend_ok = market_trend['KR'] if is_kr else market_trend['US']
            
            # 눌림목이 너무 깊거나(-8% 초과) 너무 안 눌렸으면(-3% 미만) 확신도 깎기
            curr_dist = df['Dist_From_High'].iloc[-1]
            is_good_pullback = -8.0 <= curr_dist <= -3.0

            base_conf = 75.0 if (pred_return > 0 and is_good_pullback) else 40.0
            if not trend_ok: base_conf -= 30.0 # 📌 하락장엔 무조건 탈락 (MDD 완벽 방어)
            
            conf = max(10.0, min(99.0, base_conf + (pred_return * 2)))
            
            symbol = t.replace('.KS', '')
            pred_price = float(df['Close'].iloc[-1]) * (1 + pred_return/100)
            p_str = f"{int(pred_price):,}" if symbol.isdigit() else f"${pred_price:.2f}"
            
            final_output[symbol] = {
                "pred": p_str,
                "expected_return": f"{pred_return:.2f}%",
                "confidence": f"{conf:.1f}점"
            }
            print(f"✅ {symbol} V3 분석완료 (추세: {'정상' if trend_ok else '하락장'}, 예측가: {p_str})")
            
            del model; tf.keras.backend.clear_session(); gc.collect()
            
        except Exception as e:
            print(f"❌ {t} 분석 오류: {e}")

    # 4. 서버(V3 방)로 발사
    if SERVER_IP and final_output:
        SERVER_URL = f"http://{SERVER_IP}:8080/upload?v=v3"
        try:
            res = requests.post(SERVER_URL, json=final_output, timeout=30)
            print("🚀 V3 마스터 엔진 서버 전송 완료!" if res.status_code == 200 else "⚠️ 전송 에러")
        except: print("❌ 전송 실패")

if __name__ == "__main__":
    run_v3_master_bot()

