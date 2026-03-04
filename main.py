import os
import time
import yfinance as yf
import numpy as np
import pandas as pd
import requests
import gc
import datetime
import tensorflow as tf
import FinanceDataReader as fdr
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D

# 🛡️ [보안] 깃허브 금고(Secrets)에서 서버 IP 로드
SERVER_IP = os.environ.get("MY_SERVER_IP", "34.30.112.251") 
SERVER_URL = f"http://{SERVER_IP}:8080/upload"

NUM_ENSEMBLE = 5 

def get_kospi_top_50():
    """KOSPI 시가총액 상위 50종목을 실시간으로 가져옵니다."""
    print("📊 실시간 KOSPI 시가총액 상위 50종목 데이터를 불러옵니다...")
    # KOSPI 상장 종목 전체 가져오기
    kospi_df = fdr.StockListing('KOSPI')
    # 시가총액(Marcap) 기준 내림차순 정렬 후 상위 50개 추출
    top_50_df = kospi_df.sort_values(by='Marcap', ascending=False).head(50)
    
    # yfinance 형식에 맞게 종목코드 뒤에 '.KS' 붙이기
    tickers = [f"{str(code).zfill(6)}.KS" for code in top_50_df['Code']]
    
    # 종목코드와 이름을 매핑해둘 딕셔너리 생성 (출력 시 보기 편하도록)
    ticker_names = {f"{str(code).zfill(6)}.KS": name for code, name in zip(top_50_df['Code'], top_50_df['Name'])}
    
    return tickers, ticker_names

def create_windows(X, y, window_size):
    X_win, y_win = [], []
    for i in range(window_size, len(X)):
        X_win.append(X[i-window_size:i, :])
        y_win.append(y[i, 0])
    return np.array(X_win), np.array(y_win)

def train_true_quant_bot():
    final_output = {}
    
    print("🌍 글로벌 시장 지수 다운로드 중...")
    market_df = yf.download(["^KS11", "^GSPC", "KRW=X", "^VIX", "^SOX"], period="3y", progress=False)['Close']
    market_df = market_df.rename(columns={"^KS11": "KOSPI", "^GSPC": "SP500", "KRW=X": "EXCHANGE_RATE", "^VIX": "VIX", "^SOX": "SOX_SEMI"})
    market_df.index = market_df.index.tz_localize(None) 
    market_returns = market_df.pct_change().fillna(0)

    # 📌 실시간 상위 50종목 로드
    TICKERS, TICKER_NAMES = get_kospi_top_50()

    for idx, t in enumerate(TICKERS, 1):
        stock_name = TICKER_NAMES.get(t, t)
        try:
            print(f"\n[🚀 {idx}/50] {stock_name}({t}) 퀀트 표준화 딥러닝 분석 시작...")
            
            df = yf.download(t, period="3y", progress=False) 
            
            # ⚠️ 야후 파이낸스 IP 차단(HTTP 429) 방지를 위한 휴식 (아주 중요함!)
            time.sleep(1.5) 

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = df.index.tz_localize(None)

            if len(df) < 50: # 상장된 지 얼마 안 된 종목 등 데이터가 부족하면 스킵
                print(f"   ⚠️ {stock_name} 데이터 부족으로 스킵합니다.")
                continue

            # 타겟은 수익률(%)
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

            # 문제(X)와 정답(y) 정규분포화
            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(X_raw)
            
            scaler_y = StandardScaler()
            y_scaled = scaler_y.fit_transform(y_raw)

            window_size = 15 
            X_win, y_win = create_windows(X_scaled, y_scaled, window_size)

            split = int(len(X_win) * 0.8)
            X_train, X_val = X_win[:split], X_win[split:]
            y_train, y_val = y_win[:split], y_win[split:]

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
                
                model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=0)
                
                last_input = X_scaled[-window_size:].reshape(1, window_size, len(feature_cols))
                
                pred_scaled = model.predict(last_input, verbose=0)
                pred_real_return = scaler_y.inverse_transform(pred_scaled)[0, 0]
                
                ensemble_preds.append(pred_real_return * 1.2)
                del model; gc.collect(); tf.keras.backend.clear_session()

            final_pred_return = float(np.mean(ensemble_preds))
            
            last_price = df['Close'].iloc[-1]
            pred_price = int(last_price * (1 + final_pred_return/100))
            
            std_dev = np.std(ensemble_preds)
            conf = float(max(50.0, min(99.0, 95.0 - (std_dev * 10.0))))
            
            symbol = t.split('.')[0]
            final_output[symbol] = {
                "name": stock_name,
                "pred": f"{pred_price:,}",
                "expected_return": f"{final_pred_return:.2f}%",
                "confidence": f"{conf:.1f}점"
            }
            print(f"   ✅ {stock_name} 완료: 예상가 {pred_price:,}원 ({final_pred_return:.2f}%) | 확신도 {conf:.1f}점")

        except Exception as e:
            print(f"   ❌ {stock_name}({t}) 오류 발생: {e}")

    # 최종 서버 전송 로직
    try:
        response = requests.post(SERVER_URL, json=final_output, timeout=20)
        if response.status_code == 200:
            print("\n🚀 [최종 자동화 완료] 분석 데이터가 구글 클라우드 서버로 무사히 전송되었습니다.")
        else:
            print(f"\n⚠️ 서버 전송 실패. 상태 코드: {response.status_code}")
    except Exception as e:
        print(f"\n❌ 서버 전송 중 오류 발생: {e}")

if __name__ == "__main__":
    train_true_quant_bot()
