import os, time, gc, datetime, requests
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

SERVER_IP = os.environ.get("MY_SERVER_IP")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# 🇰🇷 한국 50종목 + 🇺🇸 미국 30종목 (총 80종목)
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
    "005830.KS", "022100.KS", "010140.KS", "138040.KS", "042660.KS",
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "LLY", "AVGO",
    "V", "JPM", "UNH", "WMT", "MA", "XOM", "JNJ", "PG", "ORCL", "HD",
    "COST", "ABBV", "MRK", "CVX", "KO", "NFLX", "PEP", "BAC", "CRM", "AMD"
]

TICKER_NAMES = {
    '005930':'삼성전자', '000660':'SK하이닉스', '373220':'LG엔솔', '207940':'삼성바이오',
    'AAPL':'애플', 'MSFT':'마이크로소프트', 'NVDA':'엔비디아', 'GOOGL':'구글', 'AMZN':'아마존',
    'META':'메타', 'TSLA':'테슬라', 'BRK-B':'버크셔', 'LLY':'일라이릴리', 'AVGO':'브로드컴',
    'V':'비자', 'JPM':'JP모건', 'UNH':'유나이티드헬스', 'WMT':'월마트', 'MA':'마스터카드',
    'XOM':'엑슨모빌', 'JNJ':'존슨앤존슨', 'PG':'P&G', 'ORCL':'오라클', 'HD':'홈디포',
    'COST':'코스트코', 'ABBV':'애브비', 'MRK':'머크', 'CVX':'쉐브론', 'KO':'코카콜라',
    'NFLX':'넷플릭스', 'PEP':'펩시', 'BAC':'뱅크오브아메리카', 'CRM':'세일즈포스', 'AMD':'AMD'
}

NUM_ENSEMBLE = 5  

def create_windows(X, y, window_size):
    X_win, y_win = [], []
    for i in range(window_size, len(X)):
        X_win.append(X[i-window_size:i, :]); y_win.append(y[i, 0])
    return np.array(X_win), np.array(y_win)

def train_true_quant_bot():
    final_output = {}
    print("🌍 글로벌 시장 지수 다운로드 중...")
    market_df = yf.download(["^KS11", "^GSPC", "KRW=X", "^VIX", "^SOX"], period="3y", progress=False)['Close']
    market_df = market_df.rename(columns={"^KS11": "KOSPI", "^GSPC": "SP500", "KRW=X": "EXCHANGE_RATE", "^VIX": "VIX", "^SOX": "SOX_SEMI"})
    market_df.index = market_df.index.tz_localize(None) 
    market_returns = market_df.pct_change().fillna(0)

    for idx, t in enumerate(TICKERS, 1):
        try:
            print(f"\n[🚀 {idx}/{len(TICKERS)} - {t}] 분석 시작...")
            df = yf.download(t, period="3y", progress=False) 
            time.sleep(1) 
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.index = df.index.tz_localize(None)
            if len(df) < 50: continue

            df['Target_Return'] = df['Close'].pct_change().shift(-1) * 100 
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
            df['Momentum_5'] = df['Close'].pct_change(5) * 100
            df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).ewm(alpha=1/14).mean() / (-df['Close'].diff().clip(upper=0)).ewm(alpha=1/14).mean())))
            df['Overnight_Gap'] = (df['Open'] / df['Close'].shift(1) - 1) * 100
            df['Day_of_Week'] = df.index.dayofweek
            df = df.join(market_returns, how='left').ffill()
            
            feature_cols = ['RSI', 'Overnight_Gap', 'Day_of_Week', 'MACD', 'Momentum_5', 'KOSPI', 'SP500', 'VIX', 'SOX_SEMI']
            df_valid = df.dropna(subset=feature_cols + ['Target_Return'])
            X_raw, y_raw = df_valid[feature_cols].values, df_valid[['Target_Return']].values

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
                x = GlobalAveragePooling1D()(Attention()([x, x]))
                x = Dropout(0.2)(Dense(32, activation='relu')(x))
                model = Model(inputs=inputs, outputs=Dense(1, activation='linear')(x))
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss='mse')
                model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
                
                last_input = X_scaled[-window_size:].reshape(1, window_size, len(feature_cols))
                ensemble_preds.append(scaler_y.inverse_transform(model(last_input, training=False).numpy())[0, 0])
                del model; gc.collect(); tf.keras.backend.clear_session()

            final_pred_return = np.mean(ensemble_preds)
            last_price = float(df['Close'].iloc[-1])
            pred_price = last_price * (1 + final_pred_return/100)
            conf = max(50.0, min(99.0, 95.0 - (np.std(ensemble_preds) * 10.0)))
            
            symbol = t.split('.')[0]
            # 달러와 원화 표기 차이를 위해 포맷팅
            price_str = f"{pred_price:.2f}" if not symbol.isdigit() else f"{int(pred_price):,}"
            
            final_output[symbol] = {
                "pred": price_str,
                "expected_return": f"{final_pred_return:.2f}%",
                "confidence": f"{conf:.1f}점"
            }
            print(f"   ✅ {symbol} 분석완료")

        except Exception as e: print(f"   ❌ {t} 오류: {e}")

    if SERVER_IP:
        try: requests.post(f"http://{SERVER_IP}:8080/upload", json=final_output, timeout=30)
        except: pass

if __name__ == "__main__":
    train_true_quant_bot()
