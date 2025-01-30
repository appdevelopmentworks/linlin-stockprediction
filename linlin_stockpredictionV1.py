import streamlit as st
from datetime import date

import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

#元の株データを取得
def load_rawdata(ticker):
    data = yf.download(ticker, START, TODAY)
    return data
#予想用のデータセットにインデックスをリセット
def load_data(df):
    data = df.copy()
    data.reset_index(inplace=True)
    return data

#元データのチャート表示用
def plot_raw_data(rdf, code, name=""):
    rdf.index = pd.to_datetime(rdf.index).strftime('%m-%d-%Y')
    
    layout = {"title": {"text": "{} {}".format(code, name), "x": 0.5},
            "xaxis": {"title": "日付", "rangeslider": {"visible": True}},
            "yaxis": {"title": "価格（円）", "side": "left", "tickformat": ","}, 
            "width": 800, "height": 600,
            "plot_bgcolor": "light blue"}

    dataf = [go.Candlestick(x=rdf.index, open=rdf['Open'],high=rdf['High'], low=rdf['Low'], close=rdf['Close'],
                        increasing_line_color="red",
                        increasing_line_width=1.0,
                        increasing_fillcolor="red",
                        decreasing_line_color="green",
                        decreasing_line_width=1.0,
                        decreasing_fillcolor="green"
                        )]
    fig = go.Figure(data=dataf, layout=go.Layout(layout))
    st.plotly_chart(fig)



st.title("株価予想")
#マグニフィセント・セブン
stocks = ["GOOG","AAPL","META","AMZN","MSFT","NVDA","TSLA"]

text_input = st.text_input("ティッカーコードを入力してください：（例：7203.T）")
#stocks.insert(0, text_input)

#テキストボックスに入力あればそれを使用し
#入力なければマグニフィセント・セブンを使う
if text_input == "":
    selected_stock = st.selectbox("マグニフィセント・セブン", stocks)
else:
    selected_stock =  text_input

#予測期間スライダー
period = st.slider("予測期間", 1, 100, value=10)

#既定値に読み込み前の表示
data_load_state = st.text("データ読み込み中！")
#データの読み込みと予測用データセット作成
dataraw = load_rawdata(selected_stock)
data = load_data(dataraw)
#読み込み終了表示
data_load_state.text("データを読み込みました！")

#チャートの描画(元データ)
st.subheader("チャート")
plot_raw_data(dataraw, selected_stock)

#生データのデータフレーム表示
st.write("元のデータ")
st.write(data.tail())

#
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
#日付のみの情報に置き換える
#df_train["ds"] = pd.to_datetime(df_train["ds"]).dt.date

# データの前処理
df_train['y'] = df_train['y'].astype(str).str.strip()
df_train['y'] = df_train['y'].str.replace(r'[^\d.]', '', regex=True)
df_train['y'] = df_train['y'].str.replace(',', '', regex=True)

# 数値型への変換
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')

# 欠損値の処理 (必要であれば)
df_train['y'].fillna(df_train['y'].mean(), inplace=True)

# データ型の確認 (デバッグ用)
# print("Dtype",df_train['y'].dtype)
# print("Dtype",df_train['ds'].dtype)
# print(df_train['ds'])
# print("#####################")

model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

st.write("予測用データセット")
st.write(forecast.tail())

st.subheader("予測チャート")
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

st.subheader("時系列の傾向")
fig2 = model.plot_components(forecast)
st.write(fig2)