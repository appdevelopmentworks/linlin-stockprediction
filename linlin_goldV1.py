import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as pdta
import numpy as np
from prophet.plot import plot_plotly
from plotly import graph_objects as go
from PIL import Image

def plot_chart(df, name):
    layout = {
        "height":1500,
        "title":{"text": "{}".format(name), "x": 0.5},
        "xaxis":{"title": "日付", "rangeslider":{"visible":True}},
        "yaxis1":{"domain":[.46, 1.0], "title": "価格（円）", "side": "left", "tickformat": ","},
        "yaxis2":{"domain":[.40,.46]},
        "yaxis3":{"domain":[.30,.395], "title":"出来高", "side":"right"},
        "yaxis4":{"domain":[.20,.295], "title":"RSI", "side":"right"},
        "yaxis5":{"domain":[.10,.195], "title":"MACD", "side":"right"},
        "yaxis6":{"domain":[.00,.095], "title":"HV", "side":"right"},
        "plot_bgcolor":"light blue"
    }

    data = [
        go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                    increasing_line_color="red",
                    increasing_line_width=1.0,
                    increasing_fillcolor="red",
                    decreasing_line_color="blue",
                    decreasing_line_width=1.0,
                    decreasing_fillcolor="blue"
                    ),
        #移動平均
        go.Scatter(x=df.index, y=df['hma10'], name="HMA10",
                line={"color": "red", "width":1.2}),
        go.Scatter(x=df.index, y=df['ema5'], name="EMA5",
                line={"color": "royalblue", "width":1.2}),
            go.Scatter(x=df.index, y=df['ema13'], name="EMA13",
                line={"color": "lightseagreen", "width":1.2}),   
        go.Scatter(x=df.index, y=df['sma200'], name="SMA200",
                line={"color": "darkred", "width":1.0}),
        #出来高
        go.Bar(yaxis="y3", x=df.index, y=df['Volume'], name="Volume",marker={"color":"slategray"}),
        #RSI
        go.Scatter(yaxis="y4", x=df.index, y=df['rsiF'], name="RSI_F",
                line={"color":"magenta", "width":1}),
        go.Scatter(yaxis="y4", x=df.index, y=df['rsiS'], name="RSI_S",
                line={"color":"green", "width":1}),
        go.Scatter(yaxis="y4", x=df.index, y=df['30'], name="30",
                line={"color":"black", "width":0.5}),
        go.Scatter(yaxis="y4", x=df.index, y=df['70'], name="70",
                line={"color":"black", "width":0.5}),
        #MACD
        go.Scatter(yaxis="y5", x=df.index, y=df['macd'], name="MACD",
                line={"color":"magenta", "width":1}),
        go.Scatter(yaxis="y5", x=df.index, y=df['signal'], name="MACDSIG",
                line={"color":"green", "width":1}),
        go.Bar(yaxis="y5", x=df.index, y=df['hist'], name="MACDHIST",marker={"color":"slategrey"}),    
        
        #HV
        go.Scatter(yaxis="y6", x=df.index, y=df['hv'], name="HV",
                line={"color":"red", "width":1}),    
    ]
    fig = go.Figure(data = data, layout = go.Layout(layout))
    return fig


st.title("ゴールドラッシュ・テクニカル")
st.text("金に最適化したテクニカルチャートです。")
image = Image.open("headere.png")
st.image(image)
st.caption("HMA（ハル移動平均）とHV(ヒストリカルボラティリティー)を取り入れてみました！")
st.caption("短期トレード（スイング用）にパラメータ設定してあります")

ticker = st.text_input("TickerCode", value="GC=F")
kikan = st.slider("表示期間(日):", min_value=30, max_value=300, value=100, step=1)

df = yf.download(ticker)
df.columns = [row[0] for row in df.columns]

#Moving Average
df['hma10'] = pdta.hma(df['Close'], length=10)
df["ema5"] = pdta.ema(df['Close'], length=5)
df["ema13"] = pdta.ema(df['Close'], length=13)
df["sma200"] = pdta.sma(df['Close'], length=200)
#RSI
df['rsiF'] = pdta.rsi(df['Close'], length=3)
df['rsiS'] = pdta.rsi(df['Close'], length=5)
df['70'], df['30'] = [70 for _ in df['Close']], [30 for _ in df['Close']]
# MACD
macd = pdta.macd(df["Close"], fast=12, slow=26, signal=9)
df["macd"] = macd["MACD_12_26_9"]
df["signal"] = macd["MACDs_12_26_9"]
df["hist"] = macd["MACDh_12_26_9"]
# Historical Volatility
df['hv'] = df['Close'].pct_change().rolling(window=30).std() * np.sqrt(252) * 100
#インデックスを文字列型に（休日の抜けを無くす）
df.index = pd.to_datetime(df.index).strftime('%m-%d-%Y')


plt = plot_chart(df.tail(int(kikan)), yf.Ticker(ticker).info["shortName"])

st.plotly_chart(plt)


