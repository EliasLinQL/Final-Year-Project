<template>
  <div id="Diagram" class="box-column" :class="{ box_switch: props.stateSwitchD }"></div>
</template>

<script setup>
import { onMounted, onBeforeUnmount } from "vue";
import { createChart } from "lightweight-charts";

const props = defineProps({
  stateSwitchD: {
    type: Boolean,
    required: true,
  },
});

let chart;
let candlestickSeries;
let lineSeries;
let binanceSocket;
let priceHistory = [];
let earliestTime = null; // 记录最早的时间戳

onMounted(async () => {
  const DiagramElement = document.getElementById("Diagram");

  if (DiagramElement) {
    chart = createChart(DiagramElement, {
      width: DiagramElement.clientWidth,
      height: DiagramElement.clientHeight,
      layout: {
        backgroundColor: "#ffffff",
        textColor: "#000000",
      },
    });

    // 添加 K 线图
    candlestickSeries = chart.addCandlestickSeries();

    // 添加一条均线（移动平均线）
    lineSeries = chart.addLineSeries({ color: "blue", lineWidth: 2 });

    // **先获取最近半年的 15 分钟历史数据**
    await fetchHistoricalData();

    // **再连接 WebSocket 获取实时数据**
    connectWebSocket();

    // 监听窗口大小变化
    const resizeObserver = new ResizeObserver(() => {
      if (DiagramElement) {
        chart.resize(DiagramElement.clientWidth, DiagramElement.clientHeight);
      }
    });

    resizeObserver.observe(DiagramElement);
  } else {
    console.error("❌ Element with id 'Diagram' not found");
  }
});

// **获取最近半年的 15 分钟 K 线数据**
async function fetchHistoricalData() {
  const symbol = "BTCUSDT";
  const interval = "15m"; // **改为 15 分钟 K 线**
  const limit = 1000; // Binance API 一次最多获取 1000 根 K 线
  const sixMonthsAgo = Math.floor(Date.now() / 1000) - 180 * 24 * 60 * 60; // 半年前的 Unix 时间戳

  let endTime = Math.floor(Date.now() / 1000) * 1000; // 以当前时间为起点（毫秒）
  let allCandles = [];

  try {
    while (endTime / 1000 > sixMonthsAgo) {
      const url = `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=${limit}&endTime=${endTime}`;
      const response = await fetch(url);
      const data = await response.json();

      if (Array.isArray(data) && data.length > 0) {
        const newCandles = data.map((kline) => ({
          time: kline[0] / 1000, // 时间戳转换为秒
          open: parseFloat(kline[1]),
          high: parseFloat(kline[2]),
          low: parseFloat(kline[3]),
          close: parseFloat(kline[4]),
        }));

        allCandles = [...newCandles, ...allCandles];

        // 更新最早的时间戳，继续请求更早的数据
        endTime = data[0][0] - 1;
      } else {
        break; // 没有更多数据时，停止请求
      }
    }

    priceHistory = allCandles;
    candlestickSeries.setData(priceHistory);

    console.log(`✅ 加载完成，共加载 ${priceHistory.length} 根 15 分钟 K 线数据`);

  } catch (error) {
    console.error("🔥 Error fetching historical data:", error);
  }
}

// **WebSocket 连接，实时更新最新数据**
function connectWebSocket() {
  const symbol = "btcusdt";
  const interval = "15m"; // **订阅 15 分钟 K 线**
  const wsUrl = `wss://stream.binance.com:9443/ws/${symbol}@kline_${interval}`;

  binanceSocket = new WebSocket(wsUrl);

  binanceSocket.onmessage = (event) => {
    const message = JSON.parse(event.data);

    if (message.k) {
      const kline = message.k;

      const newCandle = {
        time: kline.t / 1000, // 时间戳转换
        open: parseFloat(kline.o),
        high: parseFloat(kline.h),
        low: parseFloat(kline.l),
        close: parseFloat(kline.c),
      };

      candlestickSeries.update(newCandle);

      // 更新价格历史数据
      priceHistory.push(newCandle);
      if (priceHistory.length > 20000) {
        priceHistory.shift(); // 只保留最近 20000 根 K 线数据
      }
    }
  };

  binanceSocket.onclose = () => {
    console.log("🔄 WebSocket closed. Reconnecting...");
    setTimeout(connectWebSocket, 5000);
  };
}

// 组件销毁时关闭 WebSocket
onBeforeUnmount(() => {
  if (binanceSocket) {
    binanceSocket.close();
  }
});
</script>

<style scoped>
.box-column {
  display: flex;
  flex-direction: column;
  background-color: #757575;
  width: 540px;
  height: 270px;
  margin: 10px;
  padding: 0;
  position: relative;
  border-radius: 18px;
  transition: all 0.3s ease;
}

.box_switch {
  width: 1260px;
  height: 630px;
}

p {
  margin: 15px;
  font-family: Microsoft YaHei;
  color: #E0E0E0;
}
</style>
