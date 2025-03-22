<template>
  <div id="Diagram" class="box-column" :class="{ box_switch: props.stateSwitchD }" ref="chartContainer">
    <div v-if="isLoading" class="loading-overlay">
      <div class="spinner"></div>
      <p class="loading-text">Loading chart...</p>
    </div>
    <p v-if="showError && !isLoading" class="msg">❌ 图像加载失败或尚未生成！</p>
  </div>
</template>


<script setup>
import { ref, watch, onMounted, onBeforeUnmount, nextTick } from 'vue';
import { createChart } from 'lightweight-charts';

const props = defineProps({
  stateSwitchD: Boolean,
  selectedCurrency: String,
  selectedModel: String,
});

const isLoading = ref(false); // ⬅ 加载状态
const chartContainer = ref(null);
let chart = null;
let candleSeries = null;
let websocket = null;
const showError = ref(false);
let lastCandleTime = null;

const initChart = () => {
  if (!chartContainer.value) return;
  chart = createChart(chartContainer.value, {
    width: chartContainer.value.clientWidth,
    height: chartContainer.value.clientHeight,
    layout: {
      background: { color: '#757575' },
      textColor: '#E0E0E0'
    },
    grid: {
      vertLines: { color: '#444' },
      horzLines: { color: '#444' }
    },
    priceScale: { borderColor: '#555' },
    timeScale: {
      borderColor: '#555',
      timeVisible: true,
      secondsVisible: false
    },
    localization: {
      locale: 'en' // ✅ 设置英文坐标轴
    }
  });
  candleSeries = chart.addCandlestickSeries();
};


// ✅ 状态切换时 resize
const resizeChart = () => {
  nextTick(() => {
    if (chart && chartContainer.value) {
      const width = chartContainer.value.clientWidth;
      const height = chartContainer.value.clientHeight;
      chart.resize(width, height);
    }
  });
};

watch(() => props.stateSwitchD, () => {
  resizeChart();
});

const fetchKlineData = async (symbol) => {
  try {
    isLoading.value = true; // 开始加载
    showError.value = false;
    const endTime = Date.now();
    const startTime = endTime - 90 * 24 * 60 * 60 * 1000;
    const interval = '15m';
    const limit = 1000;
    let allData = [];
    let fetchStart = startTime;

    while (fetchStart < endTime) {
      const url = `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${interval}&startTime=${fetchStart}&limit=${limit}`;
      const res = await fetch(url);
      const data = await res.json();
      if (!Array.isArray(data) || data.length === 0) break;

      const mapped = data.map(k => ({
        time: Math.floor(k[0] / 1000),
        open: parseFloat(k[1]),
        high: parseFloat(k[2]),
        low: parseFloat(k[3]),
        close: parseFloat(k[4]),
      }));

      allData.push(...mapped);
      fetchStart = data[data.length - 1][0] + 15 * 60 * 1000;
      await new Promise(r => setTimeout(r, 50));
    }

    if (chart) chart.remove();
    initChart();
    candleSeries.setData(allData);
    lastCandleTime = allData[allData.length - 1]?.time;
    startWebSocket(symbol);
    resizeChart();
  } catch (e) {
    showError.value = true;
    console.warn('❌ 加载失败:', e);
  } finally {
    isLoading.value = false; // ✅ 加载结束
  }
};


const startWebSocket = (symbol) => {
  if (websocket) websocket.close();
  const wsSymbol = symbol.toLowerCase();
  websocket = new WebSocket(`wss://stream.binance.com:9443/ws/${wsSymbol}@kline_15m`);
  websocket.onmessage = (event) => {
    const res = JSON.parse(event.data);
    const k = res.k;
    const newCandle = {
      time: Math.floor(k.t / 1000),
      open: parseFloat(k.o),
      high: parseFloat(k.h),
      low: parseFloat(k.l),
      close: parseFloat(k.c),
    };
    if (newCandle.time === lastCandleTime) {
      candleSeries.update(newCandle);
    } else if (newCandle.time > lastCandleTime) {
      candleSeries.update(newCandle);
      lastCandleTime = newCandle.time;
    }
  };
};

const closeWebSocket = () => {
  if (websocket) {
    websocket.close();
    websocket = null;
  }
};

watch(() => props.selectedCurrency, (currency) => {
  if (currency) {
    closeWebSocket();
    fetchKlineData(currency);
  }
});

onMounted(() => {
  if (props.selectedCurrency) {
    fetchKlineData(props.selectedCurrency);
  }
});

onBeforeUnmount(() => {
  closeWebSocket();
  if (chart) chart.remove();
});
</script>


<style scoped>
.box-column {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  background-color: #757575;
  width: 450px;
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

.msg {
  font-size: 16px;
  color: #ffcdd2;
  margin-top: 20px;
  font-weight: bold;
}

.loading-overlay {
  position: absolute;
  z-index: 10;
  background-color: rgba(0, 0, 0, 0.6);
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 5px solid rgba(255, 255, 255, 0.2);
  border-top-color: #ffffff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.loading-text {
  color: #fff;
  font-size: 16px;
  margin-top: 12px;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

</style>
