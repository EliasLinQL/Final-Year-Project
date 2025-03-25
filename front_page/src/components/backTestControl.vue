<template>
  <div class="box-column">
    <div class="box-row">
      <p class="p1">Backtesting</p>
    </div>
    <div class="box-row">
      <p class="p2">Currency:</p>
      <div class="data-area">
        <currency-checkmenu @updateSelectedCurrency="handleCurrencySelect" />
      </div>
    </div>
    <div class="box-row box-btn">
      <button class="btn" @click="goBackTest">Test</button>
    </div>
  </div>
</template>

<script setup>

/**
 * backtestControl.vue
 *
 * This component provides a simple UI for selecting a cryptocurrency and initiating a backtest.
 *
 * Features:
 * - Allows user to select a currency via a dropdown (CurrencyCheckmenu component).
 * - Triggers a backtest by sending a POST request to the Flask API (/api/run_backtest).
 * - Emits events to notify the parent component of selected currency and backtest execution.
 *
 * Emitted Events:
 * - triggerCurrencyToApp(selectedCurrency): Sends the selected currency to the parent component.
 * - triggerBackTest(): Notifies parent when backtest completes.
 */


import CurrencyCheckmenu from "@/components/currencyCheckmenu.vue";
import { ref } from 'vue';

const emit = defineEmits(["triggerBackTest", "triggerCurrencyToApp"]);
const isLoading = ref(false);
const message = ref("");

async function goBackTest() {
  isLoading.value = true;
  message.value = "Running backtest...";
  try {
    const response = await fetch("http://localhost:5000/api/run_backtest", {
      method: "POST",
    });
    const result = await response.json();
    message.value = result.message;
    emit("triggerBackTest");
  } catch (error) {
    console.error("❌ Backtest request failed:", error);
    message.value = "Failed to run backtest.";
  } finally {
    isLoading.value = false;
  }
}

function handleCurrencySelect(selected) {
  emit("triggerCurrencyToApp", selected);
}
</script>


<style scoped>
.box-column{
  display: flex;
  flex-direction: column;
  background-color: #757575;
  width: 400px;
  height: 240px;
  margin: 10px;
  padding: 0;
  position: relative;
  border-radius: 18px;
}
.box-row{
  display: flex;
  flex-direction: row;
  margin: 0;
  margin-left: 10px;
  margin-top: 5px;
}
p{
  margin: 15px;
  font-family: Microsoft YaHei;
  color: #E0E0E0;
}
.p1{
  font-size: 26px;
}
.p2{
  font-size: 18px;
  margin-top: 24px;
}
.btn{
  margin-top: 40px;
  margin-bottom: 5px;
  width: 158px;
  height: 37px;
  background-color: #3A708D;
  border: 2px solid #346884;
  border-radius: 12px;
  font-size: 20px;
  color: #E0E0E0;
  box-shadow: 1px 2px 4px #346884;
  transition: all 0.2s ease;
}
.btn:hover{
  background-color: #286487;
  opacity: 0.8;
}
.btn:active{
  background-color: #246893;
}
.box-btn{
  justify-content: space-around;
}
.data-area{
  width: 200px;
  height: 42px;
  margin-right: 20px;
  margin-left: 30px;
  margin-top: 4px;
  position: absolute;
  transform: translate(90px,12px);
}
</style>