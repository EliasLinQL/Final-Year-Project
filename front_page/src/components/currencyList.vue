<template>
  <div class="box-column">
    <div class="box-0">
      <p class="p1">Currency List</p>
      <button class="currency-submit" @click="submit">Create</button>
      <button class="currency-cancel" @click="cancel">Cancel</button>
    </div>
    <div class="box-1">
      <div
          class="box-currency"
          v-for="(currencyObj, index) in currencies"
          :key="index"
      >
        <!-- 使用 selected 属性绑定复选框 -->
        <input
            type="checkbox"
            class="check"
            v-model="currencyObj.selected"
        />

        <div class="icon"></div>

        <div class="name">
          <p class="p2">{{ currencyObj.currency.getName() }}</p>
        </div>

        <!-- 日期范围绑定 -->
        <div class="timeset">
          <DatePicker
              v-model="currencyObj.currency.dates"
              format="yyyy-MM-dd HH:mm:ss"
              range
              :teleport="true"
              style="width: 400px; height: 24px; font-size: 12px;"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";
import DatePicker from "@vuepic/vue-datepicker";
import "@vuepic/vue-datepicker/dist/main.css";
import CurrencySet from "@/bean/currencySet.js";

// 定义emit事件
const emit = defineEmits(["triggerSubmit", "triggerCancel"]);

// 初始化货币数据
const currencies = ref([
  { currency: new CurrencySet("DOGE", ["", ""]), selected: false },
  { currency: new CurrencySet("BTC", ["", ""]), selected: false },
  { currency: new CurrencySet("ETH", ["", ""]), selected: false },
  { currency: new CurrencySet("LQF", ["", ""]), selected: false },
]);


function submit() {
  const selectedCurrencies = currencies.value
      .filter((item) => item.selected)
      .map((item) => item.currency);
  emit("triggerSubmit", selectedCurrencies);
}

function cancel() {
  emit("triggerCancel");
}
</script>

<style scoped>
.box-column {
  background-color: #757575;
  width: 500px;
  height: 480px;
  margin: 10px;
  border-radius: 18px;
  display: flex;
  flex-direction: column;
}

.box-0 {
  margin: 4px 0 10px 10px;
  display: flex;
  align-items: center;
}

.box-1 {
  background-color: #a6aaa6;
  border-radius: 12px;
  margin: 4px 0 8px 16px;
  width: 460px;
  height: 372px;
  overflow: auto;
  position: relative;
}

.box-1::-webkit-scrollbar {
  width: 16px;
}

.box-1::-webkit-scrollbar-track {
  background-color: #f1f1f1;
  border-radius: 12px;
}

.box-1::-webkit-scrollbar-thumb {
  background-color: #bfc4cc;
  border-radius: 12px;
}

.box-currency {
  margin: 6px 6px;
  width: 420px;
  height: 98px;
  position: relative;
  border-radius: 10px;
  background-color: #595555;
}

.currency-submit {
  background-color: #3A708D;
  border: 2px solid #346884;
  box-shadow: 1px 2px 4px #346884;
  width: 120px;
  height: 40px;
  font-size: 20px;
  color: #E0E0E0;
  margin-left: 30px;
  margin-top: 8px;
  border-radius: 16px;
  transition: all 0.2s ease;
}

.currency-submit:hover {
  background-color: #286487;
  opacity: 0.8;
}

.currency-submit:active {
  background-color: #246893;
}

.currency-cancel {
  background-color: #3A708D;
  border: 2px solid #346884;
  box-shadow: 1px 2px 4px #346884;
  width: 120px;
  height: 40px;
  font-size: 20px;
  color: #E0E0E0;
  margin-left: 30px;
  margin-top: 8px;
  border-radius: 16px;
  transition: all 0.2s ease;
}

.currency-cancel:hover {
  background-color: #286487;
  opacity: 0.8;
}

.currency-cancel:active {
  background-color: #246893;
}

.p1 {
  font-size: 26px;
  margin-top: 20px;
  margin-bottom: 20px;
  margin-left: 10px;
}

.p2 {
  font-size: 18px;
  margin-left: 0;
  margin-top: 5px;
  margin-bottom: 10px;
}

p {
  margin: 0;
  margin-top: 4px;
  font-family: Microsoft YaHei;
  color: #E0E0E0;
}

.icon {
  position: absolute;
  background-color: #ffffff;
  height: 38px;
  width: 38px;
  left: 9px;
  border-radius: 16px;
  margin-top: 4px;
  margin-bottom: 4px;
}

.check {
  position: absolute;
  background-color: #ffffff;
  top: 6px;
  right: 7px;
  height: 28px;
  width: 28px;
}

.name {
  position: absolute;
  top: 5%;
  left: 64px;
  width: 50px;
  height: 30px;
}

.timeset {
  position: absolute;
  top: 49%;
  left: 9px;
  border-radius: 8px;
  width: 100%;
  height: 30px;
}
</style>
