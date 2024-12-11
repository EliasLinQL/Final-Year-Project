<template>
  <div class="box-column">
    <div class="box-row">
      <p class="p1">Pre Supposed Info</p>
    </div>
    <p class="pname">PreName: {{ selectedTrainSet.name }}</p>
    <div class="box-list">
      <div class="box-currency" v-for="(currencyObj, index) in selectedTrainSet.currencies" :key="index">
        <div class="icon">
          <img :src="currencyObj.getIcon()" class="currency-icon" />
        </div>
        <div class="name">
          <p class="p2">{{ currencyObj.getName() }}</p>
        </div>
        <!-- 日期范围显示 -->
        <div class="timeset">
          <p class="p3">{{ formatDate(currencyObj.getDates()[0]) }}</p>
          <p class="p3">{{ formatDate(currencyObj.getDates()[1]) }}</p>
        </div>
      </div>
    </div>
  </div>
</template>


<script setup>
import { ref, watch } from "vue";
import ModelSet from "@/bean/modelSet.js";
import CurrencySet from "@/bean/currencySet.js";

// 日期格式化函数
function formatDate(date) {
  if (!date) return "N/A";
  const d = new Date(date);
  if (isNaN(d)) return "Invalid Date";
  return d.toLocaleString("en-GB", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).replace(",", "");
}

const selectedTrainSet = ref(new ModelSet("", []));

const props = defineProps({
  trainset: { type: Object },
});

watch(
    () => props.trainset,
    (newval) => {
      if (newval) {
        selectedTrainSet.value = new ModelSet(
            newval.name,
            (newval.currencies || []).map(
                (currency) =>
                    new CurrencySet(
                        currency.name,
                        currency.dates || ["", ""],
                        currency.icon
                    )
            )
        );
      }
    },
    { immediate: true }
);

</script>

<style scoped>
.box-column {
  background-color: #6c6b6b;
  width: 340px;
  height: 920px;
  margin: 10px;
  border-radius: 18px;
  display: flex;
  flex-direction: column;
}
.box-row {
  display: flex;
  flex-direction: row;
  margin: 0;
  margin-left: 10px;
  margin-top: 5px;
}
.box-list {
  background-color: #a6aaa6;
  border-radius: 12px;
  margin: 4px 0 8px 6px;
  width: 330px;
  height: 83.6%;
  overflow: auto;
  position: relative;
}
.box-list::-webkit-scrollbar {
  width: 12px;
}

.box-list::-webkit-scrollbar-track {
  background-color: #f1f1f1;
  border-radius: 12px;
}

.box-list::-webkit-scrollbar-thumb {
  background-color: #bfc4cc;
  border-radius: 12px;
}

.box-currency {
  margin-top: 6px;
  margin-bottom: 6px;
  margin-left: 4px;
  margin-right: 0;
  width: 296px;
  height: 112px;
  position: relative;
  border-radius: 10px;
  background-color: #595555;
}
.icon {
  position: absolute;
  background-color: #504d4d;
  height: 48px;
  width: 48px;
  left: 9px;
  border-radius: 24px;
  margin-top: 8px;
  margin-bottom: 4px;
}
.currency-icon {
  width: 100%;
  height: 100%;
  border-radius: 24px;
  object-fit: cover;
}

.name {
  position: absolute;
  top: -10px;
  left: 64px;
  width: 50px;
  height: 30px;
}
.timeset {
  position: absolute;
  top: 32%;
  left: 16px;
  border-radius: 8px;
  width: 100%;
  height: 30px;
}
.p1 {
  font-size: 26px;
  margin-top: 20px;
  margin-bottom: 20px;
}
.pname{
  font-size: 20px;
  margin-bottom: 8px;
  margin-left: 24px;
  margin-top: 2px;
}
.p2 {
  font-size: 18px;
  margin-top: 20px;
}
.p3 {
  font-size: 18px;
  margin-top: 10px;
  margin-bottom: 0;
  margin-left: 56px;
}
p {
  margin: 15px;
  font-family: Microsoft YaHei;
  color: #E0E0E0;
}
</style>
