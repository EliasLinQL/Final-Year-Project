<template>
  <div class="box-column">
    <div class="box-row">
      <p class="p1">New Model Settings</p>
    </div>
    <div class="box-row">

    </div>
    <div class="box-row">
      <p class="p2">ModelSet<br>Name</p>
      <div class="data-area-name">

      </div>
    </div>
    <div class="box-row">
      <p class="p2">Target<br>Currency</p>
      <div class="box-currencychoose">
        <button class="currencylist" @click="showCurrencyList">
          {{ isCurrencyListVisible ? 'Close' : 'Show' }}
        </button>
      </div>
    </div>
    <div class="box-row box-btn">
      <button class="btn" @click="createModel">Create Model</button>
      <button class="btn" @click="Cancel">Cancel</button>
    </div>
  </div>
</template>

<script setup>
import {ref} from "vue";

const emit = defineEmits(["triggerCreateNewModel","triggerCancelCreate","triggerCurrencyList"]);
const isCurrencyListVisible = ref(false);

function showCurrencyList() {
  isCurrencyListVisible.value = !isCurrencyListVisible.value;
  emit("triggerCurrencyList");
}

function createModel() {
  emit("triggerCreateNewModel");
}
function Cancel() {
  emit("triggerCancelCreate");
}
</script>

<style scoped>
.box-column {
  background-color: #757575;
  width: 520px;
  height: 320px;
  margin: 10px;
  border-radius: 18px;
  position: relative;
}

.box-row {
  display: flex;
  flex-direction: row;
  margin: 0;
  margin-left: 10px;
  margin-top: 5px;
}

.p1 {
  font-size: 26px;
  margin-top: 20px;
  margin-bottom: 20px;
}

.p2 {
  font-size: 18px;
  margin-top: 10px;
  margin-bottom: 10px;
}
p {
  margin: 15px;
  font-family: Microsoft YaHei;
  color: #E0E0E0;
}
.data-area-name {
  width: 268px;
  height: 44px;
  background-color: #827E7E;
  margin-right: 20px;
  position: absolute;
  transform: translate(150px, 12px);
}
.box-currencychoose{
  margin-right: 20px;
  position: absolute;
  transform: translate(150px, 12px);
  z-index: 1;
}
.btn {
  margin-top: 20px;
  margin-bottom: 5px;
  width: 158px;
  height: 38px;
  background-color: #3A708D;
  border: 2px solid #346884;
  border-radius: 12px;
  font-size: 20px;
  color: #E0E0E0;
  box-shadow: 1px 2px 4px #346884;
  transition: all 0.2s ease;
}

.btn:hover {
  background-color: #286487;
  opacity: 0.8;
}

.btn:active {
  background-color: #246893;
}

.box-btn {
  justify-content: space-around;
}
.currencylist {
  width: 68px;
  height: 52px;
  background-color: rgba(86, 182, 157, 0.6);
  border: 2px solid rgba(71, 152, 130, 0.6);
  box-shadow: 1px 2px 4px rgba(84, 198, 195, 0.6);
  border-radius: 24%;
  margin-right: 20px;
  position: absolute;
  transform: translate(0px, -6px);
  transition: all 0.2s ease;
  font-size: 20px;
  color: #c1d0dd;
}

.currencylist:hover {
  background-color: rgba(102, 198, 175, 0.6);
  opacity: 0.8;
}

.currencylist:active {
  background-color: rgba(118, 214, 190, 0.6);
}
</style>
