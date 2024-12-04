<template>
  <div class="box-column">
    <div class="box-row">
      <p class="p1">New Model Settings</p>
    </div>
    <div class="box-row model-set">

    </div>
    <div class="box-row">
      <p class="p2">ModelSet<br>Name</p>
      <div class="data-area-name">

      </div>
    </div>
    <div class="box-row">
      <p class="p2">Target<br>Currency</p>
      <div class="box-currencychoose">
        <currency-checkmenu/>
      </div>
    </div>
    <div class="box-row box-btn">
      <button class="btn" @click="createModel">Create Model</button>
      <button class="btn" @click="Cancel">Cancel</button>
    </div>
  </div>
</template>

<script setup>
import CurrencyCheckmenu from "@/components/currencyCheckmenu.vue";

const emit = defineEmits(["triggerCreateNewModel","triggerCancelCreate"]);

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
  height: 480px;
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
  margin-top: 184px;
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
</style>
