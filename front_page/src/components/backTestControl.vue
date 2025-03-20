<template>
  <div class="box-column">
    <div class="box-row">
      <p class="p1">
        Backtesting
      </p>
    </div>
    <div class="box-row">
      <p class="p2">
        Currency:
      </p>
      <div class="data-area">
        <train-checkmenu/>
      </div>
    </div>
    <div class="box-row box-btn">
      <button class="btn" @click="goBackTest">Test</button>
    </div>
  </div>
</template>

<script setup>

import TrainCheckmenu from "@/components/TrainCheckmenu.vue";

const emit = defineEmits(["triggerBackTest"]);

function goBackTest() {
  emit("triggerBackTest");
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