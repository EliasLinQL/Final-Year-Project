<template>
  <div class="box-column">
    <div class="box-row">
      <p class="p1">Model Info</p>
    </div>
    <div class="box-row">
      <p class="p2">Existing Models</p>
      <div class="box-checkmenu">
        <checkmenu/>
      </div>
    </div>
    <div class="box-row">
      <p class="p2">Info</p>
      <div class="data-area">

      </div>
    </div>
    <div class="box-row">
      <p class="p2">Info</p>
      <div class="data-area">

      </div>
    </div>
    <div class="box-row">
      <p class="p2">Info</p>
      <div class="data-area">

      </div>
    </div>
    <div class="box-row box-btn">
      <button class="btn" @click="goTrain">Go Train</button>
      <button class="btn" @click="showGraph">Add</button>
    </div>
  </div>
</template>

<script setup>
import Checkmenu from "@/components/checkmenu.vue";

//定义子组件触发父组件事件
const emit = defineEmits(['triggerTrain','triggerGraph']);

//通过按钮事件来通知父组件触发对应事件
function goTrain() {
  emit("triggerTrain");
}
function showGraph(){
  emit("triggerGraph");
}

</script>

<style scoped>
.box-column{
  display: flex;
  flex-direction: column;
  background-color: #757575;
  width: 400px;
  height: 400px;
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
}
.box-btn{
  justify-content: space-around;
}
.btn{
  margin-top: 20px;
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
.data-area{
  width: 90%;
  height: 80%;
  background-color: #827E7E;
  margin-right: 20px;
  margin-top: 6px;
}
.box-checkmenu{
  position: absolute;
  left: 180px;
  z-index: 1;
}
</style>