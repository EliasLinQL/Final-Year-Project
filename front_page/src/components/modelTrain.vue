<template>
  <div class="box-column">
    <div class="box-row">
      <p class="p1">Train Details</p>
    </div>
    <div class="box-row close-text">
      <p class="p2 close-text">Target</p>
      <div class="box-checkmenu">
        <checkmenu/>
      </div>
    </div>
    <div class="box-row close-text">
      <p class="p2 close-text">Currency</p>
    </div>
    <div class="box-row">
      <p class="p2">Start</p>
      <div class="data-area">

      </div>
    </div>
    <div class="box-row">
      <p class="p2">End</p>
      <div class="data-area">

      </div>
    </div>
    <div class="box-row box-btn">
      <button class="btn" @click="endTrain">End Training</button>
      <button class="btn">Start Train</button>
    </div>
  </div>
</template>

<script setup>
import Checkmenu from "@/components/checkmenu.vue";

//定义子组件触发父组件事件
const emit = defineEmits(["triggerEndTrain"]);

//通过按钮事件来通知父组件触发对应事件
function endTrain() {
  emit("triggerEndTrain");
}
</script>

<style scoped>
  .box-column{
    background-color: #757575;
    width: 520px;
    height: 400px;
    margin: 10px;
    border-radius: 18px;
    position: relative;
  }
  .box-row{
    display: flex;
    flex-direction: row;
    margin: 0;
    margin-left: 10px;
    margin-top: 5px;
  }
  .p1{
    font-size: 26px;
    margin-top: 20px;
    margin-bottom: 20px;
  }
  .p2{
    font-size: 18px;
    margin-top: 45px;
  }
  .close-text{
    margin-top: 0;
    margin-bottom: 0;
  }
  p{
    margin: 15px;
    font-family: Microsoft YaHei;
    color: #E0E0E0;
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
  .box-btn{
    justify-content: space-around;
  }
  .box-checkmenu{
    position: absolute;
    left: 190px;
    z-index: 1;
  }
  .data-area{
    width: 268px;
    height: 42px;
    background-color: #827E7E;
    margin-right: 20px;
    margin-top: 6px;
    position: absolute;
    transform: translate(180px,32px);
  }
</style>