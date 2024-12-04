<template>
  <div class="box-column">
    <div class="box-row">
      <p class="p1">Train Details</p>
    </div>
    <div class="box-row model-set">
      <p class="p2 model-set">Model Settings</p>
      <model-checkmenu class="check-part"/>
      <button class="add-btn" @click="createModel">
        +
      </button>
    </div>
    <div class="box-row">
      <p class="p2">ModelSetName</p>
      <div class="data-area-name">

      </div>
    </div>
    <div class="box-row">
      <p class="p2">PreSupposed<br>Details</p>
      <div class="detail" @click="callDetail">
        <div :class="['icon', { 'rotate-icon': isIconRotated }]"></div>
      </div>
    </div>
    <div class="box-row box-btn">
      <button class="btn" @click="endTrain">End Training</button>
      <button class="btn" @click="startTrain">Start Train</button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import ModelCheckmenu from "@/components/modelCheckmenu.vue";

const isIconRotated = ref(false);
// 定义子组件触发父组件事件
const emit = defineEmits(["triggerEndTrain", "triggerStartTrain","triggerDetails","triggerGoCreate"]);

// 通知父组件触发对应事件
function endTrain() {
  emit("triggerEndTrain");
}
function startTrain() {
  emit("triggerStartTrain");
}
function callDetail() {
  isIconRotated.value = !isIconRotated.value;
  emit("triggerDetail");
}
function createModel(){
  emit("triggerGoCreate");
}

</script>

<style scoped>
.box-column {
  background-color: #757575;
  width: 520px;
  height: 400px;
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
  margin-top: 24px;
}

.check-part {
  position: absolute;
  left: 36%;
  top: 18%;
  z-index: 1;
}

.add-btn {
  position: absolute;
  width: 42px;
  height: 42px;
  border-radius: 16px;
  border: 2px solid #A0B5B1;
  box-shadow: 1px 2px 2px #86a5a5;
  background-color: #A0B5B1;
  top: 19%;
  left: 70%;
  font-size: 36px;
  font-weight: bold;
  color: #386698;
  line-height: 1px;
}

.add-btn:hover {
  background-color: #8DB2AB;
  opacity: 0.8;
}

.add-btn:active {
  background-color: #7DAFA5;
}

.model-set {
  margin-top: 0;
  margin-bottom: 10px;
}

p {
  margin: 15px;
  font-family: Microsoft YaHei;
  color: #E0E0E0;
}

.btn {
  margin-top: 48px;
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

.data-area-name {
  width: 268px;
  height: 44px;
  background-color: #827E7E;
  margin-right: 20px;
  position: absolute;
  transform: translate(180px, 19px);
}

.detail {
  width: 50px;
  height: 52px;
  background-color: rgba(86, 182, 157, 0.6);
  border-radius: 36%;
  margin-right: 20px;
  position: absolute;
  transform: translate(180px, 24px);
}

.icon {
  position: absolute;
  top: 15px;
  left: 13px;
  width: 1px;
  height: 2px;
  border-left: 12px solid transparent;
  border-right: 12px solid transparent;
  border-bottom: 20px solid #9bd5d5;
  transform-origin: center;
  transition: transform 0.3s ease;
  transform: rotate(-90deg);
}

.rotate-icon {
  transform: rotate(90deg);
}
</style>
