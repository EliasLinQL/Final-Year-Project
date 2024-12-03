<template>
  <div class="box-rToDBtn" @click="turn">
    <div class="switch" :class="{'switch-1': isDiagramVisible, 'switch-2': !isDiagramVisible}" />
    <p class="switch-p1" v-if="TextSwitchVisible && isDiagramVisible">Diagram</p>
    <p class="switch-p2" v-if="TextSwitchVisible && !isDiagramVisible">Relation</p>
  </div>
</template>

<script setup>
import {ref} from "vue";

const isDiagramVisible = ref(true);
const TextSwitchVisible = ref(true);
const emit = defineEmits("triggerRToD")

function turn() {
  isDiagramVisible.value = !isDiagramVisible.value;
  TextSwitchVisible.value = false; // 隐藏文字
  setTimeout(() => {
    TextSwitchVisible.value = true; // 延迟显示文字
    emit("triggerRToD");
  }, 400); // 延迟 400 毫秒
}
</script>

<style scoped>
.box-rToDBtn {
  display: flex;
  height: 72px;
  width: 180px;
  background-color: #E0E0E0;
  border-radius: 40px;
  align-items: center;
  justify-content: space-between;
  cursor: pointer;
  position: relative;
}

.switch {
  position: absolute;
  height: 64px;
  width: 64px;
  border-radius: 51%;
  background-color: #757575;
  transition: all 0.4s ease;
}

.switch-1 {
  margin-left: 6px;
}

.switch-2 {
  margin-left: 110px;
}

.box-rToDBtn:hover .switch {
  background-color: #B0B0B0;
}

p {
  margin: 18px;
  font-family: Microsoft YaHei;
  color: #707070;
  font-size: 20px;
}

.switch-p1 {
  margin-left: 80px;
}

.switch-p2 {
  margin-left: 10px;
}
</style>
