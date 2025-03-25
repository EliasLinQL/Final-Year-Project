<template>
  <div class="box-rToDBtn" @click="turn">
    <div class="switch" :class="{'switch-1': isDiagramVisible, 'switch-2': !isDiagramVisible}" />
    <p class="switch-p1" v-if="TextSwitchVisible && isDiagramVisible">TestResult</p>
    <p class="switch-p2" v-if="TextSwitchVisible && !isDiagramVisible">Diagram</p>
  </div>
</template>

<script setup>

/**
 * rToDBtn.vue
 *
 * This component provides a toggle button UI to switch between "Test Result" and "Diagram" views.
 *
 * Features:
 * - Animated toggle button with smooth visual transition.
 * - Displays contextual labels ("TestResult" / "Diagram") depending on the current state.
 * - Emits an event after the toggle animation completes, allowing the parent to respond accordingly.
 *
 * Emits:
 * - triggerRToD (): Emitted after the toggle completes, to notify the parent component of the state change.
 */


import {ref} from "vue";

const isDiagramVisible = ref(true);
const TextSwitchVisible = ref(true);
const emit = defineEmits("triggerRToD")

function turn() {
  isDiagramVisible.value = !isDiagramVisible.value;
  TextSwitchVisible.value = false;
  setTimeout(() => {
    TextSwitchVisible.value = true;
    emit("triggerRToD");
  }, 400);
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
