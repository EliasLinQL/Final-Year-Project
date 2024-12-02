<template>
  <div class="container">

    <model-info class="info" :class="{ toright: isTrainVisible }" @triggerTrain="goTrain"/>

    <transition name="train">
      <model-train class="train" v-if ="isTrainVisible" @triggerEndTrain="endTrain"/>
    </transition>
  </div>

</template>

<script setup>
import { ref } from 'vue';
import ModelInfo from "@/components/modelInfo.vue";
import ModelTrain from "@/components/modelTrain.vue";


const isTrainVisible = ref(false);


function goTrain() {
  isTrainVisible.value = true;
}
function endTrain() {
  isTrainVisible.value = false;
}
</script>

<style scoped>


.container {
  height: 100%;
  width: 100%;
  position: relative;
}

.info{
  transition: all 1.2s ease;
  transform: translateX(-200px);
  position: absolute;
  top: 10%;
  left: 50%;
}

.train {
  position: absolute;
  top: 10%;
  left: 50%;
  transform: translateX(100px);
  opacity: 1;
}


.toright{
  transform: translateX(-500px);
}


/* 为 model-train 组件的过渡效果 */
.train-enter-active,.train-leave-active {
  transition: all 1.2s ease;
}

.train-enter-from {
  transform: translateX(800px);
  opacity: 20%;
}
.train-leave-to {
  transform: translateX(800px);
  opacity: 0;
}




</style>
