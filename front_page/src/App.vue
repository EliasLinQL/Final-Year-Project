<template>
  <div class="container">

    <model-info class="info" :class="{ calltrain: isTrainVisible, callgraph: isGraphVisible }" @triggerTrain="goTrain" @triggerGraph="showGraph"/>

    <transition name="train">
      <model-train class="train" v-if ="isTrainVisible" @triggerEndTrain="endTrain"/>
    </transition>

    <transition name="showGraph-relation">
      <relation class="relation" v-if="isGraphVisible"/>
    </transition>

    <transition name="showGraph-btc">
      <back-test-control class="btc" v-if="isGraphVisible"/>
    </transition>

    <transition name="showGraph-diagram">
      <diagram class="diagram" v-if="isGraphVisible"/>
    </transition>

  </div>

</template>

<script setup>
import { ref } from 'vue';
import ModelInfo from "@/components/modelInfo.vue";
import ModelTrain from "@/components/modelTrain.vue";
import Relation from "@/components/relation.vue";
import BackTestControl from "@/components/backTestControl.vue";
import Diagram from "@/components/diagram.vue";


const isTrainVisible = ref(false);
const isGraphVisible = ref(false);

function goTrain() {
  isTrainVisible.value = true;
  if (isGraphVisible.value) {
    isGraphVisible.value = false;
  }
}
function endTrain() {
  isTrainVisible.value = false;
}
function showGraph() {
  isGraphVisible.value = true;
  if (isTrainVisible.value) {
    isTrainVisible.value = false;
  }
}
</script>

<style scoped>


  .container {
    height: 100%;
    width: 100%;
    position: relative;
  }

  /*******************/
  /* modelInfo组件相关 */
  .info{
    transition: all 1.4s ease;
    transform: translateX(-200px);
    position: absolute;
    top: 10%;
    left: 50%;
  }

  .callgraph{
    transform: translate(-48.9vw,-9vh);
  }

  /********************/
  /* modelTrain组件相关 */
  .train {
    position: absolute;
    top: 10%;
    left: 50%;
    transform: translateX(100px);
    opacity: 1;
  }

  .calltrain{
    transform: translateX(-500px);
  }

  /* 为 model-train 组件的过渡效果 */
  .train-enter-active,.train-leave-active {
    transition: all 1.4s ease;
  }

  .train-enter-from {
    transform: translateX(1000px);
    opacity: 1;
  }
  .train-leave-to {
    transform: translateX(1000px);
    opacity: 1;
  }

  /*************************/
  /* backTestControl组件相关 */
  .btc{
    position: absolute;
    top: 50%;
    left: 0.5%;
  }

  .showGraph-btc-enter-active,.showGraph-btc-leave-active {
    transition: all 1.4s ease;
  }
  .showGraph-btc-enter-from {
    transform: translate(-110%,200%);
    opacity: 1;
  }
  .showGraph-btc-leave-to {
    transform: translate(-110%,200%);
    opacity: 1;
  }

  /******************/
  /* relation组件相关 */
  .relation{
    position: absolute;
    top: 0.8%;
    left: 25%;
  }

  .showGraph-relation-enter-active, .showGraph-relation-leave-active {
    transition: all 1.2s ease;
  }
  .showGraph-relation-enter-from{
    transform: translate(40%,-120%);
    opacity: 1;
  }
  .showGraph-relation-leave-to{
    transform: translate(40%,-120%);
    opacity: 1;
  }


  /*****************/
  /* diagram组件相关 */
  .diagram{
    position: absolute;
    top: 69.5%;
    left: 25%;
  }

  .showGraph-diagram-enter-active, .showGraph-diagram-leave-active {
    transition: all 1.2s ease;
  }
  .showGraph-diagram-enter-from{
    transform: translateY(120%);
    opacity: 1;
  }
  .showGraph-diagram-leave-to{
    transform: translateY(120%);
    opacity: 1;
  }

</style>
