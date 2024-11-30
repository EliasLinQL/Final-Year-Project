<template>
<!--书写html结构-->
  <div class="container">
    <model-info/>
    <model-train/>

  </div>
</template>

<script setup>
import {ref} from 'vue';
import ModelInfo from "@/components/modelInfo.vue";
import ModelTrain from "@/components/modelTrain.vue";


</script>

<style scoped>
.container {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  height: 100vh;
  position: relative;
}
</style>


