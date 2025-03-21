<template>
  <div class="box-column" :class="{ box_switch: props.stateSwitchR }">
    <img
        v-if="props.selectedModel"
        :src="`/public/${props.selectedModel}_loss.png`"
        alt="Model Loss Image"
        class="model-image"
    />
  </div>
</template>

<script setup>
const props = defineProps({
  stateSwitchR: {
    type: Boolean,
    required: true
  },
  selectedModel: {
    type: String,
    default: ''
  }
});
</script>

<style scoped>
.box-column {
  display: flex;
  flex-direction: column;
  background-color: #757575;
  width: 1260px;
  height: 630px;
  margin: 10px;
  padding: 0;
  position: relative;
  border-radius: 18px;
}

.box_switch {
  width: 540px;
  height: 270px;
}

p {
  margin: 15px;
  font-family: Microsoft YaHei;
  color: #E0E0E0;
}

.model-image {
  width: 100%;
  height: 100%;
  object-fit: contain; /* 可改为 cover 或 fill */
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
}

</style>
