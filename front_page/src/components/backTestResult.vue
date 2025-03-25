<template>
  <div class="box-column" :class="{ box_switch: props.stateSwitchR }">
    <img
        v-if="imageUrl"
        :src="imageUrl"
        alt="Backtest Image"
        class="model-image"
        @error="handleImageError"
    />
    <p v-if="showError" class="msg">❌ Failed to load image or image not found!</p>
  </div>
</template>

<script setup>

/**
 * backTestResult.vue
 *
 * This component displays the backtest result image for a selected model and currency.
 *
 * Features:
 * - Dynamically loads and displays an image based on the selected model and currency.
 * - Automatically updates when either `selectedModel` or `selectedCurrency` changes.
 * - Handles image loading errors and displays an error message if the image is missing or fails to load.
 * - Responsive styling: adjusts layout when `stateSwitchR` is enabled.
 *
 * Props:
 * - stateSwitchR (Boolean): Triggers a compact layout mode.
 * - selectedModel (String): The currently selected model name.
 * - selectedCurrency (String): The currently selected cryptocurrency.
 */


import {  ref, watch } from 'vue';

const props = defineProps({
  stateSwitchR: {
    type: Boolean,
    required: true
  },
  selectedModel: {
    type: String,
    default: ''
  },
  selectedCurrency: {
    type: String,
    default: ''
  }
});

const imageUrl = ref('');
const showError = ref(false);

// Monitor model and currency changes, update image path
watch(
    () => [props.selectedModel, props.selectedCurrency],
    ([model, currency]) => {

      if (model && currency) {
        const url = `http://localhost:5000/api/image/${model}/${currency}_Backtest.png`;
        imageUrl.value = url;
        showError.value = false;
      }
    },
    { immediate: true }
);


// Image loading failure handling
function handleImageError(event) {
  event.target.src = '';
  showError.value = true;
  console.warn("⚠️ Failed to load image:", imageUrl.value);
}
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
  width: 450px;
  height: 226px;
}

.model-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
}

.msg {
  font-size: 16px;
  color: #ffcdd2;
  margin-top: 20px;
  text-align: center;
}
</style>
