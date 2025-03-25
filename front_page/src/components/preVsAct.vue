<template>
  <div class="box-column">
    <img
        v-if="imageUrl"
        :src="imageUrl"
        alt="Prediction Result"
        class="model-image"
        @click="showModal = true"
        @error="onImageError"
    />
    <p v-if="showError" class="msg">❌ Image loading failed or not yet generated!</p>

    <Teleport to="body">
      <div v-if="showModal" class="modal-overlay" @click="closeModal">
        <img :src="imageUrl" class="modal-image" />
      </div>
    </Teleport>
  </div>
</template>


<script setup>

/**
 * preVsAct.vue
 *
 * This component displays the predicted vs actual price chart image for a selected currency and model.
 *
 * Features:
 * - Dynamically constructs and loads an image URL based on selected model and currency.
 * - Provides graceful error handling when the image is missing or fails to load.
 * - Allows users to click on the image to view a larger version in a modal overlay.
 *
 * Props:
 * - selectedCurrency (String): The currency symbol to visualize (e.g., BTCUSDT).
 * - selectedModel (String): The name of the trained model used for prediction.
 */


import { ref, watch } from 'vue';

const props = defineProps({
  selectedCurrency: String,
  selectedModel: String,
});

const imageUrl = ref('');
const showError = ref(false);
const showModal = ref(false);

watch(
    [() => props.selectedModel, () => props.selectedCurrency],
    ([model, currency]) => {
      if (model && currency) {
        imageUrl.value = `http://localhost:5000/api/image/${model}/${currency}_Actual_vs_Predicted.png`;
        showError.value = false;
      }
    }
);

function onImageError() {
  showError.value = true;
  imageUrl.value = '';
  console.warn('❌ Image loading failed!');
}

function closeModal() {
  showModal.value = false;
}
</script>


<style scoped>
.box-column {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  background-color: #757575;
  width: 450px;
  height: 226px;
  margin: 10px;
  padding: 0;
  position: relative;
  border-radius: 18px;
  transition: all 0.3s ease;
}

.model-image {
  width: 100%;
  height: auto;
  object-fit: contain;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
  cursor: pointer;
  transition: transform 0.2s ease;
}
.model-image:hover {
  transform: scale(1.02);
}

.msg {
  font-size: 16px;
  color: #ffcdd2;
  margin-top: 20px;
  font-weight: bold;
}

.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background-color: rgba(0, 0, 0, 0.8);
  z-index: 999;
  display: flex;
  justify-content: center;
  align-items: center;
}

.modal-image {
  max-width: 90vw;
  max-height: 90vh;
  border-radius: 16px;
  box-shadow: 0 4px 12px rgba(255, 255, 255, 0.3);
  transition: all 0.3s ease;
}
</style>

