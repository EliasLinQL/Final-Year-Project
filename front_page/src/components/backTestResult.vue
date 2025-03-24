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

// 监听模型和币种变化，更新图片路径
watch(
    () => [props.selectedModel, props.selectedCurrency],
    ([model, currency]) => {
      console.log('📌 Watch Triggered:', model, currency); // ✅ 添加日志

      if (model && currency) {
        const url = `http://localhost:5000/api/image/${model}/${currency}_Backtest.png`;
        imageUrl.value = url;
        console.log("📷 imageUrl set to:", url);
        showError.value = false;
      }
    },
    { immediate: true }
);


// 图片加载失败处理
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
