<template>
  <div id="Diagram" class="box-column" :class="{ box_switch: props.stateSwitchD }">
    <img
        v-if="imageUrl"
        :src="imageUrl"
        alt="Prediction Result"
        class="model-image"
        @error="onImageError"
    />
    <p v-if="showError" class="msg">❌ 图像加载失败或尚未生成！</p>
  </div>
</template>

<script setup>
import {ref, watch} from 'vue';

const props = defineProps({
  stateSwitchD: Boolean,
  selectedCurrency: String,
  selectedModel: String,
});

const imageUrl = ref('');
const showError = ref(false);

// 👇 监听模型名 + 货币名，拼接请求路径
watch(
    [() => props.selectedModel, () => props.selectedCurrency],
    ([model, currency]) => {
      if (model && currency) {
        imageUrl.value = `http://localhost:5000/api/image/${model}/${currency}`;
        showError.value = false; // 重置错误提示
        console.log('📊 Diagram 请求图片路径：', imageUrl.value);
      }
    }
);

// 👇 图像加载失败时回调
function onImageError() {
  showError.value = true;
  imageUrl.value = '';
  console.warn('❌ 图片加载失败，可能未生成对应图像文件。');
}
</script>

<style scoped>
.box-column {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  background-color: #757575;
  width: 540px;
  height: 270px;
  margin: 10px;
  padding: 0;
  position: relative;
  border-radius: 18px;
  transition: all 0.3s ease;
}

.box_switch {
  width: 1260px;
  height: 630px;
}

.model-image {
  width: 95%;
  height: auto;
  object-fit: contain;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
}

.msg {
  font-size: 16px;
  color: #ffcdd2;
  margin-top: 20px;
  font-weight: bold;
}
</style>
