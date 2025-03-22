<template>
  <div class="box-column">
    <!-- 小图 -->
    <img
        v-if="imageUrl"
        :src="imageUrl"
        alt="Prediction Result"
        class="model-image"
        @click="showModal = true"
        @error="onImageError"
    />
    <p v-if="showError" class="msg">❌ 图像加载失败或尚未生成！</p>

    <!-- ✅ 放大图（使用 Teleport 保证最顶层展示） -->
    <Teleport to="body">
      <div v-if="showModal" class="modal-overlay" @click="closeModal">
        <img :src="imageUrl" class="modal-image" />
      </div>
    </Teleport>
  </div>
</template>


<script setup>
import { ref, watch } from 'vue';

const props = defineProps({
  selectedCurrency: String,
  selectedModel: String,
});

const imageUrl = ref('');
const showError = ref(false);
const showModal = ref(false); // 控制放大图展示

watch(
    [() => props.selectedModel, () => props.selectedCurrency],
    ([model, currency]) => {
      if (model && currency) {
        imageUrl.value = `http://localhost:5000/api/image/${model}/${currency}`;
        showError.value = false;
        console.log('📊 Diagram 请求图片路径：', imageUrl.value);
      }
    }
);

function onImageError() {
  showError.value = true;
  imageUrl.value = '';
  console.warn('❌ 图片加载失败');
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
  height: 228px;
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

/* 放大模态图样式 */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background-color: rgba(0, 0, 0, 0.8);
  z-index: 9999; /* 保证高于所有其他内容 */
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

