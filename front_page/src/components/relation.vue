<template>
  <div class="box-column" :class="{ box_switch: props.stateSwitchR }">
    <img
        v-if="props.selectedModel"
        :src="imageUrl"
        alt="Model Loss Image"
        class="model-image"
        @error="handleImageError"
    />
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue';

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

// 计算图片路径（从 Flask 后端接口加载）
const imageUrl = computed(() => {
  return props.selectedModel
      ? `http://localhost:5000/api/image/${props.selectedModel}_loss.png`
      : '';
});

// 处理加载失败时显示备用内容
function handleImageError(event) {
  event.target.src = ''; // 或换成默认图，如 /placeholder.png
  console.warn("⚠️ 图片加载失败:", imageUrl.value);
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
  object-fit: contain;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
}
</style>
