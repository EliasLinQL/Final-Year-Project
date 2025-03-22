<template>
  <div class="box-column">
    <div class="box-row">
      <p class="p1">Model Info</p>
    </div>
    <div class="box-row">
      <p class="p2">Existing Models</p>
      <div class="box-checkmenu">
        <model-checkmenu @updateSelectedModel="handleSelectedModel" />
      </div>
    </div>

    <div class="box-row box_loss">
      <img
          v-if="selectedModel"
          :src="imageUrl"
          :alt="`${selectedModel} Loss Image`"
          class="loss-image"
          @click="showModal = true"
          @error="handleImageError"
      />
    </div>

    <!-- 放大图模态框，确保显示在最上层 -->
    <Teleport to="body">
      <div v-if="showModal" class="modal-overlay" @click="closeModal">
        <img :src="imageUrl" class="modal-image" />
      </div>
    </Teleport>


    <div class="box-row box-btn">
      <button class="btn" @click="goTrain">Go Train</button>
      <button class="btn" @click="showGraph">Add</button>
    </div>
  </div>
</template>

<script setup>
import ModelCheckmenu from "@/components/modelCheckmenu.vue";
import { ref, computed } from "vue";

const emit = defineEmits(['triggerTrain', 'triggerGraph', 'triggerModelToApp']);

const selectedModel = ref(null);
const showModal = ref(false);

const imageUrl = computed(() => {
  return selectedModel.value
      ? `http://localhost:5000/api/image/${selectedModel.value}_loss.png`
      : '';
});

function handleSelectedModel(modelName) {
  selectedModel.value = modelName;
  emit('triggerModelToApp', modelName);
}

function handleImageError(event) {
  event.target.src = '';
  console.warn("⚠️ 模型图片加载失败:", imageUrl.value);
}

function goTrain() {
  emit("triggerTrain");
}
function showGraph() {
  emit("triggerGraph");
}

function closeModal() {
  showModal.value = false;
}
</script>

<style scoped>
.box-column {
  display: flex;
  flex-direction: column;
  background-color: #757575;
  width: 400px;
  height: 400px;
  margin: 10px;
  padding: 0;
  position: relative;
  border-radius: 18px;
}
.box-row {
  display: flex;
  flex-direction: row;
  margin: 0;
  margin-left: 10px;
  margin-top: 5px;
}
p {
  margin: 15px;
  font-family: Microsoft YaHei;
  color: #E0E0E0;
}
.p1 {
  font-size: 26px;
}
.p2 {
  font-size: 18px;
}
.box_loss {
  height: 180px;
  width: 75%;
  background-color: #9c9898;
  border-radius: 10px;
  justify-content: center;
  align-items: center;
  overflow: hidden;
  margin-left:  50px;
}
.loss-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  border-radius: 10px;
  cursor: pointer;
  transition: transform 0.2s ease;
}
.loss-image:hover {
  transform: scale(1.02);
}
/* ✅ 模态框放大图样式 */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  z-index: 99;
  background-color: rgba(0, 0, 0, 0.8);
  display: flex;
  justify-content: center;
  align-items: center;
}
.modal-image {
  max-width: 90vw;
  max-height: 90vh;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(255, 255, 255, 0.3);
  transition: all 0.3s ease;
}
.box-btn {
  justify-content: space-around;
}
.btn {
  margin-top: 20px;
  margin-bottom: 5px;
  width: 158px;
  height: 37px;
  background-color: #3A708D;
  border: 2px solid #346884;
  border-radius: 12px;
  font-size: 20px;
  color: #E0E0E0;
  box-shadow: 1px 2px 4px #346884;
  transition: all 0.2s ease;
}
.btn:hover {
  background-color: #286487;
  opacity: 0.8;
}
.btn:active {
  background-color: #246893;
}
.box-checkmenu {
  position: absolute;
  left: 180px;
  z-index: 1;
}
</style>
