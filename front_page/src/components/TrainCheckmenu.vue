<template>
  <div class="checkbox-menu">
    <button @click="toggleMenu" class="toggle-button">
      {{ isMenuVisible ? 'Close' : 'Open' }}
      <span class="triangle" :class="{ rotate: isMenuVisible }"></span>
    </button>

    <transition name="checkbox-fade">
      <div v-show="isMenuVisible" class="checkbox-container">
        <label v-for="(model, index) in models" :key="index">
          <input type="radio"
                 v-model="selectedModel"
                 :value="model.name" />
          {{ model.name }}
        </label>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue';

const emit = defineEmits(["triggerTrainSetCheck"])
const props = defineProps({
  trainsettings: { type: Array }
});

const isMenuVisible = ref(false);
const models = ref([]);
const selectedModel = ref(null);

function toggleMenu() {
  // 切换选单显示和隐藏状态
  isMenuVisible.value = !isMenuVisible.value;
}

watch(selectedModel, (newval) => {
  if (newval) {
    isMenuVisible.value = false;
    // 找到选中的 name 对应的 trainset
    const selectedItem = models.value.find(model => model.name === newval);
    console.log(selectedItem);
    if (selectedItem) {
      // 通过 emit 触发事件并传递对应的 trainset
      emit("triggerTrainSetCheck", selectedItem);
    }
  }
})

watch(() => props.trainsettings, (newval) => {
  if (newval && newval.length) {
    models.value = newval.map(model => ({
      name: model.name,
      currencies: model.currencies,
    }));
  }
}, {deep: true,immediate: true});
</script>

<style scoped>
.checkbox-menu {
  margin-top: 4px;
  width: 160px;
  background-color: #f4f4f4;
  padding: 2px;
  border: 1px solid #ccc;
  border-radius: 16px;
}

.toggle-button {
  background-color: #BFBBB6;
  color: white;
  border: none;
  padding: 10px;
  width: 100%;
  cursor: pointer;
  border-radius: 12px;
  text-align: center;
  display: flex;
  justify-content: space-between;
  align-items: center;
  position: relative;
}

.triangle {
  width: 0;
  height: 0;
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: 5px solid white;
  transition: transform 0.3s ease-in-out;
}

.triangle.rotate {
  transform: rotate(180deg);
}

.checkbox-container {
  margin-top: 10px;
  padding-left: 10px;
  overflow: hidden;
  max-height: 200px;
  opacity: 1;
}

.checkbox-container label {
  display: block;
  margin-bottom: 10px;
}

.checkbox-container input {
  margin-right: 10px;
}

.checkbox-fade-enter-active,
.checkbox-fade-leave-active {
  transition: all 0.4s linear;
}

.checkbox-fade-enter-from, .checkbox-fade-leave-to {
  max-height: 0;
  opacity: 80%;
}

.checkbox-fade-enter-to {
  max-height: 200px;
  opacity: 1;
}
</style>
