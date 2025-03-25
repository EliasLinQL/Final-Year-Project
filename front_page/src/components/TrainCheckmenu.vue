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

/**
 * trainCheckmenu.vue
 *
 * This component provides a dropdown-style radio menu for selecting a predefined training model set.
 *
 * Features:
 * - Displays available training presets (`trainsettings`) as radio options.
 * - Emits the selected preset object to the parent when a model is selected.
 * - Menu automatically collapses after selection.
 *
 * Props:
 * - trainsettings (Array): List of training presets, each containing `name` and `currencies`.
 *
 * Emits:
 * - triggerTrainSetCheck (Object): Emits the full selected training set object when a model is chosen.
 */


import { ref, watch} from 'vue';

const emit = defineEmits(["triggerTrainSetCheck"])
const props = defineProps({
  trainsettings: { type: Array }
});

const isMenuVisible = ref(false);
const models = ref([]);
const selectedModel = ref(null);

function toggleMenu() {
  // Switch menu display and hide status
  isMenuVisible.value = !isMenuVisible.value;
}

watch(selectedModel, (newval) => {
  if (newval) {
    isMenuVisible.value = false;
    // Find the trainset corresponding to the selected name
    const selectedItem = models.value.find(model => model.name === newval);
    if (selectedItem) {
      // Trigger events through emit and pass the corresponding trainset
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
  max-height: 200px;
  overflow-y: auto;
  overflow-x: hidden;
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
