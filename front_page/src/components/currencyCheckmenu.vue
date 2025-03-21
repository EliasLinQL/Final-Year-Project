<template>
  <div class="checkbox-menu">
    <button @click="toggleMenu" class="toggle-button">
      {{ isMenuVisible ? 'Close' : 'Open' }}
      <span class="triangle" :class="{ rotate: isMenuVisible }"></span>
    </button>

    <transition name="checkbox-fade">
      <div v-show="isMenuVisible" class="checkbox-container">
        <label v-for="(currency, index) in currencies" :key="index">
          <input
              type="radio"
              name="currencySelection"
              v-model="selectedCurrency"
              :value="currency"
          />
          {{ currency }}
        </label>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue';

const emit = defineEmits(['updateSelectedCurrency']);
const isMenuVisible = ref(false);

const currencies = ref([
  'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT',
  'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'TRXUSDT'
]);

const selectedCurrency = ref(null);

function toggleMenu() {
  isMenuVisible.value = !isMenuVisible.value;
}

watch(selectedCurrency, (newVal) => {
  emit('updateSelectedCurrency', newVal); // 发出给父组件
});
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
  overflow: auto;
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

.checkbox-fade-enter-from,
.checkbox-fade-leave-to {
  max-height: 0;
  opacity: 80%;
}

.checkbox-fade-enter-to {
  max-height: 200px;
  opacity: 1;
}
</style>
