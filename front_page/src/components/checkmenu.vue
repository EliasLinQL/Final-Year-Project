<template>
  <div class="checkbox-menu">
    <button @click="toggleMenu" class="toggle-button">
      {{ isMenuVisible ? 'Back' : 'Open' }}
      <span class="triangle" :class="{ rotate: isMenuVisible }"></span>
    </button>

    <!-- 使用 transition 包裹组件，使其具有过渡动画 -->
    <transition name="checkbox-fade">
      <div v-show="isMenuVisible" class="checkbox-container">
        <label>
          <input type="checkbox" /> Model 1
        </label>
        <label>
          <input type="checkbox" /> Model 2
        </label>
        <label>
          <input type="checkbox" /> Model 3
        </label>
        <label>
          <input type="checkbox" /> Model 4
        </label>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref } from 'vue';

const isMenuVisible = ref(false);

function toggleMenu() {
  // 切换选单显示和隐藏状态
  isMenuVisible.value = !isMenuVisible.value;
}
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
  transform: rotate(-90deg);
}

.triangle.rotate {
  transform: rotate(90deg);
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

/* 添加过渡动画 */
.checkbox-fade-enter-active,
.checkbox-fade-leave-active {
  transition: all 0.4s linear;
}

.checkbox-fade-enter-from, .checkbox-fade-leave-to {
  max-height: 0;
  opacity: 80%;
}

/* 用于展开时使 checkbox-container 最大高度为实际内容的高度 */
.checkbox-fade-enter-to {
  max-height: 200px; /* 可以根据实际情况设置 */
  opacity: 1;
}
</style>
