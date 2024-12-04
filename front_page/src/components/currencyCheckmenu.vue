<template>
  <div class="checkbox-menu">
    <button @click="toggleMenu" class="toggle-button">
      {{ isMenuVisible ? 'Close' : 'Open' }}
      <span class="triangle" :class="{ rotate: isMenuVisible }"></span>
    </button>

    <transition name="checkbox-fade">
      <div v-show="isMenuVisible" class="checkbox-container">
        <div v-for="(currency, index) in currencies" :key="index">
          <label>
            <input type="checkbox" v-model="currency.selected" />
            <div class="icon" />
            {{ currency.name }}
            <div class="timearea">
              <p>ST:{{ currency.startTime }}</p>
              <p>ET:{{ currency.endTime }}</p>
            </div>
            <button @click="toggleTimeSelection(index)" class="time-toggle-button">T</button>
          </label>
        </div>
      </div>
    </transition>
    <div class="box-timeselect" v-if="isTimeSelectVisible">
      <div class="time-selection">
        <input type="datetime-local" v-model="currencies[tid].startTime" />
        <input type="datetime-local" v-model="currencies[tid].endTime" />
      </div>
    </div>
  </div>
</template>



<script setup>
import {ref} from 'vue';

// 定义选项数据
const currencies = ref([
  { name: 'Currency1', selected: false, startTime: '', endTime: '' },
  { name: 'Currency2', selected: false, startTime: '', endTime: '' },
  { name: 'Currency3', selected: false, startTime: '', endTime: '' },
  { name: 'Currency4', selected: false, startTime: '', endTime: '' },
  { name: 'Currency6', selected: false, startTime: '', endTime: '' },
  { name: 'Currency7', selected: false, startTime: '', endTime: '' },
  { name: 'Currency5', selected: false, startTime: '', endTime: '' },
  { name: 'Currency8', selected: false, startTime: '', endTime: '' },
  { name: 'Currency9', selected: false, startTime: '', endTime: '' },
  { name: 'Currency10', selected: false, startTime: '', endTime: '' },
  { name: 'Currency11', selected: false, startTime: '', endTime: '' },
  { name: 'Currency12', selected: false, startTime: '', endTime: '' },
  { name: 'Currency13', selected: false, startTime: '', endTime: '' },
  { name: 'Currency14', selected: false, startTime: '', endTime: '' },
  { name: 'Currency15', selected: false, startTime: '', endTime: '' },

]);

const isMenuVisible = ref(false);
const tid = ref(-1);
const isTimeSelectVisible = ref(false);

function toggleMenu() {
  isMenuVisible.value = !isMenuVisible.value;
  if (isTimeSelectVisible.value) {
    isTimeSelectVisible.value = false;
    tid.value = -1;
  }
}

function toggleTimeSelection(index) {
  if (tid.value != index){
    isTimeSelectVisible.value = true;
    tid.value = index;



  }else {
    isTimeSelectVisible.value = false;
    tid.value = -1;
  }
}


</script>


<style scoped>
.checkbox-menu {
  margin-top: 4px;
  width: 320px;
  background-color: #f4f4f4;
  padding: 2px;
  border: 1px solid #ccc;
  border-radius: 16px;
}
p{
  margin: 0;
  font-size: 13px;
  white-space: nowrap;
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
  max-height: 180px;
  opacity: 1;
}

.checkbox-container label {
  display: flex;
  align-items: center;
  height: 40px;
  margin-bottom: 10px;
}

.checkbox-container input {
  margin-right: 2px;
}

.icon {
  height: 24px;
  width: 24px;
  border-radius: 50%;
  background-color: #838383;
  margin-right: 5px;
}
.timearea{
  margin-top: 6px;
  margin-left: 2px;
  height: 100%;
  width: 105px;
  display: flex;
  flex-direction: column;
  position: relative;
}
.time-toggle-button {
  margin-left: 40px;
  padding: 5px;
  background-color: #8398E0;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}
.box-timeselect{
  position: absolute;
  top: 5px;
  left: 330px;
  width: 165px;
  height: auto;
}
.time-selection {
  height: 100%;
  width: 100%;
}

.time-selection input {
  margin-right: 10px;
  padding: 5px;
  border-radius: 5px;
}

/* 添加过渡动画 */
.checkbox-fade-enter-active, .checkbox-fade-leave-active {
  transition: all 0.4s linear;
}

.checkbox-fade-enter-from, .checkbox-fade-leave-to {
  max-height: 0;
  opacity: 80%;
}

.checkbox-fade-enter-to {
  max-height: 150px;
  opacity: 1;
}
</style>
