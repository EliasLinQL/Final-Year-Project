<template>
  <div class="box-column">
    <div class="box-row">
      <p class="p1">New Model Settings</p>
    </div>
    <div class="box-row"></div>
    <div class="box-row">
      <p class="p2">ModelSet<br>Name</p>
      <div class="data-area-name">
        <input class="nameinput" type="text" v-model="modelSetName" />
      </div>
    </div>
    <div class="box-row">
      <p class="p2">Target<br>Currency</p>
      <div class="box-currencychoose">
        <button class="currencylist" @click="showCurrencyList">
          {{ isCurrencyListVisible ? 'Close' : 'Show' }}
        </button>
      </div>
    </div>
    <div class="box-row box-btn">
      <button class="btn" @click="createModel">Create Model</button>
      <button class="btn" @click="Cancel">Cancel</button>
    </div>
  </div>
</template>

<script setup>
import {ref, watch} from "vue";
import ModelSet from "@/bean/modelSet.js";

const modelSetName = ref("");

const props = defineProps({
  message: false,
  currencies: {type: Array}
})

const emit = defineEmits(["triggerCreateNewModel","triggerCancelCreate","triggerCurrencyList"]);
const isCurrencyListVisible = ref(false);

function showCurrencyList() {
  isCurrencyListVisible.value = !isCurrencyListVisible.value;
  emit("triggerCurrencyList");
}

function createModel() {
  const newModelSet = new ModelSet(modelSetName.value, props.currencies);
  emit("triggerCreateNewModel", newModelSet);

  // 如果 currencies 为空则提示
  if (!props.currencies || props.currencies.length === 0) {
    alert("Please select currencies before creating model.");
    return;
  }

  const start_date = props.currencies[0].dates[0];
  const end_date = props.currencies[0].dates[1];
  const symbols = props.currencies.map(item => item.name);

  const requestData = { start_date, end_date, symbols };

  fetch("http://localhost:5000/api/fetch_crypto_data", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(requestData),
    mode: "cors"
  })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP Error! Status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        console.log("✅ Success:", data);
        if (data.processing_status === "success") {
          alert("Data_Collection.py call successfully! Data processing completed.");
        } else {
          alert(`Data_Collection.py call successfully, but data processing failed: ${data.processing_message}`);
        }
      })
      .catch(error => {
        console.error("🔥 Frontend Error:", error);
        alert("Failed to call Data_Collection.py, Check the console.");
      });
}


function Cancel() {
  emit("triggerCancelCreate");
}

watch(props, (newval) => {
  isCurrencyListVisible.value = newval.message;
})
</script>

<style scoped>
.box-column {
  background-color: #757575;
  width: 520px;
  height: 320px;
  margin: 10px;
  border-radius: 18px;
  position: relative;
}

.box-row {
  display: flex;
  flex-direction: row;
  margin: 0;
  margin-left: 10px;
  margin-top: 5px;
}

.p1 {
  font-size: 26px;
  margin-top: 20px;
  margin-bottom: 20px;
}

.p2 {
  font-size: 18px;
  margin-top: 10px;
  margin-bottom: 10px;
}

p {
  margin: 15px;
  font-family: Microsoft YaHei;
  color: #E0E0E0;
}

.data-area-name {
  width: 268px;
  height: 44px;
  background-color: #827E7E;
  margin-right: 20px;
  position: absolute;
  transform: translate(150px, 12px);
}

.nameinput {
  margin-left: 2px;
  margin-top: 1.4px;
  width: 96%;
  height: 80%;
  background-color: #9c9898;
  border: 2px solid #6e6b6b;
  border-radius: 6px;
}


.box-currencychoose{
  margin-right: 20px;
  position: absolute;
  transform: translate(150px, 12px);
  z-index: 1;
}

.btn {
  margin-top: 20px;
  margin-bottom: 5px;
  width: 158px;
  height: 38px;
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

.box-btn {
  justify-content: space-around;
}

.currencylist {
  width: 68px;
  height: 52px;
  background-color: rgba(86, 182, 157, 0.6);
  border: 2px solid rgba(71, 152, 130, 0.6);
  box-shadow: 1px 2px 4px rgba(84, 198, 195, 0.6);
  border-radius: 24%;
  margin-right: 20px;
  position: absolute;
  transform: translate(0px, -6px);
  transition: all 0.2s ease;
  font-size: 20px;
  color: #c1d0dd;
}

.currencylist:hover {
  background-color: rgba(102, 198, 175, 0.6);
  opacity: 0.8;
}

.currencylist:active {
  background-color: rgba(118, 214, 190, 0.6);
}
</style>
