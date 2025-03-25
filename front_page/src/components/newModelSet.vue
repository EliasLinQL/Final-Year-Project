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

/**
 * newModelSet.vue
 *
 * This component allows users to define and create a new model preset by:
 * - Entering a model set name.
 * - Selecting target currencies and a date range (via external components).
 * - Validating currency listing dates against user-selected range.
 * - Sending a backend request to fetch and prepare crypto data for training.
 *
 * Features:
 * - Displays input fields for model set name and a button to trigger currency selection.
 * - Performs pre-checks to ensure date ranges are valid for all selected currencies.
 * - Handles backend communication to trigger data fetching and processing.
 * - Emits events for creation, cancellation, and currency selection visibility.
 *
 * Props:
 * - message (Boolean): Controls visibility sync with external toggle.
 * - currencies (Array): List of selected currencies with associated date ranges.
 *
 * Emits:
 * - triggerCreateNewModel (ModelSet): Emits the new model set object after successful creation.
 * - triggerCancelCreate (): Cancels the model creation process.
 * - triggerCurrencyList (): Toggles the visibility of the currency selection list.
 */


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

async function createModel() {
  if (!props.currencies || props.currencies.length === 0) {
    alert("⚠️ Please select currencies before creating model.");
    return;
  }

  const start_date = props.currencies[0].dates[0];
  const end_date = props.currencies[0].dates[1];
  const symbols = props.currencies.map(item => item.name);

  if (!start_date || !end_date) {
    alert("⚠️ Please select a valid time range before creating model.");
    return;
  }

  // Verify the launch time of each coin
  const earlySymbols = [];

  for (const symbol of symbols) {
    try {
      const url = `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=1d&limit=1&startTime=0`;
      const res = await fetch(url);
      const data = await res.json();

      if (!Array.isArray(data) || data.length === 0) continue;

      const earliestTimestamp = data[0][0];
      const earliestDate = new Date(earliestTimestamp);
      const selectedStart = new Date(start_date);

      if (selectedStart < earliestDate) {
        earlySymbols.push({
          symbol,
          earliestDateStr: earliestDate.toISOString().split('T')[0]
        });
      }

    } catch (err) {
      console.error(`⚠️ Failed to fetch start time for ${symbol}:`, err);
    }
  }

  // If there are any currencies that are launched earlier than the currency's launch time, prompt the user and interrupt
  if (earlySymbols.length > 0) {
    let message = "❌ The following currencies have a start date earlier than their listing date:\n\n";
    earlySymbols.forEach(item => {
      message += `- ${item.symbol}: Earliest available date is ${item.earliestDateStr}\n`;
    });
    alert(message);
    return;
  }

  // All coins have passed time verification → initiate backend request
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
        if (!response.ok) throw new Error(`HTTP Error! Status: ${response.status}`);
        return response.json();
      })
      .then(data => {
        if (data.status === "success") {
          alert("✅ Python backend executed successfully! Model will now be created.");
          const newModelSet = new ModelSet(modelSetName.value, props.currencies);
          emit("triggerCreateNewModel", newModelSet);
          alert(" Model created successfully!");
        } else {
          alert(`⚠️ Data fetch completed, but data processing failed: ${data.message}`);
        }
      })

      .catch(error => {
        console.error("🔥 Frontend Error:", error);
        alert("❌ Failed to call backend. Please check the server.");
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
