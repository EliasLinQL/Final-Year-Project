<template>
  <div class="box-column">
    <div class="box-0">
      <p class="p1">Currency List</p>
      <div class="alltimeset">
        <DatePicker
            v-model="allDates"
            format="yyyy-MM-dd"
            range
            :teleport="true"
            style="width: 380px; height: 24px; font-size: 12px;"
        />
      </div>
      <button class="currency-submit" @click="submit">+</button>
      <button class="currency-cancel" @click="cancel">X</button>
    </div>
      <div class="box-1">
        <div
            class="box-currency"
            v-for="(currencyObj, index) in currencies"
            :key="index"
        >
          <input
              type="checkbox"
              class="check"
              v-model="currencyObj.selected"
          />

          <div class="icon">
            <img :src="currencyObj.currency.getIcon()" class="currency-icon" />
          </div>

          <div class="name">
            <p class="p2">{{ currencyObj.currency.getName() }}</p>
          </div>

        </div>
      </div>
  </div>
</template>

<script setup>

/**
 * currencyList.vue
 *
 * This component provides a UI for selecting multiple cryptocurrencies along with a unified date range.
 *
 * Features:
 * - Displays a list of predefined cryptocurrencies, each with a checkbox and an associated icon.
 * - Allows the user to select a start and end date using a date range picker (vue-datepicker).
 * - Saves and restores user selections from localStorage.
 * - Emits the selected currencies and date range to the parent component on submit.
 * - Provides cancel functionality to abort the selection process.
 *
 * Props: (none)
 *
 * Emits:
 * - triggerSubmit (CurrencySet[]): Emits the selected currencies with their date range.
 * - triggerCancel (): Emits when the user clicks the cancel button.
 */


import {onMounted, ref} from "vue";
import DatePicker from "@vuepic/vue-datepicker";
import "@vuepic/vue-datepicker/dist/main.css";
import CurrencySet from "@/bean/currencySet.js";

const emit = defineEmits(["triggerSubmit", "triggerCancel"]);

const allDates = ref(["",""]);
// Initialize currency data
const currencies = ref([
  { currency: new CurrencySet("DOGEUSDT", ["", ""], "/currency_icon/dogecoin-doge-logo.png"), selected: false },
  { currency: new CurrencySet("BTCUSDT", ["", ""], "/currency_icon/bitcoin-btc-logo.png"), selected: false },
  { currency: new CurrencySet("ETHUSDT", ["", ""], "/currency_icon/ethereum-eth-logo.png"), selected: false },
  { currency: new CurrencySet("SOLUSDT", ["", ""], "/currency_icon/solana-sol-logo.png"), selected: false },
  { currency: new CurrencySet("XRPUSDT", ["", ""], "/currency_icon/xrp-xrp-logo.png"), selected: false },
  { currency: new CurrencySet("BNBUSDT", ["", ""], "/currency_icon/bnb-bnb-logo.png"), selected: false },
  { currency: new CurrencySet("AVAXUSDT", ["", ""], "/currency_icon/avalanche-avax-logo.png"), selected: false },
  { currency: new CurrencySet("ADAUSDT", ["", ""], "/currency_icon/cardano-ada-logo.png"), selected: false },
  { currency: new CurrencySet("LINKUSDT", ["", ""], "/currency_icon/chainlink-link-logo.png"), selected: false },
  { currency: new CurrencySet("DOTUSDT", ["", ""], "/currency_icon/polkadot-new-dot-logo.png"), selected: false },
  { currency: new CurrencySet("SHIBUSDT", ["", ""], "/currency_icon/shiba-inu-shib-logo.png"), selected: false },
  { currency: new CurrencySet("SUIUSDT", ["", ""], "/currency_icon/sui-sui-logo.png"), selected: false },
  { currency: new CurrencySet("TONUSDT", ["", ""], "/currency_icon/toncoin-ton-logo.png"), selected: false },
  { currency: new CurrencySet("TRXUSDT", ["", ""], "/currency_icon/tron-trx-logo.png"), selected: false },
  { currency: new CurrencySet("WBTCUSDT", ["", ""], "/currency_icon/wrapped-bitcoin.png"), selected: false },
]);



function submit() {
  const selectedCurrencies = currencies.value
      .filter((item) => item.selected)
      .map((item) => {
        item.currency.dates = [...allDates.value];

        item.currency.dates = item.currency.dates.map(date =>
            date instanceof Date ? date.toISOString().split("T")[0] : date
        );

        return item.currency;
      });

  // Save to local storage
  localStorage.setItem("selectedCurrencies", JSON.stringify(selectedCurrencies));

  emit("triggerSubmit", selectedCurrencies);
}



function cancel() {
  emit("triggerCancel");
}

onMounted(() => {
  const savedData = localStorage.getItem("selectedCurrencies");
  if (savedData) {
    const savedCurrencies = JSON.parse(savedData);

    // Set the value of allDates (unified time)
    if (savedCurrencies.length > 0 && savedCurrencies[0].dates) {
      const [start, end] = savedCurrencies[0].dates;
      allDates.value = [
        new Date(start),
        new Date() // Set the end time to the current time
      ];
    }

    // Update the selected status in currencies
    currencies.value.forEach((item) => {
      const match = savedCurrencies.find(
          (saved) => saved.name === item.currency.name
      );
      if (match) {
        item.selected = true;
      }
    });
  }
});
</script>

<style scoped>
.box-column {
  background-color: #757575;
  width: 610px;
  height: 480px;
  margin: 10px;
  border-radius: 18px;
  display: flex;
  flex-direction: column;
}

.box-0 {
  margin: 4px 0 10px 10px;
  display: flex;
  align-items: center;
}


.box-1 {
  background-color: #a6aaa6;
  border-radius: 12px;
  margin: 4px 0 8px 16px;
  width: 480px;
  height: 372px;
  overflow: auto;
  position: relative;
}

.box-1::-webkit-scrollbar {
  width: 20px;
}

.box-1::-webkit-scrollbar-track {
  background-color: #f1f1f1;
  border-radius: 12px;
}

.box-1::-webkit-scrollbar-thumb {
  background-color: #bfc4cc;
  border-radius: 12px;
}

.box-currency {
  margin: 6px 6px;
  width: 436px;
  height: 64px;
  position: relative;
  border-radius: 10px;
  background-color: #595555;
}

.currency-submit {
  background-color: #3A708D;
  border: 2px solid #346884;
  box-shadow: 1px 2px 4px #346884;
  width: 64px;
  height: 64px;
  font-size: 48px;
  color: #E0E0E0;
  border-radius: 16px;
  transition: all 0.2s ease;
  position: absolute;
  left: 86%;
  top: 64%;
}

.currency-submit:hover {
  background-color: #286487;
  opacity: 0.8;
}

.currency-submit:active {
  background-color: #246893;
}

.currency-cancel {
  background-color: #3A708D;
  border: 2px solid #346884;
  box-shadow: 1px 2px 4px #346884;
  width: 64px;
  height: 64px;
  font-size: 34px;
  color: #E0E0E0;
  border-radius: 16px;
  transition: all 0.2s ease;
  position: absolute;
  left: 86%;
  top: 82%;
}

.currency-cancel:hover {
  background-color: #286487;
  opacity: 0.8;
}

.currency-cancel:active {
  background-color: #246893;
}

.p1 {
  font-size: 26px;
  margin-top: 20px;
  margin-bottom: 20px;
  margin-left: 10px;
}

.p2 {
  font-size: 18px;
  margin-left: 0;
  margin-top: 5px;
  margin-bottom: 10px;
}

p {
  margin: 0;
  margin-top: 4px;
  font-family: Microsoft YaHei;
  color: #E0E0E0;
}

.icon {
  position: absolute;
  background-color: #504d4d;
  height: 38px;
  width: 38px;
  left: 9px;
  border-radius: 19px;
  margin-top: 4px;
  margin-bottom: 4px;
}

.currency-icon {
  width: 100%;
  height: 100%;
  border-radius: 16px;
  object-fit: cover;
}

.check {
  position: absolute;
  background-color: #ffffff;
  top: 6px;
  right: 7px;
  height: 28px;
  width: 28px;
}

.name {
  position: absolute;
  top: 5%;
  left: 64px;
  width: 50px;
  height: 30px;
}

.timeset {
  position: absolute;
  top: 49%;
  left: 9px;
  border-radius: 8px;
  width: 100%;
  height: 30px;
}
.alltimeset{
  position: absolute;
  left: 33%;
  top: 5.5%;
}
</style>
