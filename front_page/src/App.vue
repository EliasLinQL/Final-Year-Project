<template>

    <div class="container">
      <!------------------->
      <!--Model-Info-Part-->
      <!------------------->
      <transition name="backtest-infofade">
        <model-info class="info" v-if="!isBackTestVisible" :class="{ calltrain: isTrainVisible, callgraph: waitisGraphVisible&&!isBackTestVisible, traininfo: hasBeenTrain||hasBeenCreate }" @triggerTrain="goTrain" @triggerGraph="showGraph"/>
      </transition>

      <!-------------------->
      <!--Model-Train-Part-->
      <!-------------------->
      <transition name="train">
        <model-train class="train" v-if ="isTrainVisible"  :class="{ traintrain: hasBeenTrain||hasBeenCreate }" :newtrainset="NewTrainSet" @triggerEndTrain="endTrain" @triggerStartTrain="startTrain" @triggerGoCreate="goCreateModel" @triggerDetail="callDetails" @trigger-update-pre="updatePre"/>
      </transition>

      <transition name="callpreinfo">
        <pre-supposed-info class="preinfo" v-if="isModelSetDetailVisible" :trainset="TrainSet"/>
      </transition>

      <transition name="createnewmodel">
        <new-model-set class="newmodelset" v-if="hasBeenCreate" :message="isCurrencyListVisible" :currencies="Currencies" @trigger-create-new-model="createModel" @trigger-cancel-create="cancelCreate" @trigger-currency-list="callCurrencyList"/>
      </transition>

      <transition name="setcurrency">
        <currency-list class="currency-list" v-if="isCurrencyListVisible" @trigger-submit="currencySubmit" @trigger-cancel="currencyCancel"/>
      </transition>

      <transition name="getloss">
        <train-loss class="loss" v-if="hasBeenTrain"/>
      </transition>

      <!-------------->
      <!--Graph-Part-->
      <!-------------->
      <transition name="showGraph-relation">
        <relation class="relation" v-if="waitisGraphVisible" :class="{ swapped: !isDiagramVisible }" :state-switch-r="!isDiagramVisible"/>
      </transition>

      <transition name="showGraph-diagram">
        <diagram class="diagram" v-if="waitisGraphVisible" :class="{ swapped: !isDiagramVisible }" :state-switch-d="!isDiagramVisible"/>
      </transition>

      <transition name="showGraph-switchbtn">
        <p class="bp" v-if="isGraphVisible">Turn into</p>
      </transition>

      <transition name="showGraph-switchbtn">
        <r-to-d-btn class="r-to-d-btn" v-if="isGraphVisible" @triggerRToD="rToD"/>
      </transition>

      <transition name="showGraph-btc">
        <back-test-control class="btc" v-if="btcVisible" @triggerBackTest="goBackTest"/>
      </transition>

      <!----------------->
      <!--BackTest-Part-->
      <!----------------->
      <transition name="backtest-infoin">
        <back-test-info v-if="isBackTestVisible" class="back-test-info" @triggerEndTest="endTest"/>
      </transition>

      <transition name="backtest-tradingview">
        <trading-view-a-p-i-part v-if="isBackTestVisible" class="tradingviewpart"/>
      </transition>

    </div>

</template>

<script setup>
import {ref, watch} from 'vue';
import ModelInfo from "@/components/modelInfo.vue";
import ModelTrain from "@/components/modelTrain.vue";
import Relation from "@/components/relation.vue";
import BackTestControl from "@/components/backTestControl.vue";
import Diagram from "@/components/diagram.vue";
import RToDBtn from "@/components/rToDBtn.vue";
import TrainLoss from "@/components/trainLoss.vue";
import BackTestInfo from "@/components/backTestInfo.vue";
import TradingViewAPIPart from "@/components/tradingViewAPIPart.vue";
import NewModelSet from "@/components/newModelSet.vue";
import PreSupposedInfo from "@/components/preSupposedInfo.vue";
import CurrencyList from "@/components/currencyList.vue";

//控制参数
const isTrainVisible = ref(false);
const isGraphVisible = ref(false);
const hasBeenTrain = ref(false);
const isDiagramVisible = ref(false);
const waitisGraphVisible = ref(false);
const isBackTestVisible = ref(false);
const btcVisible = ref(false);
const hasBeenCreate = ref(false);
const isModelSetDetailVisible = ref(false);
const isCurrencyListVisible = ref(false);

//通讯数据
const Currencies = ref([]);
const NewTrainSet = ref([]);
const TrainSet = ref(null);

 // -----------------------
 // ModelInfo Function Part
 // -----------------------

function goTrain() {
  if (isDiagramVisible.value) {
    isDiagramVisible.value = false;
    setTimeout(()=>{
      isTrainVisible.value = true;
      isGraphVisible.value = false;
      btcVisible.value = false;
    },1100);
  }else {
    isTrainVisible.value = true;
    isGraphVisible.value = false;
    btcVisible.value = false;
  }
}

function showGraph() {
  isGraphVisible.value = true;
  isTrainVisible.value = false;
  hasBeenTrain.value = false;
  hasBeenCreate.value = false;
  btcVisible.value = true;
  isModelSetDetailVisible.value = false;
  isCurrencyListVisible.value = false;
}

// ------------------------
// ModelTrain Function Part
// ------------------------

function endTrain() {
  isTrainVisible.value = false;
  hasBeenTrain.value = false;
  hasBeenCreate.value = false;
  isModelSetDetailVisible.value = false;
  isCurrencyListVisible.value = false;
}

function startTrain() {
  hasBeenTrain.value = true;
  hasBeenCreate.value = false;
  isCurrencyListVisible.value = false;
}

function goCreateModel(){
  hasBeenCreate.value = true;
  hasBeenTrain.value = false;
}

function callDetails(){
  isModelSetDetailVisible.value = !isModelSetDetailVisible.value;
}

function updatePre(selectedTrainSet){
  TrainSet.value = selectedTrainSet;
}

// -------------------------
// NewModelSet Function Part
// -------------------------

function callCurrencyList(){
  isCurrencyListVisible.value = !isCurrencyListVisible.value;
}

//将数据发送到modelTrain组件
function createModel(newModelSet){
  NewTrainSet.value = newModelSet;
  hasBeenCreate.value = false;
  isCurrencyListVisible.value = false;
}

function cancelCreate(){
  hasBeenCreate.value = false;
  isCurrencyListVisible.value = false;
}

// --------------------------
// CurrencyList Function Part
// --------------------------

//提交选中货币数据到newModelSet
function currencySubmit(selectedCurrencies){
  Currencies.value = selectedCurrencies;
  isCurrencyListVisible.value = false;
}

function currencyCancel(){
  isCurrencyListVisible.value = false;
}

// ------------------------
// R To D Btn Function Part
// ------------------------

function rToD(){
  isDiagramVisible.value = !isDiagramVisible.value;
}

// -----------------------------
// BackTestControl Function Part
// -----------------------------

function goBackTest() {
  if (isDiagramVisible.value){
    isDiagramVisible.value = false;
    setTimeout(()=>{
      isBackTestVisible.value = true;
      isGraphVisible.value = false;
    },1100);
  }else {
    isBackTestVisible.value = true;
    isGraphVisible.value = false;
  }
}

// --------------------------
// BackTestInfo Function Part
// --------------------------

function endTest(){
  isBackTestVisible.value = false;
  isGraphVisible.value = true;
}


watch(isGraphVisible, (newVal) => {
  setTimeout(()=>{
   waitisGraphVisible.value = !waitisGraphVisible.value;
  },100)
})
</script>

<style scoped>
  .container {
    height: 100%;
    width: 100%;
    position: relative;
  }

  /******************/
  /* modelInfo Part */
  /******************/

  .info{
    transition: all 1.8s ease;
    transform: translate(-200px,0);
    position: absolute;
    top: 10%;
    left: 50%;
  }

  .callgraph{
    transform: translate(-48.9vw,-9vh);
  }

  .traininfo{
    transition: all 1s ease;
    transform:translate(-500px,-80px) !important;
  }

  .backtest-infofade-enter-active,.backtest-infofade-leave-active {
    transition: all 1.2s ease;
  }

  .backtest-infofade-enter-from {
    transform: translate(-1400px,-9vh);
    opacity: 1;
  }
  .backtest-infofade-leave-to {
    transform: translate(-1400px,-9vh);
    opacity: 1;
  }

  /*******************/
  /* modelTrain Part */
  /*******************/

  .train {
    position: absolute;
    top: 10%;
    left: 50%;
    transition: all 1.8s ease;
    opacity: 1;
  }

  .calltrain{
    transform: translateX(-500px);
  }

  .traintrain{
    transition: all 1s ease;
    transform: translate(0px,-80px);
  }

  /* 为 model-train 组件的过渡效果 */
  .train-enter-active,.train-leave-active {
    transition: all 1.2s ease;
  }

  .train-enter-from {
    transform: translateX(1000px);
    opacity: 1;
  }
  .train-leave-to {
    transform: translateX(1000px);
    opacity: 1;
  }

  /************************/
  /* preSupposedInfo Part */
  /************************/

  .preinfo{
    position: absolute;
    top: 0%;
    left: 80%;
    z-index: 1;
  }

  .callpreinfo-enter-active,.callpreinfo-leave-active {
    transition: all 0.8s ease;
  }

  .callpreinfo-enter-from {
    transform: translateX(400px);
    opacity: 1;
  }

  .callpreinfo-leave-to {
    transform: translateX(400px);
    opacity: 1;
  }

  /********************/
  /* newModelSet Part */
  /********************/

  .newmodelset{
    position: absolute;
    top: 46%;
    left: 50%;
    transition: all 1.2s ease;
  }

  .createnewmodel-enter-active,.createnewmodel-leave-active {
    transition: all 1.2s ease;
  }

  .createnewmodel-enter-from {
    transform: translateY(520px);
    opacity: 1;
  }

  .createnewmodel-leave-to {
    transform: translateY(520px);
    opacity: 1;
  }

  /*********************/
  /* currencyList Part */
  /*********************/

  .currency-list{
    position: absolute;
    top: 46%;
    left: 14%;
  }

  .setcurrency-enter-active,.setcurrency-leave-active {
    transition: all 0.8s ease;
  }

  .setcurrency-leave-to,.setcurrency-enter-from {
    transform: translate(-900px,160px);
    opacity: 1;
  }

  /******************/
  /* trainLoss Part */
  /******************/
  .loss{
    position: absolute;
    top: 48%;
    left: 40.2%;
  }

  .getloss-enter-active,.getloss-leave-active {
    transition: all 0.8s ease;
  }

  .getloss-enter-from {
    transform: translateY(600px);
    opacity: 1;
  }
  .getloss-leave-to {
    transform: translateY(600px);
    opacity: 1;
  }


  /************************/
  /* backTestControl Part */
  /************************/

  .btc{
    position: absolute;
    top: 50%;
    left: 0.5%;
  }

  .showGraph-btc-enter-active,.showGraph-btc-leave-active {
    transition: all 1.4s ease;
  }
  .showGraph-btc-enter-from {
    transform: translate(-110%,200%);
    opacity: 1;
  }
  .showGraph-btc-leave-to {
    transform: translate(-110%,200%);
    opacity: 1;
  }

  /*****************/
  /* relation Part */
  /*****************/

  .relation{
    position: absolute;
    top: 0.8%;
    left: 25%;
    transition: transform 1.4s ease;
  }

  .relation.swapped {
    transform: translateY(67vh);
  }

  .showGraph-relation-enter-active, .showGraph-relation-leave-active {
    transition: all 1.6s ease;
  }
  .showGraph-relation-enter-from{
    transform: translate(40%,-120%);
    opacity: 1;
  }
  .showGraph-relation-leave-to{
    transform: translate(40%,-120%);
    opacity: 1;
  }


  /****************/
  /* diagram Part */
  /****************/

  .diagram{
    position: absolute;
    top: 69.5%;
    left: 25%;
    transition: transform 1.4s ease;
  }

  .diagram.swapped {
    transform: translateY(-67.4vh);
  }

  .showGraph-diagram-enter-active, .showGraph-diagram-leave-active {
    transition: all 1.6s ease;
  }
  .showGraph-diagram-enter-from{
    transform: translateY(120%);
    opacity: 1;
  }
  .showGraph-diagram-leave-to{
    transform: translateY(120%);
    opacity: 1;
  }


  /****************/
  /* rToDBtn Part */
  /****************/

  .r-to-d-btn{
    position: absolute;
    top: 71%;
    left: 85.4%;
  }

  .bp{
    color: #E0E0E0;
    font-family: Microsoft YaHei;
    font-size: 24px;
    position: absolute;
    top: 70.5%;
    left: 78.5%;
  }

  .showGraph-switchbtn-enter-active, .showGraph-switchbtn-leave-active {
    transition: all 1.2s ease;
  }
  .showGraph-switchbtn-enter-from{
    transform: translateX(420px);
    opacity: 1;
  }
  .showGraph-switchbtn-leave-to{
    transform: translateX(420px);
    opacity: 1;
  }

  /***********************/
  /* back-test-info Part */
  /***********************/

  .back-test-info{
    position: absolute;
    top: 0.8%;
    left: 0.6%;
  }

  .backtest-infoin-enter-active,.backtest-infoin-leave-active {
    transition: all 1.2s ease;
  }

  .backtest-infoin-enter-from {
    transform: translateX(-1400px);
    opacity: 1;
  }
  .backtest-infoin-leave-to {
    transform: translateX(-1400px);
    opacity: 1;
  }

  /************************/
  /* tradingviewpart Part */
  /************************/

  .tradingviewpart{
    position: absolute;
    top: 0%;
    left: 25%;
  }

  .backtest-tradingview-enter-active,.backtest-tradingview-leave-active {
    transition: all 1.2s ease;
  }

  .backtest-tradingview-enter-from {
    transform: translateY(1400px);
    opacity: 1;
  }
  .backtest-tradingview-leave-to {
    transform: translateY(1400px);
    opacity: 1;
  }

</style>
