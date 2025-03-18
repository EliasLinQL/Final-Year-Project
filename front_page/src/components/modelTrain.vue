<template>
  <div class="box-column">
    <div class="box-row">
      <p class="p1">Train Details</p>
    </div>
    <div class="box-row model-set">
      <p class="p2 model-set">Model Settings</p>
      <train-checkmenu
          class="check-part"
          :trainsettings="trainSettings"
          @triggerTrainSetCheck="setTrainSet"
      />
      <button class="add-btn" @click="createModel">
        +
      </button>
    </div>
    <div class="box-row">
      <p class="p2">ModelSetName</p>
      <div class="data-area-name">
        {{ selectedTrainSet?.name || 'No TrainSet Selected' }}
      </div>
    </div>
    <div class="box-row">
      <p class="p2">PreSupposed<br>Details</p>
      <button class="detail" @click="callDetail">
        {{ isIconRotated ? 'Close' : 'Show' }}
      </button>
    </div>
    <div class="box-row box-btn">
      <button class="btn" @click="endTrain">End Training</button>
      <button class="btn" @click="startTrain">Start Train</button>
    </div>
  </div>
</template>

<script setup>
import {ref, watch, onMounted, onBeforeUnmount} from "vue";
import TrainCheckmenu from "@/components/TrainCheckmenu.vue";
import ModelSet from "@/bean/modelSet.js";

const isIconRotated = ref(false);
const trainSettings = ref([]);
const selectedTrainSet = ref(null);

const emit = defineEmits([
  "triggerEndTrain",
  "triggerStartTrain",
  "triggerDetail",
  "triggerGoCreate",
  "triggerUpdatePre"
]);
const props = defineProps({
  newtrainset: {type: Object},
});

function endTrain() {
  emit("triggerEndTrain");
}

function startTrain() {
  if (!selectedTrainSet.value) {
    alert("Please select a model preset before starting training!");
    return;
  }

  // const start_date = selectedTrainSet.value.currencies[0].dates[0];
  // const end_date = selectedTrainSet.value.currencies[0].dates[1];
  // const symbols = selectedTrainSet.value.currencies.map(currencies => currencies.name);
  //
  // const requestData = { start_date, end_date, symbols };

  fetch('http://localhost:5001/api/train_model', {
    method: 'POST'
  })
      .then(res => res.json())
      .then(data => {
        alert(data.message);
      })
      .catch(err => {
        console.error(err);
        alert("Training failed");
      });


  emit("triggerStartTrain");
}


function callDetail() {
  isIconRotated.value = !isIconRotated.value;
  emit("triggerDetail");
}

function createModel() {
  emit("triggerGoCreate");
}

function setTrainSet(selectedItem) {
  selectedTrainSet.value = selectedItem;
}

// 更新 LocalStorage 的函数
function updateLocalStorage() {
  localStorage.setItem("trainSettings", JSON.stringify(trainSettings.value));
}

// 从 LocalStorage 初始化 trainSettings
onMounted(() => {
  const savedSettings = localStorage.getItem("trainSettings");
  if (savedSettings) {
    trainSettings.value = JSON.parse(savedSettings);
  }
});

// 添加新数据到 trainSettings，并存储
watch(props, (newval) => {
  if (newval.newtrainset) {
    trainSettings.value.push(newval.newtrainset);
  }
});
// 当 trainSettings 改变时自动存储到 LocalStorage
watch(
    trainSettings,
    (newval) => {
      updateLocalStorage();
    },
    {deep: true}
);
//实时更新pre预览数据
watch(
    selectedTrainSet,
    (newval) => {
      emit("triggerUpdatePre",newval);
    }
);

onBeforeUnmount(() => {
  selectedTrainSet.value = new ModelSet("", []);
  emit("triggerUpdatePre", selectedTrainSet);
})
</script>

<style scoped>
.box-column {
  background-color: #757575;
  width: 520px;
  height: 400px;
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
  margin-top: 24px;
}

.check-part {
  position: absolute;
  left: 36%;
  top: 18%;
  z-index: 1;
}

.add-btn {
  position: absolute;
  width: 42px;
  height: 42px;
  border-radius: 16px;
  border: 2px solid #a0b5b1;
  box-shadow: 1px 2px 2px #86a5a5;
  background-color: #a0b5b1;
  top: 19%;
  left: 70%;
  font-size: 36px;
  font-weight: bold;
  color: #386698;
  line-height: 1px;
}

.add-btn:hover {
  background-color: #8db2ab;
  opacity: 0.8;
}

.add-btn:active {
  background-color: #7daf95;
}

.model-set {
  margin-top: 0;
  margin-bottom: 10px;
}

p {
  margin: 15px;
  font-family: Microsoft YaHei;
  color: #e0e0e0;
}

.btn {
  margin-top: 48px;
  margin-bottom: 5px;
  width: 158px;
  height: 38px;
  background-color: #3a708d;
  border: 2px solid #346884;
  border-radius: 12px;
  font-size: 20px;
  color: #e0e0e0;
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

.data-area-name {
  width: 268px;
  height: 44px;
  background-color: #827e7e;
  margin-right: 20px;
  position: absolute;
  transform: translate(180px, 19px);
  text-align: center;
  line-height: 44px;
  color: #e8e8e8;
}

.detail {
  width: 68px;
  height: 52px;
  background-color: rgba(86, 182, 157, 0.6);
  border: 2px solid rgba(71, 152, 130, 0.6);
  box-shadow: 1px 2px 4px rgba(84, 198, 195, 0.6);
  border-radius: 24%;
  margin-right: 20px;
  position: absolute;
  transform: translate(180px, 24px);
  transition: all 0.2s ease;
  font-size: 20px;
  color: #c1d0dd;
}

.detail:hover {
  background-color: rgba(102, 198, 175, 0.6);
  opacity: 0.8;
}

.detail:active {
  background-color: rgba(118, 214, 190, 0.6);
}
</style>
