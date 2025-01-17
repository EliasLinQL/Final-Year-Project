<template>
  <div id="Diagram" class="box-column" :class="{ box_switch: props.stateSwitchD }"></div>
</template>

<script setup>
import { onMounted, onBeforeUnmount } from "vue";
import { createChart } from "lightweight-charts";

const props = defineProps({
  stateSwitchD: {
    type: Boolean,
    required: true,
  },
});

let Diagram;
let resizeObserver;

onMounted(() => {
  const DiagramElement = document.getElementById("Diagram");

  if (DiagramElement) {
    const DiagramOptions = {
      layout: {
        textColor: "black",
        background: { type: "solid", color: "white" },
      },
    };

    Diagram = createChart(DiagramElement, {
      ...DiagramOptions,
      width: DiagramElement.clientWidth,
      height: DiagramElement.clientHeight,
    });

    const lineSeries1 = Diagram.addLineSeries({ color: "blue" });
    const lineSeries2 = Diagram.addLineSeries({ color: "red" });

    const data1 = [
      { value: 0, time: 1642425322 },
      { value: 8, time: 1642511722 },
      { value: 10, time: 1642598122 },
      { value: 20, time: 1642684522 },
      { value: 3, time: 1642770922 },
      { value: 43, time: 1642857322 },
      { value: 41, time: 1642943722 },
      { value: 43, time: 1643030122 },
      { value: 56, time: 1643116522 },
      { value: 46, time: 1643202922 },
    ];
    const data2 = [
      { value: 0, time: 1642425322 },
      { value: 10, time: 1642511722 },
      { value: 13, time: 1642598122 },
      { value: 18, time: 1642684522 },
      { value: 6, time: 1642770922 },
      { value: 38, time: 1642857322 },
      { value: 36, time: 1642943722 },
      { value: 41, time: 1643030122 },
      { value: 48, time: 1643116522 },
      { value: 40, time: 1643202922 },
    ];
    lineSeries1.setData(data1);
    lineSeries2.setData(data2);

    Diagram.timeScale().fitContent();

    resizeObserver = new ResizeObserver(() => {
      if (DiagramElement) {
        Diagram.resize(DiagramElement.clientWidth, DiagramElement.clientHeight);
      }
    });

    resizeObserver.observe(DiagramElement);
  } else {
    console.error("Element with id 'Diagram' not found");
  }
});

onBeforeUnmount(() => {
  if (resizeObserver) {
    resizeObserver.disconnect();
  }
});
</script>

<style scoped>
.box-column {
  display: flex;
  flex-direction: column;
  background-color: #757575;
  width: 540px;
  height: 270px;
  margin: 10px;
  padding: 0;
  position: relative;
  border-radius: 18px;
  transition: all 0.3s ease;
}
.box_switch {
  width: 1260px;
  height: 630px;
}
p {
  margin: 15px;
  font-family: Microsoft YaHei;
  color: #E0E0E0;
}
</style>
