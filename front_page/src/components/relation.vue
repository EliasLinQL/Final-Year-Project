<template>
  <div class="box-column" :class="{ box_switch: props.stateSwitchR}">

  </div>
</template>

<script setup>

const props = defineProps({
  stateSwitchR: {
    type: Boolean,
    required: true
  }
})

</script>

<style scoped>
.box-column{
  display: flex;
  flex-direction: column;
  background-color: #757575;
  width: 1260px;
  height: 630px;
  margin: 10px;
  padding: 0;
  position: relative;
  border-radius: 18px;
  transition: width 1s ease, height 1s ease;
}
.box_switch{
  width: 540px;
  height: 270px;
}
p{
  margin: 15px;
  font-family: Microsoft YaHei;
  color: #E0E0E0;
}
</style>