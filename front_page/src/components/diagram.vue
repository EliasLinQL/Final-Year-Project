<template>
  <div class="box-column" :class="{ box_switch: props.stateSwitchD}">

  </div>
</template>

<script setup>

const props = defineProps({
  stateSwitchD: {
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
  width: 540px;
  height: 270px;
  margin: 10px;
  padding: 0;
  position: relative;
  border-radius: 18px;
}
.box_switch{
  width: 1260px;
  height: 630px;
}
p{
  margin: 15px;
  font-family: Microsoft YaHei;
  color: #E0E0E0;
}
</style>