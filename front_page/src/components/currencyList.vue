<template>
  <div class="box-column">
    <div class="box-0">
      <p class="p1">Currency List</p>
    </div>
    <div class="box-row">
      <div class="box-currency">
        <input type="checkbox" class="check">
        <div class="icon">

        </div>
        <div class="name">
          <p class="p2">
            DOGE
          </p>
        </div>
        <div class="starttime">

        </div>
        <div class="endtime">

        </div>
      </div>
      <div class="box-currency">
        <input type="checkbox" class="check">
        <div class="icon">

        </div>
        <div class="name">
          <p class="p2">
            DOGE
          </p>
        </div>
        <div class="starttime">

        </div>
        <div class="endtime">

        </div>
      </div>
      <div class="box-currency">
        <input type="checkbox" class="check">
        <div class="icon">

        </div>
        <div class="name">
          <p class="p2">
            DOGE
          </p>
        </div>
        <div class="starttime">

        </div>
        <div class="endtime">

        </div>
      </div>
      <div class="box-currency">
        <input type="checkbox" class="check">
        <div class="icon">

        </div>
        <div class="name">
          <p class="p2">
            DOGE
          </p>
        </div>
        <div class="starttime">

        </div>
        <div class="endtime">

        </div>
      </div>
      <div class="box-currency">
        <input type="checkbox" class="check">
        <div class="icon">

        </div>
        <div class="name">
          <p class="p2">
            DOGE
          </p>
        </div>
        <div class="starttime">

        </div>
        <div class="endtime">

        </div>
      </div>
    </div>
    <div class="box-row">
      <div class="box-currency">
        <input type="checkbox" class="check">
        <div class="icon">

        </div>
        <div class="name">
          <p class="p2">
            DOGE
          </p>
        </div>
        <div class="starttime">

        </div>
        <div class="endtime">

        </div>
      </div>
      <div class="box-currency">
        <input type="checkbox" class="check">
        <div class="icon">

        </div>
        <div class="name">
          <p class="p2">
            DOGE
          </p>
        </div>
        <div class="starttime">

        </div>
        <div class="endtime">

        </div>
      </div>
      <div class="box-currency">
        <input type="checkbox" class="check">
        <div class="icon">

        </div>
        <div class="name">
          <p class="p2">
            DOGE
          </p>
        </div>
        <div class="starttime">

        </div>
        <div class="endtime">

        </div>
      </div>
      <div class="box-currency">
        <input type="checkbox" class="check">
        <div class="icon">

        </div>
        <div class="name">
          <p class="p2">
            DOGE
          </p>
        </div>
        <div class="starttime">

        </div>
        <div class="endtime">

        </div>
      </div>
      <div class="box-currency">
        <input type="checkbox" class="check">
        <div class="icon">

        </div>
        <div class="name">
          <p class="p2">
            DOGE
          </p>
        </div>
        <div class="starttime">

        </div>
        <div class="endtime">

        </div>
      </div>
    </div>
    <div class="box-row">
      <div class="box-currency">
        <input type="checkbox" class="check">
        <div class="icon">

        </div>
        <div class="name">
          <p class="p2">
            DOGE
          </p>
        </div>
        <div class="starttime">

        </div>
        <div class="endtime">

        </div>
      </div>
      <div class="box-currency">
        <input type="checkbox" class="check">
        <div class="icon">

        </div>
        <div class="name">
          <p class="p2">
            DOGE
          </p>
        </div>
        <div class="starttime">

        </div>
        <div class="endtime">

        </div>
      </div>
      <div class="box-currency">
        <input type="checkbox" class="check">
        <div class="icon">

        </div>
        <div class="name">
          <p class="p2">
            DOGE
          </p>
        </div>
        <div class="starttime">

        </div>
        <div class="endtime">

        </div>
      </div>
      <div class="box-currency">
        <input type="checkbox" class="check">
        <div class="icon">

        </div>
        <div class="name">
          <p class="p2">
            DOGE
          </p>
        </div>
        <div class="starttime">

        </div>
        <div class="endtime">

        </div>
      </div>
      <div class="box-currency">
        <input type="checkbox" class="check">
        <div class="icon">

        </div>
        <div class="name">
          <p class="p2">
            DOGE
          </p>
        </div>
        <div class="starttime">

        </div>
        <div class="endtime">

        </div>
      </div>
    </div>

  </div>
</template>

<script setup>
import {ref} from "vue";


</script>

<style scoped>
.box-column {
  background-color: #757575;
  width: 720px;
  height: 480px;
  margin: 10px;
  border-radius: 18px;
  display: flex;
  flex-direction: column;
}
.box-0{
  margin: 4px 0 10px 10px;
}
.box-row {
  border-radius: 12px;
  display: flex;
  flex-direction: row;
  margin: 8px 0 8px 10px;
  width: 700px;
  height: 100%;
}
.box-currency{
  margin: 2px 8px;
  width: 130px;
  height: 110px;
  position: relative;
  border-radius: 10px;
  background-color: #595555;
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
.icon{
  position: absolute;
  background-color: #ffffff;
  height: 38px;
  width: 38px;
  left: 24px;
  border-radius: 16px;
  margin-top: 2px;
}
.check{
  position: absolute;
  background-color: #ffffff;
  top: 10px;
  height: 18px;
  width: 18px;
}
.name{
  position: absolute;
  top: 5%;
  left: 64px;
  width: 50px;
  height: 30px;
}
.starttime{
  position: absolute;
  top: 39%;
  background-color: #777171;
  border-radius: 8px;
  margin-left: 2px;
  width: 120px;
  height: 30px;
}
.endtime{
  position: absolute;
  top: 69%;
  background-color: #777171;
  border-radius: 8px;
  margin-left: 2px;
  width: 120px;
  height: 30px;
}
</style>
