//引入createApp用于创建应用
import {createApp} from "vue"
//引入App根组件
import App from "./App.vue"
import modelSet from "@/bean/modelSet.js";
import currencySet from "@/bean/currencySet.js";

const app = createApp(App)

app.config.globalProperties.$modelSet = modelSet;
app.config.globalProperties.$currencySet = currencySet;

app.mount("#app")