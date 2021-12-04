// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import router from './router'
import BootstrapVue from "bootstrap-vue"
import 'element-ui/lib/theme-chalk/index.css'
import locale from 'element-ui/lib/locale/lang/en'
import ElementUI from 'element-ui'
require('./css/neo4d3.css')
Vue.config.productionTip = false
Vue.use(BootstrapVue);
Vue.use(ElementUI, {locale})
/* eslint-disable no-new */
new Vue({
  el: '#app',
  router,
  components: { App },
  template: '<App/>'
})