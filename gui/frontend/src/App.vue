<template>
  <div id="app">
      <el-row>
        <el-col :span="10">
          <el-row>
            <el-col :span="15">
              <el-input v-model="input_sent" placeholder="Please input your question"></el-input>
            </el-col>
            <el-col :span="9">
              <el-button type="primary" @click="passSent()">Pass</el-button>
            </el-col>
          </el-row>
        </el-col>
      </el-row>
      

  </div>
</template>

<script>
// import River from './components/River.vue'
import * as d3 from 'd3'
import axios from 'axios'
export default {
  name: 'App',
  components:{
    // River,
  },
  data(){
      return{
        input_sent: "",
      }
    },
  created(){

  },
  methods: {
    passSent(){
      const path = 'http://localhost:5000/passSent'
      const payload = {
          'sentence': this.input_sent,
      }
      axios.post(path, payload)
      .then((res)=>{
         console.log(res.data)
      })
      .catch((error)=>{
          console.log(error)
      })
    }
  },
  watch:{

  },
  mounted(){
    
  },
  computed:{

  }
}
</script>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
.el-row {
    margin-bottom: 20px;
    &:last-child {
      margin-bottom: 0;
    }
  }
  .el-col {
    border-radius: 4px;
  }
  .bg-purple-dark {
    background: #99a9bf;
  }
  .bg-purple {
    background: #d3dce6;
  }
  .bg-purple-light {
    background: #e5e9f2;
  }
  .grid-content {
    border-radius: 4px;
    min-height: 36px;
  }
  .row-bg {
    padding: 10px 0;
    background-color: #f9fafc;
  }
</style>
