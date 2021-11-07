<template>
  <div id="app" style="border: 1px solid lightgray; border-radius: 4px; height:800px;">
      <el-row style="background: #99a9bf; height:30px">
         <el-select v-model="table_selected_id" placeholder="Please select one table" size="mini" style="float:left" @change=changeTable()>
          <el-option
            v-for="item in table_options"
            :key="item.value"
            :label="item.label"
            :value="item.value">
          </el-option>
        </el-select>
      </el-row>
      <el-row style="height:300px">
        <el-table
      :data="tableData"
      style="width: 100%">
      <el-table-column v-for="subtitle in column_list" :key="subtitle" fixed :prop="subtitle"
        :label="subtitle">
      </el-table-column>
    </el-table>
	</el-row>
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
import table1_Data from '../public/table1.json'
import table2_Data from '../public/table2.json'
import table3_Data from '../public/table3.json'
import table4_Data from '../public/table4.json'

import * as d3 from 'd3'
import axios from 'axios'
export default {
  name: 'App',
  components:{
    // River,
  },
  data(){
      return{
        table_options: [{
          value: '1',
          label: 'Ref_Template_Types'
        }, {
          value: '2',
          label: 'Templates'
        }, {
          value: '3',
          label: 'Documents'
        }, {
          value: '4',
          label: 'Paragraphs'
        }],
        table_selected_id: '1',
        input_sent: "",
        column_list: [],
        tableData: []
      }
    },
  created(){
    this.changeTable()
    this.onload()
  },
  methods: {
    onload(){
      const path = 'http://localhost:5000/onload'
      axios.get(path)
      .then((res)=>{
         console.log(res.data)
      })
      .catch((error)=>{
          console.log(error)
      })
    },
    changeTable(){
      if (this.table_selected_id=="1"){
        this.tableData = table1_Data['data']
        this.column_list = table1_Data['columns']
      }else if(this.table_selected_id=="2"){
        this.tableData = table2_Data['data']
        this.column_list = table2_Data['columns']
      }else if(this.table_selected_id=="3"){
        this.tableData = table3_Data['data']
        this.column_list = table3_Data['columns']
      }else if(this.table_selected_id=="4"){
        this.tableData = table4_Data['data']
        this.column_list = table4_Data['columns']
      }
    
    },
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
