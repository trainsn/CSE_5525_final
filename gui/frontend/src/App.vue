<template>
  <div id="app" style="border: 1px solid lightgray; border-radius: 4px; height:800px;" :loading="div_loading">
      <el-row style="background: #99a9bf; height:30px">
        <el-col :span="3">
          <el-select v-model="table_selected_id" placeholder="Please select one table" size="mini" style="float:left" @change=changeTable()>
            <el-option
              v-for="item in table_options"
              :key="item.value"
              :label="item.label"
              :value="item.value">
            </el-option>
          </el-select>
        </el-col>
        <el-col :span="2">
          <el-button type="primary" @click="onload()" :loading="buttonLoading_init" size="mini">Initialize</el-button>
        </el-col>
        <el-col :span="2">
           <el-button type="primary" @click="startSession()" :loading="buttonLoading" size="mini">Run</el-button>
        </el-col>
      </el-row>
      <el-row>
        <el-col :span="12">
          <div style="height:750px;border-right:1px solid #E4E7ED" id="div_graph"></div>
        </el-col>
        <el-col :span="12">
          <el-row>
            <el-col :span="18"> 
              <el-input v-model="input_question" placeholder="For the given database on the left, what is your question?"></el-input>
            </el-col>
            <el-col :span="6"> 
              <el-button @click="passQuestion()" size="median">Process</el-button>
            </el-col>
          </el-row>
          <el-row>
            <el-col :span="10">
              <el-row>
                <el-col :span="9">
                  
                  
                </el-col>
              </el-row>
            </el-col>
          </el-row>
        </el-col>
      </el-row>
      <!-- <el-row style="height:300px" id="div_graph">
	    </el-row> -->
      
      <el-dialog
        title="Agent:"
        :visible.sync="dialogVisible"
        width="30%">
        <span>{{dialog_info}}</span>
        <span slot="footer" class="dialog-footer">
          <el-button @click="dialogVisible = false">Cancel</el-button>
          <el-button type="primary" @click="clickEnter()">Enter</el-button>
        </span>
      </el-dialog>
      <el-dialog
        title="Agent:"
        :visible.sync="dialogVisible2"
        width="30%">
        Decision Making Statistics: 
         <el-table :data="gridData">
          <el-table-column property="table name" label="Table Name" width="150"></el-table-column>
          <el-table-column property="column name" label="Column Name" width="200"></el-table-column>
          <el-table-column property="semantic_tag" label="Semantic Tag"></el-table-column>
          <el-table-column property="probablity" label="Probability"></el-table-column>
          <el-table-column property="flag" label="Error Detected"></el-table-column>
        </el-table>
        <span>{{agent_ques}}</span>
        <span slot="footer" class="dialog-footer">
          <el-button @click="dialogVisible2 = false">Cancel</el-button>
          <el-button type="primary" @click="clickEnter2()">Enter</el-button>
        </span>
      </el-dialog>
      
       <el-dialog
        title="Result:"
        :visible.sync="dialogVisible3"
        width="30%">
        <span>{{dialog_info3}}</span>
      </el-dialog>

  </div>
</template>

<script>
// import River from './components/River.vue'
import table1_Data from '../public/table1.json'
import table2_Data from '../public/table2.json'
import table3_Data from '../public/table3.json'
import table4_Data from '../public/table4.json'
// import * as Neo4jd3 from './js/neo4jd3'
import * as Neo4jd3 from './js/Neo4D3.js'
// import River from '@/components/River.vue'
import * as d3 from 'd3'
import axios from 'axios'
export default {
  name: 'App',
  components:{
    // River,
  },
  data(){
      return{
        table_options : [],
        table_selected_id: null,
        column_list: [],
        tableData: [],
        from_agent: '',
        dialogVisible: false,
        dialog_info:'',
        buttonLoading:false,
        div_loading: false,
        dialogVisible2: false,
        dialog_info2:'',
        dialogVisible3: false,
        dialog_info3:'',
        buttonLoading_init: false,
        graphData: null,
        graphTotal: null,
        input_question:'',
        gridData: [],
        agent_ques:'',
        count_ :0
      }
    },
  created(){
    // this.changeTable()
    // this.onload()
  },
  methods: {
    onload(){
      this.div_loading = true
      this.buttonLoading_init = true
      const path = 'http://127.0.0.1:5000/onload'
      axios.get(path)
      .then((res)=>{
         console.log(res.data)
         this.div_loading = false
         this.buttonLoading_init = false
         this.graphData = res.data[0]['neodata']
         let temp = []
         this.graphTotal = res.data
         res.data.forEach(function(d, i){
          //  console.log()
           temp.push({
             'value': i,
             'label': d['name']
           })
         })
         this.table_options = temp
         this.drawGraph()

      })
      .catch((error)=>{
          console.log(error)
      })
    },
    drawGraph(){
      var that = this
      var neo4jd3 = Neo4jd3.default('#div_graph', {
          neo4jData: this.graphData,
          nodeRadius: 30,
          infoPanel: false,
          
      });
    },
    changeTable(){
      // console.log(this.table_selected_id)
      this.graphData = this.graphTotal[this.table_selected_id]['neodata']
    
    },
    stop(data){
      this.dialog_info3 = data['sentence']+': '+data['sql']
      this.dialogVisible3 = true
    },
    clickEnter2(){
      this.count+=1
      this.dialogVisible2 = false
  
      const path = "http://127.0.0.1:5000/Inter1"
      axios.get(path)
      .then((res)=>{
        console.log(res.data)
        if (res.data.flag=="stop"){
          this.stop(res.data)
        }else{
          this.gridData = res.data['tag_seq']

          // this.drawHypo(error, tag_seq)
          // this.dialog_info2 = temp
          this.agent_ques = res.data['question'].replace(
            /[\u001b\u009b][[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><]/g, '');
          this.dialogVisible2 = true
        }
      })
      .catch((error)=>{
        console.log(error)
      })
      
      
    },
    clickEnter(){
      this.dialogVisible=false
      const path = "http://127.0.0.1:5000/Enter"
      axios.get(path)
      .then((res)=>{
          
          this.gridData = res.data['tag_seq']

          // this.drawHypo(error, tag_seq)
          // this.dialog_info2 = temp
          this.agent_ques = res.data['question'].replace(
            /[\u001b\u009b][[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><]/g, '');
          this.dialogVisible2 = true
      })
      .catch((error)=>{
        console.log(error)
      })
    },
    openEnter(sent){
      this.dialog_info = sent
      this.dialogVisible = true
    },
    passQuestion(){
      const path = "http://127.0.0.1:5000/ProcessQuestion"
      const payload = {
          'question': this.input_question,
      }
      axios.post(path, payload)
      .then((res)=>{
          this.openEnter(res.data)
      })
      .catch((error)=>{
          console.log(error)
      })
    },
    startSession(){
      this.buttonLoading = true
      const path = 'http://127.0.0.1:5000/startSession'
      axios.get(path)
      .then((res)=>{
        this.buttonLoading = false
        // this.from_agent = res.data

        let table_options_list = []
        this.table_options.forEach(function(d){
          table_options_list.push(d['label'])
        })
        this.table_selected_id = table_options_list.indexOf(res.data)
        this.graphData = this.graphTotal[this.table_selected_id]['neodata']
        // this.openQuestion()
      })
      .catch((error)=>{
          console.log(error)
      })
    }
  },
  watch:{
    graphData(){
      this.drawGraph()
    }
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
