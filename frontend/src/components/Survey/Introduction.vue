<template>
    <el-card style="border-radius: 15px;width: 100%;font-size: 1rem;font-family: SimSun;width:100%">

        <!-- <el-card style="border: 1px solid black;"> -->
        <!-- 初步调整，可以确认一下 -->
        <p style="margin-top: 2vh;padding-left: 2em;">
            请输入您所在企业/学校/代理所名称
            <el-input size="small" style="width: 10vw;" v-model="form.companyName" @input="updateCompanyName"
                @keyup.enter="getPatentNoByCompany(form.companyName)"></el-input>
            <el-button type="primary" size="small" @click="getPatentNoByCompany(form.companyName)"
                style="margin-left: 1vw;">确认</el-button>
            ，确认需填写的专利
            <br>
            或
            <br>
            直接输入邀请函上的专利号：CN
            <el-input size="small" style="width: 250px;" v-model="form.patentNo" @input="updatePatentNo"
                @keyup.enter="getPatentByNo(form.patentNo)"></el-input>

            <!-- <el-button v-if="form.patentNo&&!patentDetail.name" type="primary" @click="getPatentByNo(form.patentNo)"
                style="margin-left: 1vw;">确认</el-button> -->
            <el-button type="primary" @click="getPatentByNo(form.patentNo)" size="small"
                style="margin-left: 1vw;">确认</el-button>
            <el-button v-if="patentDetail.name&&form.patentNo" type="primary" size="small"
                @click="allowInput(patentDetail.type)" style="margin-left: 1vw;">专利正确，继续问卷</el-button>
            <br>
        </p>
        <template v-if="patentDetail.no != ''">
            <div v-if="patentDetail.name">
                <p><strong>专利名称:</strong> {{ patentDetail.name }}</p>
                <p><strong>专利概述:</strong> {{ patentDetail.summary }}</p>
                <p><strong>专利链接:</strong> <a :href="patentDetail.pdfs[0]" target="blank">{{ patentDetail.pdfs[0]
                        }}</a></p>
            </div>
        </template>
        <template v-if="searchPatents.length != 0">
            <el-table :data="searchPatents">
                <!-- <el-table-column prop="no" label="专利申请号" width="180" /> -->
                <el-table-column prop="no" label="专利申请号" width="180">
                    <template v-slot:default="scope">
                        <span @click="handleClick(scope)" style="color:#0000FF;cursor: pointer;">{{ scope.row.no
                            }}</span>
                    </template>
                </el-table-column>
                <el-table-column prop="appln_application" label="单位/申请人" width="200"/>
                <el-table-column prop="name" label="专利标题" width="400" />
            </el-table>
        </template>

        <el-dialog v-model="dialogVisible" title="确定专利信息">
            <p><strong>专利名称:</strong> {{ boxPatent.name }}</p>
            <p><strong>专利概述:</strong> {{ boxPatent.summary }}</p>
            <p><strong>专利链接:</strong> <a v-if="boxPatent.pdfs" :href="boxPatent.pdfs[0]" target="blank">{{
                    boxPatent.pdfs[0]
                    }}</a></p>
            <template #footer>
                <div class="dialog-footer">
                    <el-button @click="dialogVisible = false">取消</el-button>
                    <el-button type="primary" @click="confirm()">
                        专利无误，开始填写
                    </el-button>
                </div>
            </template>
        </el-dialog>
        <!-- </el-card> -->
        <!-- <p style="color: rgb(0,112,192)">
            填写形式：
        </p>
        <p style="color: rgb(0,112,192);text-indent: 2em;">
            如果答案前为o，请勾选单一选项；如果答案前为 口，请选择所有相关选项；
        </p>
        <p style="color: rgb(0,112,192);text-indent: 2em;">
            如果答案为矩阵，请每一行勾选相应选项
        </p>
        <p style="color: rgb(0,112,192);text-indent: 2em;">
            如果您已回答过一份问卷，A.个人基本信息与B.企业信息部分可直接跳过
        </p> -->
    </el-card>
</template>

<script setup>
import { ref, reactive, defineEmits } from 'vue';
import { surveyStore } from '../../stores/survey';
import axios from 'axios';
import { ElMessage } from 'element-plus';
import { ElMessageBox } from 'element-plus'

const surveyInfo = surveyStore().surveyInfo
const updatePatentNo = (value) => {
    if (!value.startsWith("CN")) {
        value = "CN" + value;
    }
    surveyInfo.patentNo = value;
    patentDetail.value = {}
};
const updateCompanyName = (value) => {
    surveyInfo.companyName = value;
};
const form = reactive({
    patentNo: "",
    companyName: ""
});

const patentDetail = ref({});
const getPatentByNo = async (no) => {
    if (no == '') {
        return;
    }
    if (!no.startsWith("CN")) {
      no = "CN"+no;  
    }
    let response = await axios.get('api/patents/' + no);
    if (response.data.code == 0) {
        ElMessage.error("抱歉，专利号有误或专利不存在")
    } else if (response.data.code == 1) {
        patentDetail.value = response.data.data
        if(patentDetail.value.status == 1){
            ElMessage.error("问卷已填写")
            patentDetail.value = {}
        }
    }
}

const dialogVisible = ref(false)
const boxPatent = ref({})
const handleClick = async(scope) => {
    console.log('Selected Patent No:', scope.row.no);
    updatePatentNo(scope.row.no)
    if (!scope.row.no.startsWith("CN")) {
        scope.row.no = "CN" + scope.row.no;
    }
    let response = await axios.get('api/patents/' + scope.row.no);
    if (response.data.code == 0) {
        ElMessage.error("抱歉，专利号有误或专利不存在")
    } else if (response.data.code == 1) {
        boxPatent.value = response.data.data
        console.log(boxPatent.value)
    }
    dialogVisible.value = true
    // ElMessageBox.confirm('确定填写'+scope.row.no+'的专利问卷?')
    // .then(async () => {
    //     updatePatentNo(scope.row.no)
    //     if (!scope.row.no.startsWith("CN")) {
    //         scope.row.no = "CN" + scope.row.no;
    //     }
    //     let response = await axios.get('api/patents/' + scope.row.no);
    //     if (response.data.code == 0) {
    //         ElMessage.error("抱歉，专利号有误或专利不存在")
    //     } else if (response.data.code == 1) {

    //         allowInput(response.data.data.type)
    //     }
    //     done()
    // })
    // .catch(() => {
    //     // catch error
    // })
}

const confirm = async() => {
    dialogVisible.value = false
    updatePatentNo(boxPatent.value.no)
    form.patentNo = boxPatent.value.no
    patentDetail.value = boxPatent.value
    allowInput(boxPatent.value.type)
}

const emits = defineEmits(['allow-input']);

const allowInput = (type) => {
    emits('allow-input', type);
}

const searchPatents = ref([]);
const getPatentNoByCompany = async (name) => {
    try {
        const no = "";
        const response = await axios.get(`api/patents`, {
            params: {
                company: name,
                no: ""
            }
        });
        if (response.data.data.length == 0) {
            ElMessage.error("抱歉，公司名有误或专利不存在，请确保输入公司全名")
        } else if (response.data.code == 1) {
            searchPatents.value = response.data.data.filter(p => p.status === 0);
        }
        return;
    } catch (error) {
        // 处理请求失败的情况
        console.error('请求失败:', error);
        throw error; // 可以选择抛出错误以供调用者处理
    }
}
</script>



<style scoped>
p {
    margin-bottom: 20px;
    /* 设置段落下方的外边距 */
    line-height: 40px
}

:deep(.el-radio__label) {
    white-space: normal;
    /* 换行 */
}

:deep(.el-checkbox__label) {
    white-space: normal;
    /* 换行 */
}
</style>
