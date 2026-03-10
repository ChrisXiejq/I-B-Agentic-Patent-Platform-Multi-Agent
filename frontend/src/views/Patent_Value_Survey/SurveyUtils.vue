<template>
    <div class="whole-box">
        <div style="max-width: 1400px;width:100%;justify-content: left;align-items: center;">
            <div style="display: flex;gap: 5px;">
                <p>工作人员ID：</p>
                <el-input v-model="SID" style="width: 240px;height: 50px;" placeholder="Please input" />
                <el-button style="height: 50px;" @click="checkSID">确定</el-button>
            </div>
            <div v-if="login" style="display: flex;gap: 5px;">
                <p>输入专利号删除问卷记录：</p>
                <el-input v-model="clearPatentNo" style="width: 240px;height: 50px;" placeholder="Please input" />
                <el-button style="height: 50px;" @click="clearSurvey">确定</el-button>
                <p style="color: red;font-size: large;">【危险操作】请注意核查</p>
            </div>
            <div v-if="login" style="display: flex;gap: 5px;">
                <p>输入企业/高校名称查看名下专利信息：</p>
                <el-input v-model="appln_applicant" style="width: 240px;height: 50px;" placeholder="Please input" />
                <el-button style="height: 50px;" @click="getPatentByApplicant">确定</el-button>
                <el-button style="height: 50px;" @click="clearPatentByApplicant">清除</el-button>
            </div>
            <div v-if="login" style="display: flex;gap: 5px;">
                <p>输入专利号查看专利信息：</p>
                <el-input v-model="patent_no" style="width: 240px;height: 50px;" placeholder="Please input" />
                <el-button style="height: 50px;" @click="getPatentByNo">确定</el-button>
                <el-button style="height: 50px;" @click="clearPatentByNo">清除</el-button>
            </div>
            <div v-if="searchPatents">
                <div v-if="filledPatents.length">
                    <p>已填写专利：</p>
                    <div v-for="patent in filledPatents" :key="patent.no" class="patent-item">
                        <p>{{ patent.no }} - 已填写 - {{ patent.appln_application }} - 类型: {{ patent.type }} - {{ patent.name }} </p>
                    </div>
                </div>
                <div v-if="unfilledPatents.length">
                    <p>未填写专利：</p>
                    <div v-for="patent in unfilledPatents" :key="patent.no" class="patent-item">
                        <p>{{ patent.no }} - 未填写 - {{ patent.appln_application }} - 类型: {{ patent.type }} - {{ patent.name }} </p>
                    </div>
                </div>
            </div>
            <div v-if="patentData">
                <p>专利信息：</p>
                <p>专利号: {{ patentData.no }}</p>
                <p>名称: {{ patentData.name }}</p>
                <p>摘要: {{ patentData.summary }}</p>
                <p>pdf链接: {{ patentData.pdfs }}</p>
                <p>专利类型: {{ patentData.type }}</p>
                <p>填写状态: {{ patentData.status === 1 ? '已填写' : '未填写' }}</p>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

let SID = ref("");
let clearPatentNo = ref("");
let patent_no = ref("");
let appln_applicant = ref("");
let login = ref(false);
let patentData = ref(null);
let searchPatents = ref(null);
const SIDs = ["123456", "12345"];

const filledPatents = computed(() => {
    return searchPatents.value ? searchPatents.value.filter(patent => patent.status === 1) : [];
});

const unfilledPatents = computed(() => {
    return searchPatents.value ? searchPatents.value.filter(patent => patent.status === 0) : [];
});

const checkSID = () => {
    if (SIDs.includes(SID.value)) {
        // ElMessage.success('ID存在');
        login.value = true;
    } else {
        login.value = false;
        ElMessage.error('ID不存在');
    }
}
const getPatentByApplicant = async () => {
    try {
        const response = await axios.get(`api/patents`, {
            params: {
                company: appln_applicant.value,
            }
        });
        if (response.data.data.length == 0) {
            ElMessage.error("抱歉，公司名有误或专利不存在")
        } else if (response.data.code == 1) {
            searchPatents.value = response.data.data
        }
        return;
    } catch (error) {
        // 处理请求失败的情况
        console.error('请求失败:', error);
        throw error; // 可以选择抛出错误以供调用者处理
    }
}
const getPatentByNo = async () => {
    if (patent_no.value == '') {
        return;
    }
    if (!patent_no.value.startsWith("CN")) {
        patent_no = "CN" + patent_no.value;
    }
    let response = await axios.get('api/patents/' + patent_no.value);
    if (response.data.code == 0) {
        ElMessage.error("抱歉，专利号有误或专利不存在")
    } else if (response.data.code == 1) {
        patentData.value = response.data.data;
    }
}

const clearSurvey = async () => {
    if (clearPatentNo.value == '') {
        return;
    }
    let response = await axios.get('api/patents/clear/' + clearPatentNo.value);
    if (response.data.code == 0) {
        ElMessage.error("请求失败")
    } else if (response.data.code == 1) {
        ElMessage.success("请求成功")
    }
}
const clearPatentByApplicant = () => {
    searchPatents.value = null;
}
const clearPatentByNo = () => {
    patentData.value = null;
}

</script>

<style scoped>
.demo-tabs>.el-tabs__content {
    padding: 32px;
    color: #6b778c;
    font-size: 32px;
    font-weight: 600;
}

.whole-box {
    width: 100%;
    display: flex;
    justify-content: center;
    background-color: white;
    margin-top: 20px;
    padding-left: 20px;
    padding-right: 20px;
    /* border: 1px solid #000; */
}

/* Style for portrait mode */
@media screen and (orientation: portrait) {}
</style>