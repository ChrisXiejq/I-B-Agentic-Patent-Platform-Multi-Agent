<template>
    <el-card style="border-radius: 15px;width: 100%;">
        <p style="font-family: STKaiti;color: blue;font-weight: bold;">
            （在B部分，我们想了解下您所代理专利的专利权人情况，谢谢支持！）
        </p>
        <el-form :model="form" size="large" label-position="top">

            <el-form-item class="question" style="font-weight: bolder;" label="B01.专利权人的工作状态是？">
                <el-radio-group v-model="form.pBq01" @change="handlepBq01Change">
                    <el-radio class="answer" label="在读学生" />
                    <el-radio class="answer" label="高校科研院所研究人员" />
                    <el-radio class="answer" label="创业公司合伙人/股东" />
                    <el-radio class="answer" label="受雇兼职创业" />
                    <el-radio class="answer" label="政府/国企/事业单位雇员" />
                    <el-radio class="answer" label="自由工作者" />
                    <el-radio class="answer" label="无业" />
                    <el-radio class="answer" label="不清楚" />
                </el-radio-group>
            </el-form-item>

            <el-form-item v-if="!jumpToB05&&!jumpToC" class="question" style="font-weight: bolder;"
                label="B02.专利权人所在企业/机构在上海地区大约有多少全职员工？">
                <el-text class="answer" style="font-family: Kaiti;font-weight: 100;text-indent: 2em;">2018年，大约<el-input
                        size="small" v-model="form.pBq0201" style="width: 5vw;margin-left: 0.5vw"
                        placeholder="2018 年" />人</el-text>
            </el-form-item>

            <el-form-item v-if="!jumpToB05 && !jumpToC" class="question" style="font-weight: bolder;" label="">
                <el-text class="answer" style="font-family: Kaiti;font-weight: 100;text-indent: 2em;">2023年，大约<el-input
                        size="small" v-model="form.pBq0202" style="width: 5vw;margin-left: 0.5vw;"
                        placeholder="2023 年" />人</el-text>
            </el-form-item>

            <el-form-item v-if="!jumpToB05 && !jumpToC" class="question" style="font-weight: bolder;"
                label="B03.专利权人所在企业/机构在2018年和2023年是否外包以下业务？">
                <el-table :data="tablePBQ3" style="width: 100%" :row-style="{ height: '10px' }"
                    :cell-style="{ padding: '0px' }">
                    <el-table-column class="answer" width="135%">
                        <template #default="{ row }">
                            {{ row.name }}
                        </template>
                    </el-table-column>
                    <el-table-column label="2018年">
                        <el-table-column width="90%" class="answer" header-align="center"
                            v-for="(column, colIndex) in colPBQ3.slice(0, 3)" :key="colIndex" :label="column.label">
                            <template #default="{ row }">
                                <el-checkbox class="table-container1" v-model="row.selection[colIndex]"
                                    @change="handlePBQ3(row, colIndex)"></el-checkbox>
                            </template>
                        </el-table-column>
                    </el-table-column>
                    <el-table-column label="2023年">
                        <el-table-column width="90%" class="answer" header-align="center"
                            v-for="(column, colIndex) in colPBQ3.slice(3, 6)" :key="colIndex" :label="column.label">
                            <template #default="{ row }">
                                <el-checkbox class="table-container1" v-model="row.selection[colIndex+3]"
                                    @change="handlePBQ3(row, colIndex+3)"></el-checkbox>
                            </template>
                        </el-table-column>
                    </el-table-column>
                </el-table>
            </el-form-item>

            <el-form-item v-if="!jumpToB05 && !jumpToC" class="question" style="font-weight: bolder;"
                label="B04. 专利权人所在企业/机构研发费用占营收收入比例大约是多少？">
                <el-text class="answer" style="font-family: Kaiti;font-weight: 100;text-indent: 2em;">2018年,大约<el-input
                        size="small" v-model="form.pBq0401" style="width: 5vw;margin-left: 0.5vw;"
                        placeholder="" />万人民币</el-text>
            </el-form-item>
            <el-form-item v-if="!jumpToB05 && !jumpToC" class="question" style="font-weight: bolder;" label="">
                <el-text class="answer" style="font-family: Kaiti;font-weight: 100;text-indent: 2em;">2023年,大约<el-input
                        size="small" v-model="form.pBq0402" style="width: 5vw;margin-left: 0.5vw;"
                        placeholder="" />万人民币</el-text>
            </el-form-item>

            <el-form-item class="question" v-if="!jumpToC" style="font-weight: bolder;" label="B05. 专利权人所在企业/机构是否有专门的知识产权部门/团队？">
                <el-radio-group v-model="form.pBq05" @change="handlepBq05Change">
                    <el-radio class="answer" label="是" />
                    <el-radio class="answer" label="否，暂时没有相关设置" />
                </el-radio-group>
            </el-form-item>

            <el-form-item class="question" v-if="showpBq05" style="font-weight: bolder;"
                label="B0501.如果设立了，请问知识产权部门是如何设置的？">
                <el-radio-group v-model="form.pBq0501">
                    <el-radio class="answer" label="完全独立" />
                    <el-radio class="answer" label="设置在研发部下" />
                    <el-radio class="answer" label="设置在法务部下" />
                    <el-radio class="answer" label="其他，请注明"><el-input size="small"
                            placeholder="其他，请注明"></el-input></el-radio>
                </el-radio-group>
            </el-form-item>

            <el-form-item class="question" v-if="showpBq05" style="font-weight: bolder;"
                label="B0502. 如果设立了，请问贵司的知识产权部门/团队有多少全职员工？">
                <el-text class="answer" style="font-family: Kaiti;font-weight: 100;text-indent: 2em;">大约<el-input
                        size="small" v-model="form.pBq0502" style="width: 5vw;margin-left: 0.5vw"
                        placeholder="" />人</el-text>
            </el-form-item>

        </el-form>
        <el-button type="primary" @click="submit()" style="margin-top: 1vh;margin-left: 2vw;">提交问卷（B部分）</el-button>
    </el-card>
</template>

<script setup>
import { ref, reactive, defineProps, defineEmits } from 'vue';
import { surveyStore,tableColChange } from '../../../../stores/survey';
import axios from 'axios';
import { ElMessage } from 'element-plus';
const surveyInfo = surveyStore().surveyInfo
const form = reactive({
    pBq01: "",
    pBq0201: "",
    pBq0202: "",
    pBq03: [],
    pBq0401: "",
    pBq0402: "",
    pBq05: "",
    pBq0501: "",
    pBq0502: "",
});

// 跳转B05情况
const jumpToB05 = ref(false);
// 跳转C情况
const jumpToC = ref(false);
const handlepBq01Change = (value) => {
    jumpToB05.value = value === "在读学生" || value === "高校科研院所研究人员"
    jumpToC.value = value === "自由工作者" || value === "无业" || value === "不清楚"
}

//跳转以及互斥
const showpBq05 = ref(false);
const handlepBq05Change = (value) => {
    showpBq05.value = value === "是";
};


//以下实现所有表格
const tablePBQ3 = ref([
    { name: "原材料生产", selection: [false, false, false, false, false, false] },
    { name: "研发", selection: [false, false, false, false, false, false] },
    { name: "IP 管理", selection: [false, false, false, false, false, false] },
    { name: "信息系统", selection: [false, false, false, false, false, false] },
    { name: "产品生产或服务", selection: [false, false, false, false, false, false] },
    { name: "分发/运输", selection: [false, false, false, false, false, false] },
    { name: "零售", selection: [false, false, false, false, false, false] },
]);

const colPBQ3 = [
    { label: "没有外包" },
    { label: "部分外包" },
    { label: "全部外包" },
    { label: "没有外包" },
    { label: "部分外包" },
    { label: "全部外包" },
];

// 处理单元格选中状态变化
const handlePBQ3 = (rowIndex, col) => {
    //tableColChange(tablePBQ3.value,rowIndex,col)
    form.pBq03 = tablePBQ3
};

const emits = defineEmits(['switch-tab']);

const switchToPart = (tabName) => {
    emits('switch-tab', tabName);
}

const changeTable = (table, col) => {
    let results = [];
    for (let i = 0; i < table.length; i++) {
        for (let j = 0; j < table[i].selection.length; j++) {
            if (table[i].selection[j]) {
                if(j<3){
                    results.push({ row: table[i].name, col: "2018" + col[j].label });
                }else{
                    results.push({ row: table[i].name, col: "2023" + col[j].label });
                }
                
            }
        }
    }
    return results
}

const submit = async () => {

    // 深拷贝
    let formData = JSON.parse(JSON.stringify(form));
    formData.pBq03 = changeTable(formData.pBq03, colPBQ3)

    // 将表单数据转换为对象数组
    const formDataArray = Object.entries(formData).map(([key, value]) => ({ [key]: value }));

    // 将对象数组字符串化
    const formDataString = JSON.stringify(formDataArray);

    const patentNo = surveyInfo.patentNo

    console.log(patentNo)
    console.log(formDataString);


    const invitationCode = surveyInfo.curInvitationCode

    const data = {
        invitationCode: invitationCode,
        patentNo: patentNo,
        type: "个人",
        enterprise: formDataString
    };
    let response = await axios.post('/api/survey/enterprise', data);
    if (response.status == 200) {
        if (response.data.code == 1) {
            ElMessage.success("submit successfully")
            switchToPart("C")
        }
    }
}
</script>

<style scoped>
:deep(.el-radio__label) {
    white-space: normal;
    /* 换行 */
}

:deep(.el-checkbox__label) {
    white-space: normal;
    /* 换行 */
}
.question {
    font-weight: bolder;
    font-family: SimSun;
}

.answer {
    font-family: KaiTi;
    margin-left: 2em;
}
.el-table {
    margin-left: 2.5em;
    margin-top: 1vh;
}
::v-deep .el-form-item__label {
    font-size: 17px;

}

::v-deep.el-table th {
    border: 1px solid rgb(105, 102, 102) !important;
    border-right: none !important;
    border-bottom: none !important;
    border-top: none !important;
    /* border-left: none !important; */
}

::v-deep.el-table td {
    border: 1px solid rgb(105, 102, 102);
    border-right: none !important;
    border-bottom: none !important;
    /* border-left: none !important; */
}

::v-deep .el-table {
    border-collapse: separate;
    border: none !important;
}
.table-container {
    display: flex;
    margin-left: 2vw;
}
.table-container1 {
    display: flex;
    justify-content: center;

    align-items: center;

}
</style>
