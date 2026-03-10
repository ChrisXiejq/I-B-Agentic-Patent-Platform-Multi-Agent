<template>
    <el-card style="border-radius: 15px;width: 100%;">
        <p style="font-family: STKaiti;color: blue;font-weight: bold;">
            （在B部分，我们想了解下您所在高校的科研与知识产权管理情况，部分信息需要您提供2018年与2023年的历史信息，谢谢支持！）
        </p>
        <el-form :model="form" size="large" label-position="top">
            <el-form-item class="question" style="font-weight: bolder;" label="B01.贵校是否设立了专门的知识产权部门/团队？">
                <el-radio-group v-model="form.pBq01" @change="handlePBQ01Change">
                    <el-radio class="answer" label="是" />
                    <el-radio class="answer" label="否，暂时没有相关设置" />
                </el-radio-group>
            </el-form-item>

            <el-form-item class="question" v-if="showPBQ01" style="font-weight: bolder;" label="B0101.如果设立了，请问设立的年份是？">
                <el-date-picker style="margin-left: 2vw;" size="small" v-model="form.pBq0101" type="year"
                    placeholder="选择年份" />
            </el-form-item>

            <el-form-item class="question" v-if="showPBQ01" style="font-weight: bolder;"
                label="B0102.如果设立了，请问知识产权部门是如何设置的？">
                <el-radio-group v-model="form.pBq0102">
                    <el-radio class="answer" label="完全独立" />
                    <el-radio class="answer" label="设置在研发部下" />
                    <el-radio class="answer" label="设置在法务部下" />
                    <el-radio class="answer" label="其他，请注明"><el-input size="small"
                            placeholder="其他，请注明"></el-input></el-radio>
                </el-radio-group>
            </el-form-item>

            <el-form-item class="question" v-if="showPBQ01" style="font-weight: bolder;"
                label="B0103.如果设立了，请问贵校的知识产权部门有多少全职员工？">
                <el-text class="answer" style="font-family: Kaiti;font-weight: 100;text-indent: 2em;">大约<el-input
                        size="small" v-model="form.pBq0103" style="width: 5vw;margin-left: 0.5vw"
                        placeholder="" />人</el-text>
            </el-form-item>
            <el-form-item class="question" style="font-weight: bolder;" label="B2.贵校在专利管理工作中，是否经常使用以下在线数据库/平台？">
                <el-form-item class="question blue-label" style="font-weight: bolder;" label="（请打分，1🌟为不使用，5🌟为经常使用）" />
                <el-table :data="tablePBQ2" style="width: 100%" :row-style="{ height: '10px' }"
                    :cell-style="{ padding: '0px' }">
                    <el-table-column class="answer" width="300%">
                        <template #default="{ row, $index }">
                            <template v-if="$index !== tablePBQ2.length - 1">
                                {{ row.name }}
                            </template>
                            <template v-else>
                                其他互联网资源
                                <el-input v-model="tablePBQ2[$index].name" style="height: 3vh;"
                                    placeholder="其他，请说明"></el-input>
                            </template>
                        </template>
                    </el-table-column>
                    <el-table-column width="250%" class="answer" header-align="center"
                        v-for="(column, colIndex) in colPBQ2" :key="colIndex" :label="column.label">
                        <template #default="{ row }">
                            <el-rate text-color="#ff9900" size="large" v-model="row.rate"
                                :texts="['not', 'so-so', 'relative', 'very', 'super']" show-text class="table-container"
                                @change="handlePBQ2(row, colIndex)" />
                        </template>
                    </el-table-column>
                </el-table>
            </el-form-item>

            <el-form-item class="question" style="font-weight: bolder;" label="B3.过去五年，贵校是否有过以下专利商业化经历？">
                <el-checkbox-group v-model="form.pBq3" style="display: flex;flex-wrap: wrap;"
                    @change="handlePBQ3Change">
                    <el-checkbox class="answer" label="专利转让" />
                    <el-checkbox class="answer" label="集团内部专利许可" />
                    <el-checkbox class="answer" label="与第三方专利许可" />
                    <el-checkbox class="answer" label="交叉许可、专利联盟" />
                    <el-checkbox class="answer" label="专利融资（包括质押、信托、作价入股等）" />
                    <el-checkbox class="answer" label="以专利为基础的技术合同" />
                    <el-checkbox class="answer" label="无专利商业化经历" />
                    <el-checkbox class="answer" label="其他（请注明:__________"><el-input size="small" placeholder="其他，请注明"
                            v-model="extraInput2"></el-input></el-checkbox>
                </el-checkbox-group>
            </el-form-item>

            <el-form-item class="question" v-if="showPBQ3" style="font-weight: bolder;"
                label="B201.如果有,交易信息是通过什么渠道获得的？">
                <el-checkbox-group v-model="form.pBq301" style="display: flex;flex-wrap: wrap;">
                    <el-checkbox class="answer" label="有合作的供应商" />
                    <el-checkbox class="answer" label="有合作的生产商" />
                    <el-checkbox class="answer" label="朋友介绍" />
                    <el-checkbox class="answer" label="协会推荐" />
                    <el-checkbox class="answer" label="在线交易平台" />
                    <el-checkbox class="answer" label="技术交易中心" />
                    <el-checkbox class="answer" label="中介服务机构" />
                    <el-checkbox class="answer" label="其他（请注明:__________"><el-input size="small" placeholder="其他，请注明"
                            v-model="extraInput3"></el-input></el-checkbox>
                </el-checkbox-group>
            </el-form-item>
        </el-form>
        <el-button type="primary" @click="submit()" style="margin-top: 1vh;margin-left: 2vw;">提交问卷（B部分）</el-button>
    </el-card>
</template>

<script setup>
import { ref, reactive, defineProps, defineEmits } from 'vue';
import { surveyStore, tableColChange } from '../../../../stores/survey';
import axios from 'axios';
import { ElMessage } from 'element-plus';
const surveyInfo = surveyStore().surveyInfo
const form = reactive({
    pBq01: "",
    pBq0101: "",
    pBq0102: "",
    pBq0103: "",
    pBq2: [],
    pBq3: [],
    pBq301: [],
});

const extraInput1 = ref('')
const extraInput2 = ref('')
const extraInput3 = ref('')

const showPBQ3 = ref(false)
const handlePBQ3Change = (value) => {
    if (value.includes('无专利商业化经历')) {
        form.pBq3 = ['无专利商业化经历'];
        extraInput2.value = '';
        showPBQ3.value = false
    } else {
        showPBQ3.value = form.pBq3.length > 0
    }
};

//跳转以及互斥
const showPBQ01 = ref(false);
const handlePBQ01Change = (value) => {
    showPBQ01.value = value === "是";
};

const tablePBQ2 = ref([
    { name: "中国专利信息网", rate: 0 },
    { name: "世界知识产权组织专利数据库", rate: 0 },
    { name: "USPTO、EPO 等其他域外专利管理机构公开数据平台", rate: 0 },
    { name: "商业性专利信息数据库（例如智慧芽、Incopat 等）", rate: 0 },
    { name: "上海市知识产权信息平台", rate: 0 },
    { name: "裁判文书网等法律文书公开平台", rate: 0 },
    { name: "商业性法律信息数据库（例如 lexis 等）", rate: 0 },
    { name: "", rate: 0 },
]);

const colPBQ2 = [
    { label: "不使用-->经常使用" },
];

// 处理单元格选中状态变化
const handlePBQ2 = (row, colIndex) => {
    // 取消当前行其他单元格的选中状态
    form.pBq2 = tablePBQ2
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
                if (j < 3) {
                    results.push({ row: table[i].name, col: "2018" + col[j].label });
                } else {
                    results.push({ row: table[i].name, col: "2023" + col[j].label });
                }

            }
        }
    }
    return results
}

const submit = async () => {
    //form.pBq2.pop()
    //form.pBq3.pop()
    //form.pBq301.pop()
    form.pBq2.push(extraInput1.value)
    form.pBq3.push(extraInput2.value)
    form.pBq301.push(extraInput3.value)

    // 深拷贝
    let formData = JSON.parse(JSON.stringify(form));

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
        type: "大学",
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
