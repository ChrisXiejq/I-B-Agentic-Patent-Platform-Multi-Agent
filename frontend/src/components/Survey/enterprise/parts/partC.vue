<template>
    <el-card style="border-radius: 15px;width: 100%;">
        <p style="font-family: STKaiti;color: blue;font-weight: bold;">
            （C部分，我们将根据抽取的样本专利（参见附件文档），请您提供专利的价值判断）
        </p>

        <el-form :model="form" size="large" label-position="top">
            <el-form-item class="question" style="font-weight: bolder;" label="C01.该专利属于哪一类创新？">
                <el-radio-group v-model="form.pCq01" @change="handlePCQ01Change">
                    <el-radio class="answer" label="产品创新" />
                    <el-radio class="answer" label="流程创新" />
                    <el-radio class="answer" label="都相关" />
                </el-radio-group>
            </el-form-item>

            <el-form-item class="question" v-if="showPCQ01" style="font-weight: bolder;"
                label="C0101.如果属于产品创新，相关技术产品在哪个市场阶段？">
                <el-radio-group v-model="form.pCq0101">
                    <el-radio class="answer" label="已进入市场" />
                    <el-radio class="answer" label="为引入市场做准备" />
                    <el-radio class="answer" label="远未进入市场" />
                    <el-radio class="answer" label="已退出市场" />
                </el-radio-group>
            </el-form-item>

            <el-form-item class="question" v-if="showPCQ01" style="font-weight: bolder;" label="C0102.贵公司是否生产该产品？">
                <el-radio-group v-model="form.pCq0102">
                    <el-radio class="answer" label="是" />
                    <el-radio class="answer" label="否" />
                </el-radio-group>
            </el-form-item>

            <el-form-item class="question" v-if="showPCQ01" style="font-weight: bolder;" label="C0103.如何评价该产品市场的竞争程度？">
                <el-table :data="tablePCQ0103" style="width: 100%" :row-style="{ height: '10px' }"
                    :cell-style="{ padding: '0px' }">
                    <el-table-column class="answer" width="200%">
                        <template #default="{ row }">
                            {{ row.name }}
                        </template>
                    </el-table-column>
                    <el-table-column width="150%" class="answer" header-align="center"
                        v-for="(column, colIndex) in colPCQ0103" :key="colIndex" :label="column.label">
                        <template #default="{ row }">
                            <el-checkbox class="table-container1" v-model="row.selection[colIndex]"
                                @change="handlePCQ0103(row, colIndex)"></el-checkbox>
                        </template>
                    </el-table-column>
                </el-table>
            </el-form-item>

            <el-form-item class="question" v-if="showPCQ01" style="font-weight: bolder;" label="C0104.有多少家厂商正在生产该专利产品？">
                <el-text class="answer" style="font-family: Kaiti;font-weight: 100;text-indent: 2em;">大约<el-input
                        size="small" v-model="form.pCq0104" style="width: 5vw;margin-left: 0.5vw"
                        placeholder="" />家。</el-text>
            </el-form-item>
            
            <el-form-item class="question" v-if="showPCQ01" style="font-weight: bolder;"
                label="C0105.如果形容贵公司的专利产品的市场份额？">
                <el-radio-group v-model="form.pCq0105">
                    <el-radio class="answer" label="占同类产品市场销售额 5%以下" />
                    <el-radio class="answer" label="占同类产品市场销售额 5%-20%" />
                    <el-radio class="answer" label="占同类产品市场销售额 20%-50%" />
                    <el-radio class="answer" label="占同类产品市场销售额 50%以上" />
                    <el-radio class="answer" label="不清楚" />
                </el-radio-group>
            </el-form-item>

            <el-form-item class="question" v-if="showPCQ02" style="font-weight: bolder;" label="C02.作为流程创新，该专利技术能降低多少生产成本？">
                <el-radio-group v-model="form.pCq02">
                    <el-radio class="answer" label="少于 5%" />
                    <el-radio class="answer" label="5%-20%" />
                    <el-radio class="answer" label="20%-50%" />
                    <el-radio class="answer" label="超过 50%" />
                </el-radio-group>
            </el-form-item>

            <el-form-item class="question" style="font-weight: bolder;" label="C03.如何形容该专利技术的研发成本？请估计下已经耗费的人工年？">
                <el-form-item class="question blue-label" style="font-weight: bolder;"
                    label="（1 人工年指 1 个人一年，请将人数和年数相乘填写，例如，2 个人三年：2×3=6 个人工年）" />
                <el-radio-group v-model="form.pCq03">
                    <el-radio class="answer" label="少于 1 个人工年" />
                    <el-radio class="answer" label="2-4 个人工年" />
                    <el-radio class="answer" label="5-10 个人工年" />
                    <el-radio class="answer" label="10-20 个人工年" />
                    <el-radio class="answer" label="超过 20 个人工年" />
                </el-radio-group>
            </el-form-item>

            <el-form-item class="question" style="font-weight: bolder;" label="C04.该专利技术是否有研发合作？如果有，是与哪类机构/企业进行合作？">
                <el-checkbox-group v-model="form.pCq04" style="display: flex;flex-wrap: wrap;" @change="handlePCQ04Change">
                    <el-checkbox class="answer" label="没有，完全由本单位研发" />
                    <el-checkbox class="answer" label="有，同一企业集团的公司之间合作" />0
                    <el-checkbox class="answer" label="有，与来自私营部门的客户合作" />
                    <el-checkbox class="answer" label="有，与来自公共机构（医院等）的客户合作" />
                    <el-checkbox class="answer" label="有，与大学科研院所合作" />
                    <el-checkbox class="answer" label="有，与软件、设备等供应商合作" />
                    <el-checkbox class="answer" label="有，与同行业其他企业合作" />
                    <el-checkbox class="answer" label="有，与科技孵化企业合作" />
                </el-checkbox-group>
            </el-form-item>

            <el-form-item class="question" style="font-weight: bolder;" label="C05.该专利技术的研发经费来自以下哪些渠道？">
                <el-checkbox-group v-model="form.pCq05" style="display: flex;flex-wrap: wrap;">
                    <el-checkbox class="answer" label="企业营收" />
                    <el-checkbox class="answer" label="银行商业贷款" />
                    <el-checkbox class="answer" label="风险投资" />
                    <el-checkbox class="answer" label="政府资助" />
                    <el-checkbox class="answer" label="其他，请注明"><el-input size="small" placeholder="其他，请注明"
                            v-model="extraInput1"></el-input></el-checkbox>
                </el-checkbox-group>
            </el-form-item>

            <el-form-item class="question" style="font-weight: bolder;" label="C06.您认为该技术是否解决了行业内的技术瓶颈？">
                <el-radio-group v-model="form.pCq06">
                    <el-radio class="answer" label="是" />
                    <el-radio class="answer" label="不了解" />
                    <el-radio class="answer" label="不确定" />
                </el-radio-group>
            </el-form-item>

            <el-form-item class="question" style="font-weight: bolder;" label="C07.该专利技术是否遭受过侵权？">
                <el-radio-group v-model="form.pCq07">
                    <el-radio class="answer" label="是，经常" />
                    <el-radio class="answer" label="是，偶尔" />
                    <el-radio class="answer" label="没有" />
                    <el-radio class="answer" label="不了解" />
                </el-radio-group>
            </el-form-item>

            <el-form-item class="question" style="font-weight: bolder;" label="C08.如何评价其他知识产权保护对该专利技术产品的重要性？">
                <el-form-item class="question blue-label" style="font-weight: bolder;" label="（请打分，1🌟为不重要，5🌟为非常重要）" />
                <el-table :data="tablePCQ8" style="width: 100%" :row-style="{ height: '10px' }"
                    :cell-style="{ padding: '0px' }">
                    <el-table-column class="answer" width="250%">
                        <template #default="{ row, $index }">
                            <template v-if="$index !== tablePCQ8.length - 1">
                                {{ row.name }}
                            </template>
                            <template v-else>
                                <el-input v-model="tablePCQ8[$index].name" style="height: 3vh;"
                                    placeholder="其他，请说明"></el-input>
                            </template>
                        </template>
                    </el-table-column>
                    <el-table-column width="250%" class="answer" header-align="center"
                        v-for="(column, colIndex) in colPCQ8" :key="colIndex" :label="column.label">
                        <template #default="{ row }">
                            <el-rate text-color="#ff9900" size="large" v-model="row.rate"
                                :texts="['not', 'so-so', 'relative', 'very', 'super']" show-text class="table-container"
                                @change="handlePCQ8(row, colIndex)" />
                        </template>
                    </el-table-column>
                </el-table>
            </el-form-item>

            <el-form-item class="question" style="font-weight: bolder;" label="C09.在未来三年内，贵司是否会继续开发该专利技术？">
                <el-radio-group v-model="form.pCq09">
                    <el-radio class="answer" label="是 ，会增加相关投入" />
                    <el-radio class="answer" label="是，但要减少相关研发投入" />
                    <el-radio class="answer" label="是，投入不变" />
                    <el-radio class="answer" label="否" />
                </el-radio-group>
            </el-form-item>

            <el-form-item class="question" style="font-weight: bolder;" label="C10.如果减少或者不再开发，以下原因的影响有多大？">
                <el-form-item class="question blue-label" style="font-weight: bolder;" label="（请打分，1🌟为不影响，5🌟为非常影响）" />
                <el-table :data="tablePCQ10" style="width: 100%" :row-style="{ height: '10px' }"
                    :cell-style="{ padding: '0px' }">
                    <el-table-column class="answer" width="250%">
                        <template #default="{ row }">
                            {{ row.name }}
                        </template>
                    </el-table-column>
                    <el-table-column width="250%" class="answer" header-align="center"
                        v-for="(column, colIndex) in colPCQ10" :key="colIndex" :label="column.label">
                        <template #default="{ row }">
                            <el-rate text-color="#ff9900" size="large" v-model="row.rate"
                                :texts="['not', 'so-so', 'relative', 'very', 'super']" show-text class="table-container"
                                @change="handlePCQ10(row, colIndex)" />
                        </template>
                    </el-table-column>
                </el-table>
            </el-form-item>

            <el-form-item class="question" style="font-weight: bolder;"
                label='C11.该专利是否属于专利组合的一部分？'>
                <el-radio-group v-model="form.pCq11">
                    <el-radio class="answer" label="是的，包括 2-3 项专利" />
                    <el-radio class="answer" label="是的，包括 4-6 项专利" />
                    <el-radio class="answer" label="是的，包括 7-10 项专利" />
                    <el-radio class="answer" label="是的，包括 10-20 项专利" />
                    <el-radio class="answer" label="是的，包括 20 项以上的专利" />
                    <el-radio class="answer" label="否" />
                </el-radio-group>
            </el-form-item>

            <el-form-item class="question" style="font-weight: bolder;" label="C12.假设有第三方购买该专利（专利组合），您预计最低价格是多少？">
                <el-radio-group v-model="form.pCq12">
                    <el-radio class="answer" label="<3 万 人民币" />
                    <el-radio class="answer" label="3 万-10 万 人民币" />
                    <el-radio class="answer" label="10 万-30 万 人民币" />
                    <el-radio class="answer" label="30 万-100 万 人民币" />
                    <el-radio class="answer" label="100 万-300 万 人民币" />
                    <el-radio class="answer" label="300 万-1000 万 人民币" />
                    <el-radio class="answer" label="1000 万-3000 万 人民币" />
                    <el-radio class="answer" label="3000 万-1 亿 人民币" />
                    <el-radio class="answer" label="> 1 亿人民币" />
                </el-radio-group>
            </el-form-item>
        </el-form>
        <el-button type="primary" @click="submit()" style="margin-top: 1vh;margin-left: 2vw;">提交问卷（C部分）</el-button>
    </el-card>
</template>

<script setup>
import { ref, reactive, defineProps, defineEmits } from 'vue';
import { surveyStore } from '../../../../stores/survey';
import axios from 'axios';
import { ElMessage } from 'element-plus';
const surveyInfo = surveyStore().surveyInfo
const form = reactive({
    pCq01: "",
    pCq0101: "",
    pCq0102: "",
    pCq0103: [],
    pCq0104: "",
    pCq0105: "",
    pCq02: "",
    pCq03: "",
    pCq04: [],
    pCq05: [],
    pCq06: "",
    pCq07: "",
    pCq08: [],
    pCq09: "",
    pCq10: [],
    pCq11: "",
    pCq12: "",
});

const extraInput1 = ref('')

//跳转与互斥
const showPCQ01 = ref(false);
const showPCQ02 = ref(false);
const handlePCQ01Change = (value) => {
    showPCQ01.value = value === "产品创新" || value === "都相关";
    showPCQ02.value = value === "流程创新" || value === "都相关";
};

const handlePCQ04Change = (value) => {
    if (value.includes('没有，完全由本单位研发')) {  
        form.pCq04 = ['没有，完全由本单位研发']; 
    }
};


// 以下实现所有表格
const tablePCQ0103 = ref([
    { name: "已经有2-3家龙头企业", selection: [false, false] },
    { name: "产品价格稳定", selection: [false, false] },
    { name: "不断有新企业进入", selection: [false, false] },
    { name: "产品更新换代较快", selection: [false, false] },
    { name: "市场需求还在不断扩大", selection: [false, false] },
    { name: "市场利润较高", selection: [false, false] },
]);

const colPCQ0103 = [
    { label: "2018" },
    { label: "2023" },
];

// 处理单元格选中状态变化
const handlePCQ0103 = (row, colIndex) => {
    // 取消当前行其他单元格的选中状态
    form.pCq0103 = tablePCQ0103
};

const tablePCQ8 = ref([
    { name: "商业秘密", rate: 0 },
    { name: "著作权", rate: 0 },
    { name: "商标", rate: 0 },
    { name: "地理标志", rate: 0 },
    { name: "数据保护（反不正当竞争）", rate: 0 },
    { name: "集成电路布图设计权", rate: 0 },
    { name: "", rate: 0 },
]);

const colPCQ8 = [
    { label: "不重要-->非常重要" },
];

// 处理单元格选中状态变化
const handlePCQ8 = (row, colIndex) => {
    // 取消当前行其他单元格的选中状态
    form.pCq08 = tablePCQ8
};

const tablePCQ10 = ref([
    { name: "生产/材料成本增加", rate: 0 },
    { name: "缺乏有技能的员工", rate: 0 },
    { name: "现金流出现问题", rate: 0 },
    { name: "未来产品市场需求的不确定性", rate: 0 },
    { name: "偏离研究所/公司的核心战略", rate: 0 },
    { name: "科技、税收、产业政策有变化，不能继续获得相应资助", rate: 0 },
]);

const colPCQ10 = [
    { label: "不影响-->非常影响" },
];

// 处理单元格选中状态变化
const handlePCQ10 = (row, colIndex) => {
    // 取消当前行其他单元格的选中状态
    form.pCq10 = tablePCQ10
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
                results.push({ row: table[i].name, col: col[j].label });
            }
        }
    }
    return results
}

const submit = async () => {
    //form.pCq05.pop()
    form.pCq05.push(extraInput1.value)
    // 深拷贝
    let formData = JSON.parse(JSON.stringify(form));
    formData.pCq0103 = changeTable(formData.pCq0103, colPCQ0103)
    
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
        type: "企业",
        value: formDataString
    };
    let response = await axios.post('/api/survey/value', data);
    if (response.status == 200) {
        if (response.data.code == 1) {
            ElMessage.success("submit successfully")
            switchToPart("D")
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
::v-deep .el-form-item__label {
    font-size: 17px;

}

::v-deep .el-table {
    border-collapse: separate;
    border: none !important;
}
::v-deep .blue-label .el-form-item__label {
    color: blue;
    text-indent: 2em;
}
.table-container {
    display: flex;
    margin-left: 2vw;
    /* justify-content: center;
    align-items: center; */
}
.table-container1 {
    display: flex;
    justify-content: center;

    align-items: center;

}
</style>
