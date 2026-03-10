<template>
    <el-card style="border-radius: 15px;width: 100%;">
        <el-form :model="form" size="large" label-position="top">
            <el-form-item class="question" label="A01.您与所选专利的关系是">
                <el-checkbox-group v-model="form.pAq01" style="display: flex;flex-wrap: wrap;" @change="handlePAQ01Change">
                    <el-checkbox class="answer" label="专利发明人之一" />
                    <el-checkbox class="answer" label="参与该专利技术的研发，未列发明人" />
                    <el-checkbox class="answer" label="参与该相关技术的研发，未列发明人" />
                    <el-checkbox class="answer" label="参与该专利申请" />
                    <el-checkbox class="answer" label="专利运营和其他相关事务的人员" />
                    <el-checkbox class="answer" label="以上均无，请说明"><el-input placeholder="以上均无，请说明" size="small"
                            v-model="extraInput1"></el-input></el-checkbox>
                </el-checkbox-group>
            </el-form-item>

            <el-form-item class="question" style="font-weight: bolder;" label="A02.您的性别是">
                <el-radio-group v-model="form.pAq02">
                    <el-radio class="answer" label="女" />
                    <el-radio class="answer" label="男" />
                </el-radio-group>
            </el-form-item>

            <el-form-item class="question" style="font-weight: bolder;" label="A03.您的学历背景是？">
                <el-table :data="tablePAQ3" style="width: 100%" :row-style="{ height: '10px' }"
                :cell-style="{ padding: '0px' }">
                    <el-table-column class="answer" width="180%">
                        <template #default="{ row }">
                            {{ row.name }}
                        </template>
                    </el-table-column>
                    <el-table-column width="160%" class="answer" header-align="center"
                        v-for="(column, colIndex) in colPAQ3" :key="colIndex" :label="column.label">
                        <template #default="{ row }">
                            <!-- 在每个单元格内放置一个可选中的组件 -->
                            <el-checkbox class="table-container1" v-model="row.selection[colIndex]"  @change="handlePAQ3(row, colIndex)"></el-checkbox>
                        </template>
                    </el-table-column>
                </el-table>
            </el-form-item>

            <el-form-item class="question" style="font-weight: bolder;" label="A04.您的以下语言能力如何？">
                <el-table :data="tablePAQ4" style="width: 100%" :row-style="{ height: '10px' }"
                    :cell-style="{ padding: '0px' }">
                    <el-table-column class="answer" width="100%">
                        <template #default="{ row, $index }">
                            <template v-if="$index !== tablePAQ4.length - 1">
                                {{ row.name }}
                            </template>
                            <template v-else>
                                德语，日语等第二外语
                                <el-input v-model="tablePAQ4[$index].name" style="height: 2vh;"
                                    placeholder="请注明"></el-input>
                            </template>
                        </template>
                    </el-table-column>
                    <el-table-column width="110%" class="answer" header-align="center"
                        v-for="(column, colIndex) in colPAQ4" :key="colIndex" :label="column.label">
                        <template #default="{ row }">
                            <el-checkbox class="table-container1" v-model="row.selection[colIndex]"
                                @change="handlePAQ4(row, colIndex)"></el-checkbox>
                        </template>
                    </el-table-column>
                </el-table>
            </el-form-item>

            <el-form-item class="question" style="font-weight: bolder;" label="A05.您是否有境外工作/学习经历？">
                <el-form-item class="question blue-label" style="font-weight: bolder;" label="（含疫情期间的在线学习时间）" />
                <el-table :data="tablePAQ5" style="width: 100%" :row-style="{ height: '10px' }"
                    :cell-style="{ padding: '0px' }">
                    <el-table-column class="answer" width="150%">
                        <template #default="{ row }">
                            {{ row.name }}
                        </template>
                    </el-table-column>
                    <el-table-column width="132%" class="answer" header-align="center"
                        v-for="(column, colIndex) in colPAQ5" :key="colIndex" :label="column.label">
                        <template #default="{ row }">
                            <el-checkbox class="table-container1" v-model="row.selection[colIndex]"
                                @change="handlePAQ5(row, colIndex)"></el-checkbox>
                        </template>
                    </el-table-column>
                </el-table>
            </el-form-item>


            <el-form-item class="question" style="font-weight: bolder;" label="A06.您目前的学习/工作状态是？">
                <el-radio-group v-model="form.pAq06">
                    <el-radio class="answer" label="在读学生" />
                    <el-radio class="answer" label="研究员/高校教师" />
                </el-radio-group>
            </el-form-item>

            <el-form-item class="question" style="font-weight: bolder;" label="A07.如果成功实施该专利（例如签订许可合同等），您是否能获得以下奖励？">
                <el-checkbox-group v-model="form.pAq07" style="display: flex;flex-wrap: wrap;"
                @change="handlePAQ07Change">
                    <el-checkbox class="answer" label="没有额外奖励，是必须完成的考核指标" />
                    <el-checkbox class="answer" label="更快的职业晋升" />
                    <el-checkbox class="answer" label="获得一次性现金奖励" />
                    <el-checkbox class="answer" label="获得许可费收益的提成" />
                    <el-checkbox class="answer" label="获得股权激励" />
                    <el-checkbox class="answer" label="其他（请注明:__________"><el-input placeholder="其他，请注明" size="small"
                            v-model="extraInput2"></el-input></el-checkbox>
                </el-checkbox-group>
            </el-form-item>
            
            <el-form-item class="question" v-if="showPAQ08" style="font-weight: bolder;" label="A08.如果有奖励，贵校如何进行计数？">
                <el-radio-group v-model="form.pAq08">
                    <el-radio class="answer" label="根据所完成的专利商业化合同数量" />
                    <el-radio class="answer" label="根据所完成的专利商业化合同金额" />
                    <el-radio class="answer" label="不做区分" />
                </el-radio-group>
            </el-form-item>

            <el-form-item class="question" style="font-weight: bolder;" label="A09.您认为哪一类考核/奖励有利于激励您实施专利商业化？">
                <el-form-item class="question blue-label" style="font-weight: bolder;" label="（请打分，1🌟为不重要，5🌟为非常重要）" />
                <el-table :data="tablePAQ09" style="width: 100%;" :row-style="{ height: '10px' }"
                    :cell-style="{ padding: '0px' }">
                    <el-table-column class="answer" width="250%">
                        <template #default="{ row, $index }">
                            <template v-if="$index !== tablePAQ09.length - 1">
                                {{ row.name }}
                            </template>
                            <template v-else>
                                <el-input v-model="tablePAQ09[$index].name" style="height: 3vh;"
                                    placeholder="其他，请说明"></el-input>
                            </template>
                        </template>
                    </el-table-column>
                    <el-table-column width="250%" class="answer" header-align="center"
                        v-for="(column, colIndex) in colPAQ09" :key="colIndex" :label="column.label">
                        <template #default="{ row }">
                            <!-- 在每个单元格内放置一个可选中的组件 -->
                            <el-rate text-color="#ff9900" size="large" v-model="row.rate"
                                :texts="['not', 'so-so', 'relative', 'very', 'super']" show-text class="table-container"
                                @change="handlePAQ09(row, colIndex)" />
                        </template>
                    </el-table-column>
                </el-table>
            </el-form-item>


        </el-form>
        <el-button type="primary" @click="submit()" style="margin-top: 1vh;margin-left: 2vw;">提交问卷（A部分）</el-button>
    </el-card>
</template>

<script setup>
import { ref, reactive, defineProps, defineEmits } from 'vue';
import { surveyStore,tableRowChange,tableColChange} from '../../../../stores/survey';
import axios from 'axios';
import { ElMessage } from 'element-plus';

const surveyInfo = surveyStore().surveyInfo

const form = reactive({
    pAq01: [],
    pAq02: "",
    pAq03: [],
    pAq04: [],
    pAq05: [],
    pAq06: [],
    pAq07: [],
    pAq08: "",
    pAq09: []
});

const extraInput1 = ref("")
const extraInput2 = ref("")

// 跳转逻辑以及多选题互斥逻辑
const handlePAQ01Change = (value) => {
    if (value.includes('专利发明人之一')) {  
        form.pAq01=value.filter(item => item !== '参与该专利技术的研发，未列发明人' && item !== '参与该相关技术的研发，未列发明人');  
    } 
    if(value.includes('以上均无，请说明')){
        form.pAq01=value.filter(item => item !== '专利发明人之一' && item !== '参与该专利技术的研发，未列发明人' 
        && item !== '参与该相关技术的研发，未列发明人'&& item !== '参与该专利申请'&& item !== '专利运营和其他相关事务的人员');  
    }
};

const showPAQ08 = ref(false);
const handlePAQ07Change = (value) => {
    if (value.includes('没有额外奖励，是必须完成的考核指标')) {  
        form.pAq07 = ['没有额外奖励，是必须完成的考核指标']; 
        value='没有额外奖励，是必须完成的考核指标'
        extraInput2.value = '';
    }
    showPAQ08.value = (value != "没有额外奖励，是必须完成的考核指标" && value!='');
};


// 以下处理所有表格
const tablePAQ3 = ref([
    { name: "法学", selection: [false, false, false] },
    { name: "经济学、管理学", selection: [false, false, false] },
    { name: "其他人文社会科学（文学、历史学或教育学等）", selection: [false, false, false] },
    { name: "工学", selection: [false, false, false] },
    { name: "理学", selection: [false, false, false] },
    { name: "医学",  selection: [false, false, false] },
    { name: "农学",  selection: [false, false, false] },
    { name: "无",  selection: [false, false, false] },
]);

const colPAQ3 = [
    { label: "本科或同等学历学位" },
    { label: "硕士或同等学历学位" },
    { label: "博士或同等学历学位" },
];

// 单元格选中状态变化
const handlePAQ3 = (row, colIndex) => {
    tableRowChange(tablePAQ3.value,colIndex,row)
    form.pAq03 = tablePAQ3
};


const tablePAQ4 = ref([
    { name: "普通话",  selection: [false, false, false, false, false] },
    { name: "上海话",  selection: [false, false, false, false, false] },
    { name: "粤语",  selection: [false, false, false, false, false] },
    { name: "英文",  selection: [false, false, false, false, false] },
    { name: "", selection: [false, false, false, false, false] },
]);

const colPAQ4 = [
    { label: "没有掌握" },
    { label: "能基础听说读写" },
    { label: "能交流日常事务" },
    { label: "能交流专业事务" },
    { label: "母语水平" },
];

// 处理单元格选中状态变化
const handlePAQ4 = (row, colIndex) => {
    const rowIndex = tablePAQ4.value.indexOf(row);
    tableColChange(tablePAQ4.value,rowIndex,colIndex)
    form.pAq04 = tablePAQ4
};


const tablePAQ5 = ref([
    { name: "少于三个月的学习经历",  selection: [false, false, false] },
    { name: "少于三个月的实习/工作经历", selection: [false, false, false] },
    { name: "多于三个月的学习经历", selection: [false, false, false] },
    { name: "多于三个月的实习/工作经历", selection: [false, false, false] },
    { name: "无", selection: [false, false, false] },
]);

const colPAQ5 = [
    { label: "港澳台" },
    { label: "日本、韩国或其他亚洲国家" },
    { label: "欧洲、北美洲和大洋洲" },
    { label: "其他国家和地区" },
];

// 处理单元格选中状态变化
const handlePAQ5 = (row, colIndex) => {
    tableRowChange(tablePAQ5.value,colIndex,row)
    form.pAq05 = tablePAQ5
};

const tablePAQ09 = ref([
    { name: "作为必须完成的硬性指标", rate: 0 },
    { name: "作为晋升的附加项", rate: 0 },
    { name: "签订合同后的一次性现金奖励", rate: 0 },
    { name: "基于合同许可费收益的提成", rate: 0 },
    { name: "股权激励", rate: 0 },
    { name: "增加人手、扩大部门规模", rate: 0 },
    { name: "提高部门的受重视程度", rate: 0 },
    { name: "", rate: 0 },
]);

const colPAQ09 = [
    { label: "不重要-->重要" }
];

// 处理单元格选中状态变化
const handlePAQ09 = (row, colIndex) => {
    form.pAq09 = tablePAQ09
};

const emits = defineEmits(['switch-tab']);

const switchToPart = (tabName) => {
    emits('switch-tab',  tabName);
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

    //form.pAq01.pop()
    //form.pAq07.pop()
    form.pAq01.push(extraInput1.value)
    form.pAq07.push(extraInput2.value)

    // 深拷贝
    let formData = JSON.parse(JSON.stringify(form));
    formData.pAq03 = changeTable(formData.pAq03,colPAQ3)
    formData.pAq04 = changeTable(formData.pAq04,colPAQ4)
    formData.pAq05 = changeTable(formData.pAq05,colPAQ5)

    // 将表单数据转换为对象数组
    const formDataArray = Object.entries(formData).map(([key, value]) => ({ [key]: value }));

    // 将对象数组字符串化
    const formDataString = JSON.stringify(formDataArray);

    const patentNo = surveyInfo.patentNo

    const invitationCode = surveyInfo.curInvitationCode

    const data = {
        invitationCode: invitationCode,
        patentNo: patentNo,
        type: "大学",
        identification: formDataString
    };
    let response = await axios.post('/api/survey/identification', data);
    if (response.status == 200) {
        if (response.data.code == 1) {
            ElMessage.success("submit successfully")
            switchToPart('B');
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
::v-deep .el-form-item__label {
    font-size: 17px;
}
.question{
    font-weight: bolder;
    /* font-size: 30px; */
    font-family: SimSun;
}
.answer{
    font-family: KaiTi;
    margin-left: 2em;
}

.el-table{
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
    
    align-items: center;
     */
}
.table-container1 {
    display: flex;
    justify-content: center;
    
    align-items: center;
    
}
</style>
