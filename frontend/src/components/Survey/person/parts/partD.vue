<template>
    <el-card style="border-radius: 15px;width: 100%;">
        <p style="font-family: STKaiti;color: blue;font-weight: bold;">
            （D部分，我们将根据抽取的样本专利（参见附件文档），请您提供专利的价值判断）
        </p>

        <el-form :model="form" size="large" label-position="top">
            <el-form-item class="question" style="font-weight: bolder;" label="D01.专利权人为什么申请该专利？">
                <el-checkbox-group v-model="form.pDq01" style="display: flex;flex-wrap: wrap;">
                    <el-checkbox class="answer" label="防止他人使用相关技术" />
                    <el-checkbox class="answer" label="获得高新技术企业认定" />
                    <el-checkbox class="answer" label="专利奖项申报以及商业宣传" />
                    <el-checkbox class="answer" label="希望有专利许可转让收入" />
                    <el-checkbox class="answer" label="构建技术标准" />
                    <el-checkbox class="answer" label="更好获得融资" />
                    <el-checkbox class="answer" label="其他，请注明"><el-input size="small" placeholder="其他，请注明"
                            v-model="extraInput1"></el-input></el-checkbox>
                </el-checkbox-group>
            </el-form-item>

            <el-form-item class="question" style="font-weight: bolder;" label="D02.过去三年，专利权人是否为该专利技术进行过以下宣传推广？">
                <el-checkbox-group v-model="form.pDq02" style="display: flex;flex-wrap: wrap;"
                    @change="handlePDQ02Change">
                    <el-checkbox class="answer" label="报纸、杂志等广告宣传" />
                    <el-checkbox class="answer" label="线下展会、技术交易中心" />
                    <el-checkbox class="answer" label="企业官方网站" />
                    <el-checkbox class="answer" label="投放搜索引擎广告" />
                    <el-checkbox class="answer" label="国家、地区网站相关推荐页面" />
                    <el-checkbox class="answer" label="官方（含相关协会）在线交易平台/许可平台" />
                    <el-checkbox class="answer" label="商业性在线交易平台/许可平台" />
                    <el-checkbox class="answer" label="没有进行过宣传推广" />
                    <el-checkbox class="answer" label="其他，请注明"><el-input size="small" placeholder="其他，请注明"
                            v-model="extraInput2"></el-input></el-checkbox>
                </el-checkbox-group>
            </el-form-item>

            <el-form-item class="question" style="font-weight: bolder;" label="D03.专利权人对于该专利，是否已经达成了许可交易？">
                <el-radio-group v-model="form.pDq03" @change="handlePDQ03Change">
                    <el-radio class="answer" label="是的，已达成多个许可交易。" />
                    <el-radio class="answer" label="是的，已达成一个许可交易。" />
                    <el-radio class="answer" label="还没有" />
                </el-radio-group>
            </el-form-item>

            <el-form-item class="question" v-if="showPDQ03" style="font-weight: bolder;"
                label="D0301.如果已经达成许可交易，该交易发生在哪一年？">
                <div>
                    <el-form-item class="question blue-label" style="font-weight: bolder;" label="（如有多次，回答最近一次即可）" />
                    <el-input size="small" class="answer" v-model="form.pDq0301" style="width: 240px"
                        placeholder="年份" />
                </div>
            </el-form-item>

            <el-form-item class="question" v-if="showPDQ03" style="font-weight: bolder;" label="D0302.对外许可该专利的目的是什么？">
                <el-checkbox-group v-model="form.pDq0302" style="display: flex;flex-wrap: wrap;">
                    <el-checkbox class="answer" label="高新技术企业的考核要求" />
                    <el-checkbox class="answer" label="获得许可收入" />
                    <el-checkbox class="answer" label="扩大产品市场" />
                    <el-checkbox class="answer" label="授权生产" />
                    <el-checkbox class="answer" label="合作创新" />
                    <el-checkbox class="answer" label="解决侵权纠纷" />
                    <el-checkbox class="answer" label="获得市场地位" />
                    <el-checkbox class="answer" label="提高企业形象、商业宣传" />
                    <el-checkbox class="answer" label="其他，请注明"><el-input size="small" placeholder="其他，请注明"
                            v-model="extraInput3"></el-input></el-checkbox>
                </el-checkbox-group>
            </el-form-item>

            <el-form-item class="question" v-if="showPDQ03" style="font-weight: bolder;" label="D0303.对外许可的企业类型是？">
                <el-checkbox-group v-model="form.pDq0303" style="display: flex;flex-wrap: wrap;">
                    <el-checkbox class="answer" label="国有企业" />
                    <el-checkbox class="answer" label="民营企业" />
                    <el-checkbox class="answer" label="外资企业（含外商独资、中外合资和中外合作）" />
                    <el-checkbox class="answer" label="港澳台资企业" />
                    <el-checkbox class="answer" label="高校科研院所" />
                    <el-checkbox class="answer" label="不愿披露" />
                </el-checkbox-group>
            </el-form-item>

            <el-form-item class="question" v-if="showPDQ03" style="font-weight: bolder;" label="D0304.对外许可的企业规模是？">
                <el-checkbox-group v-model="form.pDq0304" style="display: flex;flex-wrap: wrap;">
                    <el-checkbox class="answer" label="大型企业（>1000 人）" />
                    <el-checkbox class="answer" label="中型企业（100-1000 人）" />
                    <el-checkbox class="answer" label="高校科研院所" />
                    <el-checkbox class="answer" label="不愿披露" />
                </el-checkbox-group>
            </el-form-item>

            <el-form-item class="question" v-if="showPDQ03" style="font-weight: bolder;" label="D0305.该交易的许可类型是？">
                <el-checkbox-group v-model="form.pDq0305" style="display: flex;flex-wrap: wrap;">
                    <el-checkbox class="answer" label="独占许可" />
                    <el-checkbox class="answer" label="排他许可" />
                    <el-checkbox class="answer" label="普通许可" />
                    <el-checkbox class="answer" label="交叉许可" />
                    <el-checkbox class="answer" label="不方便透露" />
                    <el-checkbox class="answer" label="不清楚" />
                </el-checkbox-group>
            </el-form-item>

            <el-form-item class="question" v-if="showPDQ03" style="font-weight: bolder;" label="D0306.该交易的许可定价是？">
                <el-checkbox-group v-model="form.pDq0306" style="display: flex;flex-wrap: wrap;">
                    <el-checkbox class="answer" label="完全免费" />
                    <el-checkbox class="answer" label="一次性总付" />
                    <el-checkbox class="answer" label="协议使用费率" />
                    <el-checkbox class="answer" label="部分预付+使用费率" />
                    <el-checkbox class="answer" label="不方便透露" />
                    <el-checkbox class="answer" label="不清楚" />
                </el-checkbox-group>
            </el-form-item>

            <el-form-item class="question" v-if="showPDQ04" style="font-weight: bolder;" label="D04.如果尚未达成许可交易，阻碍是什么？">
                <el-checkbox-group v-model="form.pDq04" style="display: flex;flex-wrap: wrap;">
                    <el-checkbox class="answer" label="专利技术仍在开发中，没办法许可" />
                    <el-checkbox class="answer" label="没有发现市场需求" />
                    <el-checkbox class="answer" label="有市场需求，但是没有渠道联系到被许可人" />
                    <el-checkbox class="answer" label="有潜在许可对象，但是许可价格谈不拢" />
                    <el-checkbox class="answer" label="不相信许可对象的合同履行能力" />
                    <el-checkbox class="answer" label="担心技术泄露" />
                    <el-checkbox class="answer" label="担心市场占有率收到损失" />
                    <el-checkbox class="answer" label="不愿意许可" />
                </el-checkbox-group>
            </el-form-item>

            <el-form-item class="question" style="font-weight: bolder;" label="D05.未来三年，专利权人是尝试对外许可该专利？">
                <el-radio-group v-model="form.pDq05" @change="handlePDQ05Change">
                    <el-radio class="answer" label="是的，我们希望能对外许可" />
                    <el-radio class="answer" label="否" />
                </el-radio-group>
            </el-form-item>

            <el-form-item class="question" v-if="showPDQ05" style="font-weight: bolder;" label="D0501.如果有计划许可，目的是什么？">
                <el-checkbox-group v-model="form.pDq0501" style="display: flex;flex-wrap: wrap;">
                    <el-checkbox class="answer" label="高新技术企业的专利商业化考核要求" />
                    <el-checkbox class="answer" label="获得许可收入" />
                    <el-checkbox class="answer" label="扩大产品市场" />
                    <el-checkbox class="answer" label="合作创新" />
                    <el-checkbox class="answer" label="授权生产" />
                    <el-checkbox class="answer" label="解决侵权纠纷" />
                    <el-checkbox class="answer" label="提高企业形象，商业宣传" />
                    <el-checkbox class="answer" label="其他（请注明______）"><el-input size="small" placeholder="其他，请注明"
                            v-model="extraInput4"></el-input></el-checkbox>
                </el-checkbox-group>
            </el-form-item>

            <el-form-item class="question" v-if="showPDQ05" style="font-weight: bolder;"
                label="D0502.如果有计划许可，对外许可的可能企业类型是？">
                <el-checkbox-group v-model="form.pDq0502" style="display: flex;flex-wrap: wrap;">
                    <el-checkbox class="answer" label="国有企业" />
                    <el-checkbox class="answer" label="民营企业" />
                    <el-checkbox class="answer" label="港澳台资企业" />
                    <el-checkbox class="answer" label="高校科研院所" />
                    <el-checkbox class="answer" label="外资企业（含外商独资、中外合资和中外合作）" />
                    <el-checkbox class="answer" label="不愿披露" />
                </el-checkbox-group>
            </el-form-item>

            <el-form-item class="question" v-if="showPDQ05" style="font-weight: bolder;"
                label="D0503.如果有计划许可，对外许可的可能企业类型是？">
                <el-checkbox-group v-model="form.pDq0503" style="display: flex;flex-wrap: wrap;">
                    <el-checkbox class="answer" label="大型企业（>1000 人）" />
                    <el-checkbox class="answer" label="中型企业（100-1000 人）" />
                    <el-checkbox class="answer" label="高校科研院所" />
                    <el-checkbox class="answer" label="不愿披露" />
                </el-checkbox-group>
            </el-form-item>

            <el-form-item class="question" v-if="showPDQ05" style="font-weight: bolder;" label="D0504. 对外许可的可能许可类型是？">
                <el-checkbox-group v-model="form.pDq0504" style="display: flex;flex-wrap: wrap;">
                    <el-checkbox class="answer" label="独占许可" />
                    <el-checkbox class="answer" label="排他许可" />
                    <el-checkbox class="answer" label="普通许可" />
                    <el-checkbox class="answer" label="交叉许可" />
                    <el-checkbox class="answer" label="不方便透露" />
                </el-checkbox-group>
            </el-form-item>

            <el-form-item class="question" v-if="showPDQ05" style="font-weight: bolder;" label="D0505. 对外许可的可能许可定价是？">
                <el-checkbox-group v-model="form.pDq0505" style="display: flex;flex-wrap: wrap;">
                    <el-checkbox class="answer" label="完全免费" />
                    <el-checkbox class="answer" label="一次性总付" />
                    <el-checkbox class="answer" label="协议使用费率" />
                    <el-checkbox class="answer" label="部分预付+使用费率" />
                    <el-checkbox class="answer" label="不方便透露" />
                </el-checkbox-group>
            </el-form-item>

            <el-form-item class="question" v-if="showPDQ0506" style="font-weight: bolder;" label="D0506.不对外许可的理由是？">
                <el-checkbox-group v-model="form.pDq0506" style="display: flex;flex-wrap: wrap;">
                    <el-checkbox class="answer" label="专利技术仍在开发中，没办法许可" />
                    <el-checkbox class="answer" label="没有发现市场需求" />
                    <el-checkbox class="answer" label="有市场需求，但是没有渠道联系到被许可人" />
                    <el-checkbox class="answer" label="有潜在许可对象，但是许可价格谈不拢" />
                    <el-checkbox class="answer" label="不相信许可对象的合同履行能力" />
                    <el-checkbox class="answer" label="担心技术泄露" />
                    <el-checkbox class="answer" label="担心市场占有率收到损失" />
                    <el-checkbox class="answer" label="不愿意许可" />
                </el-checkbox-group>
            </el-form-item>

            <el-form-item class="question" style="font-weight: bolder;" label="D06.未来三年，贵司对该专利是否有其他商业化计划？">
                <el-checkbox-group v-model="form.pDq06" style="display: flex;flex-wrap: wrap;"
                    @change="handlePDQ06Change">
                    <el-checkbox class="answer" label="专利转让" />
                    <el-checkbox class="answer" label="集团内部专利许可" />
                    <el-checkbox class="answer" label="与第三方专利许可" />
                    <el-checkbox class="answer" label="交叉许可、专利联盟" />
                    <el-checkbox class="answer" label="专利融资（包括质押、信托、作价入股等）" />
                    <el-checkbox class="answer" label="以专利为基础的技术合同" />
                    <el-checkbox class="answer" label="未来无其他商业化计划" />
                    <el-checkbox class="answer" label="其他，请注明"><el-input size="small" placeholder="其他，请注明"
                            v-model="extraInput5"></el-input></el-checkbox>
                </el-checkbox-group>
            </el-form-item>
        </el-form>
        <el-button type="primary" @click="submit()" style="margin-top: 1vh;margin-left: 2vw;">提交问卷（D部分）</el-button>
    </el-card>
</template>

<script setup>
import { ref, reactive, defineProps, defineEmits } from 'vue';
import { surveyStore } from '../../../../stores/survey';
import axios from 'axios';
import { ElMessage } from 'element-plus';
const surveyInfo = surveyStore().surveyInfo
const form = reactive({
    pDq01: [],
    pDq02: [],
    pDq03: "",
    pDq0301: "",
    pDq0302: [],
    pDq0303: [],
    pDq0304: [],
    pDq0305: [],
    pDq0306: [],
    pDq04: [],
    pDq05: "",
    pDq0501: [],
    pDq0502: [],
    pDq0503: [],
    pDq0504: [],
    pDq0505: [],
    pDq0506: [],
    pDq06: [],
});

const extraInput1 = ref('')
const extraInput2 = ref('')
const extraInput3 = ref('')
const extraInput4 = ref('')
const extraInput5 = ref('')

//跳转与互斥
const showPDQ03 = ref(false);
const showPDQ04 = ref(false);
const handlePDQ03Change = (value) => {
    showPDQ03.value = (value === "是的，已达成多个许可交易。") || (value === "是的，已达成一个许可交易。");
    showPDQ04.value = value === "还没有";
};

const showPDQ05 = ref(false);
const showPDQ0506 = ref(false);
const handlePDQ05Change = (value) => {
    showPDQ05.value = value === "是的，我们希望能对外许可";
    showPDQ0506.value = value === "否";
};

const handlePDQ02Change = (value) => {
    if (value.includes('没有进行过宣传推广')) {  
        form.pDq02 = ['没有进行过宣传推广']; 
        extraInput2.value = '';
    }
};

const handlePDQ06Change = (value) => {
    if (value.includes('未来无其他商业化计划')) {  
        form.pDq06 = ['未来无其他商业化计划']; 
        extraInput2.value = '';
    }
};

const emits = defineEmits(['switch-tab']);

const switchToPart = (tabName) => {
    emits('switch-tab', tabName);
}

const submit = async () => {
    //form.pDq0302.pop()
    //form.pDq0501.pop()
    //form.pDq06.pop()
    //form.pDq01.pop()
    form.pDq0302.push(extraInput1.value)
    form.pDq0501.push(extraInput2.value)
    form.pDq06.push(extraInput3.value)
    form.pDq01.push(extraInput4.value)
    // 将表单数据转换为对象数组
    const formDataArray = Object.entries(form).map(([key, value]) => ({ [key]: value }));

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
        usage: formDataString
    };
    let response = await axios.post('/api/survey/usage', data);
    if (response.status == 200) {
        if (response.data.code == 1) {
            ElMessage.success("submit successfully")
            switchToPart("E")
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

::v-deep .blue-label .el-form-item__label {
    color: blue;
    text-indent: 2em;
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
::v-deep .el-form-item__label {
    font-size: 17px;

}
</style>
