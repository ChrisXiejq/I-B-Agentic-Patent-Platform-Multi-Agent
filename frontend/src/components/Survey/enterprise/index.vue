<template>
    <div class="whole-box">
        <div style="max-width: 1400px;">
            <el-container style="margin-top: 10px;margin-right: 5px; width: 100%;">
                <el-tabs style="width: 100%;" v-model="activeName" tab-position="top" class="demo-tabs"
                    @tab-click="handleClick" type="card" stretch="true">
                    <el-tab-pane label="个人信息" name="A">
                        <part-a @switch-tab="switchTab" />
                    </el-tab-pane>
                    <el-tab-pane label="企业研发与知识产权管理" name="B">
                        <part-b @switch-tab="switchTab" />
                    </el-tab-pane>
                    <el-tab-pane label="专利技术价值" name="C">
                        <part-c @switch-tab="switchTab" />
                    </el-tab-pane>
                    <el-tab-pane label="专利许可运用" name="D">
                        <part-d @switch-tab="switchTab" />
                    </el-tab-pane>
                    <el-tab-pane label="知识产权政策" name="E">
                        <part-e @switch-tab="switchTab" />
                    </el-tab-pane>
                </el-tabs>
            </el-container>
        </div>
    </div>
</template>

<script>
import partA from './parts/partA.vue'
import partB from './parts/partB.vue'
import partC from './parts/partC.vue'
import partD from './parts/partD.vue'
import partE from './parts/partE.vue'
import { surveyStore } from '../../../stores/survey';
import { ElMessage } from 'element-plus';
import { Text } from 'vue'
export default {
    components: {
        partA,
        partB,
        partC,
        partD,
        partE
    },
    data() {
        const surveyInfo = surveyStore().surveyInfo

        return {
            surveyInfo,
            activeName: "A",
            patentNo: "",
            dialogVisible: false, // 控制对话框显示的属性
            invitationCode: '', // 存储输入的邀请码
            disableInput: true,
            showRecommendationDialog: false,//用于控制建议弹窗的显示
        }
    },
    mounted() {
        this.checkScreenWidth();

        window.addEventListener('resize', this.checkScreenWidth);
    },
    methods: {
        handleClose(done) {
            // 可以在这里添加关闭对话框前的操作，比如确认用户是否真的想关闭对话框
            done(); // 这会实际关闭对话框
        },
        checkInvitationCode() {
            const survey = surveyStore();
            // 这里实现邀请码的检查逻辑
            if (this.surveyInfo.curInvitationCode === survey.surveyInfo.invitationCode
                || this.surveyInfo.curInvitationCode === survey.surveyInfo.testInvitationCode) {
                this.dialogVisible = false; // 如果邀请码正确，关闭对话框
                this.disableInput = false;
            } else {
                // 如果邀请码错误，给出提示，不关闭对话框
                this.$message.error('邀请码错误，请重试');
            }
        },
        showInput() {
            this.dialogVisible = true
        },
        switchTab(tabName) {
            // 更新 activeName 属性来切换标签页
            console.log(tabName)
            this.activeName = tabName;
            window.scrollTo({
                top: 0,
                behavior: "smooth" // 使用平滑滚动
            });
        },
        allowInput(value) {
            this.disableInput = false;
            ElMessage.success("请开始填写问卷")
        },
        checkScreenWidth() {
            if (window.innerWidth < 500) {
                this.showRecommendationDialog = true;
            } else {
                this.showRecommendationDialog = false;
            }
        },
        handleRecommendationDialogClose(done) {
            this.showRecommendationDialog = false;
            done();
        },
    },
    beforeUnmount() {
        window.removeEventListener('resize', this.checkScreenWidth);
    }

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
}

/* Style for portrait mode */
@media screen and (orientation: portrait) {}
</style>