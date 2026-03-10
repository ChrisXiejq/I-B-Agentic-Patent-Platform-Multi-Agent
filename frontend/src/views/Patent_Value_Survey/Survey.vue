<template>
    <div class="whole-box">
        <div style="max-width: 1400px;">
            <el-container
                style="font-family:SimHei;font-size: 2rem;font-weight: bolder;display: flex;justify-content: center;align-items: center; width: 100%; overflow: auto;margin-bottom: 2vh;">
                开放专利与创新调查：上海2024
            </el-container>
            <el-dialog title="提示" v-model="showRecommendationDialog"
                style="font-family: SimSun;width: 300px;align-items: center;justify-content: center;position: absolute; left: 20px;"
                :before-close="handleRecommendationDialogClose">
                <p style=" margin-top: -10px">为方便阅读，请用手机横屏或PC端填写，谢谢</p>
            </el-dialog>
            <!-- <el-dialog :modal="false" center :close-on-click-modal="false" :close-on-press-escape="false"
                style="font-family: SimSun;width: 400px;align-items: center;justify-content: center;"
                title="欢迎参加问卷调研，请输入邀请码" v-model="dialogVisible" :before-close="handleClose">
                <el-card style="gap: 6px;border: none;align-items: center;justify-content: center;display: flex;"
                    shadow="never">
                    <el-container style="font-family: KaiTi">请填入预先告知您的邀请码，或询问工作人员</el-container>
                    <el-input v-model="surveyInfo.curInvitationCode" style="width: 250px;margin-top: 30px;"
                        placeholder="请输入邀请码"></el-input>
                    <el-container>
                        <span slot="footer" style="margin-top: 30px;">
                            <el-button type="primary" @click="checkInvitationCode">确认</el-button>
                        </span>
                    </el-container>
                </el-card>
            </el-dialog> -->
            <el-container style="margin-top: 10px; width: 100%;">
                <el-tabs style="width: 100%;" v-model="activeName" tab-position="top" class="demo-tabs"
                    @tab-click="handleClick" stretch="true">
                    <el-tab-pane label="专利确认" name="专利信息确认">
                        <el-button type="primary" @click="showInput" v-show="false"
                            style="margin-bottom: 20px;">输入邀请码后确认填写资格</el-button>
                        <introduction @allow-input="allowInput" />
                    </el-tab-pane>
                    <el-tab-pane label="企业版问卷" name="enterprise" :disabled="disableEnterpriseInput">
                        <enterprise />
                    </el-tab-pane>
                    <el-tab-pane label="高校版问卷" name="campus" :disabled="disableCampusInput">
                        <campus />
                    </el-tab-pane>
                    <el-tab-pane label="个人版问卷" name="person" :disabled="disablePersonInput">
                        <person />
                    </el-tab-pane>
                </el-tabs>
            </el-container>
        </div>
    </div>
</template>

<script>
import introduction from '../../components/Survey/Introduction.vue'
import campus from '../../components/Survey/campus/index.vue'
import enterprise from '../../components/Survey/enterprise/index.vue'
import person from '../../components/Survey/person/index.vue'
import { surveyStore } from '../../stores/survey';
import { ElMessage } from 'element-plus';
import { Text } from 'vue'
export default {
    components: {
        introduction,
        campus,
        enterprise,
        person,
    },
    data() {
        const surveyInfo = surveyStore().surveyInfo

        return {
            surveyInfo,
            activeName: "专利信息确认",
            patentNo: "",
            dialogVisible: false, // 控制对话框显示的属性
            invitationCode: '', // 存储输入的邀请码
            disableEnterpriseInput: true,
            disableCampusInput: true,
            disablePersonInput: true,
            showRecommendationDialog: false,//用于控制建议弹窗的显示
            showRecommendationAgain: false,
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
        allowInput(value){
            if(value == '企业'){
                this.disableEnterpriseInput = false
                this.disableCampusInput = true
                this.disablePersonInput = true
                this.activeName = "enterprise"
                ElMessage.success("请填写企业版问卷")
            } else if(value == '大学'){
                this.disableCampusInput = false
                this.disableEnterpriseInput = true
                this.disablePersonInput = true
                this.activeName = "campus"
                ElMessage.success("请填写高校版问卷")
            } else if(value == '个人'){
                this.disablePersonInput = false
                this.disableEnterpriseInput = true
                this.disableCampusInput = true
                this.activeName = "person"
                ElMessage.success("请填写个人版问卷")
            }
        },
        checkScreenWidth() {
            if (window.innerWidth < 500 && this.showRecommendationAgain == false ) {
                this.showRecommendationDialog = true;
            } else {
                this.showRecommendationDialog = false;
            }
        },
        handleRecommendationDialogClose(done) {
            this.showRecommendationDialog = false;
            this.showRecommendationAgain = true;
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

/* 响应式设计调整 */
@media(max-width: 500px) {
    .whole-box {
    width: 100%;
    display: flex;
    justify-content: center;
    background-color: white;
}
}
</style>