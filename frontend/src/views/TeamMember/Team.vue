<template>
    <body>
        <!-- <div class="update-notice">
            <div class="content">
                内容即将更新，敬请期待
            </div>
        </div> -->
        <div v-show="false">
            “开放专利与创新”上海调查研究团队由同济大学上海国际知识产权学院（SICIP）青年教师组成，是SICIP与德国马普创新与竞争研究所（MPIIC）联合科研的一部分。
        </div>
        <div class="worker-box">
            <div class="team-row">
                <div class="profile-box">
                    <div class="profile-card">
                        <div style="display: flex;">
                            <img src="../../public/teamimg/worker/刘夏.png">
                            <div class="profile-name">
                                <h4>{{ $t('team.LiuXia1') }}</h4>
                                <small>{{ $t('team.LiuXia2') }}</small>
                            </div>
                        </div>
                        <div style="display: flex; flex-direction: column;justify-content: center;">
                            <p class="profile-p">{{ $t('team.LiuXiaP') }}</p>
                            <a class="profile-href" href="https://sicip.tongji.edu.cn/lx/list.htm" target="_blank">{{
                                $t('team.more') }}</a>
                        </div>
                    </div>
                </div>
                <div class="profile-box">
                    <div class="profile-card">
                        <div style="display: flex;">
                            <img src="../../public/teamimg/worker/王晓璐.png">
                            <div class="profile-name">
                                <h4>{{ $t('team.WangXiaolu1') }}</h4>
                                <small>{{ $t('team.WangXiaolu2') }}</small>
                            </div>
                        </div>
                        <div style="display: flex; flex-direction: column;justify-content: center;">
                            <p class="profile-p">{{ $t('team.WangXiaoluP') }}</p>
                            <a class="profile-href" href="https://www.lucyxiaoluwang.com/" target="_blank">{{
                                $t('team.more') }}</a>
                        </div>
                    </div>
                </div>
                <div class="profile-box">
                    <div class="profile-card">
                        <div style="display: flex;">
                            <img src="../../public/teamimg/worker/DietmarHarhoff.png">
                            <div class="profile-name">
                                <h4>{{ $t('team.DietmarHarhoff1') }}</h4>
                                <small>{{ $t('team.DietmarHarhoff2') }}</small>
                            </div>
                        </div>
                        <div style="display: flex; flex-direction: column;justify-content: center;">
                            <p class="profile-p">{{ $t('team.ChenLiP') }}</p>
                            <a class="profile-href" href="https://www.gu.se/om-universitetet/hitta-person/lichen"
                                target="_blank">{{ $t('team.more') }}</a>
                        </div>
                    </div>
                </div>
                <div class="profile-box">
                    <div class="profile-card">
                        <div style="display: flex;">
                            <img src="../../public/teamimg/worker/王洁.png">
                            <div class="profile-name">
                                <h4>{{ $t('team.WangJie1') }}</h4>
                                <small>{{ $t('team.WangJie2') }}</small>
                            </div>
                        </div>
                        <div style="display: flex; flex-direction: column;justify-content: center;">
                            <p class="profile-p">{{ $t('team.WangJieP') }}</p>
                            <a class="profile-href" href="https://faculty.tongji.edu.cn/JieWang/zh_CN/" target="_blank">{{
                                $t('team.more') }}</a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <!-- <div class="assistants-box">
            <h2>Student Assistant</h2>
            <div class="team-row">
                <div class="assistant-box">
                    <h4>Student Assistant</h4>
                    <small> Student Assistant</small>
                    <img src="../../public/teamimg/assistants/1.png">
                    <p class="assistant-p">Student AssistantStudent AssistantStudent AssistantStudent AssistantStudent
                        AssistantStudent AssistantStudent Assistant</p>
                </div>
                <div class="assistant-box">
                    <h4>Student Assistant</h4>
                    <small> Student Assistant</small>
                    <img src="../../public/teamimg/assistants/2.png">
                    <p class="assistant-p">Student AssistantStudent AssistantStudent AssistantStudent AssistantStudent
                        AssistantStudent AssistantStudent Assistant</p>
                </div>
                <div class="assistant-box">
                    <h4>Student Assistant</h4>
                    <small> Student Assistant</small>
                    <img src="../../public/teamimg/assistants/3.png">
                    <p class="assistant-p">Student AssistantStudent AssistantStudent AssistantStudent AssistantStudent
                        AssistantStudent AssistantStudent Assistant</p>
                </div>
                <div class="assistant-box">
                    <h4>Student Assistant</h4>
                    <small> Student Assistant</small>
                    <img src="../../public/teamimg/assistants/4.png">
                    <p class="assistant-p">Student AssistantStudent AssistantStudent AssistantStudent AssistantStudent
                        AssistantStudent AssistantStudent Assistant</p>
                </div>
                <div class="assistant-box">
                    <h4>Student Assistant</h4>
                    <small> Student Assistant</small>
                    <img src="../../public/teamimg/assistants/5.png">
                    <p class="assistant-p">Student AssistantStudent AssistantStudent AssistantStudent AssistantStudent
                        AssistantStudent AssistantStudent Assistant</p>
                </div>
                <div class="assistant-box">
                    <h4>Student Assistant</h4>
                    <small> Student Assistant</small>
                    <img src="../../public/teamimg/assistants/6.png">
                    <p class="assistant-p">Student AssistantStudent AssistantStudent AssistantStudent AssistantStudent
                        AssistantStudent AssistantStudent Assistant</p>
                </div>
            </div>
        </div> -->
    </body>
</template>

<script setup>

</script>

<style lang="scss" scoped>
body {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    // background-color: #f3f3f3;
}

.title-box {
    text-align: center;
    margin: auto 20px;
    padding: 30px;
}

.title-box p {
    margin-top: 10px;
    line-height: 1.6;
    color: #34495e;
}

.title-box h1 {
    color: #2c3e50;
    font-size: 2.5rem;
}

.worker-box {
    display: flex;
    flex-direction: column;
    flex-wrap: wrap;
    /* justify-content: center; */
    align-items: center;
    margin: 10px auto;
    gap: 30px;
}

.profile-name {
    padding: 30px;
    white-space: nowrap;
    width: 80px;
}

.team-row {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    margin: auto;
    width: 100%;
    gap: 20px;
}

.profile-box {
    width: 90%;
    max-width: 1200px;
    background: #fff;
    border-radius: 10px;
    margin-bottom: 30px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.12);
    padding: 30px;
    position: relative;
    transition: transform 0.3s;
}

.profile-box:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
}

.profile-box img {
    width: 90px;
    height: 120px;
    border-radius: 50%;
    border: 3px solid #eee;
}

.profile-box h4 {
    color: #2980b9;
    margin-bottom: 5px;
}

.profile-card {
    display: flex;
}

.profile-box small {
    display: block;
    margin-bottom: 10px;
    color: #7f8c8d;
}

.profile-box .profile-p {
    margin: -10px auto 0px;
    padding: 10px;
    text-align: start;
    line-height: 1.6;
    /* 增加行间距 */
    color: #2c3e50c2;
    max-width: 900px;
    /* 限制最大宽度 */
    font-size: 1rem;
    /* 调整字体大小 */
}

.profile-href {
    padding-left: 10px;
    text-align: start;
    color: #449cf4f9;
}

.assistants-box {
    display: flex;
    flex-direction: column;
    flex-wrap: wrap;
    max-width: 1200px;
    /* border: 1px solid #000; */
    align-items: center;
    margin: 20px auto;
    gap: 30px;
}

.assistant-box {
    width: 300px;
    background: #fff;
    border-radius: 5px;
    margin-bottom: 30px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    /* 调整阴影 */
    padding: 20px;
    position: relative;
    transition: transform 0.3s;
    /* 添加过渡效果 */
}

.assistant-box:hover {
    transform: scale(1.05);
    /* 鼠标悬停时放大 */
}

.assistant-box img {
    width: 100px;
    position: absolute;
    top: 20px;
    right: 20px;
    border-radius: 50%;
    /* 圆形图片 */
    border: 3px solid #eee;
    /* 图片边框 */
}

.assistant-box .assistant-p {
    margin: 120px auto 20px;
    /* 调整间距 */
    text-align: center;
    line-height: 24px;
    color: #555;
}

h4,
small {
    color: #333;
}

@media (max-width: 768px) {

    .profile-box,
    .team-row,
    .title-box,
    .worker-box {
        width: 100%;
        // padding: 15px;
    }

    .profile-card {
        flex-direction: column;
    }
}

.update-notice {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 20vh; /* 确保占据足够的屏幕高度 */
}

.content {
  padding: 20px 40px;
  background-color: #ffffff; /* 突出显示的背景颜色 */
//   border-radius: 10px; /* 圆角边框 */
  box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* 轻微的阴影效果 */
  font-size: 1.5rem; /* 较大的字体尺寸 */
  color: #333; /* 深色字体 */
  text-align: center; /* 文本居中 */
  max-width: 80%; /* 限制最大宽度，确保在大屏幕上也能看起来很好 */
}
</style>