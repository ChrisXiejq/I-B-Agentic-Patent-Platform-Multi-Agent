<template>

    <body>
        <div class="whole-box" style="font-family: SimSun;">
            <div class="content-box">
                <div v-show="false" style="font-size: large;">
                    &nbsp&nbsp&nbsp&nbsp“开放专利与创新”上海调查研究团队由同济大学上海国际知识产权学院（SICIP）青年教师组成，是SICIP与德国马普创新与竞争研究所（MPIIC）联合科研的一部分。
                </div>
                <div class="title">
                    <h3>项目联系人</h3>
                </div>
                <div class="profile-box">
                    <div style="width: 130px">
                    <img class="profile-pic" src="../../public/teamimg/worker/刘夏.png">
                    </div>
                    <div class="profile-name">
                        <p>{{ $t('team.LiuXia1') }}</p>
                        <p>同济大学上海国际知识产权学院副教授</p>
                        <a href="liuxia0510@tongji.edu.cn">liuxia0510@tongji.edu.cn</a>
                    </div>
                </div>

                <div class="title">
                    <h3>学术指导</h3>
                </div>
                <div class="profile-box">
                    <div style="width: 130px">
                    <img class="profile-pic" src="../../public/teamimg/worker/DietmarHarhoff.png">
                    </div>
                    <div class="profile-name">
                        <p>Dietmar Harhoff</p>
                        <p>Director/Professor</p>
                        <p>Max Planck Institute for Innovation and Competition</p>
                    </div>
                </div>

                <div class="profile-box">
                    <div style="width: 130px">
                    <img class="profile-pic" src="../../public/teamimg/worker/王晓璐.png">
                    </div>
                    <div class="profile-name">
                        <p>Lucy Xiaolu Wang,</p>
                        <p>Assistant Professor</p>
                        <p>University of Massachusetts Amherst
Faculty Research Fellow，MPIIC
</p>
                        <a href="https://www.lucyxiaoluwang.com">https://www.lucyxiaoluwang.com</a>
                    </div>
                </div>

                <div class="profile-box">
                    <div style="width: 130px">
                    <img class="profile-pic" src="../../public/teamimg/worker/王洁.png">
                    </div>
                    <div class="profile-name">
                        <p>王洁</p>
                        <p>同济大学软件学院助理教授</p>
                    </div>
                </div>
                <div>
                    <div class="title">
                        <h3>研究助理</h3>
                    </div>
                    <h4>
                        ● 文字编辑
                    </h4>
                    <div> 国知院：陈雪妮、段明赫、周泽鑫</div>
                    <h4>
                        ● 网站/数据库
                        </h4>
                    <div> 软件学院：李天昊、谢嘉麒</div>
                    <div> 经济与管理学院：谭紫萱、余同洲</div>
                    <div> 电子信息与工程学院：苏惠</div>
                    <div> 国知院：朱啸宇、李浩宇</div>
                    <h4>
                        ● 调研组织
                    </h4>
                    <div> 国知院：叶知秋、刘佳欣、张玉涵</div>
                </div>
            </div>
        </div>


    </body>
</template>

<script setup>

</script>

<style lang="scss" scoped>
.whole-box {
    width: 100%;
    display: flex;
    max-width: 1400px;
    margin-top: 30px;
    justify-content: center;
    align-items: center;
    background-color: white;
    // padding: 20px;
}

.content-box {
    width: 100%;
    max-width: 800px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    // padding-left: 30px;
}

.profile-box {
    display: flex;
    gap: 50px;
    .profile-pic {
        height: 160px;
        border: 3px solid #eee;
    }
}

.title {
    h3 {
        color: #34495e;
    }
}
</style>