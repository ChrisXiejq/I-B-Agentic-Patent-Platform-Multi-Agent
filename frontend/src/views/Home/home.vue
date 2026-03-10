<template>
    <div class="whole-box">
        <el-card
            style="max-width: 1400px; margin-top: 20px; border-radius: 15px;width: 100%;font-size: 1.3rem;font-family: SimSun;display: flex;flex-direction: column;">
            <p>尊敬的专家：</p>

            <p style="text-indent: 2em;">
                您好！本次问卷是“开放专利与创新”研究的一部分，探索我国专利的经济价值、专利许可影响因素、政策需求问题，获得国家社会科学基金项目（21&ZD65）、国家自然科学基金（72304209、72072129、72074169）支持。
            </p>

            <p style="text-indent: 2em;">我们共向上海地区超过 7000 名持有近 2 万个高价值专利的管理者（包括发明人、企业 IP 管理者、专利代理人）等发放调查问卷。问卷会耗时
                15-20
                分钟。您可以填写纸质版问卷，也可登录<a href="www.opi-shanghai.com"> www.opi-shanghai.com</a>
                填写在线问卷。我们后续也将在网站上公布调查研究成果。
            </p>

            <p style="text-indent: 2em;">您的意见对知识产权研究非常重要！问卷内容不涉及任何个人隐私或商业秘密，我们也不会以任何方式直接披露您的问卷填写信息。
            </p>
            <p style="text-indent: 2em;">衷心感谢您的支持与协助！
            </p>
        </el-card>
    </div>
</template>

<script setup>

</script>



<style scoped>
p {
    margin-bottom: 20px;
    /* 设置段落下方的外边距 */
    line-height: 50px
}

.whole-box {
    width: 100%;
    display: flex;
    justify-content: center;
    background-color: white;
    margin-top: 20px;
}


:deep(.el-radio__label) {
    white-space: normal;
    /* 换行 */
}

:deep(.el-checkbox__label) {
    white-space: normal;
    /* 换行 */
}
</style>
