<template>
    <div class="whole-box">
        <el-card
            style="max-width: 1400px; margin-top: 20px; border-radius: 15px;width: 100%;font-size: 1.3rem;font-family: SimSun;display: flex;flex-direction: column;">
            <h2 style="width: 100%;text-align: center;">数据使用</h2>
            <p style="text-indent: 2em;">“开放专利与创新”调查研究课题组所获得的所有原始数据由课题组保存，不向任何人提供。脱敏处理后的数据（以下简称“脱敏数据”）依据以下规则对外提供：</p>
            <p style="text-indent: 2em;">
                1.对同济大学上海国际知识产权学院师生：同济大学上海国际知识产权学院师生可以在与学院相关的研究中,在课题组提供的计算机环境下使用脱敏数据，不得复制、传播脱敏数据。</p>
            <p style="text-indent: 2em;">2.对其他研究者：希望使用脱敏数据从事相关学术研究的研究者，经过中心批准后，可以在课题组提供的计算机环境下使用脱敏数据，不得复制、传播脱敏数据。
            </p>

            <p style="text-indent: 2em;">
                3.对期刊审稿人：期刊审稿人希望进行数据验证的，经过课题组批准后，可以在课题组提供的计算机环境下使用脱敏数据，不得复制、传播脱敏数据。由于特殊情况需要远程访问的，中心可以就论文中使用的部分脱敏数据提供一个月内的远程访问权限。审稿人请通过论文作者向中心申请数据验证。
            </p>
            <p style="width: 100%;text-align: end;">数据使用联系人： 2333769@tongji.edu.cn</p>
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
