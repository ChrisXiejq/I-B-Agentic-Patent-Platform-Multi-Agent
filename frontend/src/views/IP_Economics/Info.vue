<template>
    <body>
        <div class="whole-box">
            <!--<div class="side-box">-->
            <!--    <p>这是侧边栏eeeeeeeeeeeee</p>-->
            <!--</div>-->
            <div class="content-box">
                <h4>{{ title }}</h4>
                <p>{{ _abstract }}</p>
                <a :href="url">{{ url }}</a>
            </div>
            
        </div>
    </body>
</template>

<script setup>
import { useRouter, useRoute } from 'vue-router';
import { ref, onMounted, getCurrentInstance, watch } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

const router = useRouter();
const route = useRoute()

import { useGeneralStore } from '../../stores/general';
const store = useGeneralStore()

const _id = route.params._id;
let { ctx } = getCurrentInstance()
let response = ''
let title = ref('')
let _abstract = ref('')
let url = ref('')
watch(() => store.changeLanguage, () => {
    if (ctx.$i18n.locale == 'zn') {
        title.value = response.data.data.title_zn
        _abstract.value = response.data.data.abstract_zn
    }
    else if (ctx.$i18n.locale == 'en') {
        title.value = response.data.data.title_en
        _abstract.value = response.data.data.abstract_en
    }
    else if (ctx.$i18n.locale == 'de') {
        title.value = response.data.data.title_de
        _abstract.value = response.data.data.abstract_de
    }

});

onMounted(() => {
    const pathSegments = route.path.split('/');
    getInfo(pathSegments[pathSegments.length - 2])
})

const getInfo = async (str) => {
    
    response = await axios.get('/api/economic/'+str+'/' + _id)
    console.log(response)
    url.value = response.data.data.url
    if (ctx.$i18n.locale == 'zn') {
        title.value = response.data.data.title_zn
        _abstract.value = response.data.data.abstract_zn
    }
    else if (ctx.$i18n.locale == 'en') {
        title.value = response.data.data.title_en
        _abstract.value = response.data.data.abstract_en
    }
    else if (ctx.$i18n.locale == 'de') {
        title.value = response.data.data.title_de
        _abstract.value = response.data.data.abstract_de
    }
    console.log(_abstract.value)
}
</script>

<style lang="scss" scoped>
body {
    margin-top: 20px;
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    // border: 1px solid black;
}

.whole-box {
    max-width: 1200px;
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    // border: 1px solid black;
    
    a {
        word-break: break-all; // 允许在任何字符间断行
        color: #4a90e2; // 设置链接颜色，可根据需要调整
        text-decoration: underline; // 设置下划线，可根据需要调整
    }
}

.side-box {
    // width: 30%;
    padding: 20px;
    background-color: #f5f5f5; // 浅灰色背景
    border: 1px solid #e0e0e0; // 浅灰色边框
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); // 轻微的阴影效果
    border-radius: 5px; // 圆角边框
    // border: 1px solid black;
    p {
        font-size: 16px;
        color: #333333;
        margin-bottom: 10px; // 段落间距

        &:hover {
            color: #4a90e2; // 悬停时的颜色变化
        }
    }
}

.content-box {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    text-align: start;
    margin-left: 10px; // 与侧边栏间距
    // border: 1px solid black;
}

</style>