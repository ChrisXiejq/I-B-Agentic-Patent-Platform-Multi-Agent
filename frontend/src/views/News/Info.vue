<template>
    <div class="whole-box">
        <div class="content-box">
            <!-- <vue-office-docx :src="url" style="max-width: 100%;overflow:auto;margin-top: 5vh;margin-left: 15%;"
                @rendered="renderedHandler" @error="errorHandler" /> -->
            <pdf :src="url" style="max-width: 1400px;overflow:auto;width: 100%;"></pdf>
        </div>
    </div>
    <!-- <pdf :src="url" style="max-width: 1400px;overflow:auto;margin-top: 5vh;margin-left: 15%;width: 70%;"></pdf>
    <a style="width: 100%; max-width: 1200px;margin-left: 15%;" :href="url">{{ title }}</a> -->
</template>


<script setup>
import { useRouter, useRoute } from 'vue-router';
import { ref, onMounted, getCurrentInstance, watch } from 'vue'
import axios from 'axios';
import { useGeneralStore } from '../../stores/general';
//引入VueOfficeDocx组件
import VueOfficeDocx from '@vue-office/docx'
//引入相关样式
import '@vue-office/docx/lib/index.css'
// import pdf from 'vue-pdf'
import pdf from 'pdf-vue3';

const store = useGeneralStore();
let news = ref({});
let url = ref("");
let title = ref("")
const route = useRoute()
let { ctx } = getCurrentInstance()

const getDetail = async () => {
    try {
        const response = await axios.get('api/ibnews/singleNews/' + route.params.id);
        if (response.status === 200) {
            news.value = response.data.data;
            updateLanguageValues();
        }
    } catch (error) {
        console.error('Error fetching news details:', error);
    }
};

const updateLanguageValues = () => {
    if (ctx.$i18n.locale == 'zn') {
        url.value = news.value.urlZn;
        title.value = news.value.titleZn;
    } else if (ctx.$i18n.locale == 'en') {
        url.value = news.value.urlEn;
        title.value = news.value.titleEn;
    } else if (ctx.$i18n.locale == 'de') {
        url.value = news.value.urlDn;
        title.value = news.value.titleDn;
    }
};

onMounted(() => {
    getDetail();
});

watch(() => store.changeLanguage, () => {
    console.log("Language changed");
    updateLanguageValues();
});
</script>

<style lang="scss" scoped>
.whole-box {
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    // box-shadow: 0 4px 8px $shadow-color;
    background-color: white;
    margin-top:30px
    // border: 1px solid #000;
}
.content-box {
    width: 100%;
    max-width: 1400px;
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    // border: 1px solid #000;
}

.set-vertical {
    display: flex;
    flex-direction: column;
}

.set-horizonal {
    display: flex;
    flex-direction: row;
}

.el-carousel__item h3 {
    display: flex;
    opacity: 0.75;
    margin: 0;
}

.news-item {
    width: 95vw;
    box-sizing: border-box;
    border: 1px solid #ddd;
    margin-top: 2vh;
    height: 25vh;
}

.news-item:hover {
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    /* 悬浮时的阴影效果 */
}

:deep(.docx-wrapper) {
    background-color: #fff;
    padding: 0;
}

:deep(.docx-wrapper > section.docx) {
    width: 100% !important;
    padding: 0px !important;
    min-height: auto !important;
    box-shadow: none;
    margin-bottom: 0px;
}
</style>