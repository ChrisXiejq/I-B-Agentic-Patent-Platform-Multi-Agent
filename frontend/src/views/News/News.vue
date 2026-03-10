<template>
    <div class="whole-box">
        <div class="set-vertical" style="max-width: 1400px;">
            <el-container>
                <el-carousel style="width: 100%;margin-top: 5vh;" indicator-position="outside">
                    <el-carousel-item class="set-vertical" style="cursor: pointer;" v-for="item in news" :key="item">
                        <el-container style="height: 200px;background-color: #99a9bf;" @click="getNewsDetail(item.id)">
                            <img :src="item.cover" style="width: 100%; height: 100%; object-fit: cover;">
                        </el-container>
                        <el-card style="margin-top: 0;margin-bottom: 1vh;" @click="getNewsDetail(item.id)">
                            <h5 style="font-family: arial;">{{ item.title }}</h5>
                        </el-card>
                    </el-carousel-item>
                </el-carousel>
            </el-container>
            <el-container style="max-width: 1400px; width: 100%;margin-top: 20px;" class="set-vertical">
                <div v-for="item in news" :key="item.id" class="news-item set-horizonal"
                    style="max-width: 1400px;border-radius: 20px;cursor: pointer;width: 100%;" @click="getNewsDetail(item.id)">
                    <el-container style="width: 30%;background-color:#99a9bf; ;">
                        <img :src="item.cover" style="width: 100%; height: 100%; object-fit: cover;">
                    </el-container>
                    <el-card style="width: 70%;">
                        <h5 style="font-family: arial;font-size: 1.5em;">{{ item.title }}</h5>
                        <p>{{ item.time }}</p>
                    </el-card>
                </div>
            </el-container>
        </div>
    </div>
</template>

<script setup>
import { useRouter, useRoute } from 'vue-router';
import { ref, onMounted, getCurrentInstance, watch } from 'vue'
import axios from 'axios';
import { useGeneralStore } from '../../stores/general';


const store = useGeneralStore();
let news = ref([]);
let response = '';
const route = useRoute()
let { ctx } = getCurrentInstance()

const formatDate = (dateString) => {
    const date = new Date(dateString);
    const year = date.getFullYear();
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const day = date.getDate().toString().padStart(2, '0');
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    const seconds = date.getSeconds().toString().padStart(2, '0');

    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
}

const getNews = async () => {
    try {
        response = await axios.get('api/ibnews');
        if (response.status === 200) {
            news.value = response.data.data;
            news.value.forEach(item => {
                item.time = formatDate(item.time);  // 格式化时间
            });
            updateLanguageValues();
        }
    } catch (error) {
        console.error('Error fetching news details:', error);
    }
};

const router = useRouter()
const getNewsDetail = (id) => {
    router.push({
        path: `/news/${id}`
    })
}

const updateLanguageValues = () => {
    news.value.forEach(item => {
        if (ctx.$i18n.locale == 'zn') {
            item.title = item.titleZn;
        } else if (ctx.$i18n.locale == 'en') {
            item.title = item.titleEn;
        } else if (ctx.$i18n.locale == 'de') {
            item.title = item.titleDn;
        }
    });
};

onMounted(() => {
    getNews();
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
    background-color: white;
    margin-top: 20px;
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
    // width: 95vw;
    box-sizing: border-box;
    border: 1px solid #ddd;
    margin-top: 2vh;
}

.news-item:hover {
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    /* 悬浮时的阴影效果 */
}
</style>