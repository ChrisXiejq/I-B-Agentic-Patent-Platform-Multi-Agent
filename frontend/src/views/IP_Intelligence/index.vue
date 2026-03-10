<template>
    <body>
        <div v-if="!showmore" class="update-notice">
            <div class="content">
                内容即将更新，敬请期待
            </div>
        </div>
        <div  v-if="showmore" class="header-container">
            <img class="head-img" src="../../public/img/Intelligence-head.jpg"/>
            <div class="text-overlay">
                <h2>{{ $t('topnav.IP_Intelligence') }}/</h2>
                <h3>{{ $t(ChosedItem) }}</h3>
            </div>
        </div>
        <div  v-if="showmore" class="whole-box">
            <el-menu class="menu" default-active="1" @select="handleSelect">
                <el-menu-item index="1">{{ $t('intelligence.course') }}</el-menu-item>
                <el-menu-item index="2">{{ $t('intelligence.PatentPledge') }}</el-menu-item>
                <el-menu-item index="3">{{ $t('intelligence.TOP2500RD') }}</el-menu-item>
            </el-menu>
            <div class="content-box">
                <div class="list-box">
                    <div class="item-box">
                        <div v-for="content in contentList" :key="content">
                            <ContentItem :content="content" @click="contentInfo(content._id)"></ContentItem>
                        </div>
                    </div>
                    <div style="display: flex;justify-content: center;align-items: flex-end;  margin: 20px;">
                        <el-pagination @current-change="handleCurrentChange" background layout="prev, pager, next"
                            :total=totalElements :current-page="intelligenceStore.chosedPage" />
                    </div>
                </div>
            </div>
        </div>
    </body>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { useRouter } from 'vue-router'
const router = useRouter()
import ContentItem from '../../components/public/ContentItem.vue'
import { useIntelligenceStore } from '../../stores/Intelligence.js'
import { ElMessage } from 'element-plus'
const intelligenceStore = useIntelligenceStore()
const showmore = false
const menuList = [
    'intelligence.course',
    'intelligence.PatentPledge',
    'intelligence.TOP2500RD'
]
const typelist =[
    'intelligence',
    'patentpledge',
    'top2500RD'
]
const urllist = [
    'courses',
    'courses',
    'top2500RD'
]
const BaseUrl = '/api/intelligence'
let contentList = ref([])
let totalElements = ref(contentList.value.length)
let ChosedItem = ref(menuList[0])
const pagesize = 10
onMounted(() => {
    intelligenceStore.chosedPage = 1
    getContentList(urllist[menuList.indexOf(ChosedItem.value)])
})
//处理换页
const handleCurrentChange = async (newPage) => {
    intelligenceStore.chosedPage = newPage
    getContentList(urllist[menuList.indexOf(ChosedItem.value)])
}
//跳转详情页
const contentInfo = (_id) => {
    router.push({
        path: `/intelligence/${urllist[menuList.indexOf(ChosedItem.value)]}/${_id}`
    })
}
//获取内容
const getContentList = async (source) => {
    contentList.value = []
    const url = `${BaseUrl}/${source}?page=${intelligenceStore.chosedPage}&size=${pagesize}&type=${typelist[menuList.indexOf(ChosedItem.value)]}`
    let response = await axios.get(url)
    console.log(response)
    if (response.status == 200) {
        console.log(response.data.data)
        contentList.value = response.data.data.lists
        totalElements.value = response.data.data.count
        console.log(contentList.value)
    }
}
const handleSelect = (index) => {
    ChosedItem.value = menuList[index - 1]
    getContentList(urllist[menuList.indexOf(ChosedItem.value)])
}
const renderedHandler = () => {
    console.log("渲染完成")
}
const errorHandler = () => {
    console.log("渲染失败")
}
</script>

<style lang="scss" scoped>
/* 在此处定义你的颜色方案 */
$primary-color: #4a90e2;
$secondary-color: #f3f3f3;
$text-color: #333;
$shadow-color: rgba(0, 0, 0, 0.1);

body {
    // background-color: $secondary-color;
    margin-top: 10px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.header-container {
    position: relative; // 父容器设置为相对定位
    width: 100%;
    max-width: 1400px;
}

.head-img {
    width: 100%;
    opacity: 0.8;
    object-fit: cover;
    object-position: center;
}

.text-overlay {
    position: absolute; // 文本部分设置为绝对定位
    bottom: 0; // 定位到左下角
    left: 0;
    padding: 30px; // 根据需要调整文本的内边距
    color: #fff; // 根据图片颜色调整文本颜色以确保可读性
}
.whole-box {
    width: 100%;
    display: flex;
    max-width: 1400px;
    // justify-content: center;
    // box-shadow: 0 4px 8px $shadow-color;
    background-color: white;
    padding: 20px;
    // border-radius: 8px;
}

.content-box {
    width: 100%;
    max-width: 1400px;
    display: flex;
    flex-direction: column;
    // justify-content: center;
    // align-items: center;
    // border: 1px solid #000;
    padding-left: 30px;
}

.list-box {
    // border: 1px solid #000;
    // width: 80%;
    min-height: 80vh;
    display: flex;
    flex-direction: column;
    gap: 10px;

    // 添加 flex-grow 使内容部分填充空间
    >div:first-child {
        flex-grow: 1;
        overflow-y: auto; // 如果内容很多，可以滚动
    }

    // 分页器的容器
    >div:last-child {
        // 保持在底部
    }
}


.menu {
    width: 300px;
    // border-right: 1px solid $shadow-color;
}

.el-menu-item {
    font-size: large;
    color: $text-color;

    &:hover,
    &.is-active {
        background-color: $primary-color;
        color: white;
    }
}

/* 响应式设计调整 */
@media(max-width: 720px) {
    .el-menu-item {
        font-size: medium;
    }

    .menu {
        display: flex;
        width: 100%;
        border-right: none;
        margin-bottom: 10px;
    }

    .el-menu-item {
        /*width: 33%;*/
        /*border: 1px solid black;*/
        font-size: 14px;
    }

    .whole-box {
        flex-direction: column;
        padding: 5px;
    }

    .content-box {
        padding-left: 5px;
    }
    .text-overlay{
        padding-bottom: 0px;
        padding-left: 10px;
        font-size: smaller;
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