<template>
    <div class="item-box">
        <el-icon><ArrowRight /></el-icon>
        <!-- <p style="white-space: nowrap;" v-for="author in authors">{{ author }}&nbsp;&nbsp;</p> -->
        <!-- <div v-if="authors.length > 0" class="vertical-line"><p>&nbsp;&nbsp;</p></div> -->
        <p>{{ title }}</p>
    </div>
</template>

<script setup>
import {ref, onMounted,getCurrentInstance,watch} from 'vue'
import { useGeneralStore } from '../../stores/general';
const store = useGeneralStore()
let { ctx } = getCurrentInstance()
let authors = ref([])
const props = defineProps({
    content: Object,
});

let title = ref('')
watch(() => store.changeLanguage, () => {
    updateContent(ctx.$i18n.locale);
});

const updateContent = (locale) => {
    authors.value = [];
    if (locale === 'en') {
        props.content.authors.forEach(lecturer => authors.value.push(lecturer.name_en));
        title.value = props.content.title_en;
    } else if (locale === 'zn') {
        props.content.authors.forEach(lecturer => authors.value.push(lecturer.name_zn));
        title.value = props.content.title_zn;
    } else if (locale === 'de') {
        props.content.authors.forEach(lecturer => authors.value.push(lecturer.name_de));
        title.value = props.content.title_de;
    }
};

onMounted(() =>{
    authors.value = []
    title.value = ''
    updateContent(ctx.$i18n.locale);
})
</script>

<style lang="scss" scoped>
.item-box {
    display: flex;
    align-items: center;
    padding: 10px 15px; // 增加内边距以提高可读性
    background-color: #fff; // 使用白色背景
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); // 添加轻微的阴影以提升层次感
    border-radius: 8px; // 添加圆角
    transition: all 0.3s ease; // 平滑过渡效果

    &:hover {
        // transform: translateY(-2px); // 悬停时轻微上移，增加交互感
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15); // 悬停时加强阴影
        background-color: #4a91e220;
        p {
            color: #000; // 悬停时改变字体颜色
        }
    }
}

p {
    margin: 0; // 移除默认的段落外边距
    color: #333; // 使用深灰色以提高可读性
    font-size: 16px;
    //white-space: nowrap; // 保持标题不换行
    //text-overflow: ellipsis; // 文本过长时用省略号表示
    overflow: hidden; // 隐藏溢出的文本
    //max-width: calc(100% - 40px); // 预留空间给箭头和竖线
}

.el-icon {
    margin-right: 8px; // 图标和文本之间的间距
}

.vertical-line {
    width: 2px;
    margin: 0 12px; // 竖线两侧的间距
    height: 20px; // 调整竖线高度以适应内容
    background-color: #ddd; // 使用更淡的颜色以减少视觉干扰
}

@media (max-width: 720px) {
    .item-box {
        padding: 8px 10px; // 在小屏幕上减少内边距
    }

    p {
        font-size: 14px; // 在小屏幕上减小字体大小
    }

    .el-icon, .vertical-line {
        display: none; // 在小屏幕上隐藏图标和竖线以节省空间
    }
}
</style>

<!-- <style lang="scss" scoped>
.item-box{
    width: 100%;
    height: auto;
    display: flex;
    align-items: center;
    gap:10px;
    // border: 1px solid black;
}
p{
    font-size: large;
}
.vertical-line {
  height: 10px; /* 竖线的高度 */
  width: 2px; /* 竖线的宽度 */
  background-color: rgba(0, 0, 0, 0.342); /* 竖线的颜色 */
}
@media (max-width: 720px) {
    body, p {
        font-size: 12px;
    }
}
</style> -->