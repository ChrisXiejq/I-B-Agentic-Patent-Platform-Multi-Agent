<template>
    <div class="whole-box">
        <vue-office-docx :src="docx" style="height: 85vh; max-width: 1200px;overflow:auto" @rendered="renderedHandler" @error="errorHandler" />
        <a style="width: 100%; max-width: 1200px;" :href="docx">{{$t('intelligence.download')}}</a>
    </div>
</template>

<script setup>
import { useRouter, useRoute } from 'vue-router';
import { ref, onMounted,getCurrentInstance,watch } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

const router = useRouter();
const route = useRoute()

//引入VueOfficeDocx组件
import VueOfficeDocx from '@vue-office/docx'
//引入相关样式
import '@vue-office/docx/lib/index.css'

let docx = ref('')
import { useGeneralStore } from '../../stores/general';
const store = useGeneralStore()

const _id = route.params._id;
let type = ref('')
let { ctx } = getCurrentInstance()
let response =''

watch(() => store.changeLanguage, () => {
    if(ctx.$i18n.locale == 'zn')
        docx.value = response.data.data.url_zn
    else if(ctx.$i18n.locale == 'en')
        docx.value = response.data.data.url_en
    else if(ctx.$i18n.locale == 'de')
        docx.value = response.data.data.url_de
});

onMounted(() => {
    const pathSegments = route.path.split('/');
    type.value = pathSegments[pathSegments.length - 2];
    getInfo()
    
})

const getInfo =async ()=>{
    response = await axios.get('/api/intelligence/'+ type.value + '/'+ _id)
    // console.log(response)
    if(ctx.$i18n.locale == 'zn')
        docx.value = response.data.data.url_zn
    else if(ctx.$i18n.locale == 'en')
        docx.value = response.data.data.url_en
    else if(ctx.$i18n.locale == 'de')
        docx.value = response.data.data.url_de
}
</script>

<style lang="scss" scoped>
.whole-box{
    margin-top: 20px;
    width: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.vue-office-docx{
    width: 100%;
    
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