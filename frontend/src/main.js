import { createApp } from 'vue'
import App from './App.vue'

import { createPinia } from 'pinia'
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

import router from './router'
import { createI18n } from 'vue-i18n';
import enMessages from './locales/en.json';
import znMessages from './locales/zn.json';
import deMessages from './locales/de.json';

import axios from 'axios'

const i18n = createI18n({
    locale: 'zn', // 设置默认语言
    fallbackLocale: 'zn',
    messages: {
        en: enMessages, // 导入英语消息
        zn: znMessages, // 导入中文消息
        de: deMessages // 导入德文消息
    }
});

const pinia = createPinia()
const app = createApp(App)

pinia.use(piniaPluginPersistedstate)

for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component)
}
app.use(pinia)
app.use(ElementPlus)
app.use(router)
app.use(i18n)
app.mount('#app')
