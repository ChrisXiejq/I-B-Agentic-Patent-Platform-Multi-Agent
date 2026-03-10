### 国际化

文本内容存储在locals文件夹

没研究出来watch语言切换的方法，就在UseGeneralStore里加了个changeLanguage

watch(() => store.changeLanguage, () => {

​    //updateContent(ctx.$i18n.locale);

});

具体如何使用详见Components/TopNav.vue（使用locales文件夹中的文本）

components/public/ContentItem.vue（内容随语言实时切换）