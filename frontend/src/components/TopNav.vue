<template>
    <div id="whole-box">
        <div class="nav-container">
            <div v-show="true" class="nav-logo">
                <img id="logo-img" src="../public/img/蓝色-前.png">
                <!-- <p>Innovation&Behavior</p> -->
                <!-- <p>占个logo位</p> -->
            </div>
            <div class="nav-box">
                <div class="nav-menu-horizontal">
                    <el-menu mode="horizontal" default-active="0" :ellipsis="false">
                        <el-menu-item index="0" @click="redirectToHome" :class="{ 'blue-text': menutextcolor === 0 }">{{
                        $t('topnav.Home') }}</el-menu-item>
                        <el-sub-menu index="5" :class="{ 'blue-text': menutextcolor === 5 }">
                            <template #title>
                                {{ $t('topnav.Patent_Value_Survey') }}
                            </template>
                            <el-menu-item-group>
                                <el-menu-item index="5-1" @click="redirectToSurveyMethod">{{
                        $t('topnav.Survey_Method') }}</el-menu-item>
                                <el-menu-item index="5-2" @click="redirectToSurvey">{{
                        $t('topnav.Survey_Fill') }}</el-menu-item>
                                <el-menu-item index="5-3"
                                    @click="redirectToSurveyResults">{{ $t('topnav.Survey_Results') }} </el-menu-item>
                                <el-menu-item index="5-4" @click="redirectToData">{{
                        $t('topnav.DataUsageInstructions') }}</el-menu-item>
                        <el-menu-item index="5-5" @click="redirectToUtils">{{
                        $t('topnav.Survey_Utils') }}</el-menu-item>
                            </el-menu-item-group>
                        </el-sub-menu>
                        <el-menu-item index="2" @click="redirectToNews" :class="{ 'blue-text': menutextcolor === 2 }">{{
                        $t('topnav.News') }}</el-menu-item>
                        <el-menu-item index="3" @click="redirectToEconomics"
                            :class="{ 'blue-text': menutextcolor === 3 }">{{
                        $t('topnav.IP_Economics') }}</el-menu-item>
                        <el-menu-item index="4" @click="redirectToIntelligence"
                            :class="{ 'blue-text': menutextcolor === 4 }">{{
                        $t('topnav.IP_Intelligence') }}</el-menu-item>
                        <!-- <el-menu-item index="1" @click="redirectToPlatform" :class="{ 'blue-text': menutextcolor === 1 }">{{
                            $t('topnav.OpenLicensingPlatform') }}</el-menu-item> -->
                        <el-menu-item index="6" @click="redirectToTeam" :class="{ 'blue-text': menutextcolor === 6 }">{{
                        $t('topnav.Team') }}</el-menu-item>
                    </el-menu>
                </div>
                <div class="nav-search">
                    <!-- <div class="search-box"> -->
                    <el-input style="max-width: 200px;" v-model="store.searchQuery" :prefix-icon="Search"
                        @keyup.enter="handleSearch">
                    </el-input>
                    <!-- </div> -->

                    <el-dropdown id="language-dropdonw" trgger="hover" @command="handleLanguageChange"
                        style="border: none;box-shadow: none;">
                        <span class="el-dropdown-link">
                            {{ $t('topnav.language') }}<i class="el-icon-arrow-down el-icon--right"></i>
                        </span>
                        <template #dropdown>
                            <el-dropdown-menu slot="dropdown">
                                <el-dropdown-item command="en">{{ $t('topnav.English') }}</el-dropdown-item>
                                <el-dropdown-item command="zn">{{ $t('topnav.Chinese') }}</el-dropdown-item>
                                <el-dropdown-item command="de">{{ $t('topnav.German') }}</el-dropdown-item>
                            </el-dropdown-menu>
                        </template>
                    </el-dropdown>
                </div>
                <div class="nav-menu-dropdown">
                    <el-dropdown trgger="hover" style="border: none;box-shadow: none;">
                        <span class="el-dropdown-link">
                            {{ $t('topnav.menu') }}
                        </span>
                        <template #dropdown>
                            <el-dropdown-menu slot="dropdown">
                                <el-dropdown-item index="1" @click="redirectToHome"
                                    :class="{ 'blue-text': menutextcolor === 1 }">{{
                        $t('topnav.Home') }}</el-dropdown-item>
                                <el-dropdown trgger="hover" style="border: none;box-shadow: none;">
                                    <span class="el-dropdown-link">
                                        {{ $t('topnav.Patent_Value_Survey') }}
                                    </span>
                                    <template #dropdown>
                                        <el-dropdown-menu slot="dropdown">
                                            <el-dropdown-item index="5" @click="redirectToSurveyMethod"
                                                :class="{ 'blue-text': menutextcolor === 5 }">{{
                        $t('topnav.Survey_Method') }}</el-dropdown-item>
                                            <el-dropdown-item index="7" @click="redirectToSurvey"
                                                :class="{ 'blue-text': menutextcolor === 7 }">{{
                        $t('topnav.Survey_Fill') }}</el-dropdown-item>
                                            <el-dropdown-item index="8" @click="redirectToSurveyResults"
                                                :class="{ 'blue-text': menutextcolor === 8 }">{{
                        $t('topnav.Survey_Results') }}</el-dropdown-item>
                                            <el-dropdown-item index="9" @click="redirectToData"
                                                :class="{ 'blue-text': menutextcolor === 9 }">{{
                        $t('topnav.DataUsageInstructions') }}</el-dropdown-item>
                        <el-dropdown-item index="10" @click="redirectToUtils"
                                                :class="{ 'blue-text': menutextcolor === 10 }">{{
                        $t('topnav.Survey_Utils') }}</el-dropdown-item>
                                        </el-dropdown-menu>
                                    </template>
                                </el-dropdown>
                                <!-- <el-dropdown-item index="1" @click="redirectToPlatform"
                                    :class="{ 'blue-text': menutextcolor === 1 }">{{
                                        $t('topnav.OpenLicensingPlatform') }}</el-dropdown-item> -->
                                <el-dropdown-item index="2" @click="redirectToNews"
                                    :class="{ 'blue-text': menutextcolor === 2 }">{{
                        $t('topnav.News') }}</el-dropdown-item>
                                <el-dropdown-item index="3" @click="redirectToEconomics"
                                    :class="{ 'blue-text': menutextcolor === 3 }">{{
                        $t('topnav.IP_Economics') }}</el-dropdown-item>
                                <el-dropdown-item index="4" @click="redirectToIntelligence"
                                    :class="{ 'blue-text': menutextcolor === 4 }">{{
                        $t('topnav.IP_Intelligence') }}</el-dropdown-item>

                                <el-dropdown-item index="6" @click="redirectToTeam"
                                    :class="{ 'blue-text': menutextcolor === 6 }">{{
                                    $t('topnav.Team') }}</el-dropdown-item>
                                <el-dropdown trgger="hover" @command="handleLanguageChange"
                                    style="border: none;box-shadow: none;">
                                    <span class="el-dropdown-link">
                                        {{ $t('topnav.language') }}<i class="el-icon-arrow-down el-icon--right"></i>
                                    </span>
                                    <template #dropdown>
                                        <el-dropdown-menu slot="dropdown">
                                            <el-dropdown-item command="en">{{ $t('topnav.English') }}</el-dropdown-item>
                                            <el-dropdown-item command="zn">{{ $t('topnav.Chinese') }}</el-dropdown-item>
                                            <el-dropdown-item command="de">{{ $t('topnav.German') }}</el-dropdown-item>
                                        </el-dropdown-menu>
                                    </template>
                                </el-dropdown>
                            </el-dropdown-menu>
                        </template>
                    </el-dropdown>
                </div>
            </div>
        </div>
    </div>
</template>


<script setup>
import { ref, computed, getCurrentInstance } from 'vue'
import { useRouter } from 'vue-router'
import { useGeneralStore } from '../stores/general.js'
import { Search } from '@element-plus/icons-vue'

const store = useGeneralStore()
const router = useRouter()

//切换语言
let { ctx } = getCurrentInstance()
const handleLanguageChange = (language) => {
    ctx.$i18n.locale = language;
    store.changeLanguage = !store.changeLanguage
}

const handleSearch = () => {
    // 执行搜索逻辑
    console.log('Searching for:', store.searchQuery)
}

const redirectToHome = () => {
    router.push('/');
}

const redirectToPlatform = () => {
    router.push('/platform');
}
const redirectToNews = () => {
    router.push('/news');
}
const redirectToEconomics = () => {
    router.push('/economics');
}
const redirectToIntelligence = () => {
    router.push('/intelligence');
}
const redirectToSurvey = () => {
    router.push('/survey');
}
const redirectToTeam = () => {
    router.push('/team');
}
const redirectToSurveyMethod = () => {
    router.push('/survey-method');
};

const redirectToSurveyResults = () => {
    router.push('/survey-results');
};

const redirectToUtils = () => {
    router.push('/survey-utils');
};
const redirectToData = () => {
    router.push('/data-usage-instructions');
};

</script>

<style scoped>
#whole-box {
    display: flex;
    height: 120px;
    width: 100%;
    justify-content: center;
    /* border: 1px solid black; */
}

.nav-logo {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 10px;
    /* padding-left: 20px; */
    font-size: xx-large;
    font-weight: bolder;
    /* border: 1px solid black; */
}

#logo-img {
    /* padding-left: 30px; */
    width: 300px;
}

.nav-container {
    width: 100%;
    display: flex;
    max-width: 1400px;
    justify-content: space-between;
    align-items: center;
    /* border: 1px solid black; */
    padding-top: 10px;
    width: 100%;
    background-color: #fff;
}

.el-menu-item {
    font-size: large;
}

.nav-box {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 10px;
    align-items: flex-end;
    /* justify-content: center; */
    /* border: 1px solid #000; */
}

.nav-menu {
    /* border: 1px solid #000; */
    /* width: 100%; */
    display: flex;
    align-items: center;
}

.nav-search {
    /* width: 100%; */
    /* border: 1px solid #170202; */
    display: flex;
    justify-content: end;
    align-items: center;
}

.el-dropdown-link {
    cursor: pointer;
    display: flex;
    align-items: center;
    word-wrap: nowrap;
    margin-left: 5px;
    text-wrap: nowrap;
    padding: 5px 10px;
    color: #449cf4f9;
    border-radius: 4px;
    background-color: #f9f9f9;
    transition: background-color 0.3s, border-color 0.3s;
}

.el-dropdown-link:hover {
    background-color: #e6e6e6;
    /* 鼠标悬浮时的背景色 */
    border-color: #b3b3b3;
    /* 鼠标悬浮时的边框色 */
}


@media (max-width: 1060px) {
    .el-menu-item {
        font-size: 10px;
        padding: 10px;
    }
}

@media(min-width: 720px) {

    .nav-menu-horizontal {
        display: flex;
    }

    .nav-menu-dropdown {
        display: none;
    }

}

@media(max-width: 720px) {
    /* .nav-menu-horizontal {
        display: none;
    }

    #language-dropdonw{
        display: none;
    }
    .nav-menu-dropdown {
        display: block;
    } */
    /* .nav-menu-dropdown {
        display: none;
    } */

    /* .nav-logo {
        display: none;
    } */

    /* .nav-box {
        align-items: center;
    } */
}

@media(max-width: 720px) {
    #logo-img {
        width: 160px;
    }

    .nav-logo {
        display: flex;
        padding-left: 0px;
        padding-right: 5px;
        margin-bottom: 0px;
        font-size: medium;
    }

    .nav-menu-horizontal {
        display: none;
    }

    .nav-menu-dropdown {
        display: block;
    }

    #language-dropdonw {
        display: none;
    }

    .nav-container {
        justify-content: space-between;
    }

    .nav-box {
        flex-direction: row;
        justify-content: flex-end;
        align-items: center;
    }

    .el-header {
        height: 80px;
    }

    #whole-box {
        height: 80px;
        margin-top: -20px;
    }
}
</style>