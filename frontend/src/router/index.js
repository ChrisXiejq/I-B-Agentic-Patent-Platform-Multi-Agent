import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  {
    // Home
    path: '/',
    name: 'home',
    component: () => import('../views/Home/home.vue')
  },
  {
    // OpenLicensingPlatform
    path: '/platform',
    name: 'OpenLicensingPlatform',
    component: () => import('../views/OpenLicensingPlatform/Platform.vue')
  },
  {
    // News
    path: '/news',
    name: 'news',
    component: () => import('../views/News/News.vue')
  },
  {
    // News_Info
    path: '/news/:id',
    name: 'newsInfo',
    component: () => import('../views/News/Info.vue')
  },
  {
    // TeamMember
    path: '/team',
    name: 'team',
    component: () => import('../views/TeamMember/TeamMember.vue')
  },
  {
    // IP_Economics
    path: '/economics',
    name: 'economics',
    component: () => import('../views/IP_Economics/index.vue')
  },
  {
    // IP_Economic_Article_Info
    path: '/economic/articles/:_id',
    name: 'economicArticleInfo',
    component: () => import('../views/IP_Economics/Info.vue')
  },
  {
    // IP_Economic_Courses_Info
    path: '/economic/courses/:_id',
    name: 'economicCoursesInfo',
    component: () => import('../views/IP_Economics/courseInfo.vue')
  },
  {
    // IP_Intelligence
    path: '/intelligence',
    name: 'intelligence',
    component: () => import('../views/IP_Intelligence/index.vue')
  },
  {
    // IP_Intelligence_Course_Info
    path: '/intelligence/courses/:_id',
    name: 'intelligenceCourseInfo',
    component: () => import('../views/IP_Intelligence/Info.vue')
  },
  {
    // IP_Intelligence_PatentPledge_Info
    path: '/intelligence/patentpledge/:_id',
    name: 'intelligencePatentPledgeInfo',
    component: () => import('../views/IP_Intelligence/Info.vue')
  },
  {
    // Patent_Value_Survey
    path: '/survey-method',
    name: 'survey-method',
    component: () => import('../views/Patent_Value_Survey/Survey-method.vue')
  },
  {
    // Patent_Value_Survey
    path: '/survey',
    name: 'survey',
    component: () => import('../views/Patent_Value_Survey/Survey.vue')
  },
  {
    // Patent_Value_Survey
    path: '/survey-results',
    name: 'survey-results',
    component: () => import('../views/Patent_Value_Survey/Survey-results.vue')
  },
  {
    // SurveyUtils
    path: '/survey-utils',
    name: 'survey-utils',
    component: () => import('../views/Patent_Value_Survey/SurveyUtils.vue')
  },
  {
    // DataUsageInstructions
    path: '/data-usage-instructions',
    name: 'data-usage-instructions',
    component: () => import('../views/Patent_Value_Survey/DataUsageInstructions.vue')
  }
]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

// router.beforeEach((to, from, next) => {

// })

export default router
