import { defineStore } from 'pinia'

export const useGeneralStore = defineStore('general', {
  state: () => ({
    chosedPage: 1, //指在分页列表中选择的页数
    searchQuery: '',
    changeLanguage: true,
    patentNo:'',
  }),
//   persist: true,
})
