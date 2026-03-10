import { defineStore } from 'pinia'

export const useIntelligenceStore = defineStore('intelligence', {
  state: () => ({
    chosedPage: 1
  }),
//   persist: true,
})
