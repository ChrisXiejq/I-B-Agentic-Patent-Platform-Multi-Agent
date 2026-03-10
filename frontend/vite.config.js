import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    open: true,//启动项目自动弹出浏览器
    port: 8192,//启动端口
    proxy: {
      '/api': {
        // target: 'http://49.235.179.94:8190',	
         target: 'http://localhost:8190',
        // target: 'http://110.40.206.206:8190',	
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      },
    },
    // client: {
    //   overlay: false   //关闭Uncaught runtime errors
    // }

  },
  define: {
    'process.env': {},
  },
})