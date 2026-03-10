<template>
    <body>
        <div class="update-notice">
            <div class="content">
                内容即将更新，敬请期待
            </div>
        </div>
    </body>
</template>

<script setup>
</script>

<style lang="scss" scoped>
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