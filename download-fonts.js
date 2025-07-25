#!/usr/bin/env node

/**
 * 字体文件下载脚本
 * 用于自动下载项目所需的字体文件到 public/fonts 目录
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

// 字体文件下载配置
const FONTS_CONFIG = {
  'NotoSansJP-Regular.woff2': 'https://fonts.gstatic.com/s/notosansjp/v52/-F6jfjtqLzI2JPCgQBnw7HFyzSD-AsregP8VFBEj75vY0rw-oME.woff2',
  'NotoSansJP-Medium.woff2': 'https://fonts.gstatic.com/s/notosansjp/v52/-F6jfjtqLzI2JPCgQBnw7HFyzSD-AsregP8VFBEj75vY0rw-oME.woff2',
  'NotoSansJP-Bold.woff2': 'https://fonts.gstatic.com/s/notosansjp/v52/-F6jfjtqLzI2JPCgQBnw7HFyzSD-AsregP8VFBEj75vY0rw-oME.woff2',
  'NotoSerifJP-Regular.woff2': 'https://fonts.gstatic.com/s/notoserifjp/v20/xn77YHs72GKoTvER4Gn3b5eMZBaPbBpBhgXsOwD-fO4.woff2',
  'NotoSerifJP-Medium.woff2': 'https://fonts.gstatic.com/s/notoserifjp/v20/xn77YHs72GKoTvER4Gn3b5eMZBaPbBpBhgXsOwD-fO4.woff2',
  'NotoSerifJP-Bold.woff2': 'https://fonts.gstatic.com/s/notoserifjp/v20/xn77YHs72GKoTvER4Gn3b5eMZBaPbBpBhgXsOwD-fO4.woff2',
  'GeistMono-Regular.woff2': 'https://github.com/vercel/geist-font/raw/main/packages/next/src/mono/GeistMono-Regular.woff2'
};

const FONTS_DIR = path.join(__dirname, 'public', 'fonts');

// 确保字体目录存在
if (!fs.existsSync(FONTS_DIR)) {
  fs.mkdirSync(FONTS_DIR, { recursive: true });
  console.log('✅ 创建字体目录:', FONTS_DIR);
}

// 下载字体文件函数
function downloadFont(filename, url) {
  return new Promise((resolve, reject) => {
    const filePath = path.join(FONTS_DIR, filename);
    
    // 检查文件是否已存在
    if (fs.existsSync(filePath)) {
      console.log(`⏭️  跳过已存在的文件: ${filename}`);
      resolve();
      return;
    }

    console.log(`📥 开始下载: ${filename}`);
    
    const file = fs.createWriteStream(filePath);
    
    https.get(url, (response) => {
      if (response.statusCode === 200) {
        response.pipe(file);
        
        file.on('finish', () => {
          file.close();
          console.log(`✅ 下载完成: ${filename}`);
          resolve();
        });
      } else {
        fs.unlink(filePath, () => {});
        reject(new Error(`下载失败: ${filename}, 状态码: ${response.statusCode}`));
      }
    }).on('error', (err) => {
      fs.unlink(filePath, () => {});
      reject(new Error(`下载错误: ${filename}, ${err.message}`));
    });
  });
}

// 主函数
async function main() {
  console.log('🚀 开始下载字体文件...');
  console.log('📁 目标目录:', FONTS_DIR);
  console.log('');

  try {
    const downloadPromises = Object.entries(FONTS_CONFIG).map(([filename, url]) => 
      downloadFont(filename, url)
    );

    await Promise.all(downloadPromises);
    
    console.log('');
    console.log('🎉 所有字体文件下载完成!');
    console.log('💡 提示: 请重启开发服务器以应用字体更改');
    
  } catch (error) {
    console.error('❌ 下载过程中出现错误:', error.message);
    console.log('');
    console.log('🔧 解决方案:');
    console.log('1. 检查网络连接');
    console.log('2. 手动从以下地址下载字体文件:');
    console.log('   - Google Fonts: https://fonts.google.com/');
    console.log('   - Noto Fonts: https://github.com/googlefonts/noto-fonts');
    console.log('   - Geist Font: https://github.com/vercel/geist-font');
    console.log('3. 将字体文件放置到 public/fonts/ 目录');
    process.exit(1);
  }
}

// 运行脚本
if (require.main === module) {
  main();
}

module.exports = { downloadFont, FONTS_CONFIG, FONTS_DIR };