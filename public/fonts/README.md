# 字体文件说明

为了避免动态网络下载字体，请将以下字体文件放置在此目录中：

## 需要的字体文件

### Noto Sans JP (无衬线字体)
- `NotoSansJP-Regular.woff2` (400 weight)
- `NotoSansJP-Medium.woff2` (500 weight) 
- `NotoSansJP-Bold.woff2` (700 weight)

### Noto Serif JP (衬线字体)
- `NotoSerifJP-Regular.woff2` (400 weight)
- `NotoSerifJP-Medium.woff2` (500 weight)
- `NotoSerifJP-Bold.woff2` (700 weight)

### Geist Mono (等宽字体)
- `GeistMono-Regular.woff2`

## 字体文件获取方式

### 方法1：从 Google Fonts 下载
1. 访问 [Google Fonts](https://fonts.google.com/)
2. 搜索并下载 "Noto Sans JP" 和 "Noto Serif JP"
3. 选择需要的字重：Regular (400)、Medium (500)、Bold (700)
4. 下载 WOFF2 格式的字体文件

### 方法2：从 GitHub 下载
- Noto 字体：https://github.com/googlefonts/noto-fonts
- Geist 字体：https://github.com/vercel/geist-font

## 文件结构

放置完成后，目录结构应该如下：

```
public/fonts/
├── NotoSansJP-Regular.woff2
├── NotoSansJP-Medium.woff2
├── NotoSansJP-Bold.woff2
├── NotoSerifJP-Regular.woff2
├── NotoSerifJP-Medium.woff2
├── NotoSerifJP-Bold.woff2
├── GeistMono-Regular.woff2
└── README.md (本文件)
```

## 注意事项

1. 确保字体文件名与上述列表完全一致
2. 推荐使用 WOFF2 格式，文件更小，加载更快
3. 如果某些字体文件缺失，应用会回退到系统默认字体
4. 字体文件放置完成后，重启开发服务器以确保更改生效

## 验证

字体文件放置完成后，可以通过浏览器开发者工具的 Network 面板验证是否还有对 Google Fonts 的网络请求。如果配置正确，应该不会看到任何对 `fonts.googleapis.com` 或 `fonts.gstatic.com` 的请求。