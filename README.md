# Docker CodeFormer

快速体验 & 上手 CodeFormer。

![](.github/preview.jpg)

## 使用教程

《[Stable Diffusion 硬核生存指南：WebUI 中的 CodeFormer](https://soulteary.com/2023/08/02/stable-diffusion-hardcore-survival-guide-codeformer-in-webui.html)》


## 快速上手

从 `Docker CodeFormer` 项目下载代码，并进入项目目录：

```bash
git clone https://github.com/soulteary/docker-codeformer.git

cd docker-codeformer
```

执行项目中的镜像构建工具：

```bash
scripts/build.sh
```

在完成基础镜像构建之后，可以从[网盘下载](https://pan.baidu.com/s/1rxaHRyYuff1gbt-g1y6DNA?pwd=soul) `weights.zip`。

模型应用运行需要的所有模型都在这里了，下载完毕后，解压缩模型压缩包，将 `CodeFormer`、`facelib`、`realesrgan` 三个目录放置到 `weights` 目录中，完整的项目结构这样的：

```bash
├── LICENSE
├── README.md
├── assets
│   └── image
├── docker
│   └── Dockerfile
├── scripts
│   └── build.sh
├── src
│   ├── app.py
│   └── code-fix.py
└── weights
    ├── CodeFormer
    ├── facelib
    └── realesrgan
```

准备好模型文件之后，使用下面的命令启动模型应用：

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -it -v `pwd`/weights/:/app/CodeFormer/weights -p 7860:7860 soulteary/docker-codeformer
```

稍等片刻，我们将看到类似下面的日志：

```bash
Running on local URL:  http://0.0.0.0:7860

To create a public link, set `share=True` in `launch()`.
```

接着，我们就可以打开浏览器访问 `http://localhost:7860` 或者 `http://你的IP地址:7860` 来试试看啦。


## 处理图片对比

![](.github/case1.jpg)

![](.github/case2.jpg)

![](.github/case3.jpg)

## 相关项目

- [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer)
