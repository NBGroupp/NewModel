# 数据处理

## 数据文件

- **corpuses,7z:** 包含了国家平衡语料库、中文维基以及百度百科语料，为未清洗的原始数据，因为文件大小超过了100M，因此需要 git-lfs 来同步或者直接在网页上点击下载
- **vocab.10000:** 最终模型使用的词典，从中文维基建立，字为单位

## 代码文件

- **功能函数代码**

  [**langconv.py**](https://github.com/csdz/nstools/blob/master/zhtools/langconv.py)、[**zh-wiki.py**](https://github.com/csdz/nstools/blob/master/zhtools/zh_wiki.py) - 简繁转换

  **data_util.py** - 主要功能函数

- **处理流程代码**

  **data_process.py**

- **运行配置文件**

  **config.py**

- **多个不同混错比例数据集一次运行代码（linux）**

  **run.py**：修改代码中的 error_ratios 元组设置不同比例，脚本每跑完一个混错数据集就会打包生成 .7z 文件
