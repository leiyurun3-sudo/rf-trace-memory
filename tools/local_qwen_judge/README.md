# R85 本地 Qwen-VL AI 评审（历史 v6）

此版本作为评测开发历史保留，分数未用于当前论文比较。当前 factual v2 推理程序见 `../factual_qwen_v2`，修正后的汇总见 `../factual_qwen_audit`。

v6 将适用性改为程序管理：模型只评分适用项，彻底移除重复预测适用性造成的冲突。推理仅在服务器本地进行，不上传图片或调用推理 API。

## 一键启动

将 ZIP 解压到服务器后，在解压出的 `A_Qwen_LocalJudge` 目录执行：

```bash
bash start_qwen_judge.sh
```

默认复用服务器 `.venv/bin/python`（可用 `QWEN_PYTHON` 指定），完成后自动生成 `/root/autodl-tmp/A_Qwen_LocalJudge_results.zip`。

脚本先检查 `/root/autodl-tmp/MINT_Flow_Jupyter_Ready/R84_R85_FinalNight/R85_IndependentComparison`，再检查两个常见备用路径；找不到时明确退出。首次下载 Qwen2.5-VL-7B-Instruct 到持久目录 `/root/autodl-tmp/local-models/Qwen2.5-VL-7B-Instruct`，记录解析到的模型 commit；下载完成后强制离线推理。需要镜像时运行：`HF_ENDPOINT=https://hf-mirror.com bash start_qwen_judge.sh`。可选环境变量 `R85_ROOT`、`QWEN_MODEL_DIR`、`QWEN_JUDGE_OUT` 和 `QWEN_REPEAT_COUNT` 覆盖默认值。默认结果目录为 `/root/autodl-tmp/A_Qwen_LocalJudge_results`，默认对固定随机抽取的 20 个主任务各重复评分一次，设为 0 可关闭。

需要 Linux、CUDA、现有 PyTorch、transformers、Pillow、accelerate 和 huggingface_hub；服务器 torch 2.8、transformers 4.55.4 为目标环境。脚本不安装或升级依赖；缺少依赖时在独立环境补齐。包内 `r85_plan.json` 取自 R85 结果计划，包含完整 520 项任务映射与源文件校验值；即使服务器目录留有旧的 440 项计划，也按 520 项运行。原生输入逐项检查：源图使用 R85 plan 的 `source_image`，生成图使用 `results/<job-id>/raw.png`；任一缺失时写出清单并停止，避免用预览替代。输入完整后，实际将本地模型和处理器加载到 GPU 并报告显存，再开始评审；可用显存仅供参考，不设 30 GiB 硬门槛。GPU 模型加载不成功时停止。

## 协议与输出

先只给模型源图和源/目标描述，冻结 requested_edit、protected_requirements、subject_applicable 和 pattern_applicable。主体替换或数量变化时，源主体保持要求可由该源任务 rubric 标为不适用；subject 与 pattern 的适用性对所有方法和控制固定。输出评审只收到两张图和对应文本、冻结 rubric，方法名保存在私有映射中。每项输出为 0、1 或带原因的 null；仅 rubric 明确 false 才视为 NA，适用性未知或评分不确定保持未决。

joint 要求 edit=1、artifact=0；subject 和 pattern 各自在适用时要求为1，明确 NA 时略去。任何已知失败都会使 joint=0，即使另一维未知；全部必要项通过则为1，其余为未决。各维比例的分母排除对应的明确 NA；某维全部 NA 时适用分母为0，比例为空。未决项目不填造结果，输出未决数和将未决分别按0、1处理的输出级上下界。

保护要求允许为空列表。源任务 rubric 仍有六个固定字段。输出阶段模型只返回 edit、artifact 和 rubric 明确 true 的 subject/pattern；每项只含 value 与 reason。程序从冻结 rubric 附加 applicable，并把明确不适用或适用性未知的项写成 value=null，原因注明来自冻结 rubric。它们不是模型判出的成功或失败。模型额外返回的适用性和非适用项只留在原始回复中，不改变冻结规则。

格式错误仍最多三次尝试，保存所有提示、原始回答和错误。适用项缺失的分数和理由不会补写。连续三个任务失败停止；全部源 rubric 有效后才开始输出评分。v6 使用新协议指纹，旧协议和原始记录保留。仅在模型、输入、任务与源 rubric 一致时迁移 v5 已验证 rubric 和已成功的评分，记录原协议和原文件；失败评分重新生成。可见 `ok cached` 表示已有效复用。进度写入 progress.json，逐项报告实际耗时和剩余时间。


`ratings.csv` 含各项状态和原因；`arm_summary.csv` 给各方法、维度、适用分母、NA、未决、上下界和按源聚类 bootstrap 区间；`paired_summary.csv` 给 CORR25 与各方法的 source 配对 joint 差值和区间；`summary.json` 包含相同汇总及控制信息。先对每源可用 seed 取均值，再对 source 重采样；RFEDIT 只有 seed0。缺失 seed 不作失败。

`controls_private.json` 与输出中的 `SOURCE_CONTROL` 是每个任务的源图对源图校准，需结合任务解释，不自动判失败。`repeat_responses/` 和 `repeat_audit.csv` 独立保存重复评分及一致性，不计入主任务；重复采用贪心解码，只检查同一模型设置下复现性。`rubrics/`、`responses/` 保存原始回答、提示词、revision 和结果；`jobs_private.json` 保存本地方法映射。内容、冻结 rubric、模型或协议变化会更换缓存指纹；失败项可重试。模型可见的图像依然可能泄露方法特征，匿名与打乱不能保证消除偏差。

可选手工预检：`python judge.py preflight --model /path/to/model --load-model`；手工运行参数见 `bash run_local_judge.sh`。模型下载参考 [Qwen 官方模型页](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) 和 [Transformers 文档](https://huggingface.co/docs/transformers/v4.51.1/en/model_doc/qwen2_5_vl)。
