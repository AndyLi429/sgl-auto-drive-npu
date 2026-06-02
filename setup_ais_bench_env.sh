#!/usr/bin/env bash
# AISBench 一键环境初始化脚本
# 功能：克隆仓库 + 安装依赖 + 下载解压数据集
#       （AIME2024 / AIME2025 / AIME2026 / GPQA / GSM8K / C-Eval / LongBench / ShareGPT）
# 用法：bash setup_env.sh [选项]
#   --install-dir <路径>   仓库克隆到的目录（默认: ~/benchmark）
#   --skip-clone           跳过 git clone（已有仓库时使用）
#   --skip-install         跳过 pip 安装
#   --hf-token <TOKEN>     HuggingFace token（下载 LongBench/ShareGPT 用）
#   --hf-endpoint <URL>    HF 镜像地址（默认: https://hf-mirror.com）
# 示例：
#   bash setup_env.sh                              # 全新环境一键初始化
#   bash setup_env.sh --skip-clone --skip-install  # 只补下数据集

set -euo pipefail

# ────────────────────────────────────────────────────────────────
# 参数解析
# ────────────────────────────────────────────────────────────────
SKIP_CLONE=false
SKIP_INSTALL=false
HF_TOKEN=""
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
INSTALL_DIR="${HOME}/benchmark"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-clone)    SKIP_CLONE=true; shift ;;
        --skip-install)  SKIP_INSTALL=true; shift ;;
        --install-dir)   INSTALL_DIR="$2"; shift 2 ;;
        --hf-token)      HF_TOKEN="$2"; shift 2 ;;
        --hf-endpoint)   HF_ENDPOINT="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

export HF_ENDPOINT

# ────────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

command -v wget  &>/dev/null || error "缺少工具: wget"
command -v unzip &>/dev/null || error "缺少工具: unzip"
command -v git   &>/dev/null || error "缺少工具: git"
unset http_proxy https_proxy ALL_PROXY all_proxy

# ────────────────────────────────────────────────────────────────
# 0. 克隆仓库
# ────────────────────────────────────────────────────────────────
if [[ "$SKIP_CLONE" == false ]]; then
    info "======== 克隆 AISBench 仓库 ========"

    # 跳过 SSL 证书校验（服务器证书链不完整时 git 会失败）
    export GIT_SSL_NO_VERIFY=1

    if [[ -d "$INSTALL_DIR/.git" ]]; then
        warn "$INSTALL_DIR 已是 git 仓库，跳过 clone，执行 git pull"
        git -C "$INSTALL_DIR" pull --ff-only
    else
        # GitHub 在国内时通时断，最多重试 3 次
        CLONE_OK=false
        for attempt in 1 2 3; do
            info "git clone 尝试 ${attempt}/3 ..."
            if git clone https://gitee.com/aisbench/benchmark.git "$INSTALL_DIR"; then
                CLONE_OK=true
                break
            fi
            warn "第 ${attempt} 次失败，等待 5 秒后重试..."
            sleep 5
        done

        if [[ "$CLONE_OK" == false ]]; then
            error "git clone 连续 3 次失败。请检查网络或手动克隆后加 --skip-clone 重试:\n  git clone https://gitee.com/aisbench/benchmark.git $INSTALL_DIR"
        fi
        info "克隆完成: $INSTALL_DIR"
    fi
else
    warn "--skip-clone 已指定，跳过克隆"
fi

REPO_ROOT="$INSTALL_DIR"
DATASETS_DIR="${REPO_ROOT}/ais_bench/datasets"

# ────────────────────────────────────────────────────────────────
# 1. 安装 Python 依赖
# ────────────────────────────────────────────────────────────────
if [[ "$SKIP_INSTALL" == false ]]; then
    info "======== 安装依赖 ========"
    cd "$REPO_ROOT"
    [[ ! -f "setup.py" && ! -f "pyproject.toml" ]] && error "未找到项目文件，请确认 --install-dir 路径正确: $REPO_ROOT"

    # 若环境设置了 SOCKS 代理但缺少 PySocks，pip 会报 "Missing dependencies for SOCKS support"
    # 临时清除代理，仅对本脚本的 pip 调用生效，不影响系统全局设置
    _SAVED_ALL_PROXY="${ALL_PROXY:-}"
    _SAVED_all_proxy="${all_proxy:-}"
    _SAVED_HTTPS_PROXY="${HTTPS_PROXY:-}"
    _SAVED_https_proxy="${https_proxy:-}"
    _SAVED_HTTP_PROXY="${HTTP_PROXY:-}"
    _SAVED_http_proxy="${http_proxy:-}"
    unset ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy

    PIP_MIRROR="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"

    info "安装核心包"
    pip3 install -e ./ --use-pep517 $PIP_MIRROR

    info "安装 API 推理依赖（vLLM / MindIE / TGI / Triton）"
    pip3 install -r requirements/api.txt -q $PIP_MIRROR

    info "安装扩展依赖"
    pip3 install -r requirements/extra.txt -q $PIP_MIRROR

    # 恢复代理环境变量（后续 wget/HF 下载可能需要）
    [[ -n "$_SAVED_ALL_PROXY"   ]] && export ALL_PROXY="$_SAVED_ALL_PROXY"
    [[ -n "$_SAVED_all_proxy"   ]] && export all_proxy="$_SAVED_all_proxy"
    [[ -n "$_SAVED_HTTPS_PROXY" ]] && export HTTPS_PROXY="$_SAVED_HTTPS_PROXY"
    [[ -n "$_SAVED_https_proxy" ]] && export https_proxy="$_SAVED_https_proxy"
    [[ -n "$_SAVED_HTTP_PROXY"  ]] && export HTTP_PROXY="$_SAVED_HTTP_PROXY"
    [[ -n "$_SAVED_http_proxy"  ]] && export http_proxy="$_SAVED_http_proxy"
else
    warn "--skip-install 已指定，跳过依赖安装"
fi

# ────────────────────────────────────────────────────────────────
# 准备数据集根目录
# ────────────────────────────────────────────────────────────────
info "======== 准备数据集目录: $DATASETS_DIR ========"
mkdir -p "$DATASETS_DIR"
cd "$DATASETS_DIR"

# wget 不支持 socks5h:// 协议，数据集下载阶段全程清除 SOCKS 代理
unset ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy

# ────────────────────────────────────────────────────────────────
# 2. AIME 2024  →  aime/aime.jsonl
#    对应数据集任务: aime2024_gen / aime2024_gen_0_shot_chat_prompt / aime2024_gen_0_shot_str
#    数据集任务默认路径: ais_bench/datasets/aime/aime.jsonl
# ────────────────────────────────────────────────────────────────
info "======== 下载 AIME2024 数据集 ========"
if [[ -f "aime/aime.jsonl" ]]; then
    warn "aime/aime.jsonl 已存在，跳过"
else
    mkdir -p aime
    if [[ -f "aime/aime.zip" ]] && ! unzip -t aime/aime.zip &>/dev/null; then
        warn "aime/aime.zip 已损坏，删除重新下载"
        rm -f aime/aime.zip
    fi
    [[ ! -f "aime/aime.zip" ]] && \
        wget --show-progress -O aime/aime.zip \
            "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime.zip"
    unzip -q aime/aime.zip -d aime/ && rm -f aime/aime.zip
    info "AIME2024 完成: $(ls aime/)"
fi

# ────────────────────────────────────────────────────────────────
# 2b. AIME 2025  →  aime2025/aime2025.jsonl
#    对应数据集任务: aime2025_gen / aime2025_gen_0_shot_chat_prompt / aime2025_gen_0_shot_llmjudge
#    数据集任务默认路径: ais_bench/datasets/aime2025/aime2025.jsonl
#    来源：opencompass OSS（解压后自带 aime2025/ 目录）
# ────────────────────────────────────────────────────────────────
info "======== 下载 AIME2025 数据集 ========"
if [[ -f "aime2025/aime2025.jsonl" ]]; then
    warn "aime2025/aime2025.jsonl 已存在，跳过"
else
    if [[ -f "aime2025.zip" ]] && ! unzip -t aime2025.zip &>/dev/null; then
        warn "aime2025.zip 已损坏，删除重新下载"
        rm -f aime2025.zip
    fi
    [[ ! -f "aime2025.zip" ]] && \
        wget --show-progress -O aime2025.zip \
            "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime2025.zip"
    unzip -q aime2025.zip && rm -f aime2025.zip
    info "AIME2025 完成: $(ls aime2025/)"
fi

# ────────────────────────────────────────────────────────────────
# 2c. AIME 2026  →  aime2026/aime2026.jsonl
#    对应数据集任务: aime2026_gen / aime2026_gen_0_shot_chat_prompt / aime2026_gen_0_shot_str
#    数据集任务默认路径: ais_bench/datasets/aime2026/aime2026.jsonl
#    来源：ModelScope evalscope/aime26（无 OSS 直链，需用 modelscope 拉取后整理为 jsonl）
#    jsonl 字段要求: problem（题干） + answer（整数答案），与配置文件
#    aime2026_gen_0_shot_chat_prompt.py 的 input_columns=["problem"], output_column="answer" 对齐
# ────────────────────────────────────────────────────────────────
info "======== 下载 AIME2026 数据集（ModelScope）========"
if [[ -f "aime2026/aime2026.jsonl" ]]; then
    warn "aime2026/aime2026.jsonl 已存在，跳过"
else
    mkdir -p aime2026

    # modelscope 不在基础依赖中，缺失时按需安装（aliyun 源对 modelscope 拉取更稳定）
    if ! python3 -c "import modelscope" &>/dev/null; then
        info "未检测到 modelscope，开始安装 ..."
        pip3 install -q modelscope \
            -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
            || warn "modelscope 安装失败，AIME2026 将无法自动下载"
    fi

    if python3 -c "import modelscope" &>/dev/null; then
        info "使用 modelscope 拉取 evalscope/aime26 并整理为 aime2026.jsonl ..."
        python3 - <<PYEOF
import json, os
from modelscope.msdatasets import MsDataset

out = "${DATASETS_DIR}/aime2026/aime2026.jsonl"
os.makedirs(os.path.dirname(out), exist_ok=True)

# 不同版本/构建的 split 名可能不同，依次尝试
ds, last_err = None, None
for split in ["test", "train", "validation"]:
    try:
        ds = MsDataset.load("evalscope/aime26", split=split)
        break
    except Exception as e:
        last_err = e
if ds is None:
    raise SystemExit(f"无法从 ModelScope 加载 evalscope/aime26: {last_err}")

def pick(row, keys):
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return v
    return None

n = 0
with open(out, "w", encoding="utf-8") as f:
    for row in ds:
        problem = pick(row, ["problem", "question", "Problem", "Question"])
        answer  = pick(row, ["answer", "gold_answer", "Answer", "solution"])
        if problem is None or answer is None:
            continue
        f.write(json.dumps({"problem": problem, "answer": str(answer)}, ensure_ascii=False) + "\n")
        n += 1
print(f"AIME2026 写入 {n} 条 -> {out}")
PYEOF
    fi

    if [[ -f "aime2026/aime2026.jsonl" ]]; then
        info "AIME2026 完成: $(wc -l < aime2026/aime2026.jsonl) 题"
    else
        warn "AIME2026 自动下载失败，请手动准备："
        warn "  1) 打开 https://modelscope.cn/datasets/evalscope/aime26/summary"
        warn "  2) 将题目整理为 jsonl（每行 {\"problem\": ..., \"answer\": ...}）"
        warn "  3) 保存到 ${DATASETS_DIR}/aime2026/aime2026.jsonl"
    fi
fi

# ────────────────────────────────────────────────────────────────
# 3. GPQA  →  gpqa/gpqa_diamond.csv 等
# ────────────────────────────────────────────────────────────────
info "======== 下载 GPQA 数据集 ========"
if [[ -f "gpqa/gpqa_diamond.csv" ]]; then
    warn "gpqa/gpqa_diamond.csv 已存在，跳过"
else
    # 检测残留的损坏 zip，自动清理重下
    if [[ -f "gpqa.zip" ]] && ! unzip -t gpqa.zip &>/dev/null; then
        warn "gpqa.zip 已损坏，删除重新下载"
        rm -f gpqa.zip
    fi
    [[ ! -f "gpqa.zip" ]] && \
        wget --show-progress -O gpqa.zip \
            "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gpqa.zip"
    unzip -q gpqa.zip && rm -f gpqa.zip
    info "GPQA 完成: $(ls gpqa/)"
fi

# ────────────────────────────────────────────────────────────────
# 4. C-Eval  →  ceval/formal_ceval/{dev,val,test}/*.csv
#    来源：魔塔社区 ModelScope
# ────────────────────────────────────────────────────────────────
info "======== 下载 C-Eval 数据集 ========"
if [[ -d "ceval/formal_ceval/val" && -n "$(ls ceval/formal_ceval/val/*.csv 2>/dev/null)" ]]; then
    warn "ceval/formal_ceval/val/ 已存在，跳过"
else
    mkdir -p ceval/formal_ceval
    [[ ! -f "ceval/formal_ceval/ceval-exam.zip" ]] && \
        wget --show-progress -O ceval/formal_ceval/ceval-exam.zip \
            "https://www.modelscope.cn/datasets/opencompass/ceval-exam/resolve/master/ceval-exam.zip"
    unzip -q ceval/formal_ceval/ceval-exam.zip -d ceval/formal_ceval/ && \
        rm -f ceval/formal_ceval/ceval-exam.zip
    info "C-Eval 完成: dev/$(ls ceval/formal_ceval/dev/ | wc -l)题 val/$(ls ceval/formal_ceval/val/ | wc -l)题 test/$(ls ceval/formal_ceval/test/ | wc -l)题"
fi

# ────────────────────────────────────────────────────────────────
# 5. GSM8K  →  gsm8k/test.jsonl 等
# ────────────────────────────────────────────────────────────────
info "======== 下载 GSM8K 数据集 ========"
if [[ -f "gsm8k/test.jsonl" ]]; then
    warn "gsm8k/test.jsonl 已存在，跳过"
else
    if [[ -f "gsm8k.zip" ]] && ! unzip -t gsm8k.zip &>/dev/null; then
        warn "gsm8k.zip 已损坏，删除重新下载"
        rm -f gsm8k.zip
    fi
    [[ ! -f "gsm8k.zip" ]] && \
        wget --show-progress -O gsm8k.zip \
            "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip"
    unzip -q gsm8k.zip && rm -f gsm8k.zip
    info "GSM8K 完成: $(ls gsm8k/)"
fi

# ────────────────────────────────────────────────────────────────
# 5. LongBench  →  LongBench/data/*.jsonl
#    官方文档只提供 HuggingFace 下载，无 OSS 直链
# ────────────────────────────────────────────────────────────────
info "======== 下载 LongBench 数据集（HuggingFace）========"
if [[ -f "LongBench/data/hotpotqa.jsonl" ]]; then
    warn "LongBench/data/ 已存在，跳过"
else
    mkdir -p LongBench

    # 国内默认走 hf-mirror.com，可通过 --hf-endpoint 覆盖
    : "${HF_ENDPOINT:=https://hf-mirror.com}"
    export HF_ENDPOINT

    info "使用 huggingface_hub Python 下载（镜像: $HF_ENDPOINT）"
    python3 - <<PYEOF
from huggingface_hub import snapshot_download
token = "${HF_TOKEN}" if "${HF_TOKEN}" else None
snapshot_download(
    repo_id="zai-org/LongBench",
    repo_type="dataset",
    local_dir="${DATASETS_DIR}/LongBench",
    token=token,
    endpoint="${HF_ENDPOINT}",
    ignore_patterns=["*.git*"],
)
print("LongBench 下载完成")
PYEOF

    # snapshot_download 下载的是 data.zip，需要手动解压
    if [[ -f "LongBench/data.zip" ]]; then
        info "解压 LongBench/data.zip ..."
        unzip -q LongBench/data.zip -d LongBench/
        rm -f LongBench/data.zip
    fi

    FILE_COUNT=$(ls LongBench/data/*.jsonl 2>/dev/null | wc -l | xargs)
    if [[ "${FILE_COUNT}" -eq 0 ]]; then
        warn "LongBench data/*.jsonl 未找到，请检查网络或手动下载："
        warn "  ${HF_ENDPOINT}/datasets/zai-org/LongBench"
    else
        info "LongBench 完成: ${FILE_COUNT} 个 jsonl 文件"
    fi
fi

# ────────────────────────────────────────────────────────────────
# 6. ShareGPT  →  sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json
#    README 给的是 /blob/ 页面地址，实际下载须用 /resolve/ 直链
# ────────────────────────────────────────────────────────────────
info "======== 下载 ShareGPT 数据集 ========"
SHAREGPT_FILE="sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
if [[ -f "$SHAREGPT_FILE" ]]; then
    warn "${SHAREGPT_FILE} 已存在，跳过"
else
    mkdir -p sharegpt
    # /resolve/main/ 是 HuggingFace 文件直链，README 里的 /blob/ 是 HTML 页面
    SHAREGPT_URL="${HF_ENDPOINT}/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

    WGET_EXTRA_ARGS=()
    [[ -n "$HF_TOKEN" ]] && WGET_EXTRA_ARGS+=(--header="Authorization: Bearer ${HF_TOKEN}")

    info "下载: $SHAREGPT_URL"
    wget -q --show-progress "${WGET_EXTRA_ARGS[@]}" -O "$SHAREGPT_FILE" "$SHAREGPT_URL"
    info "ShareGPT 完成: $(du -sh "$SHAREGPT_FILE" | cut -f1)"
fi

# ────────────────────────────────────────────────────────────────
# 7. 验证汇总
# ────────────────────────────────────────────────────────────────
info "======== 数据集验证 ========"
check_file() {
    local name="$1" path="$2"
    if [[ -e "${DATASETS_DIR}/${path}" ]]; then
        echo -e "  ${GREEN}✓${NC} ${name}"
    else
        echo -e "  ${RED}✗${NC} ${name}  【未找到: ${path}】"
    fi
}

check_file "AIME2024"  "aime/aime.jsonl"
check_file "AIME2025"  "aime2025/aime2025.jsonl"
check_file "AIME2026"  "aime2026/aime2026.jsonl"
check_file "GPQA"      "gpqa/gpqa_diamond.csv"
check_file "C-Eval"    "ceval/formal_ceval/val/accountant_val.csv"
check_file "GSM8K"     "gsm8k/test.jsonl"
check_file "LongBench" "LongBench/data/hotpotqa.jsonl"
check_file "ShareGPT"  "sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "第一步：查找并修改配置文件"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  运行前必须修改模型配置文件，配置推理服务的 IP、端口等参数。"
echo "  使用 --search 可以打印出对应配置文件的绝对路径："
echo ""
echo -e "  ${GREEN}# 精度评测 - 查找配置文件路径${NC}"
echo "  cd $REPO_ROOT"
echo "  ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_0_shot_cot_chat_prompt --search"
echo ""
echo -e "  ${GREEN}# NPU/MindIE 服务 - 查找配置文件路径${NC}"
echo "  ais_bench --models mindie_stream_api_general --datasets gsm8k_gen_0_shot_cot_str_perf --mode perf --search"
echo ""
echo "  执行后会输出如下表格，找到 Config File Path 列的路径，用编辑器打开修改："
echo ""
echo "  ┌──────────────┬─────────────────────────────────┬──────────────────────────────────────────────────┐"
echo "  │ Task Type    │ Task Name                       │ Config File Path                                 │"
echo "  ├──────────────┼─────────────────────────────────┼──────────────────────────────────────────────────┤"
echo "  │ --models     │ vllm_api_general_chat           │ .../configs/models/vllm_api/vllm_api_general_chat.py │"
echo "  │ --datasets   │ gsm8k_gen_0_shot_cot_chat_prompt│ .../configs/datasets/gsm8k/gsm8k_gen_0_shot_...  │"
echo "  └──────────────┴─────────────────────────────────┴──────────────────────────────────────────────────┘"
echo ""
echo -e "  ${YELLOW}需要修改的关键参数（模型配置文件中）：${NC}"
echo "    host_ip    = \"127.0.0.1\"   # 推理服务 IP（不要用 0.0.0.0）"
echo "    host_port  = 8080           # 推理服务端口"
echo "    model      = \"\"             # 服务端加载的模型名（空字符串自动获取）"
echo "    max_out_len = 512           # 最大输出 token 数"
echo "    batch_size  = 1            # 并发请求数"
echo ""
echo -e "  ${YELLOW}性能测评额外注意：${NC}"
echo "    stream     = True           # 性能模式必须开启流式"
echo "    ignore_eos = True           # 固定输出长度压测时设为 True"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "第二步：执行评测"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "  ${GREEN}# 精度评测${NC}"
echo "  ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_0_shot_cot_chat_prompt"
echo "  ais_bench --models vllm_api_general_chat --datasets aime2024_gen_0_shot_chat_prompt"
echo "  ais_bench --models vllm_api_general_chat --datasets aime2025_gen_0_shot_chat_prompt"
echo "  ais_bench --models vllm_api_general_chat --datasets aime2026_gen_0_shot_chat_prompt"
echo "  ais_bench --models vllm_api_general_chat --datasets gpqa_gen_0_shot_cot_chat_prompt"
echo "  ais_bench --models vllm_api_general_chat --datasets ceval_gen_0_shot_cot_chat_prompt"
echo "  ais_bench --models vllm_api_general_chat --datasets longbench"
echo ""
echo -e "  ${GREEN}# 性能评测（NPU/MindIE）${NC}"
echo "  ais_bench --models mindie_stream_api_general --datasets gsm8k_gen_0_shot_cot_str_perf --mode perf"
echo ""
echo -e "  ${GREEN}# 调试模式（直接打印日志，不写文件）${NC}"
echo "  ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_0_shot_cot_chat_prompt --debug"
echo ""
echo "  结果文件位于: ${REPO_ROOT}/outputs/default/<timestamp>/summary/"
