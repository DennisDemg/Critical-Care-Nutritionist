# icu_nutrition.py
# v7 (final): Hybrid decision mode (online state) - NO stay_id
# - subtype computed ONLY ONCE via subgroup_tool (cached)
# - HybridPolicy uses payload["subtype"] instead of training/predicting subtype
# - Sidebar collects: day + state features + Day1/SG features (with units)
# - Chat can still accept key=value parsing to fill hybrid_ctx incrementally

import os
import re
import certifi
import requests
from typing import Dict, Any, Tuple

import streamlit as st

from guideline_tool import init_guideline_qa

# ========= Hybrid tool (online state) =========
try:
    from hybrid_policy import HybridPolicy
except Exception as e:
    HybridPolicy = None
    HYBRID_IMPORT_ERR = str(e)
else:
    HYBRID_IMPORT_ERR = None

# ========= Subgroup tool =========
try:
    from subgroup_tool.subgroup_tool import run as run_subgroup
except Exception:
    run_subgroup = None

# =========================
# Config
# =========================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-ab941f4690a24c03be80db5dcc15b0ed")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

KB_ROOT = "icu_nutrition_kb"
WORKFLOW_DIR = os.path.join(KB_ROOT, "workflow")
CONSULT_DIR = os.path.join(KB_ROOT, "consult")
ASPEN_MD_PATH = os.path.join(CONSULT_DIR, "paper_aspen.md")


# =========================
# Unified default helpers
# =========================
def clamp_default(vmin: float, vmax: float, default: float) -> float:
    """Ensure Streamlit number_input default value is within [vmin, vmax]."""
    try:
        vmin = float(vmin)
        vmax = float(vmax)
        default = float(default)
    except Exception:
        return 0.0
    if vmin > vmax:
        vmin, vmax = vmax, vmin
    return float(min(max(default, vmin), vmax))


def clamp_default_int(vmin: int, vmax: int, default: int) -> int:
    try:
        vmin = int(vmin)
        vmax = int(vmax)
        default = int(default)
    except Exception:
        return 0
    if vmin > vmax:
        vmin, vmax = vmax, vmin
    return int(min(max(default, vmin), vmax))


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def safe_int(x):
    try:
        return int(float(x))
    except Exception:
        return None


# =========================
# Variable specs (units + bounds + defaults)
# =========================
STATE_FEATURES = [
    # BIN
    "gender", "vent", "rt",
    # NORM
    "age", "charlson_comorbidity_index", "height",
    "hr", "meanbp", "rr", "temp",
    "sodium_chem", "chloride_chem", "calcium",
    "hemoglobin", "sofa", "albumin",
    # LOG
    "vaso_dose", "weight", "gcs", "wbc", "glucose_chem", "potassium_chem",
    "platelet", "ptt", "pt", "pao2_fio2", "pco2", "lactate",
    "spo2", "ast", "alt", "bilirubin", "output_total", "bun",
    "creatinine", "inr", "TCI", "TPI", "BCR",
]

# Day1/SG 的10项：我们统一用这套（小写）存入 ctx
DAY1_FEATURES = [
    "age",
    "charlson_comorbidity_index",
    "hemoglobin",
    "pao2_fio2",          # PF_ratio
    "sofa",
    "bun",
    "creatinine_day1",
    "albumin",
    "output_total",       # Urine_output
    "BCR",
]

STATE_SPECS: Dict[str, Dict[str, Any]] = {
    # BIN (0/1)
    "gender": {"label": "gender（性别）", "unit": "0=Female, 1=Male", "type": "int", "min": 0, "max": 1, "default": 1, "step": 1},
    "vent": {"label": "vent（机械通气）", "unit": "0/1", "type": "int", "min": 0, "max": 1, "default": 0, "step": 1},
    "rt": {"label": "rt（肾替代治疗）", "unit": "0/1", "type": "int", "min": 0, "max": 1, "default": 0, "step": 1},

    # NORM
    "age": {"label": "age", "unit": "years", "type": "float", "min": 0, "max": 120, "default": 60, "step": 1},
    "charlson_comorbidity_index": {"label": "charlson_comorbidity_index", "unit": "score", "type": "float", "min": 0, "max": 40, "default": 2, "step": 1},
    "height": {"label": "height", "unit": "cm", "type": "float", "min": 80, "max": 230, "default": 170, "step": 1},

    "hr": {"label": "hr", "unit": "bpm", "type": "float", "min": 0, "max": 250, "default": 90, "step": 1},
    "meanbp": {"label": "meanbp", "unit": "mmHg", "type": "float", "min": 0, "max": 200, "default": 75, "step": 1},
    "rr": {"label": "rr", "unit": "breaths/min", "type": "float", "min": 0, "max": 80, "default": 20, "step": 1},
    "temp": {"label": "temp", "unit": "°C", "type": "float", "min": 25, "max": 45, "default": 37.0, "step": 0.1, "format": "%.1f"},

    "sodium_chem": {"label": "sodium_chem", "unit": "mmol/L", "type": "float", "min": 80, "max": 200, "default": 140, "step": 1},
    "chloride_chem": {"label": "chloride_chem", "unit": "mmol/L", "type": "float", "min": 50, "max": 200, "default": 105, "step": 1},
    "calcium": {"label": "calcium", "unit": "mmol/L", "type": "float", "min": 0.5, "max": 5.0, "default": 2.1, "step": 0.1, "format": "%.2f"},

    "hemoglobin": {"label": "hemoglobin", "unit": "g/dL", "type": "float", "min": 0, "max": 25, "default": 9.5, "step": 0.1, "format": "%.2f"},
    "sofa": {"label": "sofa", "unit": "score", "type": "float", "min": 0, "max": 30, "default": 8, "step": 1},
    "albumin": {"label": "albumin", "unit": "g/dL", "type": "float", "min": 0, "max": 6, "default": 2.8, "step": 0.1, "format": "%.2f"},

    # LOG
    "vaso_dose": {"label": "vaso_dose（等效去甲）", "unit": "μg/kg/min", "type": "float", "min": 0, "max": 5, "default": 0.2, "step": 0.01, "format": "%.3f"},
    "weight": {"label": "weight", "unit": "kg", "type": "float", "min": 0, "max": 300, "default": 70, "step": 0.5, "format": "%.1f"},
    "gcs": {"label": "gcs", "unit": "score", "type": "float", "min": 3, "max": 15, "default": 14, "step": 1},
    "wbc": {"label": "wbc", "unit": "10^9/L", "type": "float", "min": 0, "max": 200, "default": 12, "step": 0.1, "format": "%.1f"},
    "glucose_chem": {"label": "glucose_chem", "unit": "mmol/L", "type": "float", "min": 0, "max": 60, "default": 8.5, "step": 0.1, "format": "%.1f"},
    "potassium_chem": {"label": "potassium_chem", "unit": "mmol/L", "type": "float", "min": 0, "max": 10, "default": 4.0, "step": 0.1, "format": "%.1f"},

    "platelet": {"label": "platelet", "unit": "10^9/L", "type": "float", "min": 0, "max": 2000, "default": 150, "step": 1},
    "ptt": {"label": "ptt", "unit": "s", "type": "float", "min": 0, "max": 300, "default": 35, "step": 0.1, "format": "%.1f"},
    "pt": {"label": "pt", "unit": "s", "type": "float", "min": 0, "max": 200, "default": 14, "step": 0.1, "format": "%.1f"},

    "pao2_fio2": {"label": "pao2_fio2（PF ratio）", "unit": "mmHg", "type": "float", "min": 0, "max": 800, "default": 260, "step": 1},
    "pco2": {"label": "pco2", "unit": "mmHg", "type": "float", "min": 0, "max": 200, "default": 42, "step": 1},
    "lactate": {"label": "lactate", "unit": "mmol/L", "type": "float", "min": 0, "max": 30, "default": 1.6, "step": 0.1, "format": "%.1f"},

    "spo2": {"label": "spo2", "unit": "% (0–100)", "type": "float", "min": 0, "max": 100, "default": 97, "step": 1},
    "ast": {"label": "ast", "unit": "U/L", "type": "float", "min": 0, "max": 5000, "default": 40, "step": 1},
    "alt": {"label": "alt", "unit": "U/L", "type": "float", "min": 0, "max": 5000, "default": 35, "step": 1},
    "bilirubin": {"label": "bilirubin", "unit": "mg/dL（按你的数据单位）", "type": "float", "min": 0, "max": 100, "default": 1.0, "step": 0.1, "format": "%.2f"},

    "output_total": {"label": "output_total（尿量）", "unit": "mL/24h", "type": "float", "min": 0, "max": 20000, "default": 1800, "step": 10},
    "bun": {"label": "bun", "unit": "mg/dL（按你的数据单位）", "type": "float", "min": 0, "max": 300, "default": 12, "step": 0.1, "format": "%.1f"},
    "creatinine": {"label": "creatinine（当日）", "unit": "mg/dL", "type": "float", "min": 0, "max": 30, "default": 1.2, "step": 0.1, "format": "%.2f"},
    "inr": {"label": "inr", "unit": "ratio", "type": "float", "min": 0, "max": 20, "default": 1.1, "step": 0.1, "format": "%.2f"},

    "TCI": {"label": "TCI（热卡达标率）", "unit": "% (0–100)", "type": "float", "min": 0, "max": 100, "default": 30, "step": 1},
    "TPI": {"label": "TPI（蛋白达标率）", "unit": "% (0–100)", "type": "float", "min": 0, "max": 100, "default": 25, "step": 1},
    "BCR": {"label": "BCR（BUN/Cr）", "unit": "ratio", "type": "float", "min": 0, "max": 200, "default": 10, "step": 0.1, "format": "%.2f"},
}

DAY1_SPECS: Dict[str, Dict[str, Any]] = {
    "age": STATE_SPECS["age"],
    "charlson_comorbidity_index": STATE_SPECS["charlson_comorbidity_index"],
    "hemoglobin": STATE_SPECS["hemoglobin"],
    "pao2_fio2": STATE_SPECS["pao2_fio2"],
    "sofa": STATE_SPECS["sofa"],
    "bun": STATE_SPECS["bun"],
    "creatinine_day1": {"label": "creatinine_day1（Day1）", "unit": "mg/dL", "type": "float", "min": 0, "max": 30, "default": 1.2, "step": 0.1, "format": "%.2f"},
    "albumin": STATE_SPECS["albumin"],
    "output_total": STATE_SPECS["output_total"],
    "BCR": STATE_SPECS["BCR"],
}


# =========================
# Rendering helper (unified)
# =========================
def render_number_inputs(
    specs: Dict[str, Dict[str, Any]],
    keys: list,
    key_prefix: str,
    n_cols: int = 2
) -> Dict[str, Any]:
    cols = st.columns(n_cols)
    out: Dict[str, Any] = {}
    for i, k in enumerate(keys):
        s = specs[k]
        label = f"{s.get('label', k)}  [{s.get('unit','')}]".strip()
        with cols[i % n_cols]:
            if s.get("type") == "int":
                vmin = int(s.get("min", 0))
                vmax = int(s.get("max", 1))
                default = clamp_default_int(vmin, vmax, int(s.get("default", vmin)))
                out[k] = st.number_input(
                    label,
                    min_value=min(vmin, vmax),
                    max_value=max(vmin, vmax),
                    value=default,
                    step=int(s.get("step", 1)),
                    key=f"{key_prefix}{k}",
                )
            else:
                vmin = float(s.get("min", 0.0))
                vmax = float(s.get("max", 1e9))
                default = clamp_default(vmin, vmax, float(s.get("default", 0.0)))
                step = float(s.get("step", 0.1))
                fmt = s.get("format", None)
                kwargs = {}
                if fmt:
                    kwargs["format"] = fmt
                out[k] = st.number_input(
                    label,
                    min_value=float(min(vmin, vmax)),
                    max_value=float(max(vmin, vmax)),
                    value=float(default),
                    step=step,
                    key=f"{key_prefix}{k}",
                    **kwargs,
                )
    return out


# =========================
# Guideline QA
# =========================
@st.cache_resource(show_spinner=False)
def get_aspen_qa():
    return init_guideline_qa(
        ASPEN_MD_PATH,
        llm_api_key=DEEPSEEK_API_KEY,
        llm_base_url="https://api.deepseek.com",
        llm_model="deepseek-chat",
        cache_dir=".guideline_cache"
    )


@st.cache_resource(show_spinner=False)
def get_hybrid_policy():
    if HybridPolicy is None:
        raise RuntimeError(f"HybridPolicy import failed: {HYBRID_IMPORT_ERR}")
    return HybridPolicy().load()


# =========================
# KB Loader
# =========================
def load_workflow_kb():
    if not os.path.isdir(WORKFLOW_DIR):
        raise FileNotFoundError(f"未找到 workflow 知识库文件夹：{WORKFLOW_DIR}")
    md_files = sorted([f for f in os.listdir(WORKFLOW_DIR) if f.lower().endswith(".md")])
    if not md_files:
        raise FileNotFoundError(f"{WORKFLOW_DIR} 中未找到任何 .md 文件")
    parts = []
    for fname in md_files:
        path = os.path.join(WORKFLOW_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        parts.append(f"\n\n===== {fname} =====\n\n{content}")
    return "\n".join(parts).strip()


@st.cache_data(show_spinner=False)
def get_workflow_kb_text():
    return load_workflow_kb()


# =========================
# DeepSeek API call
# =========================
def build_messages(system_prompt, history, user_prompt):
    messages = [{"role": "system", "content": system_prompt}]
    for role, content in history:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def call_deepseek(messages):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"model": "deepseek-chat", "messages": messages}

    session = requests.Session()
    session.trust_env = False

    r = session.post(
        DEEPSEEK_API_URL,
        headers=headers,
        json=payload,
        timeout=60,
        verify=certifi.where()
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# =========================
# Prompts
# =========================
SYSTEM_PROMPT_WORKFLOW = """
你是一名重症营养治疗师。
你的回答必须且只能基于【提供的知识库内容】。
【输出】临床查房式语言，三段式：判断/建议/复评。
""".strip()

SYSTEM_PROMPT_CONSULT = """
你是一名 ICU 营养治疗专家，用于回答【知识咨询型问题】。
如知识库未覆盖，允许“（通用知识补充）”标注后补充。
""".strip()

CONTROL_PROMPT_WORKFLOW = """
给建议前必须确认当前时间节点（至少：ICU第几天/喂养第几天 或 启动EN多久）。
""".strip()


# =========================
# Hybrid helpers
# =========================
SUBTYPE_MEANING_CN = {
    0: "A 亚型",
    1: "B 亚型",
    2: "C 亚型",
}


def parse_kv_from_text(text: str) -> Dict[str, Any]:
    """
    Parse loose key=value from chat to fill hybrid_ctx.
    Supports:
      day=2 vaso_dose=0.2 lactate=1.6 creatinine=1.2
      age=60 cci=2 ... etc.
    """
    out: Dict[str, Any] = {}
    t = text.strip()

    # day patterns like "ICU day 2"
    m = re.search(r'\b(?:ICU\s*day|day)\s*[:=]?\s*(\d+)\b', t, re.IGNORECASE)
    if m:
        out["day"] = safe_int(m.group(1))

    allow_keys = set(STATE_FEATURES + DAY1_FEATURES + ["day"])
    alias = {
        "cci": "charlson_comorbidity_index",
        "hb": "hemoglobin",
        "pf": "pao2_fio2",
        "pf_ratio": "pao2_fio2",
        "alb": "albumin",
        "urine_output": "output_total",
        "creatinine_d1": "creatinine_day1",
        "cre_day1": "creatinine_day1",
    }

    for mm in re.finditer(r'\b([A-Za-z_][A-Za-z0-9_]*)\b\s*[:=]\s*([-+]?\d*\.?\d+)', t):
        k = alias.get(mm.group(1), mm.group(1))
        v = mm.group(2)
        if k in allow_keys:
            if k in {"gender", "vent", "rt"}:
                out[k] = safe_int(v)
            else:
                out[k] = safe_float(v)

    return out


def merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if v is None:
            continue
        dst[k] = v
    return dst


def compute_subtype_once_from_day1(day1_ctx: Dict[str, Any]) -> int:
    """
    Use subgroup_tool (already trained) to compute subtype label: 0/1/2.
    Cache result in st.session_state.cached_subtype.
    """
    if "cached_subtype" in st.session_state and st.session_state.cached_subtype is not None:
        return int(st.session_state.cached_subtype)

    if run_subgroup is None:
        raise RuntimeError("subgroup_tool 未加载（run_subgroup=None），无法计算 subtype。")

    # subgroup_tool expects the 10 features in "SUBGROUP_FEATURES" schema (Title case)
    payload = {
        "Age": float(day1_ctx["age"]),
        "CCI": float(day1_ctx["charlson_comorbidity_index"]),
        "Hemoglobin": float(day1_ctx["hemoglobin"]),
        "Albumin": float(day1_ctx["albumin"]),
        "Creatinine": float(day1_ctx["creatinine_day1"]),
        "BUN": float(day1_ctx["bun"]),
        "BCR": float(day1_ctx["BCR"]),
        "PF_ratio": float(day1_ctx["pao2_fio2"]),
        "Urine_output": float(day1_ctx["output_total"]),
        "SOFA": float(day1_ctx["sofa"]),
    }

    out = run_subgroup(payload)
    if not out or not out.get("ok"):
        raise RuntimeError(f"subgroup_tool 运行失败：{out}")

    pred_label = int(out.get("pred_label"))
    if pred_label not in (0, 1, 2):
        raise RuntimeError(f"subgroup_tool 返回 pred_label 非法：{pred_label}")

    st.session_state.cached_subtype = pred_label
    return pred_label


def build_hybrid_payload_from_ctx(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    HybridPolicy.recommend() expects (B方案):
      - day
      - subtype (0/1/2)  [from subgroup_tool]
      - current-day STATE_FEATURES (can be partial; missing imputed by model)
    """
    # auto BCR for day1 if needed
    if ctx.get("BCR") is None and ctx.get("bun") is not None and ctx.get("creatinine_day1") is not None:
        try:
            bun = float(ctx["bun"])
            cr1 = float(ctx["creatinine_day1"])
            if cr1 > 0:
                ctx["BCR"] = bun / cr1
        except Exception:
            pass

    # compute subtype once (cached)
    day1_ctx = {k: ctx.get(k, None) for k in DAY1_FEATURES}
    miss_day1 = [k for k, v in day1_ctx.items() if v is None]
    if miss_day1:
        raise KeyError(f"缺少 Day1/SG 计算 subtype 的字段：{miss_day1}")

    subtype = compute_subtype_once_from_day1(day1_ctx)

    payload: Dict[str, Any] = {"day": int(ctx["day"]), "subtype": int(subtype)}

    for k in STATE_FEATURES:
        if k in ctx and ctx[k] is not None:
            payload[k] = float(ctx[k])

    return payload


def compute_hybrid_summary(policy: Any, ctx: Dict[str, Any], debug: bool = False) -> Tuple[str, Dict[str, Any]]:
    payload = build_hybrid_payload_from_ctx(ctx)
    out = policy.recommend(payload)

    rec_kcal_bin = int(out.get("rec_kcal_bin"))
    rec_pro_bin = int(out.get("rec_pro_bin"))
    rec_tci = float(out.get("rec_TCI_pct"))
    rec_tpi = float(out.get("rec_TPI_pct"))
    kcal_range = out.get("rec_TCI_bin_range", "")
    pro_range = out.get("rec_TPI_bin_range", "")

    subtype = int(out.get("subtype", payload["subtype"]))
    state = out.get("state", None)
    top5 = out.get("state_prob_top5", [])

    top5_str = ""
    if isinstance(top5, list) and len(top5) > 0:
        try:
            top5_str = ", ".join([f"s{d['state']}={float(d['prob']):.2f}" for d in top5])
        except Exception:
            top5_str = ""

    kcal_rule = int(out.get("kcal_rule_trigger", 0))
    pro_rule = int(out.get("pro_rule_trigger", 0))

    text = f"""当前基于血流动力学与代谢状态综合评估，建议采用分层推进的营养达标策略。

处理建议：
- Hybrid 推荐：热卡档位 **kcal_bin={rec_kcal_bin}**（{kcal_range}），蛋白档位 **pro_bin={rec_pro_bin}**（{pro_range}）。
- 对应推荐目标：TCI≈**{rec_tci:.1f}%**，TPI≈**{rec_tpi:.1f}%**。

随访与复评：
- 24 小时内复评：血流动力学（vaso_dose）、乳酸趋势、肾功能与耐受情况，并据此更新次日推荐。

模型输出摘要：
- day={int(ctx["day"])} | subtype={subtype}（{SUBTYPE_MEANING_CN.get(subtype,"NA")}） | state={state}""" + (f" | top5：{top5_str}" if top5_str else "") + f"""
- 规则触发：热卡={'是' if kcal_rule else '否'}（vaso_dose<0.3 且 lactate<2），蛋白={'是' if pro_rule else '否'}（+ creatinine<5）
""".strip()

    detail = dict(out)
    if debug:
        detail["ctx_used"] = ctx
        detail["payload_used"] = payload
        detail["cached_subtype"] = st.session_state.get("cached_subtype", None)
    return text, detail


# =========================
# Streamlit UI
# =========================
st.title("重症营养治疗师")

def reset_round():
    st.session_state.history = []
    st.session_state.patient_context = ""
    st.session_state.hybrid_ctx = {}
    st.session_state.hybrid_last = None
    st.session_state.cached_subtype = None
    st.session_state.history.append(("assistant", "已开始新患者查房。请先在左侧填写时间轴/喂养状态，或直接描述患者当前情况。"))
    st.toast("已清空本次查房，开始新患者。", icon="✅")


# init session state
if "history" not in st.session_state:
    st.session_state.history = []
if "patient_context" not in st.session_state:
    st.session_state.patient_context = ""
if "hybrid_ctx" not in st.session_state:
    st.session_state.hybrid_ctx = {}
if "hybrid_last" not in st.session_state:
    st.session_state.hybrid_last = None
if "cached_subtype" not in st.session_state:
    st.session_state.cached_subtype = None

# load workflow KB
try:
    kb_text = get_workflow_kb_text()
except Exception as e:
    st.error(f"Workflow 知识库加载失败：{e}")
    st.stop()

user_input = st.chat_input("请输入患者情况或问题")
mode = st.radio("工作模式", ["治疗建议", "专家咨询", "混合决策（达标率）"], index=0)
debug_mode = st.toggle("开发者模式", value=False)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Control Panel")

    if st.button("清空本次查房 / 新患者", use_container_width=True):
        reset_round()
        st.rerun()

    st.session_state.patient_context = st.text_area(
        "更新当前时间点与喂养状态（可选，但强烈建议）",
        value=st.session_state.patient_context,
        height=160,
        key="patient_context_textarea",
        placeholder=(
            "示例：\n"
            "ICU Day 2；EN started 10h ago\n"
            "Current rate 25 ml/h; last 24h EN 400 ml\n"
            "Last FIS 2 at 2h ago; no vomiting; mild distension\n"
        ),
    )

    st.divider()

    # ---- Hybrid inputs
    with st.expander("Hybrid 输入（day + state + Day1/SG）", expanded=True):
        day_val = st.number_input("day（ICU第几天）", min_value=1, max_value=30, value=1, step=1, key="hy_day")

        st.caption("State（用于在线预测 state；请尽量完整填写）")
        state_vals = render_number_inputs(STATE_SPECS, STATE_FEATURES, key_prefix="st_", n_cols=2)

        st.divider()
        st.caption("Day1/SG（用于 subtype；只需提供一次）")
        day1_vals = render_number_inputs(DAY1_SPECS, DAY1_FEATURES, key_prefix="d1_", n_cols=2)

        # auto-compute Day1 BCR if 0 but bun & creatinine_day1 > 0
        try:
            if float(day1_vals.get("BCR", 0.0)) == 0.0:
                bun = float(day1_vals.get("bun", 0.0))
                cr1 = float(day1_vals.get("creatinine_day1", 0.0))
                if bun > 0 and cr1 > 0:
                    day1_vals["BCR"] = bun / cr1
        except Exception:
            pass

        # show cached subtype
        cached = st.session_state.get("cached_subtype", None)
        if cached is None:
            st.info("subtype：未计算（将首次 Hybrid 运行时自动计算并缓存）。")
        else:
            st.success(f"subtype：已缓存 = {int(cached)}（{SUBTYPE_MEANING_CN.get(int(cached),'NA')}）")

        colA, colB = st.columns(2)
        with colA:
            if st.button("写入 混合决策 上下文", use_container_width=True):
                ctx_add = {"day": int(day_val)}
                for k in STATE_FEATURES:
                    ctx_add[k] = state_vals.get(k, None)
                for k in DAY1_FEATURES:
                    ctx_add[k] = day1_vals.get(k, None)
                st.session_state.hybrid_ctx = merge_dict(st.session_state.hybrid_ctx, ctx_add)
                st.toast("已写入 混合决策 上下文。", icon="✅")

        with colB:
            if st.button("计算 subtype（缓存一次）", use_container_width=True):
                if run_subgroup is None:
                    st.error("subgroup_tool 未加载（run_subgroup=None）。")
                else:
                    try:
                        # build day1 ctx then cache subtype
                        day1_ctx = {k: day1_vals.get(k, None) for k in DAY1_FEATURES}
                        # ensure BCR
                        if (day1_ctx.get("BCR") is None or float(day1_ctx.get("BCR", 0.0)) == 0.0) and day1_ctx.get("bun") is not None and day1_ctx.get("creatinine_day1") is not None:
                            bun = float(day1_ctx["bun"])
                            cr1 = float(day1_ctx["creatinine_day1"])
                            if cr1 > 0:
                                day1_ctx["BCR"] = bun / cr1
                        subtype = compute_subtype_once_from_day1(day1_ctx)
                        st.success(f"已缓存 subtype={subtype}（{SUBTYPE_MEANING_CN.get(subtype,'NA')}）")
                    except Exception as e:
                        st.error(f"计算 subtype 失败：{e}")

        st.divider()

        if st.button("立即运行 混合决策（侧栏）", use_container_width=True):
            if HybridPolicy is None:
                st.error(f"混合决策 模块未加载：{HYBRID_IMPORT_ERR}")
            else:
                try:
                    pol = get_hybrid_policy()

                    ctx = {"day": int(day_val)}
                    for k in STATE_FEATURES:
                        ctx[k] = state_vals.get(k, None)
                    for k in DAY1_FEATURES:
                        ctx[k] = day1_vals.get(k, None)

                    answer, detail = compute_hybrid_summary(pol, ctx, debug=debug_mode)
                    st.success("混合决策 已完成。结果已写入聊天窗口。")
                    st.session_state.history.append(("assistant", answer))
                    st.session_state.hybrid_last = detail
                    if debug_mode:
                        st.json(detail)
                except Exception as e:
                    st.error(f"混合决策 运行失败：{e}")




# =========================
# Chat handling
# =========================
if user_input:
    st.session_state.history.append(("user", user_input))

    # Expert consult
    if mode == "专家咨询":
        try:
            aspen_qa = get_aspen_qa()
            result = aspen_qa.ask(user_input, top_k=6)
            raw_answer = result["answer"]
            answer_for_user = raw_answer if debug_mode else re.sub(r"\[\d+\]", "", raw_answer).strip()
            st.session_state.history.append(("assistant", answer_for_user))
            if debug_mode:
                with st.expander("Citations（调试可见）", expanded=False):
                    st.json(result.get("citations", []))
        except requests.exceptions.SSLError as e:
            st.error(f"SSL 错误：{e}")
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP 错误：{e}\n\n返回内容：{getattr(e.response, 'text', '')}")
            st.stop()
        except Exception as e:
            st.error(f"调用 ASPEN 指南问答失败：{e}")
            st.stop()

    # Hybrid decision
    elif mode == "混合决策（达标率）":
        if HybridPolicy is None:
            st.session_state.history.append(("assistant", f"混合决策 模块未加载成功：{HYBRID_IMPORT_ERR}"))
        else:
            parsed = parse_kv_from_text(user_input)
            st.session_state.hybrid_ctx = merge_dict(st.session_state.hybrid_ctx, parsed)
            ctx = st.session_state.hybrid_ctx

            # minimal requirement: day + DAY1 (for subtype); state can be partial
            need_keys = ["day"] + DAY1_FEATURES
            missing = [k for k in need_keys if ctx.get(k, None) is None]
            if missing:
                st.session_state.history.append(
                    ("assistant",
                     "Hybrid 计算缺少必要字段：\n- " + "\n- ".join(missing) +
                     "\n\n请在侧栏“混合决策 输入”补全后再运行，或在聊天中用 key=value 补全。")
                )
            else:
                try:
                    pol = get_hybrid_policy()
                    answer, detail = compute_hybrid_summary(pol, ctx, debug=debug_mode)
                    st.session_state.history.append(("assistant", answer))
                    st.session_state.hybrid_last = detail
                    if debug_mode:
                        with st.expander("混合决策（调试）", expanded=False):
                            st.json(detail)
                except Exception as e:
                    st.session_state.history.append(("assistant", f"混合决策 计算失败：{e}"))

    # Workflow decision (KB)
    else:
        user_prompt = f"""
{CONTROL_PROMPT_WORKFLOW}

【患者时间轴与喂养状态】
{st.session_state.get("patient_context", "")}

【知识库内容】
{kb_text}

【用户输入】
{user_input}
""".strip()

        try:
            messages = build_messages(
                SYSTEM_PROMPT_WORKFLOW,
                st.session_state.history[:-1],
                user_prompt
            )
            answer = call_deepseek(messages)
        except requests.exceptions.SSLError as e:
            st.error(f"SSL 错误：{e}")
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP 错误：{e}\n\n返回内容：{getattr(e.response, 'text', '')}")
            st.stop()
        except Exception as e:
            st.error(f"调用 DeepSeek 失败：{e}")
            st.stop()

        reminder = "提示：如需使用 混合决策（达标率）推荐，请切换到顶部“混合决策（达标率）”模式。"
        st.session_state.history.append(("assistant", f"{answer}\n\n{reminder}"))


# =========================
# Render chat
# =========================
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.write(msg)
