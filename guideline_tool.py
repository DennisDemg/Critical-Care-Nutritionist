import os
import re
import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import os
import re
import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import requests
import certifi

from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# Helpers
# =========================
def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _split_markdown_by_heading(text: str) -> List[Tuple[str, str]]:
    """按 #/##/### 标题切分为 (title, content) blocks。"""
    blocks: List[Tuple[str, str]] = []
    current_title = "ROOT"
    buf: List[str] = []

    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line.strip())
        if m:
            if buf:
                blocks.append((current_title, "\n".join(buf).strip()))
                buf = []
            current_title = m.group(2).strip()
        else:
            buf.append(line)

    if buf:
        blocks.append((current_title, "\n".join(buf).strip()))

    return [(t, c) for (t, c) in blocks if c]


def _chunk_sliding_window(content: str, chunk_size_chars: int, overlap_chars: int) -> List[str]:
    out = []
    start = 0
    L = len(content)
    while start < L:
        end = min(L, start + chunk_size_chars)
        piece = content[start:end].strip()
        if piece:
            out.append(piece)
        if end == L:
            break
        start = max(0, end - overlap_chars)
    return out


def _file_fingerprint(path: str) -> str:
    st = os.stat(path)
    s = f"{os.path.abspath(path)}|{st.st_mtime_ns}|{st.st_size}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# =========================
# Core classes
# =========================
@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    title: str
    content: str


class GuidelineQA:
    """
    TF-IDF 检索版指南问答工具（Windows 稳定）：
    - ask(query): 基于检索片段回答，并返回 citations（调试可见）
    """
    def __init__(
        self,
        chunks: List[Chunk],
        vectorizer: TfidfVectorizer,
        tfidf_matrix,  # sparse matrix
        llm_api_key: str,
        llm_base_url: str,
        llm_model: str,
        timeout: int = 60
    ):
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix

        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url.rstrip("/")
        self.llm_model = llm_model
        self.timeout = timeout

        self.session = requests.Session()
        self.session.trust_env = False

    def _retrieve(self, query: str, top_k: int = 6) -> List[Tuple[float, Chunk]]:
        if top_k <= 0:
            top_k = 1

        q_vec = self.vectorizer.transform([query])  # (1, V)
        # cosine on L2-normalized TF-IDF: dot product is cosine
        sims = (self.tfidf_matrix @ q_vec.T).toarray().ravel()  # (N,)

        if sims.size == 0:
            return []

        top_k = min(top_k, sims.shape[0])
        top_idx = np.argsort(sims)[::-1][:top_k]

        res: List[Tuple[float, Chunk]] = []
        for i in top_idx:
            res.append((float(sims[i]), self.chunks[int(i)]))
        return res

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.llm_base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2
        }

        r = self.session.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=self.timeout,
            verify=certifi.where()
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def ask(self, query: str, top_k: int = 6) -> Dict:
        hits = self._retrieve(query, top_k=top_k)
        if not hits:
            return {
                "answer": "未检索到与问题相关的指南片段，无法基于当前指南内容回答。",
                "citations": []
            }

        ctx_parts = []
        citations = []
        for rank, (score, ch) in enumerate(hits, start=1):
            ctx_parts.append(
                f"[{rank}] score={score:.3f}\n"
                f"SOURCE={os.path.basename(ch.source_file)}\n"
                f"TITLE={ch.title}\n"
                f"CHUNK_ID={ch.chunk_id}\n"
                f"CONTENT:\n{ch.content}\n"
            )
            citations.append({
                "rank": rank,
                "score": round(score, 3),
                "source_file": os.path.basename(ch.source_file),
                "title": ch.title,
                "chunk_id": ch.chunk_id,
                "excerpt": (ch.content[:300] + "…") if len(ch.content) > 300 else ch.content
            })

        ctx = "\n\n".join(ctx_parts).strip()

        system_prompt = (
            "你是一名 ICU 营养治疗专家，用于回答‘咨询型问题’（非流程决策）。\n"
            "\n"
            "【总原则】\n"
            "1) 优先依据【指南片段】作答：若片段覆盖问题的关键点（指征、禁忌、要点、监测、并发症等），应明确引用并以其为主。\n"
            "2) 若【指南片段】未覆盖用户问题的某部分，允许使用通用医学/重症营养常识进行补充，使回答可落地；但必须明确标注为“通用知识补充（非本指南片段原文）”。\n"
            "3) 不要求‘片段不足就拒答’，而是：先给出可操作的通用建议，再说明哪些细节需要参考本指南/本院规范或进一步信息。\n"
            "\n"
            "【输出要求】\n"
            "- 使用临床查房式语言，简明、可执行。\n"
            "- 优先用 3 段式：\n"
            "  (1) 结论/判断（1句）\n"
            "  (2) 建议要点（2-5条要点，条目化）\n"
            "  (3) 复评/监测（1-2条）\n"
            "- 避免输出过多背景教学。\n"
            "\n"
            "【安全与边界】\n"
            "- 不输出个体化处方剂量（例如精确 kcal/kg、g/kg、具体药物剂量）除非【指南片段】明确给出且与问题直接相关。\n"
            "- 遇到高风险/需要本地流程确认的内容（如静脉营养适应证细节、感染控制、管路管理等），要提示需结合医院规范。\n"
            "\n"
            "【引用规则】\n"
            "- 如果使用了【指南片段】的信息，在答案末尾用[1][2]标注对应片段编号。\n"
            "- 如果某条建议完全来自通用知识补充，则在该条后标注“（通用知识补充）”，且末尾不强行引用片段编号。\n"
        )

        user_prompt = (
            f"【用户问题】\n{query}\n\n"
            f"【指南片段】\n{ctx}\n\n"
            "请用中文作答，并严格遵守约束。"
        )

        answer = self._call_llm(system_prompt, user_prompt)
        return {"answer": answer, "citations": citations}


def init_guideline_qa(
    md_path: str,
    *,
    chunk_size_chars: int = 1200,
    overlap_chars: int = 150,
    llm_api_key: Optional[str] = None,
    llm_base_url: str = "https://api.deepseek.com",
    llm_model: str = "deepseek-chat",
    cache_dir: str = ".guideline_cache"
) -> GuidelineQA:
    if not os.path.isfile(md_path):
        raise FileNotFoundError(f"找不到指南 md 文件：{md_path}")

    llm_api_key = llm_api_key or os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not llm_api_key:
        raise ValueError("缺少 API Key：请设置环境变量 DEEPSEEK_API_KEY/LLM_API_KEY，或直接传入 llm_api_key。")

    os.makedirs(cache_dir, exist_ok=True)

    # 1) chunk
    text = _read_text(md_path)
    blocks = _split_markdown_by_heading(text)

    chunks: List[Chunk] = []
    cid = 0
    for title, content in blocks:
        pieces = _chunk_sliding_window(content, chunk_size_chars, overlap_chars)
        for p in pieces:
            chunks.append(Chunk(
                chunk_id=f"chunk_{cid}",
                source_file=md_path,
                title=title,
                content=p
            ))
            cid += 1

    if not chunks:
        raise ValueError("未生成任何 chunk，请检查 md 内容是否为空或切分参数是否合适。")

    # 2) TF-IDF（可缓存）
    fp = _file_fingerprint(md_path)
    cache_npz = os.path.join(cache_dir, f"tfidf_{os.path.basename(md_path)}_{fp}.npz")
    cache_meta = os.path.join(cache_dir, f"tfidf_{os.path.basename(md_path)}_{fp}.json")

    contents = [c.content for c in chunks]

    if os.path.isfile(cache_npz) and os.path.isfile(cache_meta):
        # 直接重建 vectorizer + sparse matrix
        with open(cache_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)

        vectorizer = TfidfVectorizer(
            lowercase=False,
            token_pattern=meta["token_pattern"],
            ngram_range=tuple(meta["ngram_range"]),
            max_features=meta["max_features"]
        )
        # 需要 fit 以恢复词表（用同样的文本 fit）
        vectorizer.fit(contents)

        # 载入稀疏矩阵（CSR）
        loader = np.load(cache_npz, allow_pickle=True)
        data = loader["data"]
        indices = loader["indices"]
        indptr = loader["indptr"]
        shape = tuple(loader["shape"])
        from scipy.sparse import csr_matrix
        tfidf_matrix = csr_matrix((data, indices, indptr), shape=shape)

    else:
        # 对中文/英文混合，token_pattern 用默认的会对中文不友好
        # 这里用一个更宽松的规则：中文单字/英文词/数字都保留
        token_pattern = r"(?u)\b\w+\b|[\u4e00-\u9fff]"
        vectorizer = TfidfVectorizer(
            lowercase=False,
            token_pattern=token_pattern,
            ngram_range=(1, 2),
            max_features=80000
        )
        tfidf_matrix = vectorizer.fit_transform(contents)  # L2 norm by default

        # 缓存 sparse matrix
        from scipy.sparse import csr_matrix
        tfidf_matrix = csr_matrix(tfidf_matrix)

        np.savez_compressed(
            cache_npz,
            data=tfidf_matrix.data,
            indices=tfidf_matrix.indices,
            indptr=tfidf_matrix.indptr,
            shape=np.array(tfidf_matrix.shape)
        )
        with open(cache_meta, "w", encoding="utf-8") as f:
            json.dump({
                "token_pattern": token_pattern,
                "ngram_range": [1, 2],
                "max_features": 80000
            }, f, ensure_ascii=False, indent=2)

    return GuidelineQA(
        chunks=chunks,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model
    )
