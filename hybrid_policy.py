# hybrid_policy.py
# Online Hybrid Recommendation Module (Scheme B + subtype from subgroup_tool)
# - NO stay_id needed
# - State is computed ONLINE by: Step1-3 scaler (z-score space) -> optional Step4 PCA -> nearest centroid => state id
# - Action is chosen by: Step5 Q + soft bonuses (threshold rules / subtype day4-7 / day-progress targets day1-3)
# - subtype 不再在这里训练/预测：由外部 subgroup_tool 计算后传入 payload["subtype"] (0/1/2)
#
# Required files:
#   - step5_mdp_daily_policy.npz
#   - step1_3_daily_prepared.npz
#   - step4_states_daily.npz

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from sklearn.neighbors import NearestNeighbors


# =========================
# Path helpers
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent

def resolve_path(p: str) -> Path:
    p = Path(p)
    if p.is_absolute() and p.exists():
        return p
    cand = SCRIPT_DIR / p
    if cand.exists():
        return cand
    cand2 = Path.cwd() / p
    if cand2.exists():
        return cand2
    matches = list(SCRIPT_DIR.rglob(p.name))
    if matches:
        for m in matches:
            if m.name == p.name:
                return m
        return matches[0]
    raise FileNotFoundError(
        f"[ERROR] File not found: {p} (searched in script_dir, cwd, and script_dir recursive)"
    )

def to_numeric(df: pd.DataFrame, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def normalize_q(q_row: np.ndarray) -> np.ndarray:
    q = q_row.astype(float).copy()
    qmin, qmax = float(np.min(q)), float(np.max(q))
    if qmax > qmin:
        return (q - qmin) / (qmax - qmin)
    return np.zeros_like(q)


# =========================
# Default configs
# =========================
DEFAULT_ART_STEP5   = "step5_mdp_daily_policy.npz"
DEFAULT_ART_STEP1_3 = "step1_3_daily_prepared.npz"
DEFAULT_ART_STEP4   = "step4_states_daily.npz"


# =========================
# Soft bonus configs
# =========================
DAY_COL  = "day"
VASO_COL = "vaso_dose"
LAC_COL  = "lactate"
CRE_COL  = "creatinine"

# ---- day-progress target bins (soft)
DAY_TARGET_KCAL = {1: 2, 2: 3, 3: 3}  # day>=4 -> 4
DAY_TARGET_PRO  = {1: 2, 2: 3, 3: 3}  # day>=4 -> 4
TARGET_DEFAULT  = 4
W_DAY = 0.12

# ---- threshold rule target bins (soft)
TARGET_KCAL_BIN = 4
TARGET_PRO_BIN  = 4
W_RULE = 0.15

# ---- subtype postprocess (soft)
SUBTYPE_DAY_FROM = 4
SUBTYPE_DAY_TO   = 7
W_SUB = 0.10
W_SUB2_PENALTY_BIN5 = 0.25
W_SUB2_BONUS_BIN4    = 0.08
W_SUB2_BONUS_BIN3    = 0.04
W_SUB2_PENALTY_BIN1  = 0.02


# =========================
# HybridPolicy (Scheme B)
# =========================
class HybridPolicy:
    """
    Scheme B:
      - state: centroids nearest neighbor, computed online from current-day features (no stay_id)
      - subtype: MUST be provided by caller (e.g., subgroup_tool), as payload["subtype"] in {0,1,2}
                if not provided, default subtype=0

    Usage:
        hp = HybridPolicy().load()
        out = hp.recommend(payload)

    Required payload:
        - day (int)

    Strongly recommended payload (for accurate state):
        - all Step1-3 feature columns: bin_cols + norm_cols + log_cols
          (can be partial; missing -> treated as NaN then -> 0 in z-space)

    Optional payload:
        - subtype: int 0/1/2  (from subgroup_tool)
    """

    def __init__(
        self,
        step5_path: str = DEFAULT_ART_STEP5,
        step1_3_path: str = DEFAULT_ART_STEP1_3,
        step4_path: str = DEFAULT_ART_STEP4,
    ):
        self.step5_path = step5_path
        self.step1_3_path = step1_3_path
        self.step4_path = step4_path

        # Step5 RL artifacts
        self.Q = None
        self.unique_pairs = None
        self.unique_dose = None
        self.optimal_policy = None

        # Step1-3 scaler params
        self.bin_cols = None
        self.norm_cols = None
        self.log_cols = None
        self.all_cols = None
        self.col_index = None

        self.cmu = None
        self.csigma = None
        self.dmu = None
        self.dsigma = None
        self.weighted_cols = {}

        # Step4 centroids + PCA params
        self.centroids = None
        self.nn = None  # NearestNeighbors
        self.use_pca = False
        self.pca_components = None
        self.pca_mean = None

    # -------------------------
    # loading
    # -------------------------
    def load(self):
        self._load_step5_artifacts()
        self._load_step1_3_scaler()
        self._load_step4_centroids()
        return self

    def _load_step5_artifacts(self):
        p = resolve_path(self.step5_path)
        d5 = np.load(p, allow_pickle=True)
        self.Q = d5["Q"].astype(float)
        self.unique_pairs = d5["unique_pairs"].astype(int)
        self.unique_dose = d5["unique_dose"].astype(float)
        self.optimal_policy = d5["optimal_policy"].astype(int)

        if self.Q.ndim != 2:
            raise ValueError("Q must be 2D")
        if self.unique_pairs.shape[0] != self.Q.shape[1]:
            raise ValueError("unique_pairs rows must match n_actions")
        print(f"[OK] Loaded Step5 artifacts: Q={self.Q.shape}, actions={self.Q.shape[1]}")

    def _load_step1_3_scaler(self):
        p = resolve_path(self.step1_3_path)
        d = np.load(p, allow_pickle=True)

        self.bin_cols  = d["bin_cols"].astype(object).tolist()
        self.norm_cols = d["norm_cols"].astype(object).tolist()
        self.log_cols  = d["log_cols"].astype(object).tolist()

        self.cmu = d["cmu"].astype(float)
        self.csigma = d["csigma"].astype(float)
        self.dmu = d["dmu"].astype(float)
        self.dsigma = d["dsigma"].astype(float)

        keys = d.get("weighted_cols_keys", np.array([], dtype=object)).astype(object).tolist()
        vals = d.get("weighted_cols_vals", np.array([], dtype=float)).astype(float).tolist()
        self.weighted_cols = {k: float(v) for k, v in zip(keys, vals)}

        self.all_cols = self.bin_cols + self.norm_cols + self.log_cols
        self.col_index = {c: i for i, c in enumerate(self.all_cols)}

        print(f"[OK] Loaded Step1-3 scaler: bin={len(self.bin_cols)}, norm={len(self.norm_cols)}, log={len(self.log_cols)}")

    def _load_step4_centroids(self):
        p = resolve_path(self.step4_path)
        d = np.load(p, allow_pickle=True)

        self.centroids = d["centroids"].astype(float)

        self.use_pca = bool(d["use_pca"][0]) if "use_pca" in d.files else False
        self.pca_components = d.get("pca_components", np.array([], dtype=float)).astype(float)
        self.pca_mean = d.get("pca_mean", np.array([], dtype=float)).astype(float)

        if self.use_pca:
            if self.pca_components.size == 0 or self.pca_mean.size == 0:
                raise ValueError("[ERROR] step4 indicates use_pca=True but pca params are missing in npz.")

        self.nn = NearestNeighbors(n_neighbors=1).fit(self.centroids)
        print(f"[OK] Loaded Step4: centroids={self.centroids.shape}, use_pca={self.use_pca}")

    # -------------------------
    # state: centroid online
    # -------------------------
    def _zscore_transform_one(self, payload: Dict[str, Any]) -> np.ndarray:
        """
        Reproduce Step1-3 ZScalerLikeMatlab.transform for a single row.
        Missing values -> NaN -> later set to 0 in z-space.
        """
        x = {c: payload.get(c, np.nan) for c in self.all_cols}
        X = pd.DataFrame([x])
        X = to_numeric(X, self.all_cols)

        # BIN: x - 0.5
        Xb = X[self.bin_cols].to_numpy(float) - 0.5

        # NORM: (x - cmu) / csigma
        Xn_raw = X[self.norm_cols].to_numpy(float)
        Xn = (Xn_raw - self.cmu) / self.csigma

        # LOG: (log(0.1+x) - dmu) / dsigma
        Xl_raw = X[self.log_cols].to_numpy(float)
        Xl = (np.log(0.1 + Xl_raw) - self.dmu) / self.dsigma

        Z = np.concatenate([Xb, Xn, Xl], axis=1)
        Z = np.real(Z)
        Z[np.isinf(Z)] = np.nan

        # NaN -> 0 in z-space
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

        # weights (e.g., weight * 2)
        for col, w in self.weighted_cols.items():
            if col in self.col_index:
                Z[:, self.col_index[col]] *= float(w)

        return Z[0]

    def _pca_transform_one(self, z: np.ndarray) -> np.ndarray:
        if not self.use_pca:
            return z
        return (z - self.pca_mean) @ self.pca_components.T

    def predict_state_from_payload(self, payload: Dict[str, Any]) -> Tuple[int, np.ndarray]:
        """
        Returns:
          - state int
          - pseudo-prob vector over states (debug only)
        """
        if self.centroids is None or self.nn is None:
            raise RuntimeError("Centroids not loaded. Call load().")

        z = self._zscore_transform_one(payload)
        x = self._pca_transform_one(z)

        _, ind = self.nn.kneighbors([x], n_neighbors=1)
        s = int(ind[0, 0])

        # pseudo prob (debug): softmax(-distance)
        dists = np.linalg.norm(self.centroids - x[None, :], axis=1)
        dd = dists - dists.min()
        prob = np.exp(-dd)
        prob = prob / (prob.sum() + 1e-12)

        return s, prob

    # -------------------------
    # hybrid integration
    # -------------------------
    @staticmethod
    def _rule_flags(vaso_dose: float, lactate: float, creatinine: float):
        kcal_rule = False if (np.isnan(vaso_dose) or np.isnan(lactate)) else ((vaso_dose < 0.3) and (lactate < 2.0))
        pro_rule = False if (np.isnan(vaso_dose) or np.isnan(lactate) or np.isnan(creatinine)) else (
            (vaso_dose < 0.3) and (lactate < 2.0) and (creatinine < 5.0)
        )
        return kcal_rule, pro_rule

    @staticmethod
    def _day_targets(day: int) -> Tuple[int, int]:
        day = int(day)
        kcal_t = DAY_TARGET_KCAL.get(day, TARGET_DEFAULT if day >= 4 else DAY_TARGET_KCAL.get(3, 3))
        pro_t  = DAY_TARGET_PRO.get(day,  TARGET_DEFAULT if day >= 4 else DAY_TARGET_PRO.get(3, 3))
        if day >= 4:
            kcal_t = TARGET_DEFAULT
            pro_t  = TARGET_DEFAULT
        return int(kcal_t), int(pro_t)

    def _choose_action(self, q_row, day: int, subtype: int, vaso: float, lac: float, cre: float) -> Tuple[int, Dict[str, Any]]:
        score = normalize_q(q_row)
        kcal_bins = self.unique_pairs[:, 0].astype(int)
        pro_bins  = self.unique_pairs[:, 1].astype(int)

        # --- rule soft bonus
        kcal_rule, pro_rule = self._rule_flags(vaso, lac, cre)
        rule_bonus = np.zeros_like(score)

        if kcal_rule:
            rule_bonus += W_RULE * (kcal_bins == TARGET_KCAL_BIN).astype(float)
        if pro_rule:
            rule_bonus += W_RULE * (pro_bins == TARGET_PRO_BIN).astype(float)

        score = score + rule_bonus

        # --- day-progress targets (soft)
        day_kcal_t, day_pro_t = self._day_targets(day)
        day_bonus = W_DAY * ((kcal_bins == day_kcal_t).astype(float) + (pro_bins == day_pro_t).astype(float))
        score = score + day_bonus

        # --- subtype postprocess day4-7 (soft)
        sub_bonus = np.zeros_like(score)
        if SUBTYPE_DAY_FROM <= int(day) <= SUBTYPE_DAY_TO:
            if subtype == 1:
                # higher is better
                sub_bonus += W_SUB * ((kcal_bins - 1) / 4.0 + (pro_bins - 1) / 4.0)
            elif subtype == 2:
                # too high harmful: penalize bin5, slight prefer 3-4, avoid bin1
                sub_bonus -= W_SUB2_PENALTY_BIN5 * ((kcal_bins == 5).astype(float) + (pro_bins == 5).astype(float))
                sub_bonus += W_SUB2_BONUS_BIN4 * ((kcal_bins == 4).astype(float) + (pro_bins == 4).astype(float))
                sub_bonus += W_SUB2_BONUS_BIN3 * ((kcal_bins == 3).astype(float) + (pro_bins == 3).astype(float))
                sub_bonus -= W_SUB2_PENALTY_BIN1 * ((kcal_bins == 1).astype(float) + (pro_bins == 1).astype(float))

        score = score + sub_bonus

        a = int(np.argmax(score))
        dbg = {
            "kcal_rule_trigger": int(kcal_rule),
            "pro_rule_trigger": int(pro_rule),
            "day_target_kcal_bin": int(day_kcal_t),
            "day_target_pro_bin": int(day_pro_t),
            "chosen_action_id": a,
            "chosen_kcal_bin": int(kcal_bins[a]),
            "chosen_pro_bin": int(pro_bins[a]),
        }
        return a, dbg

    @staticmethod
    def _bin_ranges():
        kcal_bin_range = {1: "<=17", 2: "(17,43]", 3: "(43,83]", 4: "(83,116]", 5: ">116"}
        pro_bin_range  = {1: "<=10", 2: "(10,32]", 3: "(32,80]", 4: "(80,120]", 5: ">120"}
        return kcal_bin_range, pro_bin_range

    # -------------------------
    # public API
    # -------------------------
    def recommend(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Required:
          - payload['day']

        Optional:
          - payload['subtype'] in {0,1,2}  (from subgroup_tool)
          - feature columns (bin+norm+log) for accurate state
        """
        if self.Q is None:
            raise RuntimeError("Step5 artifacts not loaded. Call load().")

        if DAY_COL not in payload:
            raise KeyError("payload missing 'day'")
        day = int(payload[DAY_COL])

        subtype = int(payload.get("subtype", 0))
        if subtype not in (0, 1, 2):
            subtype = 0

        # state (centroid)
        s, state_prob = self.predict_state_from_payload(payload)

        # RL baseline action
        a_rl = int(self.optimal_policy[s])
        rl_kcal_bin = int(self.unique_pairs[a_rl, 0])
        rl_pro_bin  = int(self.unique_pairs[a_rl, 1])
        rl_TCI_pct  = float(self.unique_dose[a_rl, 0])
        rl_TPI_pct  = float(self.unique_dose[a_rl, 1])

        # hybrid choose with soft bonuses
        vaso = float(payload.get(VASO_COL, np.nan))
        lac  = float(payload.get(LAC_COL, np.nan))
        cre  = float(payload.get(CRE_COL, np.nan))

        a_rec, dbg = self._choose_action(self.Q[s, :], day, subtype, vaso, lac, cre)
        rec_kcal_bin = int(self.unique_pairs[a_rec, 0])
        rec_pro_bin  = int(self.unique_pairs[a_rec, 1])
        rec_TCI_pct  = float(self.unique_dose[a_rec, 0])
        rec_TPI_pct  = float(self.unique_dose[a_rec, 1])

        kcal_rng, pro_rng = self._bin_ranges()

        out = {
            "ok": True,
            "day": day,
            "subtype": int(subtype),

            # state
            "state": int(s),
            "state_prob_top5": self._topk_prob(state_prob, k=5),

            # RL baseline
            "rl_action_id": int(a_rl),
            "rl_kcal_bin": rl_kcal_bin,
            "rl_pro_bin": rl_pro_bin,
            "rl_TCI_pct": rl_TCI_pct,
            "rl_TPI_pct": rl_TPI_pct,

            # hybrid recommendation
            "rec_action_id": int(a_rec),
            "rec_kcal_bin": rec_kcal_bin,
            "rec_pro_bin": rec_pro_bin,
            "rec_TCI_pct": rec_TCI_pct,
            "rec_TPI_pct": rec_TPI_pct,
            "rec_TCI_bin_range": kcal_rng.get(rec_kcal_bin, ""),
            "rec_TPI_bin_range": pro_rng.get(rec_pro_bin, ""),

            # debug flags
            **dbg,
        }
        return out

    @staticmethod
    def _topk_prob(prob: np.ndarray, k: int = 5):
        idx = np.argsort(prob)[::-1][:k]
        return [{"state": int(i), "prob": float(prob[i])} for i in idx]


# =========================
# Quick CLI test
# =========================
if __name__ == "__main__":
    hp = HybridPolicy().load()

    demo = {
        "day": 2,
        "subtype": 1,  # <- 假设 subgroup_tool 输出 1

        # BIN
        "gender": 1, "vent": 0, "rt": 0,

        # NORM
        "age": 60, "charlson_comorbidity_index": 2, "height": 170,
        "hr": 95, "meanbp": 75, "rr": 22, "temp": 37.2,
        "sodium_chem": 140, "chloride_chem": 105, "calcium": 2.1,
        "hemoglobin": 9.5, "sofa": 8, "albumin": 2.8,

        # LOG
        "vaso_dose": 0.2, "weight": 70, "gcs": 14, "wbc": 12, "glucose_chem": 8.5, "potassium_chem": 4.0,
        "platelet": 150, "ptt": 35, "pt": 14, "pao2_fio2": 260, "pco2": 42, "lactate": 1.6,
        "spo2": 97, "ast": 40, "alt": 35, "bilirubin": 10, "output_total": 1800, "bun": 12,
        "creatinine": 1.2, "inr": 1.1,
        "TCI": 30, "TPI": 25, "BCR": 10.0,
    }

    out = hp.recommend(demo)
    print(json.dumps(out, ensure_ascii=False, indent=2))
