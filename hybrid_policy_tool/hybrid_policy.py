# hybrid_policy_tool/hybrid_policy.py
# HybridPolicyEngine: load Step4/Step5/Step1_3 artifacts + subtype XGB model,
# then produce daily Hybrid recommendation (soft rules + soft subtype preference).
#
# Place assets in: hybrid_policy_tool/assets/
#   - step4_states_daily.npz
#   - step5_mdp_daily_policy.npz
#   - step1_3_daily_prepared.npz
#   - (optional) step6_subtype_model/subtype_xgb.json + subtype_meta.json
#
# Author: (generated) GPT
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # keep engine usable without pandas
    pd = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


# =========================
# Constants (bin explanations)
# =========================
KCAL_BIN_RANGE = {1: "<=17", 2: "(17,43]", 3: "(43,83]", 4: "(83,116]", 5: ">116"}
PRO_BIN_RANGE  = {1: "<=10", 2: "(10,32]", 3: "(32,80]", 4: "(80,120]", 5: ">120"}

SUBTYPE_MEANING = {
    0: "insensitive to nutrition dose",
    1: "higher is better",
    2: "too high is harmful",
}


# =========================
# Dataclass for output
# =========================
@dataclass
class HybridResult:
    ok: bool
    msg: str = ""

    # identifiers
    stay_id: Optional[int] = None
    day: Optional[int] = None
    split: Optional[str] = None  # derivation/validation if known
    state: Optional[int] = None  # assigned MDP state

    # patient-level subtype
    subtype: Optional[int] = None

    # rule flags
    kcal_rule_trigger: int = 0
    pro_rule_trigger: int = 0

    # RL baseline (from optimal_policy)
    rl_action_id: Optional[int] = None
    rl_kcal_bin: Optional[int] = None
    rl_pro_bin: Optional[int] = None
    rl_TCI_pct: Optional[float] = None
    rl_TPI_pct: Optional[float] = None

    # Hybrid recommendation (post-processed argmax over Q)
    rec_action_id: Optional[int] = None
    rec_kcal_bin: Optional[int] = None
    rec_pro_bin: Optional[int] = None
    rec_TCI_pct: Optional[float] = None
    rec_TPI_pct: Optional[float] = None

    # extra debug
    debug: Optional[Dict[str, Any]] = None


# =========================
# Engine
# =========================
class HybridPolicyEngine:
    """
    Standalone engine to produce Hybrid daily recommendation.

    Key idea:
      - Use Step4 state assignment (state_d / state_v) aligned with Step1_3 keys (stay_id_d/day_d etc.)
      - Use Step5 artifacts: Q(s,a), optimal_policy(s), unique_pairs(a)->(kcal_bin,pro_bin), unique_dose(a)->(TCI%,TPI%)
      - Hybrid action = argmax_a [ normalize(Q[s,a]) + soft bonuses ]
        where bonuses include:
          (A) hemodynamic threshold soft bonus:
              if vaso_dose<0.3 & lactate<2    -> prefer kcal_bin==4  (soft)
              if vaso_dose<0.3 & lactate<2 & creatinine<5 -> prefer pro_bin==4 (soft)
          (B) subtype soft preference (Day4-7 only):
              subtype==1: add small score proportional to (kcal_bin+pro_bin) (prefer higher)
              subtype==2: penalize bin5, small bonus bin3/4, slight penalize bin1 (avoid extremes)
              subtype==0: no adjustment
    """

    # ---- defaults (match your Step6 v2) ----
    TARGET_KCAL_BIN = 4
    TARGET_PRO_BIN  = 4

    W_RULE = 0.15
    W_SUB  = 0.10

    SUBTYPE_DAY_FROM = 4
    SUBTYPE_DAY_TO   = 7

    W_SUB2_PENALTY_BIN5 = 0.25
    W_SUB2_BONUS_BIN4   = 0.08
    W_SUB2_BONUS_BIN3   = 0.04
    W_SUB2_PENALTY_BIN1 = 0.02

    # Early days: soft progressive target (your “方式2：目标档位随 day 变化（soft）”)
    # If you already implemented progressive in Step6, you can keep this consistent here.
    EARLY_DAY_TO = 3
    W_EARLY = 0.10
    EARLY_TARGET_BY_DAY = {
        1: {"kcal": 2, "pro": 2},
        2: {"kcal": 2, "pro": 3},
        3: {"kcal": 3, "pro": 3},
    }

    # ---- subtype model features (fixed per your spec) ----
    subtype_features = [
        "age",
        "charlson_comorbidity_index",
        "hemoglobin",
        "pao2_fio2",
        "sofa",
        "bun",
        "creatinine",
        "albumin",
        "output_total",
        "BCR",
    ]

    def __init__(self, assets_dir: str | Path):
        self.assets_dir = Path(assets_dir).resolve()

        # paths
        self.p_step4 = self._resolve("step4_states_daily.npz")
        self.p_step5 = self._resolve("step5_mdp_daily_policy.npz")
        self.p_step1 = self._resolve("step1_3_daily_prepared.npz")

        # subtype model (optional)
        self.subtype_dir = self.assets_dir / "step6_subtype_model"
        self.p_subtype_model = self.subtype_dir / "subtype_xgb.json"
        self.p_subtype_meta  = self.subtype_dir / "subtype_meta.json"

        # load artifacts
        self._load_step1_3()
        self._load_step4()
        self._load_step5()
        self._build_key_index()

        # load subtype model if available
        self._subtype_clf = None
        self._subtype_meta = None
        self._try_load_subtype_model()

    # =========================
    # Path helpers
    # =========================
    def _resolve(self, fname: str) -> Path:
        p = self.assets_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"[HybridPolicyEngine] Missing asset: {p}")
        return p

    # =========================
    # Load artifacts
    # =========================
    def _load_step1_3(self):
        d = np.load(self.p_step1, allow_pickle=True)
        # expected
        self.stay_id_d = d["stay_id_d"]
        self.day_d = d["day_d"].astype(int)
        self.stay_id_v = d["stay_id_v"]
        self.day_v = d["day_v"].astype(int)

    def _load_step4(self):
        d = np.load(self.p_step4, allow_pickle=True)
        self.state_d = d["state_d"].astype(int)
        self.state_v = d["state_v"].astype(int) if "state_v" in d.files else None

        # sanity checks against keys
        if len(self.state_d) != len(self.stay_id_d):
            raise RuntimeError(f"[HybridPolicyEngine] state_d length {len(self.state_d)} != stay_id_d length {len(self.stay_id_d)}")
        if self.state_v is not None and len(self.state_v) != len(self.stay_id_v):
            raise RuntimeError(f"[HybridPolicyEngine] state_v length {len(self.state_v)} != stay_id_v length {len(self.stay_id_v)}")

    def _load_step5(self):
        d = np.load(self.p_step5, allow_pickle=True)
        self.Q = d["Q"].astype(float)                      # shape (n_states, n_actions)
        self.unique_pairs = d["unique_pairs"].astype(int)  # shape (n_actions, 2): (kcal_bin, pro_bin)
        self.unique_dose = d["unique_dose"].astype(float)  # shape (n_actions, 2): (TCI%, TPI%)
        self.optimal_policy = d["optimal_policy"].astype(int)  # shape (n_states,)

        # derived
        self.n_states = int(self.Q.shape[0])
        self.n_actions = int(self.Q.shape[1])

    def _build_key_index(self):
        """
        Build mapping (stay_id, day) -> state, split.
        Use both derivation and validation keys, same as step6 v2.
        """
        self.key_map: Dict[Tuple[int, int], Tuple[int, str]] = {}

        for sid, day, st in zip(self.stay_id_d, self.day_d, self.state_d):
            try:
                k = (int(sid), int(day))
                self.key_map[k] = (int(st), "derivation")
            except Exception:
                continue

        if self.state_v is not None:
            for sid, day, st in zip(self.stay_id_v, self.day_v, self.state_v):
                try:
                    k = (int(sid), int(day))
                    self.key_map[k] = (int(st), "validation")
                except Exception:
                    continue

    # =========================
    # Subtype model (optional)
    # =========================
    def _try_load_subtype_model(self):
        if XGBClassifier is None:
            return
        if not (self.p_subtype_model.exists() and self.p_subtype_meta.exists()):
            return

        try:
            meta = json.loads(self.p_subtype_meta.read_text(encoding="utf-8"))
            clf = XGBClassifier()
            clf.load_model(str(self.p_subtype_model))
            self._subtype_meta = meta
            self._subtype_clf = clf
        except Exception:
            self._subtype_meta = None
            self._subtype_clf = None

    def predict_subtype_from_day1(self, day1_payload: Dict[str, Any]) -> int:
        """
        Predict subtype (0/1/2) using Day1 features.
        If subtype model missing, default subtype=0.
        Missing values filled with median of provided values (fallback 0.0).
        """
        if self._subtype_clf is None or self._subtype_meta is None:
            return 0

        feats = self._subtype_meta.get("features", self.subtype_features)
        x = []
        for f in feats:
            v = day1_payload.get(f, None)
            try:
                vv = float(v)
                if vv != vv:
                    vv = np.nan
            except Exception:
                vv = np.nan
            x.append(vv)

        X = np.array([x], dtype=float)
        # fill nan with median of non-nan in row; if all nan -> 0
        row = X[0]
        if np.all(np.isnan(row)):
            row = np.zeros_like(row)
        else:
            med = np.nanmedian(row)
            row = np.where(np.isnan(row), med, row)
        X[0] = row

        pred = self._subtype_clf.predict(X)
        try:
            return int(pred[0])
        except Exception:
            return 0

    # =========================
    # Core scoring logic
    # =========================
    @staticmethod
    def _normalize_q(q_row: np.ndarray) -> np.ndarray:
        q = q_row.astype(float).copy()
        qmin = float(np.min(q))
        qmax = float(np.max(q))
        if qmax > qmin:
            return (q - qmin) / (qmax - qmin)
        return np.zeros_like(q)

    @staticmethod
    def _rule_flags(vaso_dose: float, lactate: float, creatinine: float) -> Tuple[bool, bool]:
        def isn(x): return (x is None) or (not np.isfinite(float(x)))
        if isn(vaso_dose) or isn(lactate):
            kcal_rule = False
        else:
            kcal_rule = (float(vaso_dose) < 0.3) and (float(lactate) < 2.0)

        if isn(vaso_dose) or isn(lactate) or isn(creatinine):
            pro_rule = False
        else:
            pro_rule = (float(vaso_dose) < 0.3) and (float(lactate) < 2.0) and (float(creatinine) < 5.0)

        return kcal_rule, pro_rule

    def _apply_soft_bonuses(
        self,
        score: np.ndarray,
        day: int,
        subtype: int,
        vaso_dose: float,
        lactate: float,
        creatinine: float,
    ) -> Tuple[np.ndarray, bool, bool, Dict[str, Any]]:
        """
        Apply:
          - threshold soft bonus (rule)
          - subtype soft bonus (Day4-7)
          - early progressive soft bonus (Day1-3)
        """
        debug: Dict[str, Any] = {}
        score2 = score.copy()

        kcal_bins = self.unique_pairs[:, 0].astype(int)
        pro_bins  = self.unique_pairs[:, 1].astype(int)

        kcal_rule, pro_rule = self._rule_flags(vaso_dose, lactate, creatinine)
        debug["kcal_rule"] = kcal_rule
        debug["pro_rule"] = pro_rule

        # (A) hemodynamic rules (soft)
        if kcal_rule:
            score2 += self.W_RULE * (kcal_bins == self.TARGET_KCAL_BIN).astype(float)
        if pro_rule:
            score2 += self.W_RULE * (pro_bins == self.TARGET_PRO_BIN).astype(float)

        # (B) subtype postprocess day4-7 (soft)
        if self.SUBTYPE_DAY_FROM <= int(day) <= self.SUBTYPE_DAY_TO:
            if int(subtype) == 1:
                # prefer higher bins (scaled 0..1)
                score2 += self.W_SUB * ((kcal_bins - 1) / 4.0 + (pro_bins - 1) / 4.0)
                debug["subtype_bonus"] = "subtype1_high_is_better"
            elif int(subtype) == 2:
                # avoid bin5, prefer 3-4, slight avoid 1
                score2 -= self.W_SUB2_PENALTY_BIN5 * ((kcal_bins == 5).astype(float) + (pro_bins == 5).astype(float))
                score2 += self.W_SUB2_BONUS_BIN4   * ((kcal_bins == 4).astype(float) + (pro_bins == 4).astype(float))
                score2 += self.W_SUB2_BONUS_BIN3   * ((kcal_bins == 3).astype(float) + (pro_bins == 3).astype(float))
                score2 -= self.W_SUB2_PENALTY_BIN1 * ((kcal_bins == 1).astype(float) + (pro_bins == 1).astype(float))
                debug["subtype_bonus"] = "subtype2_avoid_too_high"
            else:
                debug["subtype_bonus"] = "none"

        # (C) early days progressive (soft)
        if 1 <= int(day) <= self.EARLY_DAY_TO:
            tgt = self.EARLY_TARGET_BY_DAY.get(int(day), None)
            if tgt:
                score2 += self.W_EARLY * (kcal_bins == int(tgt["kcal"])).astype(float)
                score2 += self.W_EARLY * (pro_bins == int(tgt["pro"])).astype(float)
                debug["early_target"] = tgt

        return score2, kcal_rule, pro_rule, debug

    def choose_action(
        self,
        state: int,
        day: int,
        subtype: int,
        vaso_dose: float,
        lactate: float,
        creatinine: float,
        return_debug: bool = False,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Choose Hybrid action for given state + patient context.
        """
        s = int(state)
        if s < 0 or s >= self.n_states:
            raise ValueError(f"state out of range: {s}")

        base = self._normalize_q(self.Q[s, :])
        score, kcal_rule, pro_rule, dbg = self._apply_soft_bonuses(
            base, day=int(day), subtype=int(subtype),
            vaso_dose=float(vaso_dose), lactate=float(lactate), creatinine=float(creatinine)
        )
        a = int(np.argmax(score))

        if return_debug:
            dbg = dbg or {}
            dbg["state"] = s
            dbg["chosen_action"] = a
            dbg["rl_action"] = int(self.optimal_policy[s])
            dbg["q_min"] = float(np.min(self.Q[s, :]))
            dbg["q_max"] = float(np.max(self.Q[s, :]))
            dbg["score_top5"] = list(np.argsort(score)[-5:][::-1].astype(int))
            return a, dbg

        return a, {}

    # =========================
    # Public API
    # =========================
    def lookup_state(self, stay_id: int, day: int) -> Optional[Tuple[int, str]]:
        return self.key_map.get((int(stay_id), int(day)), None)

    def recommend_one(
        self,
        stay_id: int,
        day: int,
        subtype: int,
        vaso_dose: float,
        lactate: float,
        creatinine: float,
        debug: bool = False,
    ) -> HybridResult:
        """
        Main entry: given (stay_id, day) + patient-level subtype + day-level variables,
        return RL baseline and Hybrid recommendation.
        """
        k = (int(stay_id), int(day))
        hit = self.key_map.get(k, None)
        if hit is None:
            return HybridResult(ok=False, msg=f"Key not found: stay_id={stay_id}, day={day}")

        state, split = hit
        s = int(state)

        # RL baseline
        a_rl = int(self.optimal_policy[s])
        rl_k = int(self.unique_pairs[a_rl, 0])
        rl_p = int(self.unique_pairs[a_rl, 1])
        rl_tci = float(self.unique_dose[a_rl, 0])
        rl_tpi = float(self.unique_dose[a_rl, 1])

        # Hybrid choose
        a_rec, dbg = self.choose_action(
            state=s, day=int(day), subtype=int(subtype),
            vaso_dose=float(vaso_dose), lactate=float(lactate), creatinine=float(creatinine),
            return_debug=debug
        )
        rec_k = int(self.unique_pairs[a_rec, 0])
        rec_p = int(self.unique_pairs[a_rec, 1])
        rec_tci = float(self.unique_dose[a_rec, 0])
        rec_tpi = float(self.unique_dose[a_rec, 1])

        kcal_rule, pro_rule = self._rule_flags(float(vaso_dose), float(lactate), float(creatinine))

        return HybridResult(
            ok=True,
            msg="ok",
            stay_id=int(stay_id),
            day=int(day),
            split=str(split),
            state=int(s),
            subtype=int(subtype),
            kcal_rule_trigger=int(kcal_rule),
            pro_rule_trigger=int(pro_rule),
            rl_action_id=int(a_rl),
            rl_kcal_bin=int(rl_k),
            rl_pro_bin=int(rl_p),
            rl_TCI_pct=float(rl_tci),
            rl_TPI_pct=float(rl_tpi),
            rec_action_id=int(a_rec),
            rec_kcal_bin=int(rec_k),
            rec_pro_bin=int(rec_p),
            rec_TCI_pct=float(rec_tci),
            rec_TPI_pct=float(rec_tpi),
            debug=dbg if debug else None
        )

    # Optional convenience
    def explain_bins(self, kcal_bin: int, pro_bin: int) -> Dict[str, str]:
        return {
            "kcal_bin": str(int(kcal_bin)),
            "kcal_range": KCAL_BIN_RANGE.get(int(kcal_bin), "NA"),
            "pro_bin": str(int(pro_bin)),
            "pro_range": PRO_BIN_RANGE.get(int(pro_bin), "NA"),
        }

    def version_info(self) -> Dict[str, Any]:
        return {
            "assets_dir": str(self.assets_dir),
            "n_states": int(self.n_states),
            "n_actions": int(self.n_actions),
            "subtype_model_loaded": bool(self._subtype_clf is not None),
            "subtype_features": list(self.subtype_features),
            "constants": {
                "TARGET_KCAL_BIN": self.TARGET_KCAL_BIN,
                "TARGET_PRO_BIN": self.TARGET_PRO_BIN,
                "W_RULE": self.W_RULE,
                "W_SUB": self.W_SUB,
                "SUBTYPE_DAY_FROM": self.SUBTYPE_DAY_FROM,
                "SUBTYPE_DAY_TO": self.SUBTYPE_DAY_TO,
                "W_EARLY": self.W_EARLY,
                "EARLY_TARGET_BY_DAY": self.EARLY_TARGET_BY_DAY,
            }
        }
