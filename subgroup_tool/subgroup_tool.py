import json
import os
from typing import Dict, Any

# 你可以直接把 SubgroupXGBPredictor 复制到 subgroup_predictor.py
# 然后从那里 import
from .subgroup_predictor import SubgroupXGBPredictor



class SubgroupClassifierTool:
    """
    主智能体调用的 Tool：输入10个入ICU特征 -> 输出三分类亚组
    """

    def __init__(self, model_json_path: str, meta_json_path: str):
        self.predictor = SubgroupXGBPredictor(
            model_json_path,
            meta_json_path
        )

        # label 名称映射：如果 meta 里没有，就用默认 Cluster_1/2/3
        self.label_names = self.predictor.meta.get("label_names") or ["Cluster_1", "Cluster_2", "Cluster_3"]

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # 1) 预测
            out = self.predictor.predict(payload)  # 你已有predict方法的话
            # 兼容：如果你的 predictor 返回的是 (label, proba) 或 dict
            if isinstance(out, dict):
                pred = int(out["pred_label"])
                proba = out.get("proba")
            else:
                pred, proba = out  # (pred_label, proba_vector)

            # 2) 处理 proba
            proba_dict = None
            if proba is not None:
                # 如果 proba 是 list/np array
                if isinstance(proba, (list, tuple)):
                    proba_dict = {self.label_names[i]: float(proba[i]) for i in range(len(proba))}
                else:
                    # 已经是 dict
                    proba_dict = {k: float(v) for k, v in proba.items()}

            pred_name = self.label_names[pred] if pred < len(self.label_names) else f"Cluster_{pred}"

            return {
                "ok": True,
                "pred_label": pred,
                "pred_name": pred_name,
                "proba": proba_dict,
                "input_used": payload,
                "feature_order": getattr(self.predictor, "feature_names", None),
            }

        except Exception as e:
            return {"ok": False, "error": str(e)}


# ====== 给主智能体的统一入口 ======
_TOOL_INSTANCE = None

def run(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    主智能体调用这个函数即可：run(payload)->result
    """
    global _TOOL_INSTANCE
    base_dir = os.path.dirname(__file__)
    model_json = os.path.join(base_dir, "subgroup_xgb_model.json")
    meta_json  = os.path.join(base_dir, "subgroup_model_meta.json")

    if _TOOL_INSTANCE is None:
        _TOOL_INSTANCE = SubgroupClassifierTool(model_json, meta_json)

    return _TOOL_INSTANCE.run(payload)
