import json
import numpy as np
import xgboost as xgb

class SubgroupXGBPredictor:
    def __init__(self, model_json_path: str, meta_json_path: str):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_json_path)

        with open(meta_json_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        # 训练时用的特征顺序（非常关键）
        DEFAULT_FEATURES = [
            "Age", "CCI", "Hemoglobin", "Albumin", "Creatinine",
            "BUN", "BCR", "PF_ratio", "Urine_output", "SOFA"
        ]

        # 兼容不同 meta 写法
        self.feature_names = (
                self.meta.get("feature_names")
                or self.meta.get("features")
                or self.meta.get("columns")
                or DEFAULT_FEATURES
        )

        # 有些人会把 features 存成 dict / list of dict，这里统一成 list[str]
        if isinstance(self.feature_names, dict):
            self.feature_names = list(self.feature_names.keys())
        elif isinstance(self.feature_names, list) and len(self.feature_names) > 0 and isinstance(self.feature_names[0],
                                                                                                 dict):
            # 例如 [{"name":"Age"}, ...]
            self.feature_names = [d.get("name") for d in self.feature_names if "name" in d]

        # 如果你保存了 label 偏移/映射，这里读出来
        self.label_offset = self.meta.get("label_offset", 0)  # 例如训练时 y=label-1 => offset=1
        # 或者 label_map / inv_label_map
        self.inv_label_map = self.meta.get("inv_label_map", None)

    def _vectorize(self, x: dict) -> np.ndarray:
        row = []
        missing = []
        for k in self.feature_names:
            v = x.get(k, None)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                missing.append(k)
                row.append(np.nan)
            else:
                row.append(float(v))
        if missing:
            # 你可以改成填充默认值/中位数；这里先直接报错最安全
            raise ValueError(f"Missing required features: {missing}")
        return np.array(row, dtype=float).reshape(1, -1)

    def predict(self, x: dict) -> dict:
        X = self._vectorize(x)

        proba = self.model.predict_proba(X)[0]  # shape (3,)
        pred0 = int(np.argmax(proba))           # 0/1/2

        # 映射回原始标签（如果你训练时做了 -1）
        if self.inv_label_map:
            pred_label = self.inv_label_map[str(pred0)] if isinstance(self.inv_label_map, dict) else pred0
        else:
            pred_label = pred0 + int(self.label_offset)

        return {
            "pred_label": int(pred_label),
            "pred_0based": pred0,
            "proba": [float(p) for p in proba],
            "feature_names": self.feature_names
        }



