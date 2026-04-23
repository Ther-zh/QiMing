from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, List
from PIL import Image

from utils.config_loader import config_loader

class MockQwenMultimodal:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Mock Qwen多模态大模型
        
        Args:
            config: 配置字典
        """
        self.config = config
        self._risk_rules = None
        self._turn_idx = 0
        print("[Mock LLM] 初始化成功")

    def _get_risk_rules(self) -> Dict[str, Any]:
        if self._risk_rules is None:
            try:
                self._risk_rules = config_loader.get_risk_rules() or {}
            except Exception:
                self._risk_rules = {}
        return self._risk_rules

    @staticmethod
    def _norm_direction(direction: Optional[str]) -> str:
        if not direction:
            return "前方"
        d = str(direction)
        for k in ("前", "后", "左", "右"):
            if k in d:
                return f"{k}侧" if k in ("左", "右") else ("前方" if k == "前" else "后方")
        return d

    @staticmethod
    def _intent(prompt: str) -> str:
        p = (prompt or "").strip()
        if any(k in p for k in ("过马路", "红绿灯", "斑马线", "路口")):
            return "cross"
        if any(k in p for k in ("导航", "去", "到", "怎么走", "往哪")):
            return "nav"
        if any(k in p for k in ("障碍", "避让", "绕开", "躲")):
            return "avoid"
        if any(k in p for k in ("路况", "前面", "前方", "有什么", "安全吗", "危险")):
            return "scene"
        return "scene"

    def _distance_coeff(self, dist_m: float) -> float:
        rules = self._get_risk_rules()
        table = rules.get("distance_coefficients") or {}
        # 默认表（防止规则文件缺失）
        if not table:
            table = {"0-1": 3.0, "1-2": 2.5, "2-3": 2.0, "3-5": 1.5, "5-10": 1.0, "10-20": 0.5, "20+": 0.1}

        d = max(0.0, float(dist_m or 0.0))
        for key, coef in table.items():
            try:
                if key.endswith("+"):
                    lo = float(key[:-1])
                    if d >= lo:
                        return float(coef)
                else:
                    lo_s, hi_s = key.split("-", 1)
                    lo, hi = float(lo_s), float(hi_s)
                    if lo <= d < hi:
                        return float(coef)
            except Exception:
                continue
        return 1.0

    def _speed_coeff(self, speed: float) -> float:
        rules = self._get_risk_rules()
        table = rules.get("speed_coefficients") or {}
        if not table:
            table = {"0-5": 0.5, "5-10": 1.0, "10-15": 1.5, "15-20": 2.0, "20-30": 2.5, "30+": 3.0}
        s = max(0.0, float(speed or 0.0))
        for key, coef in table.items():
            try:
                if key.endswith("+"):
                    lo = float(key[:-1])
                    if s >= lo:
                        return float(coef)
                else:
                    lo_s, hi_s = key.split("-", 1)
                    lo, hi = float(lo_s), float(hi_s)
                    if lo <= s < hi:
                        return float(coef)
            except Exception:
                continue
        return 1.0

    def _category_weight(self, category: str) -> float:
        rules = self._get_risk_rules()
        weights = rules.get("category_weights") or {}
        if not weights:
            weights = {"person": 10, "car": 15, "truck": 20, "bicycle": 8, "motorcycle": 12, "traffic_light": 5, "crosswalk": 3, "obstacle": 8, "construction": 10, "unknown": 5}
        return float(weights.get(category, weights.get("unknown", 5)))

    def _score_target(self, t: Dict[str, Any]) -> float:
        cat = str(t.get("category") or "unknown")
        dist = float(t.get("distance") or 0.0)
        spd = float(t.get("speed") or 0.0)
        base = self._category_weight(cat)
        return base * self._distance_coeff(dist) * (1.0 + 0.35 * self._speed_coeff(spd))

    def _risk_level(self, score: float) -> str:
        rules = self._get_risk_rules()
        levels = rules.get("risk_levels") or {}
        # 阈值越高越危险：level1 > level2 > level3 > level4
        order = ["level1", "level2", "level3", "level4"]
        for lv in order:
            try:
                th = float((levels.get(lv) or {}).get("threshold", 0))
            except Exception:
                th = 0.0
            if score >= th:
                return lv
        return "level4"

    @staticmethod
    def _brief_targets(targets: List[Dict[str, Any]], top_k: int = 2) -> str:
        def _one(t: Dict[str, Any]) -> str:
            cat = str(t.get("category") or "目标")
            dist = float(t.get("distance") or 0.0)
            direction = MockQwenMultimodal._norm_direction(t.get("direction"))
            return f"{direction}{dist:.1f}米{cat}"

        parts = [_one(t) for t in targets[:top_k]]
        return "、".join(parts)
    
    def inference(self, input_data: Tuple[Optional[Image.Image], Dict[str, Any], str]) -> str:
        """
        模拟多模态推理
        
        Args:
            input_data: Tuple[image, metadata, prompt]，包含图像、环境元数据和用户指令
            
        Returns:
            生成的口语化文本（≤100字）
        """
        image, metadata, prompt = input_data
        metadata = metadata or {}

        print(f"[Mock LLM] 接收到的prompt: {prompt}")

        targets = metadata.get("targets", [])

        # 1) 计算风险：取最危险目标 + 总体风险等级
        scored = []
        for t in targets:
            try:
                scored.append((self._score_target(t), t))
            except Exception:
                continue
        scored.sort(key=lambda x: x[0], reverse=True)
        top_score, top_t = (scored[0] if scored else (0.0, None))
        risk_lv = self._risk_level(float(top_score))

        # 2) 识别意图（来自 ASR/预设 prompt）
        intent = self._intent(prompt)

        # 3) 演示模式：输出稳定、短、可控
        demo_mode = bool(self.config.get("demo_mode", True))
        verbose = bool(self.config.get("demo_verbose", False))
        self._turn_idx += 1

        # 4) 生成话术
        if not targets:
            if intent == "cross":
                return "路口信息不足，先停一下，左右听车流再过。"
            if intent == "nav":
                return "我暂时没看到明显目标，建议保持直行，注意脚下台阶。"
            return "目前画面目标较少，前方可慢速前进，注意脚下。"

        # 目标摘要（只说最关键的 1-2 个）
        top_targets = [t for _, t in scored[:2] if isinstance(t, dict)]
        brief = self._brief_targets(top_targets, top_k=2)

        # 风险话术
        if risk_lv in ("level1", "level2"):
            action = "先停" if risk_lv == "level1" else "放慢"
            if intent == "cross":
                return f"{action}，{brief}，确认安全再过马路。"
            if intent == "avoid":
                return f"{action}，{brief}，建议向空旷一侧绕开。"
            if intent == "nav":
                return f"{action}，{brief}，先避开危险再继续导航。"
            return f"{action}，{brief}，注意避让。"

        # 低风险/安全：给出更像“导盲提示”的建议
        if intent == "cross":
            if any(str(t.get("category")) == "traffic_light" for t in targets):
                return f"前方有信号灯，{brief}，等绿灯再过。"
            if any(str(t.get("category")) == "crosswalk" for t in targets):
                return f"看到斑马线，{brief}，先听车流再过。"
            return f"路口附近有目标：{brief}，建议慢速通过。"

        if intent == "nav":
            # 演示：给个稳定的“直行-微调”建议
            base = f"前方目标：{brief}，保持直行。"
            if demo_mode and (self._turn_idx % 3 == 0):
                base = f"前方目标：{brief}，略向右调整后直行。"
            return base[:100]

        if intent == "avoid":
            return f"前方有：{brief}，保持慢速，贴边绕行。"

        # scene
        msg = f"前方情况：{brief}，整体安全，慢速前进。"
        if verbose:
            msg = f"{msg}（mock风险={top_score:.1f}/{risk_lv}）"
        return msg[:100]
    
    def release(self):
        """
        释放资源
        """
        print("[Mock LLM] 资源已释放")
