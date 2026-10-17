import os
import math
import logging
import numpy as np

logger = logging.getLogger("RiskModel")

def _safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def _safe_int(val, default=0):
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

class PeatFireRiskModel:
    """
    High-Throughput Peatland Fire Risk Inference Model (CPU Optimized)
    Evaluates multivariant biophysical and anthropogenic factors:
    1. Ground Water Level (GWL in cm) - Critical boundary < -40 cm (PP 57/2016)
    2. Days Without Rain (HTH)
    3. Vapor Pressure Deficit (VPD in kPa)
    4. Peat Depth (Meters)
    5. Proximity to Drainage Canals (Meters)
    6. Proximity to Access Roads (Meters)

    Supports:
    - Trained XGBoost Classifier model (models/karhutla_xgb_v1.json)
    - Fallback to physics-based deterministic empirical formula if model not present
    """

    def __init__(self, model_path=None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "models", "karhutla_xgb_v1.json")

        self.model_path = model_path
        self.xgb_model = None
        self.is_ml_active = False

        self._load_model_if_available()

    def _load_model_if_available(self):
        """Attempts to load the trained XGBoost model from disk."""
        if os.path.exists(self.model_path):
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier()
                model.load_model(self.model_path)
                self.xgb_model = model
                self.is_ml_active = True
                self.hardware_descriptor = "CPU (Trained XGBoost Karhutla Engine v1 - Histogram)"
                logger.info("Successfully loaded XGBoost model from %s", self.model_path)
            except Exception as e:
                logger.warning("Failed to load XGBoost model from %s: %s. Using deterministic fallback.", self.model_path, e)
                self.xgb_model = None
                self.is_ml_active = False
                self.hardware_descriptor = "CPU (Vectorized Deterministic Peatland Engine)"
        else:
            self.xgb_model = None
            self.is_ml_active = False
            self.hardware_descriptor = "CPU (Vectorized Deterministic Peatland Engine)"
            logger.info("No trained XGBoost model at %s. Running in deterministic heuristic mode.", self.model_path)

    def predict_batch(self, feature_list):
        """
        Receives list of H3FeatureVector-like objects or dicts.
        Returns list of dicts: {h3_index, risk_score, risk_category, sar_anomaly_flag}
        """
        if not feature_list:
            return []

        # Parse feature vectors
        parsed = []
        for feat in feature_list:
            if isinstance(feat, dict):
                h3_index = str(feat.get('h3_index') or '')
                depth = _safe_float(feat.get('peat_depth_meters'), 2.0)
                gwl = _safe_float(feat.get('ground_water_level_cm'), -35.0)
                dwr = _safe_int(feat.get('days_without_rain'), 5)
                vpd = _safe_float(feat.get('vapor_pressure_deficit'), 2.0)
                dist_road = _safe_float(feat.get('distance_to_road_meters'), 1500.0)
                dist_canal = _safe_float(feat.get('distance_to_canal_meters'), 800.0)
            else:
                h3_index = str(getattr(feat, 'h3_index', ''))
                depth = _safe_float(getattr(feat, 'peat_depth_meters', None), 2.0)
                gwl = _safe_float(getattr(feat, 'ground_water_level_cm', None), -35.0)
                dwr = _safe_int(getattr(feat, 'days_without_rain', None), 5)
                vpd = _safe_float(getattr(feat, 'vapor_pressure_deficit', None), 2.0)
                dist_road = _safe_float(getattr(feat, 'distance_to_road_meters', None), 1500.0)
                dist_canal = _safe_float(getattr(feat, 'distance_to_canal_meters', None), 800.0)

            parsed.append((h3_index, depth, gwl, dwr, vpd, dist_road, dist_canal))

        # Attempt XGBoost ML Batch Inference if active
        scores = None
        if self.is_ml_active and self.xgb_model is not None:
            try:
                # Contract order: [peat_depth_meters, ground_water_level_cm, days_without_rain, vapor_pressure_deficit, distance_to_road_meters, distance_to_canal_meters]
                X = np.array([[p[1], p[2], p[3], p[4], p[5], p[6]] for p in parsed], dtype=np.float32)
                probs = self.xgb_model.predict_proba(X)[:, 1]
                scores = [round(float(np.clip(prob, 0.010, 0.995)), 3) for prob in probs]
            except Exception as e:
                logger.error("XGBoost batch inference failed: %s. Reverting to deterministic formula.", e)
                scores = None

        results = []
        for idx, p in enumerate(parsed):
            h3_index, depth, gwl, dwr, vpd, dist_road, dist_canal = p

            if scores is not None:
                score = scores[idx]
            else:
                score = self._compute_score(gwl, dwr, vpd, depth, dist_road, dist_canal)

            category = self._classify_category(score)
            sar_anomaly = (category in ("EXTREME", "HIGH")) and (score >= 0.70 or gwl < -45.0)

            results.append({
                "h3_index": h3_index,
                "risk_score": score,
                "risk_category": category,
                "sar_anomaly_flag": sar_anomaly
            })

        return results

    def _compute_score(self, gwl, dwr, vpd, depth, dist_road, dist_canal):
        """Deterministic physics-based empirical formula (PP 57/2016 aligned)."""
        # 1. GWL factor (negative cm: -15cm is wet, -40cm is threshold, -65cm is severely dried)
        f_gwl = max(0.0, min(1.0, (-gwl - 18.0) / 42.0))

        # 2. Drought factor: Days without rain (0 to 24 days)
        f_rain = max(0.0, min(1.0, dwr / 20.0))

        # 3. Atmospheric drying: VPD (1.0 to 4.0 kPa)
        f_vpd = max(0.0, min(1.0, (vpd - 1.2) / 2.8))

        # 4. Drainage canal proximity (closer canal = increased drained dome subsidence)
        f_canal = max(0.0, min(1.0, (1800.0 - dist_canal) / 1800.0))

        # 5. Road proximity (anthropogenic human access / slash-and-burn)
        f_road = max(0.0, min(1.0, (3000.0 - dist_road) / 3000.0))

        # 6. Peat depth (deep peat domes store higher combustible fuel load)
        f_depth = max(0.0, min(1.0, depth / 4.5))

        # Core weighted combination
        base_score = (
            (0.42 * f_gwl) +
            (0.22 * f_rain) +
            (0.18 * f_vpd) +
            (0.08 * f_canal) +
            (0.06 * f_road) +
            (0.04 * f_depth)
        )

        # Non-linear synergy: when GWL drops beyond critical PP 57/2016 threshold (< -40cm) AND dwr > 12 days
        if gwl < -40.0 and dwr >= 12:
            base_score = min(1.0, base_score * 1.15)

        return round(float(np.clip(base_score, 0.010, 0.995)), 3)

    def _classify_category(self, score):
        if score >= 0.750:
            return "EXTREME"
        elif score >= 0.550:
            return "HIGH"
        elif score >= 0.300:
            return "MEDIUM"
        else:
            return "LOW"
