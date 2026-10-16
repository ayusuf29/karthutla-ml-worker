import os
import sys
import time
import logging
from concurrent import futures
import grpc

# Add local directory to path for generated protobuf imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import fire_risk_service_pb2
import fire_risk_service_pb2_grpc
from risk_model import PeatFireRiskModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MLWorker")


class FireRiskInferenceServiceServicer(fire_risk_service_pb2_grpc.FireRiskInferenceServiceServicer):

    def __init__(self):
        self.model = PeatFireRiskModel()
        logger.info("PeatFireRiskModel initialized on CPU.")

    def BatchPredictIgnitionRisk(self, request, context):
        start_time = time.perf_counter()
        region_id = getattr(request, 'region_id', 'UNKNOWN')
        features = request.features

        logger.info(
            "Received BatchPredictIgnitionRisk for region '%s' with %d H3 hexagons",
            region_id, len(features)
        )

        # Execute CPU-based inference scoring
        inference_results = self.model.predict_batch(features)

        # Build Protobuf response
        response = fire_risk_service_pb2.BatchRiskResponse()
        for item in inference_results:
            risk_item = response.results.add()
            risk_item.h3_index = item["h3_index"]
            risk_item.risk_score = item["risk_score"]
            risk_item.risk_category = item["risk_category"]
            risk_item.sar_anomaly_flag = item["sar_anomaly_flag"]

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        response.compute_time_ms = elapsed_ms
        response.accelerator_hardware = self.model.hardware_descriptor

        logger.info(
            "Completed batch inference for %d cells in %d ms (%s)",
            len(features), elapsed_ms, response.accelerator_hardware
        )
        return response

    def AnalyzeSarClearingAnomaly(self, request, context):
        h3_index = request.h3_index
        pre_bytes = len(request.pre_event_sar_tile) if request.pre_event_sar_tile else 0
        post_bytes = len(request.post_event_sar_tile) if request.post_event_sar_tile else 0

        logger.info(
            "Received AnalyzeSarClearingAnomaly for H3 cell '%s' (pre: %d bytes, post: %d bytes)",
            h3_index, pre_bytes, post_bytes
        )

        # Heuristic mock anomaly detection:
        # In actual Sentinel-1 pipeline, this computes backscatter ratio (Sigma0 VV/VH decibel difference)
        is_clearing = (hash(h3_index) % 5 == 0) or (abs(post_bytes - pre_bytes) > 100)
        confidence = 0.94 if is_clearing else 0.15

        response = fire_risk_service_pb2.SarAnomalyResponse()
        response.h3_index = h3_index
        response.clearing_detected = is_clearing
        response.confidence = confidence

        logger.info(
            "SAR Anomaly Result for %s: clearing_detected=%s, confidence=%.2f",
            h3_index, is_clearing, confidence
        )
        return response


from google.protobuf import json_format

def make_dual_handler(servicer_method, req_cls, resp_cls):
    def handler(request_bytes, context):
        is_json = request_bytes and request_bytes.lstrip().startswith(b'{')
        if is_json:
            req = req_cls()
            json_format.Parse(request_bytes.decode('utf-8'), req, ignore_unknown_fields=True)
        else:
            req = req_cls.FromString(request_bytes)

        resp = servicer_method(req, context)

        if is_json:
            return json_format.MessageToJson(resp, preserving_proto_field_name=True).encode('utf-8')
        else:
            return resp.SerializeToString()

    return grpc.unary_unary_rpc_method_handler(
        handler,
        request_deserializer=lambda b: b,
        response_serializer=lambda b: b
    )

def serve(port=50051):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )
    servicer = FireRiskInferenceServiceServicer()
    rpc_method_handlers = {
        'BatchPredictIgnitionRisk': make_dual_handler(
            servicer.BatchPredictIgnitionRisk,
            fire_risk_service_pb2.BatchRiskRequest,
            fire_risk_service_pb2.BatchRiskResponse
        ),
        'AnalyzeSarClearingAnomaly': make_dual_handler(
            servicer.AnalyzeSarClearingAnomaly,
            fire_risk_service_pb2.SarAnomalyRequest,
            fire_risk_service_pb2.SarAnomalyResponse
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'sentinel.ml.FireRiskInferenceService', rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))

    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    server.start()

    logger.info("=========================================================")
    logger.info("Nusantara Fire Sentinel - ML Inference Worker (CPU Mode)")
    logger.info("Listening on gRPC port: %d (%s)", port, listen_addr)
    logger.info("Target: Localhost Laptop Environment (No GPU required)")
    logger.info("=========================================================")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        server.stop(grace=2)


if __name__ == "__main__":
    port = int(os.environ.get("GRPC_PORT", 50051))
    serve(port)
