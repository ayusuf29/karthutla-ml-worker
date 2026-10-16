import sys
import os
import grpc

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import fire_risk_service_pb2
import fire_risk_service_pb2_grpc

def test_inference(port=None):
    if port is None:
        port = int(os.environ.get("ML_PORT", "50052"))
    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = fire_risk_service_pb2_grpc.FireRiskInferenceServiceStub(channel)

    # 1. Test BatchPredictIgnitionRisk
    request = fire_risk_service_pb2.BatchRiskRequest(region_id="KUBU_RAYA_TEST")
    
    # Add a normal cell
    f1 = request.features.add()
    f1.h3_index = "888c95c325fffff"
    f1.peat_depth_meters = 2.5
    f1.ground_water_level_cm = -20.0
    f1.days_without_rain = 3
    f1.vapor_pressure_deficit = 1.4
    f1.distance_to_road_meters = 2000.0
    f1.distance_to_canal_meters = 1200.0

    # Add a critical EXTREME cell (GWL < -45 cm, drought 20 days)
    f2 = request.features.add()
    f2.h3_index = "888c95d881fffff"
    f2.peat_depth_meters = 4.2
    f2.ground_water_level_cm = -55.0
    f2.days_without_rain = 22
    f2.vapor_pressure_deficit = 3.8
    f2.distance_to_road_meters = 350.0
    f2.distance_to_canal_meters = 150.0

    print("Sending batch risk request with 2 cells...")
    response = stub.BatchPredictIgnitionRisk(request)
    print(f"Batch prediction completed in {response.compute_time_ms} ms on {response.accelerator_hardware}")
    for res in response.results:
        print(f"  Cell {res.h3_index}: Score={res.risk_score:.3f}, Category={res.risk_category}, SAR Alert={res.sar_anomaly_flag}")

    # 2. Test AnalyzeSarClearingAnomaly
    sar_req = fire_risk_service_pb2.SarAnomalyRequest(
        h3_index="888c95d881fffff",
        pre_event_sar_tile=b"fake_tile_pre_data",
        post_event_sar_tile=b"fake_tile_post_data_with_anomaly_delta"
    )
    print("\nSending SAR clearing anomaly request...")
    sar_resp = stub.AnalyzeSarClearingAnomaly(sar_req)
    print(f"  SAR Result for {sar_resp.h3_index}: Clearing Detected={sar_resp.clearing_detected}, Confidence={sar_resp.confidence:.2f}")

    print("\n[SUCCESS] gRPC ML Microservice test passed!")

if __name__ == "__main__":
    test_inference()
