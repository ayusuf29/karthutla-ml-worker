import os
import sys
import re
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent
    local_proto = base_dir / "proto"
    proto_dir = local_proto if local_proto.exists() else base_dir.parent / "proto"
    proto_file = proto_dir / "fire_risk_service.proto"

    if not proto_file.exists():
        print(f"Error: Proto file not found at {proto_file}")
        sys.exit(1)

    print(f"Compiling {proto_file} into {base_dir}...")
    
    # Run grpc_tools.protoc
    import grpc_tools.protoc
    res = grpc_tools.protoc.main([
        "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={base_dir}",
        f"--grpc_python_out={base_dir}",
        str(proto_file)
    ])

    if res != 0:
        print(f"Compilation failed with exit code: {res}")
        sys.exit(res)

    print("Successfully generated fire_risk_service_pb2.py and fire_risk_service_pb2_grpc.py")

    # Patch import in fire_risk_service_pb2_grpc.py for seamless local import
    grpc_out = base_dir / "fire_risk_service_pb2_grpc.py"
    if grpc_out.exists():
        content = grpc_out.read_text(encoding="utf-8")
        # Ensure import works whether imported from package or root
        patched = re.sub(
            r"import fire_risk_service_pb2 as fire__risk__service__pb2",
            r"try:\n    import fire_risk_service_pb2 as fire__risk__service__pb2\nexcept ImportError:\n    from . import fire_risk_service_pb2 as fire__risk__service__pb2",
            content
        )
        grpc_out.write_text(patched, encoding="utf-8")
        print("Patched grpc import fallback in fire_risk_service_pb2_grpc.py")

if __name__ == "__main__":
    main()
