# usage: python acsilo_grpc/code_gen.py

from grpc_tools import protoc

protoc.main(
    (
        "",
        "-I..",
        "--python_out=.",
        "--grpc_python_out=.",
        "acsilo_grpc/aggregation_server.proto",
    )
)
