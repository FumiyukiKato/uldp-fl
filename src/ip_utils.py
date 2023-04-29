AGGREGATION_SERVER_ID = 0
AGGREGATION_SERVER_IP = "127.0.0.1"
DEFAULT_NODE_IP = "127.0.0.1"

GRPC_BASE_PORT = 8890
LOCAL_HOST_IP = "0.0.0.0"


def build_ip_table(node_ids: list[int], ip_config_map: dict[int, str] = None):
    ip_table = dict()
    ip_table[AGGREGATION_SERVER_ID] = AGGREGATION_SERVER_IP
    if ip_config_map:
        for node_id in node_ids:
            ip_table[node_id] = ip_config_map[node_id]
        return ip_table
    for node_id in node_ids:
        ip_table[node_id] = DEFAULT_NODE_IP
    return ip_table


def resolve_port_from_receiver_id(receiver_id: int):
    return GRPC_BASE_PORT + receiver_id


def create_silo_client_id_mapping(n_silos: int):
    silo_client_id_mapping = {}
    for silo_id in range(n_silos):
        silo_client_id_mapping[silo_id] = silo_id + 1
    return silo_client_id_mapping
