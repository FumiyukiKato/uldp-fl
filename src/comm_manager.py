from abc import abstractmethod
import pickle
import time
import grpc
from concurrent import futures
import threading
import queue

from acsilo_grpc import aggregation_server_pb2_grpc, aggregation_server_pb2
import ip_utils
from message_type import FLMessage, GRPCMessage
from mylogger import logger

lock = threading.Lock()


class GRPCCommManager:
    def __init__(
        self,
        host,
        port,
        client_id,
        comm_nodes,
        ip_config_map=None,
    ):
        # host is the ip address of server
        self.host = host
        self.port = str(port)
        self.client_id = client_id
        self._observers: list[GRPCCommManager] = []
        self.message_handler_dict = dict()

        if client_id == ip_utils.AGGREGATION_SERVER_ID:
            self.node_type = "server"
            logger.info("############# THIS IS FL SERVER ################")
            max_workers = len(comm_nodes)
            self.ip_config = ip_utils.build_ip_table(comm_nodes, ip_config_map)
        else:
            self.node_type = "client"
            logger.info("------------- THIS IS FL CLIENT ----------------")
            max_workers = 1
            # client should know own ip address and server's ip address
            self.ip_config = ip_utils.build_ip_table([self.client_id], ip_config_map)
        self.opts = [
            ("grpc.max_send_message_length", 1000 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1000 * 1024 * 1024),
            ("grpc.enable_http_proxy", 0),
        ]
        self.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=self.opts,
        )
        self.grpc_servicer = GRPCCommServicer(host, port, client_id)
        aggregation_server_pb2_grpc.add_gRPCCommManagerServicer_to_server(
            self.grpc_servicer, self.grpc_server
        )

        # starts a grpc_server on local machine using ip address "0.0.0.0"
        self.grpc_server.add_insecure_port("{}:{}".format(ip_utils.LOCAL_HOST_IP, port))

        self.grpc_server.start()
        self.is_running = True
        logger.info("Grpc server started. Listening on port " + str(port))

    def run(self):
        self.register_message_receive_handlers()
        self.handle_receive_message()
        logger.info("Finished...")

    @abstractmethod
    def register_message_receive_handlers(self) -> None:
        pass

    def register_message_receive_handler(self, msg_type, handler_callback_func):
        self.message_handler_dict[msg_type] = handler_callback_func

    def receive_message(self, msg_type, msg_params) -> None:
        if msg_params.get_sender_id() == msg_params.get_receiver_id():
            logger.debug(
                "communication backend is alive (loop_forever, sender 0 to receiver 0)"
            )
        else:
            logger.debug(
                "receive_message. msg_type = %s, sender_id = %d, receiver_id = %d"
                % (
                    str(msg_type),
                    msg_params.get_sender_id(),
                    msg_params.get_receiver_id(),
                )
            )
        try:
            handler_callback_func = self.message_handler_dict[msg_type]
            handler_callback_func(msg_params)
        except KeyError:
            raise Exception(
                "KeyError. msg_type = {}. Not found callback.".format(msg_type)
            )

    def send_message(self, msg: GRPCMessage):
        logger.debug("msg = {}".format(msg))
        # payload = msg.to_json()

        logger.debug("pickle.dumps(msg) START")
        pickle_dump_start_time = time.time()
        msg_pkl = pickle.dumps(msg)
        logger.debug(f"PickleDumpsTime: {time.time() - pickle_dump_start_time}")
        logger.debug("Pickled Message size: {}".format(len(msg_pkl)))
        logger.debug("pickle.dumps(msg) END")

        receiver_id = msg.get_receiver_id()
        # lookup ip of receiver from self.ip_config table
        receiver_ip = self.ip_config[receiver_id]
        channel_url = "{}:{}".format(
            receiver_ip, ip_utils.resolve_port_from_receiver_id(receiver_id)
        )

        channel = grpc.insecure_channel(channel_url, options=self.opts)
        stub = aggregation_server_pb2_grpc.gRPCCommManagerStub(channel)

        request = aggregation_server_pb2.CommRequest()
        logger.debug(
            "sending message to {} (receiver id {})".format(channel_url, receiver_id)
        )

        request.client_id = self.client_id

        request.message = msg_pkl

        stub.sendMessage(request)
        logger.debug("sent successfully")
        channel.close()

    def handle_receive_message(self):
        self._notify_connection_ready()
        self.message_handling_subroutine()

        # Cannont run message_handling_subroutine in new thread
        # Related https://stackoverflow.com/a/70705165

        # thread = threading.Thread(target=self.message_handling_subroutine)
        # thread.start()

    def message_handling_subroutine(self):
        while self.is_running:
            if self.grpc_servicer.message_q.qsize() > 0:
                lock.acquire()
                msg_pkl = self.grpc_servicer.message_q.get()
                msg = pickle.loads(msg_pkl)
                msg_type = msg.get_type()
                self.receive_message(msg_type, msg)
                lock.release()
            time.sleep(0.001)
        return

    def finish(self):
        logger.info("__finish")
        self.stop_receive_message()

    def stop_receive_message(self):
        self.grpc_server.stop(None)
        self.is_running = False

    def _notify_connection_ready(self):
        msg_params = GRPCMessage()
        msg_params.sender_id = self.client_id
        msg_params.receiver_id = self.client_id
        msg_type = FLMessage.MSG_TYPE_CONNECTION_IS_READY
        self.receive_message(msg_type, msg_params)


# Call Protobuf functions
class GRPCCommServicer(aggregation_server_pb2_grpc.gRPCCommManagerServicer):
    def __init__(self, host, port, client_id):
        self.host = host
        self.port = port
        self.client_id = client_id

        if self.client_id == ip_utils.AGGREGATION_SERVER_ID:
            self.node_type = "server"
        else:
            self.node_type = "client"

        self.message_q = queue.Queue()

    def sendMessage(self, request, context):
        context_ip = context.peer().split(":")[1]
        logger.debug(
            "client_{} got something from client_{} from ip address {}".format(
                self.client_id, request.client_id, context_ip
            )
        )

        response = aggregation_server_pb2.CommResponse()
        lock.acquire()
        self.message_q.put(request.message)
        lock.release()
        return response

    def handleReceiveMessage(self, request, context):
        pass
