import time
from typing import List, Optional, Tuple, Dict
import torch

from message_type import FLMessage, GRPCMessage
from comm_manager import GRPCCommManager
from aggregator import Aggregator
import ip_utils
from mylogger import logger


class FLServer:
    def __init__(
        self,
        seed: int,
        model: torch.nn.Module,
        train_dataset: List[Tuple[torch.Tensor, int]],
        test_dataset: List[Tuple[torch.Tensor, int]],
        n_silos: int,
        device: str,
        n_total_round: int,
        n_silo_per_round: int,
        silo_client_id_mapping: Dict[int, int],
        agg_strategy: str,
        clipping_bound: float = None,
        sigma: float = None,
        delta: float = None,
        dataset_name: Optional[str] = None,
    ):
        aggregator = Aggregator(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            n_silos=n_silos,
            n_silo_per_round=n_silo_per_round,
            device=device,
            base_seed=seed,
            strategy=agg_strategy,
            clipping_bound=clipping_bound,
            sigma=sigma,
            delta=delta,
            dataset_name=dataset_name,
        )

        self.server_manger = ServerManager(
            aggregator,
            n_total_round,
            n_silo_per_round,
            silo_client_id_mapping,
        )

    def run(self):
        self.server_manger.run()

    def get_results(self) -> dict:
        results = dict()
        results["global_comm"] = self.server_manger.aggregator.get_comm_results()
        return results


class ServerManager(GRPCCommManager):
    ONLINE_STATUS_FLAG = "ONLINE"
    RUN_FINISHED_STATUS_FLAG = "FINISHED"

    def __init__(
        self,
        aggregator,
        n_total_round: int,
        n_silo_per_round: int,
        silo_client_id_mapping: dict[int, int],
    ):
        client_ids = list(silo_client_id_mapping.values())
        super().__init__(
            host=ip_utils.LOCAL_HOST_IP,
            port=ip_utils.resolve_port_from_receiver_id(ip_utils.AGGREGATION_SERVER_ID),
            client_id=ip_utils.AGGREGATION_SERVER_ID,
            comm_nodes=client_ids,
            ip_config_map=None,
        )
        self.round_idx = 0
        self.n_silo_per_round = n_silo_per_round
        self.aggregator: Aggregator = aggregator
        self.n_total_round = n_total_round
        self.client_online_mapping = {}
        self.client_finished_mapping = {}

        self.is_initialized = False
        self.data_silo_id_list = None
        self.silo_client_id_mapping = silo_client_id_mapping
        self.client_silo_id_mapping = {
            client_id: silo_id for silo_id, client_id in silo_client_id_mapping.items()
        }

    def get_sender_id(self):
        return self.client_id

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        super().register_message_receive_handler(
            FLMessage.MSG_TYPE_CONNECTION_IS_READY, self.handle_message_connection_ready
        )

        super().register_message_receive_handler(
            FLMessage.MSG_TYPE_C2S_CLIENT_STATUS,
            self.handle_message_client_status_update,
        )

        super().register_message_receive_handler(
            FLMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.handle_message_receive_model_from_client,
        )

    def handle_message_connection_ready(self, msg_params):
        if not self.is_initialized:
            logger.debug("Start the round 0 and check client status...")
            self.silo_id_list_in_this_round = self.aggregator.silo_selection()

            # check silo status in case that some silos start earlier than the server
            for silo_id in self.silo_id_list_in_this_round:
                try:
                    self.send_message_check_client_status(
                        self.silo_client_id_mapping[silo_id],
                        silo_id,
                    )
                    logger.debug(
                        "Connection ready for client"
                        + str(self.silo_client_id_mapping[silo_id])
                    )
                except Exception as e:
                    logger.debug(
                        str(e)
                        + ": Connection not ready for client"
                        + str(self.silo_client_id_mapping[silo_id])
                    )

    def handle_message_client_status_update(self, msg_params):
        client_status = msg_params.get(FLMessage.MSG_ARG_KEY_CLIENT_STATUS)
        logger.debug(
            f"received client status {client_status} from client_id = {msg_params.get_sender_id()}"
        )
        if client_status == ServerManager.ONLINE_STATUS_FLAG:
            self.process_online_status(client_status, msg_params)
        elif client_status == ServerManager.RUN_FINISHED_STATUS_FLAG:
            self.process_finished_status(client_status, msg_params)

    def send_message_check_client_status(self, receive_id, silo_id):
        message = GRPCMessage(
            FLMessage.MSG_TYPE_S2C_CHECK_CLIENT_STATUS, self.get_sender_id(), receive_id
        )
        message.add_params(FLMessage.MSG_ARG_KEY_SILO_ID, str(silo_id))
        super().send_message(message)

    def process_online_status(self, client_status, msg_params):
        if self.client_online_mapping.get(str(msg_params.get_sender_id())) is None:
            self.client_online_mapping[str(msg_params.get_sender_id())] = True
        else:
            logger.error("client is duplicated, check the specified silo_id")
            super().finish()
            raise Exception("client is duplicated, check the specified silo_id")

        logger.debug(
            "self.client_online_mapping = {}".format(self.client_online_mapping)
        )

        # 最初のみ，silo_sectionの結果に関わらず全員チェックする
        all_client_is_online = True
        for silo_id, client_id in self.silo_client_id_mapping.items():
            if not self.client_online_mapping.get(str(client_id), False):
                all_client_is_online = False
                break

        logger.debug(
            "sender_id = %d, all_client_is_online = %s"
            % (msg_params.get_sender_id(), str(all_client_is_online))
        )

        if all_client_is_online:
            # send initialization message to all clients to start training
            self.send_init_msg()
            self.is_initialized = True

    def process_finished_status(self, client_status, msg_params):
        self.client_finished_mapping[str(msg_params.get_sender_id())] = True

        all_client_is_finished = True
        for silo_id in self.silo_id_list_in_this_round:
            client_id = self.silo_client_id_mapping[silo_id]
            if not self.client_finished_mapping.get(str(client_id), False):
                all_client_is_finished = False
                break

        logger.debug(
            "sender_id = %d, all_client_is_finished = %s"
            % (msg_params.get_sender_id(), str(all_client_is_finished))
        )

        if all_client_is_finished:
            time.sleep(5)
            super().finish()

    def send_init_msg(self):
        global_model_params = self.aggregator.get_global_model_params()
        logger.debug("Start global FL round 0...")
        for silo_id in self.silo_id_list_in_this_round:
            client_id = self.silo_client_id_mapping[silo_id]
            self.send_message_init_config(
                client_id, global_model_params, silo_id, round_idx=0
            )

    def send_message_init_config(
        self,
        receive_id,
        global_model_params,
        silo_id,
        round_idx,
    ):
        message = GRPCMessage(
            FLMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id
        )
        message.add_params(FLMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(FLMessage.MSG_ARG_KEY_SILO_ID, str(silo_id))
        message.add_params(FLMessage.MSG_ARG_KEY_ROUND_IDX, round_idx)
        super().send_message(message)

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(FLMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(FLMessage.MSG_ARG_KEY_MODEL_PARAMS)
        n_local_sample = msg_params.get(FLMessage.MSG_ARG_KEY_NUM_SAMPLES)
        eps = msg_params.get(FLMessage.MSG_ARG_KEY_EPSILON)

        self.aggregator.add_local_trained_result(
            self.client_silo_id_mapping[sender_id], model_params, n_local_sample, eps
        )
        b_all_received = self.aggregator.check_whether_all_receive(
            self.silo_id_list_in_this_round
        )
        logger.debug("b_all_received = " + str(b_all_received))
        if b_all_received:
            logger.debug("all received, start aggregate for round %d" % self.round_idx)
            global_model_params = self.aggregator.aggregate(
                self.silo_id_list_in_this_round,
                self.round_idx,
            )

            self.aggregator.test_global(self.round_idx)
            logger.info(
                "\n\n========== end {}-th round training ===========\n".format(
                    self.round_idx
                )
            )
            self.round_idx += 1

            if self.round_idx == self.n_total_round:
                logger.debug("all round finished, send finish message to all silos")
                self.send_finish_message_to_all_client()
                super().finish()
                return

            logger.debug("Start global FL round {}...".format(self.round_idx))

            self.silo_id_list_in_this_round = self.aggregator.silo_selection()

            for silo_id in self.silo_id_list_in_this_round:
                receiver_id = self.silo_client_id_mapping[silo_id]
                self.send_message_sync_model_to_client(
                    receiver_id,
                    global_model_params,
                    silo_id,
                    self.round_idx,
                )

    def send_finish_message_to_all_client(self):
        for client_id in self.silo_client_id_mapping.values():
            self.send_finish_message(client_id)

    def send_message_sync_model_to_client(
        self,
        receive_id,
        global_model_params,
        silo_id,
        round_idx,
    ):
        logger.debug("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = GRPCMessage(
            FLMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(FLMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(FLMessage.MSG_ARG_KEY_SILO_ID, str(silo_id))
        message.add_params(FLMessage.MSG_ARG_KEY_ROUND_IDX, round_idx)
        self.send_message(message)

    def send_finish_message(
        self,
        receive_id,
    ):
        logger.debug("send_finish_message. receive_id = %d" % receive_id)
        message = GRPCMessage(
            FLMessage.MSG_TYPE_S2C_FINISH,
            self.get_sender_id(),
            receive_id,
        )
        self.send_message(message)
