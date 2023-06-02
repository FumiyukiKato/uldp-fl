from typing import Dict, List, Optional, Tuple
import torch

from comm_manager import GRPCCommManager
from local_trainer import ClassificationTrainer
from message_type import FLMessage, GRPCMessage
import ip_utils
from mylogger import logger


class FLSilo:
    def __init__(
        self,
        seed: int,
        model: torch.nn.Module,
        agg_strategy: str,
        local_train_dataset: List[Tuple[torch.Tensor, int]],
        local_test_dataset: List[Tuple[torch.Tensor, int]],
        user_histogram: Optional[Dict[int, int]],
        user_ids_of_local_train_dataset: Optional[List[int]],
        local_learning_rate: float,
        local_batch_size: int,
        client_optimizer: str,
        local_epochs: int,
        device: str,
        silo_id: int,
        client_id: int,
        n_total_round: int,
        weight_decay: Optional[float] = None,
        local_sigma: Optional[float] = None,
        local_delta: Optional[float] = None,
        local_clipping_bound: Optional[float] = None,
        group_k: Optional[int] = None,
        user_weights: Optional[Dict[int, float]] = None,
        dataset_name: Optional[str] = None,
    ):
        local_trainer = ClassificationTrainer(
            base_seed=seed,
            model=model,
            silo_id=silo_id,
            device=device,
            agg_strategy=agg_strategy,
            local_train_dataset=local_train_dataset,
            local_test_dataset=local_test_dataset,
            user_histogram=user_histogram,
            user_ids_of_local_train_dataset=user_ids_of_local_train_dataset,
            client_optimizer=client_optimizer,
            local_learning_rate=local_learning_rate,
            local_batch_size=local_batch_size,
            weight_decay=weight_decay,
            local_epochs=local_epochs,
            local_sigma=local_sigma,
            local_delta=local_delta,
            local_clipping_bound=local_clipping_bound,
            group_k=group_k,
            user_weights=user_weights,
            dataset_name=dataset_name,
        )
        self.client_manager = SiloManager(
            local_trainer,
            n_total_round,
            silo_id,
            client_id,
        )

    def run(self):
        self.client_manager.run()

    def get_results(self):
        results = dict()
        results["local_comm"] = dict()
        results["local_comm"][
            self.client_manager.silo_id
        ] = self.client_manager.local_trainer.get_comm_results()
        return results


class SiloManager(GRPCCommManager):
    ONLINE_STATUS_FLAG = "ONLINE"
    RUN_FINISHED_STATUS_FLAG = "FINISHED"

    def __init__(
        self,
        local_trainer,
        n_total_round: int,
        silo_id: int,
        client_id: int,
    ):
        super().__init__(
            host=ip_utils.LOCAL_HOST_IP,
            port=ip_utils.resolve_port_from_receiver_id(client_id),
            client_id=client_id,
            comm_nodes=[ip_utils.AGGREGATION_SERVER_ID],
            ip_config_map=None,
        )
        self.local_trainer: ClassificationTrainer = local_trainer
        self.num_rounds = n_total_round
        self.local_round_idx = 0
        self.silo_id = silo_id

        self.has_sent_online_msg = False
        self.is_initialized = False

    def run(self):
        super().run()

    def __train(self, global_round_index: int):
        logger.info(
            "####### Training ########### global round = %d (local = %d) #######"
            % (global_round_index, self.local_round_idx)
        )
        weights, n_local_sample = self.local_trainer.train(global_round_index)
        cpu_weights = {k: v.to("cpu") for k, v in weights.items()}
        self.local_trainer.test_local(global_round_index)
        self.send_model_to_server(
            ip_utils.AGGREGATION_SERVER_ID,
            cpu_weights,
            n_local_sample,
            self.local_trainer.get_latest_epsilon(),
        )

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            FLMessage.MSG_TYPE_CONNECTION_IS_READY, self.handle_message_connection_ready
        )

        self.register_message_receive_handler(
            FLMessage.MSG_TYPE_S2C_CHECK_CLIENT_STATUS, self.handle_message_check_status
        )

        self.register_message_receive_handler(
            FLMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init
        )
        self.register_message_receive_handler(
            FLMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.handle_message_receive_model_from_server,
        )

        self.register_message_receive_handler(
            FLMessage.MSG_TYPE_S2C_FINISH,
            self.handle_message_finish,
        )

    def handle_message_connection_ready(self, msg_params):
        if not self.has_sent_online_msg:
            self.has_sent_online_msg = True
            self.send_client_status(ip_utils.AGGREGATION_SERVER_ID)

    def handle_message_check_status(self, msg_params):
        self.send_client_status(ip_utils.AGGREGATION_SERVER_ID)

    def handle_message_init(self, msg_params):
        if self.is_initialized:
            return

        self.is_initialized = True

        global_model_params = msg_params.get(FLMessage.MSG_ARG_KEY_MODEL_PARAMS)
        silo_id = msg_params.get(FLMessage.MSG_ARG_KEY_SILO_ID)
        logger.info("SILO ID = %s" % str(silo_id))

        self.local_trainer.set_model_params(global_model_params)
        self.local_round_idx = 0

        self.__train(global_round_index=0)
        self.local_round_idx += 1

    def handle_message_receive_model_from_server(self, msg_params):
        logger.debug("handle_message_receive_model_from_server.")
        model_params = msg_params.get(FLMessage.MSG_ARG_KEY_MODEL_PARAMS)
        global_round_idx = msg_params.get(FLMessage.MSG_ARG_KEY_ROUND_IDX)
        self.local_trainer.set_model_params(model_params)
        if global_round_idx < self.num_rounds:
            self.__train(global_round_idx)
            self.local_round_idx += 1
        else:
            logger.error("global round index is over the limit.")
            self.send_client_status(
                ip_utils.AGGREGATION_SERVER_ID, SiloManager.RUN_FINISHED_STATUS_FLAG
            )
            super().finish()

    def handle_message_finish(self, msg_params):
        logger.debug(" ==================== cleanup ====================")
        self.cleanup()

    def cleanup(self):
        self.finish()

    def send_client_status(self, receive_id, status=ONLINE_STATUS_FLAG):
        logger.debug("send_client_status")
        message = GRPCMessage(
            FLMessage.MSG_TYPE_C2S_CLIENT_STATUS, self.client_id, receive_id
        )

        message.add_params(FLMessage.MSG_ARG_KEY_CLIENT_STATUS, status)
        self.send_message(message)

    def send_model_to_server(self, receive_id, weights, n_local_sample, eps):
        message = GRPCMessage(
            FLMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.client_id,
            receive_id,
        )
        message.add_params(FLMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(FLMessage.MSG_ARG_KEY_NUM_SAMPLES, n_local_sample)
        message.add_params(FLMessage.MSG_ARG_KEY_EPSILON, eps)
        self.send_message(message)
