import time
from typing import List, Optional, Tuple, Dict
import torch
import copy
from coordinator import Coordinator
from dataset import TCGA_BRCA

from message_type import FLMessage, GRPCMessage
from comm_manager import GRPCCommManager
from aggregator import Aggregator
import ip_utils
from method_group import (
    METHOD_GROUP_ENHANCED_WEIGHTED_WITH_SAMPLING,
    METHOD_GROUP_ENHANCED_WEIGHTED_WITHOUT_SAMPLING,
    METHOD_GROUP_SAMPLING_WITH_ENHANCED_WEIGHTING,
    METHOD_GROUP_SAMPLING_WITHOUT_ENHANCED_WEIGHTING,
    METHOD_GROUP_NO_WEIGHTS,
    METHOD_GROUP_SECURE_WEIGHTING,
    METHOD_GROUP_ULDP_GROUPS,
    METHOD_GROUP_WEIGHTS,
)
from mylogger import logger
from secure_aggregation import SecureAggregator


class FLServer:
    def __init__(
        self,
        seed: int,
        model: torch.nn.Module,
        train_dataset: List[Tuple[torch.Tensor, int]],
        test_dataset: List[Tuple[torch.Tensor, int]],
        n_users: int,
        n_silos: int,
        device: str,
        n_total_round: int,
        n_silo_per_round: int,
        silo_client_id_mapping: Dict[int, int],
        agg_strategy: str,
        clipping_bound: float = None,
        sigma: float = None,
        delta: float = None,
        global_learning_rate: float = None,
        sampling_rate_q: float = None,
        dataset_name: Optional[str] = None,
        group_k: Optional[int] = None,
        is_secure: bool = False,
    ):
        if is_secure:
            if agg_strategy not in METHOD_GROUP_SECURE_WEIGHTING:
                raise ValueError(f"agg_strategy {agg_strategy} is not supported.")

            aggregator = SecureAggregator(
                model=copy.deepcopy(model),
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                n_users=n_users,
                n_silos=n_silos,
                n_silo_per_round=n_silo_per_round,
                device=device,
                base_seed=seed,
                strategy=agg_strategy,
                clipping_bound=clipping_bound,
                sigma=sigma,
                delta=delta,
                global_learning_rate=global_learning_rate,
                dataset_name=dataset_name,
                sampling_rate_q=sampling_rate_q,
            )
        else:
            aggregator = Aggregator(
                model=copy.deepcopy(model),
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                n_users=n_users,
                n_silos=n_silos,
                n_silo_per_round=n_silo_per_round,
                device=device,
                base_seed=seed,
                strategy=agg_strategy,
                clipping_bound=clipping_bound,
                sigma=sigma,
                delta=delta,
                global_learning_rate=global_learning_rate,
                dataset_name=dataset_name,
                sampling_rate_q=sampling_rate_q,
            )

        coordinator = Coordinator(
            base_seed=seed,
            n_silos=n_silos,
            n_users=n_users,
            group_k=group_k,
            sampling_rate_q=sampling_rate_q,
            agg_strategy=agg_strategy,
        )

        self.server_manger = ServerManager(
            aggregator,
            coordinator,
            n_total_round,
            n_silo_per_round,
            silo_client_id_mapping,
            is_secure,
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
        aggregator: Aggregator,
        coordinator: Coordinator,
        n_total_round: int,
        n_silo_per_round: int,
        silo_client_id_mapping: dict[int, int],
        is_secure: bool = False,
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
        self.aggregator: SecureAggregator = aggregator
        self.coordinator = coordinator
        self.n_total_round = n_total_round
        self.is_secure = is_secure
        self.client_online_mapping = {}
        self.client_finished_mapping = {}

        self.is_initialized = False
        self.data_silo_id_list = None
        self.silo_client_id_mapping = silo_client_id_mapping
        self.client_silo_id_mapping = {
            client_id: silo_id for silo_id, client_id in silo_client_id_mapping.items()
        }
        self.ready_set = set()

    def get_sender_id(self):
        return self.client_id

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        super().register_message_receive_handler(
            FLMessage.MSG_TYPE_CONNECTION_IS_READY, self.handle_message_connection_ready
        )

        super().register_message_receive_handler(
            FLMessage.MSG_TYPE_C2S_USER_HISTOGRAM,
            self.handle_message_receive_user_hist,
        )

        super().register_message_receive_handler(
            FLMessage.MSG_TYPE_C2S_COMPLETE_PREPARATION,
            self.handle_message_receive_complete_preparation,
        )

        super().register_message_receive_handler(
            FLMessage.MSG_TYPE_C2S_DH_KEY_EXCHANGE,
            self.handle_message_dh_key_exchange,
        )

        super().register_message_receive_handler(
            FLMessage.MSG_TYPE_C2S_SHARED_RANDOM_SEEDS,
            self.handle_message_shared_random_seeds,
        )

        super().register_message_receive_handler(
            FLMessage.MSG_TYPE_C2S_BLINDED_USER_HISTOGRAM,
            self.handle_message_blinded_user_histogram,
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
            for silo_id, client_id in self.silo_client_id_mapping.items():
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
            self.is_initialized = True
            # ----------- Before starting the training round ------------
            if self.is_secure:
                self.broadcast_start_secure_protocol()
            else:
                self.send_init_user_histogram_msg()

    def handle_message_receive_user_hist(self, msg_params):
        silo_id = msg_params.get(FLMessage.MSG_ARG_KEY_SILO_ID)
        user_hist = msg_params.get(FLMessage.MSG_ARG_KEY_USER_HIST)

        self.coordinator.set_user_hist_by_silo_id(silo_id, user_hist)

        if self.coordinator.is_ready():
            self.send_init_user_weights()

    def handle_message_dh_key_exchange(self, msg_params):
        silo_id = msg_params.get(FLMessage.MSG_ARG_KEY_SILO_ID)
        dh_pubkey = msg_params.get(FLMessage.MSG_ARG_KEY_DH_PUBKEY)

        client_keys = self.aggregator.receive_dh_pubkey(silo_id, dh_pubkey)
        if client_keys is not None:
            self.broadcast_keys_to_clients()

    def handle_message_shared_random_seeds(self, msg_params):
        shared_random_seed_per_silo = msg_params.get(
            FLMessage.MSG_ARG_KEY_SHARED_RANDOM_SEEDS
        )
        self.broadcast_shared_random_seed(shared_random_seed_per_silo)

    def handle_message_blinded_user_histogram(self, msg_params):
        silo_id = msg_params.get(FLMessage.MSG_ARG_KEY_SILO_ID)
        blinded_user_histogram = msg_params.get(
            FLMessage.MSG_ARG_KEY_BLINDED_USER_HISTOGRAM
        )
        inversed_blinded_user_histogram = (
            self.aggregator.receive_blinded_user_histogram(
                silo_id, blinded_user_histogram
            )
        )

        if inversed_blinded_user_histogram is not None:
            if (
                self.aggregator.strategy
                in METHOD_GROUP_ENHANCED_WEIGHTED_WITHOUT_SAMPLING
            ):
                encrypted_inversed_blinded_user_histogram = (
                    self.aggregator.get_encrypt_inversed_blinded_user_histogram()
                )
                self.broadcast_encrypted_weights(
                    encrypted_inversed_blinded_user_histogram
                )
            else:
                global_model_params = self.aggregator.get_global_model_params()
                self._run_training_round(
                    round_idx=0, global_model_params=global_model_params
                )

    def handle_message_receive_complete_preparation(self, msg_params):
        silo_id = msg_params.get(FLMessage.MSG_ARG_KEY_SILO_ID)
        self.ready_set.add(silo_id)

        if len(self.ready_set) == self.coordinator.n_silos:
            global_model_params = self.aggregator.get_global_model_params()
            self._run_training_round(
                round_idx=0, global_model_params=global_model_params
            )

    def process_finished_status(self, client_status, msg_params):
        self.client_finished_mapping[str(msg_params.get_sender_id())] = True

        all_client_is_finished = True
        for silo_id, client_id in self.silo_client_id_mapping.items():
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

    def send_init_user_histogram_msg(self):
        logger.debug("Start to collect user histogram...")
        for silo_id, client_id in self.silo_client_id_mapping.items():
            message = GRPCMessage(
                FLMessage.MSG_TYPE_S2C_USER_HISTOGRAM, self.get_sender_id(), client_id
            )
            message.add_params(FLMessage.MSG_ARG_KEY_SILO_ID, str(silo_id))
            super().send_message(message)

    def send_init_user_weights(self):
        user_weights_per_silo = {}
        if self.aggregator.strategy in METHOD_GROUP_ULDP_GROUPS:
            if self.aggregator.dataset_name == TCGA_BRCA:
                # To calculate Cox-loss for TCGA-BRCA, we need at least 2 samples per user
                min_count = 2
            else:
                min_count = 1
            bounded_user_hist_per_silo = self.coordinator.build_user_bound_histograms(
                self.coordinator.original_user_hist_dct, min_count
            )
            user_weights_per_silo = bounded_user_hist_per_silo
        elif self.aggregator.strategy in METHOD_GROUP_NO_WEIGHTS:
            user_weights_per_silo = self.coordinator.build_user_weights(weighted=False)
        elif self.aggregator.strategy in METHOD_GROUP_WEIGHTS:
            user_weights_per_silo = self.coordinator.build_user_weights(weighted=True)

        for silo_id, client_id in self.silo_client_id_mapping.items():
            self.send_message_init_config(
                client_id,
                user_weights_per_silo[silo_id],
                silo_id,
            )

    def send_message_init_config(
        self,
        receive_id,
        user_weights,
        silo_id,
    ):
        message = GRPCMessage(
            FLMessage.MSG_TYPE_S2C_TRAINING_PREPARATION,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(FLMessage.MSG_ARG_KEY_USER_WEIGHTS, user_weights)
        message.add_params(FLMessage.MSG_ARG_KEY_SILO_ID, str(silo_id))
        super().send_message(message)

    def _run_training_round(self, round_idx: int, global_model_params):
        logger.debug("Start global FL round {}...".format(round_idx))
        self.silo_id_list_in_this_round = self.aggregator.silo_selection()

        if self.is_secure:
            encrypted_inversed_blinded_user_histogram = None
            if self.aggregator.strategy in METHOD_GROUP_ENHANCED_WEIGHTED_WITH_SAMPLING:
                encrypted_inversed_blinded_user_histogram = self.aggregator.get_encrypt_inversed_blinded_user_histogram_with_userlevel_subsampling(
                    round_idx
                )
            for silo_id in self.silo_id_list_in_this_round:
                receiver_id = self.silo_client_id_mapping[silo_id]
                self.send_message_sync_model_to_client(
                    receiver_id,
                    global_model_params,
                    silo_id,
                    round_idx,
                    user_weights=encrypted_inversed_blinded_user_histogram,
                )
        else:
            user_weights_per_silo = {}
            if (
                self.aggregator.strategy
                in METHOD_GROUP_SAMPLING_WITHOUT_ENHANCED_WEIGHTING
            ):
                user_weights_per_silo = self.coordinator.build_user_weights(
                    weighted=False, is_sample=True
                )
            elif (
                self.aggregator.strategy
                in METHOD_GROUP_SAMPLING_WITH_ENHANCED_WEIGHTING
            ):
                user_weights_per_silo = self.coordinator.build_user_weights(
                    weighted=True, is_sample=True
                )

            for silo_id in self.silo_id_list_in_this_round:
                receiver_id = self.silo_client_id_mapping[silo_id]
                user_weights = user_weights_per_silo.get(silo_id)
                self.send_message_sync_model_to_client(
                    receiver_id,
                    global_model_params,
                    silo_id,
                    round_idx,
                    user_weights=user_weights,
                )

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(FLMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(FLMessage.MSG_ARG_KEY_MODEL_PARAMS)
        _ = msg_params.get(FLMessage.MSG_ARG_KEY_NUM_SAMPLES)
        eps = msg_params.get(FLMessage.MSG_ARG_KEY_EPSILON)

        self.aggregator.add_local_trained_result(
            self.client_silo_id_mapping[sender_id], model_params, eps
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

            self._run_training_round(self.round_idx, global_model_params)

    def send_finish_message_to_all_client(self):
        for client_id in self.silo_client_id_mapping.values():
            self.send_finish_message(client_id)

    def send_message_sync_model_to_client(
        self,
        receive_id,
        global_model_params,
        silo_id,
        round_idx,
        user_weights=None,
    ):
        logger.debug("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = GRPCMessage(
            FLMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(FLMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(FLMessage.MSG_ARG_KEY_SILO_ID, str(silo_id))
        if user_weights is not None:
            message.add_params(FLMessage.MSG_ARG_KEY_USER_WEIGHTS, user_weights)
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

    def broadcast_start_secure_protocol(self):
        for silo_id, client_id in self.silo_client_id_mapping.items():
            message = GRPCMessage(
                FLMessage.MSG_TYPE_S2C_START_SECURE_PROTOCOL,
                self.get_sender_id(),
                client_id,
            )
            message.add_params(FLMessage.MSG_ARG_KEY_SILO_ID, str(silo_id))
            super().send_message(message)

    def broadcast_keys_to_clients(self):
        for silo_id, client_id in self.silo_client_id_mapping.items():
            message = GRPCMessage(
                FLMessage.MSG_TYPE_S2C_KEY_DISTRIBUTION,
                self.get_sender_id(),
                client_id,
            )
            message.add_params(
                FLMessage.MSG_ARG_KEY_PAILLIER_PUBKEY,
                self.aggregator.paillier_public_key,
            )
            message.add_params(
                FLMessage.MSG_ARG_KEY_DH_PUBKEY_DCT, self.aggregator.client_keys
            )
            super().send_message(message)

    def broadcast_shared_random_seed(self, shared_random_seed_per_silo):
        for silo_id, client_id in self.silo_client_id_mapping.items():
            message = GRPCMessage(
                FLMessage.MSG_TYPE_S2C_SHARED_RANDOM_SEED,
                self.get_sender_id(),
                client_id,
            )
            message.add_params(
                FLMessage.MSG_ARG_KEY_SHARED_RANDOM_SEED,
                shared_random_seed_per_silo.get(silo_id),
            )
            super().send_message(message)

    def broadcast_encrypted_weights(self, encrypted_weights):
        for silo_id, client_id in self.silo_client_id_mapping.items():
            message = GRPCMessage(
                FLMessage.MSG_TYPE_S2C_ENCRYPTED_WEIGHTS,
                self.get_sender_id(),
                client_id,
            )
            message.add_params(
                FLMessage.MSG_ARG_KEY_ENCRYPTED_WEIGHTS, encrypted_weights
            )
            super().send_message(message)
