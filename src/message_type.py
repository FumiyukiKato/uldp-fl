import json
import sys

# The source code is based on a simplified version of FedML (https://github.com/FedML-AI/FedML).


class FLMessage(object):
    """
    message type definition
    """

    # connection info
    MSG_TYPE_CONNECTION_IS_READY = 0

    # server to client
    MSG_TYPE_S2C_INIT_CONFIG = 1
    MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT = 2
    MSG_TYPE_S2C_CHECK_CLIENT_STATUS = 6
    MSG_TYPE_S2C_FINISH = 7
    MSG_TYPE_S2C_USER_HISTOGRAM = 9
    MSG_TYPE_S2C_TRAINING_PREPARATION = 12
    MSG_TYPE_S2C_START_SECURE_PROTOCOL = 13
    MSG_TYPE_S2C_KEY_DISTRIBUTION = 15
    MSG_TYPE_S2C_SHARED_RANDOM_SEED = 17
    MSG_TYPE_S2C_ENCRYPTED_WEIGHTS = 19

    # client to server
    MSG_TYPE_C2S_SEND_MODEL_TO_SERVER = 3
    MSG_TYPE_C2S_SEND_STATS_TO_SERVER = 4
    MSG_TYPE_C2S_CLIENT_STATUS = 5
    MSG_TYPE_C2S_FINISHED = 8
    MSG_TYPE_C2S_USER_HISTOGRAM = 10
    MSG_TYPE_C2S_COMPLETE_PREPARATION = 11
    MSG_TYPE_C2S_DH_KEY_EXCHANGE = 14
    MSG_TYPE_C2S_SHARED_RANDOM_SEEDS = 16
    MSG_TYPE_C2S_BLINDED_USER_HISTOGRAM = 18

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_USER_WEIGHTS = "user_weights"
    MSG_ARG_KEY_USER_HIST = "user_hist"
    MSG_ARG_KEY_MODEL_PARAMS_URL = "model_params_url"
    MSG_ARG_KEY_MODEL_PARAMS_KEY = "model_params_key"
    MSG_ARG_KEY_SILO_ID = "silo_id"
    MSG_ARG_KEY_DH_PUBKEY = "dh_pubkey"
    MSG_ARG_KEY_DH_PUBKEY_DCT = "dh_pubkey_dct"
    MSG_ARG_KEY_PAILLIER_PUBKEY = "paillier_public_key"
    MSG_ARG_KEY_SHARED_RANDOM_SEEDS = "shared_random_seeds"
    MSG_ARG_KEY_SHARED_RANDOM_SEED = "shared_random_seed"
    MSG_ARG_KEY_BLINDED_USER_HISTOGRAM = "blinded_user_histogram"
    MSG_ARG_KEY_ENCRYPTED_WEIGHTS = "encrypted_weights"

    MSG_ARG_KEY_TRAIN_CORRECT = "train_correct"
    MSG_ARG_KEY_TRAIN_ERROR = "train_error"
    MSG_ARG_KEY_TRAIN_NUM = "train_num_sample"

    MSG_ARG_KEY_TEST_CORRECT = "test_correct"
    MSG_ARG_KEY_TEST_ERROR = "test_error"
    MSG_ARG_KEY_TEST_NUM = "test_num_sample"

    MSG_ARG_KEY_CLIENT_STATUS = "client_status"
    MSG_ARG_KEY_ROUND_IDX = "round_idx"
    MSG_ARG_KEY_EPSILON = "epsilon"

    MSG_ARG_KEY_EVENT_NAME = "event_name"
    MSG_ARG_KEY_EVENT_VALUE = "event_value"
    MSG_ARG_KEY_EVENT_MSG = "event_msg"

    """
        MLOps related message 
    """
    # Client Status
    MSG_MLOPS_CLIENT_STATUS_IDLE = "IDLE"
    MSG_MLOPS_CLIENT_STATUS_UPGRADING = "UPGRADING"
    MSG_MLOPS_CLIENT_STATUS_INITIALIZING = "INITIALIZING"
    MSG_MLOPS_CLIENT_STATUS_TRAINING = "TRAINING"
    MSG_MLOPS_CLIENT_STATUS_STOPPING = "STOPPING"
    MSG_MLOPS_CLIENT_STATUS_FINISHED = "FINISHED"

    # Server Status
    MSG_MLOPS_SERVER_STATUS_IDLE = "IDLE"
    MSG_MLOPS_SERVER_STATUS_STARTING = "STARTING"
    MSG_MLOPS_SERVER_STATUS_RUNNING = "RUNNING"
    MSG_MLOPS_SERVER_STATUS_STOPPING = "STOPPING"
    MSG_MLOPS_SERVER_STATUS_KILLED = "KILLED"
    MSG_MLOPS_SERVER_STATUS_FAILED = "FAILED"
    MSG_MLOPS_SERVER_STATUS_FINISHED = "FINISHED"

    # Client OS
    MSG_CLIENT_OS_ANDROID = "android"
    MSG_CLIENT_OS_IOS = "iOS"
    MSG_CLIENT_OS_Linux = "linux"


class GRPCMessage(object):
    MSG_ARG_KEY_OPERATION = "operation"
    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    MSG_OPERATION_SEND = "send"
    MSG_OPERATION_RECEIVE = "receive"
    MSG_OPERATION_BROADCAST = "broadcast"
    MSG_OPERATION_REDUCE = "reduce"

    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_MODEL_PARAMS_URL = "model_params_url"
    MSG_ARG_KEY_MODEL_PARAMS_KEY = "model_params_key"

    def __init__(self, type="default", sender_id=0, receiver_id=0):
        self.type = str(type)
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.msg_params = {}
        self.msg_params[GRPCMessage.MSG_ARG_KEY_TYPE] = type
        self.msg_params[GRPCMessage.MSG_ARG_KEY_SENDER] = sender_id
        self.msg_params[GRPCMessage.MSG_ARG_KEY_RECEIVER] = receiver_id

    def init(self, msg_params):
        self.msg_params = msg_params

    def init_from_json_string(self, json_string):
        self.msg_params = json.loads(json_string)
        self.type = self.msg_params[GRPCMessage.MSG_ARG_KEY_TYPE]
        self.sender_id = self.msg_params[GRPCMessage.MSG_ARG_KEY_SENDER]
        self.receiver_id = self.msg_params[GRPCMessage.MSG_ARG_KEY_RECEIVER]

    def init_from_json_object(self, json_object):
        self.msg_params = json_object
        self.type = self.msg_params[GRPCMessage.MSG_ARG_KEY_TYPE]
        self.sender_id = self.msg_params[GRPCMessage.MSG_ARG_KEY_SENDER]
        self.receiver_id = self.msg_params[GRPCMessage.MSG_ARG_KEY_RECEIVER]

    def get_sender_id(self):
        return self.sender_id

    def get_receiver_id(self):
        return self.receiver_id

    def add_params(self, key, value):
        self.msg_params[key] = value

    def get_params(self):
        return self.msg_params

    def add(self, key, value):
        self.msg_params[key] = value

    def get(self, key):
        if key not in self.msg_params.keys():
            return None
        return self.msg_params[key]

    def get_type(self):
        return self.msg_params[GRPCMessage.MSG_ARG_KEY_TYPE]

    def to_string(self):
        return self.msg_params

    def to_json(self):
        json_string = json.dumps(self.msg_params)
        print("json string size = " + str(sys.getsizeof(json_string)))
        return json_string

    def get_content(self):
        print_dict = self.msg_params.copy()
        msg_str = str(self.__to_msg_type_string()) + ": " + str(print_dict)
        return msg_str

    def __to_msg_type_string(self):
        type = self.msg_params[GRPCMessage.MSG_ARG_KEY_TYPE]
        return type
