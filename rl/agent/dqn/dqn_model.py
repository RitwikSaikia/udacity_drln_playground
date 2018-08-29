from abc import abstractmethod

DEFAULT_LEARNING_RATE = 5e-3


class _AbstractDqnModel:

    def __init__(self, input_shape, output_shape, lr=DEFAULT_LEARNING_RATE) -> None:
        super().__init__()
        self.lr = lr
        self.input_shape = input_shape
        self.output_shape = output_shape

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()

    @abstractmethod
    def train(self, X, Y):
        raise NotImplementedError()

    @abstractmethod
    def get_weights(self):
        raise NotImplementedError()

    @abstractmethod
    def set_weights(self, weights):
        raise NotImplementedError()

    @abstractmethod
    def save_model(self, filename):
        raise NotImplementedError()

    @abstractmethod
    def load_model(self, filename):
        raise NotImplementedError()