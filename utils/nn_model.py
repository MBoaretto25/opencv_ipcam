import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile


class NNModel:

    def __init__(self, params):
        self._frozen_graph = params['frozen_graph']
        self._input_node = params['input_node']
        self._output_node = params['output_node']

        self._model = None
        self._initialize_model()

    def _initialize_model(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._model = tf.Session(config=config)

        with gfile.FastGFile(self._frozen_graph, "rb") as pb_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(pb_file.read())

            # Using trt for performance gain
            trt_graph = trt.create_inference_graph(
                input_graph_def=graph_def,
                outputs=[self._output_node],
                max_batch_size=8,
                max_workspace_size_bytes=1 << 32,
                precision_mode="FP16"
            )
            tf.import_graph_def(trt_graph, name="")
            self._model_input_node = self._model.graph.get_tensor_by_name(self._input_node)
            self._model_output_node = self._model.graph.get_tensor_by_name(self._output_node)

    def predict(self, batch):
        # Perform prediction
        predictions = self._model.run(
            self._model_output_node,
            {self._model_input_node: batch}
        )[0]
        predictions = self._decode_prediction(predictions)
        return predictions

    def _decode_prediction(self, predictions):
        # TODO
        # Step for decoding predictions of the nn_model
        return predictions
