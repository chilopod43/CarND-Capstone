import tensorflow as tf
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        # load classifier
        self.graph = tf.Graph()
        with self.graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile("./light_classification/ssd_mobilenet.pb", 'rb') as f:
                frozen_graph = f.read()
                graph_def.ParseFromString(frozen_graph)
                tf.import_graph_def(graph_def, name='')

        self.image = self.graph.get_tensor_by_name('image_tensor:0')
        self.scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.graph.get_tensor_by_name('detection_classes:0')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # implement light color prediction
        with self.graph.as_default():
            unsqueezed_image = np.expand_dims(image, axis=0)

            (scores, classes) = sess.run([self.scores, self.classes],
              feed_dict={image_tensor: unsqueezed_image})
        
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        
        state = TrafficLight.UNKNOWN
        if scores[0] > self.threshold:
            if classes[0] == 1:
                state = TrafficLight.GREEN
            elif classes[0] == 2:
                state = TrafficLight.RED
            elif classes[0] == 3:
                state = TrafficLight.YELLOW

        return state
