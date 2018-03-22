import cv2
import os
import time
from time import gmtime, strftime
import demo_library as demo_lib
import threading
import logging
from Queue import Queue, Empty
import yaml

###################
# User Parameters #
###################
# Set up user parameters
with open("./configure.yml", 'r') as stream:
    user_parameters = yaml.load(stream)

VIDEO_SOURCE = user_parameters['video_source']
WINDOW_NAME = user_parameters['window_name']
WRITE_VIDEO_OUTPUT = user_parameters['write_video_output']
VIDEO_FPS = user_parameters['video_fps']

##################################
# Video and Queue I/O Parameters #
##################################
VIDEO_W, VIDEO_H = 640, 480
FRAME_QUEUE_MAX_SIZE = 3
QUEUE_WAIT_TIME = 4
# VIDEO_SOURCE = 0
WEIGHT_PATH = './Openpose/model/pose_model.pth.tar'

if WRITE_VIDEO_OUTPUT:
    # Create directories for output if not exists
    output_dir = './outputs/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if VIDEO_SOURCE in [0, 1]:
        current_time = strftime("%Y%m%d_%H%M", gmtime())
        VIDEO_OUTPUT_FILENAME = ('{}/{}_pose_demo.mp4'
                                 .format(output_dir, current_time))
    else:
        VIDEO_OUTPUT_FILENAME = (output_dir
                                 + VIDEO_SOURCE.split('.')[0]
                                 + '_out.mp4')

LOGGING_LEVEL = logging.INFO
logging.basicConfig(
        level=LOGGING_LEVEL,
        format=('[%(asctime)s] {%(filename)s:%(lineno)d} '
                '%(levelname)s - %(message)s'),
        )
logger = logging.getLogger(__name__)


class FrameReader(threading.Thread):
    """ Thread to read frame from webcam.
    Need to be stopped by calling stop() in IdentifyAndDisplayer."""

    def __init__(self, cap, frame_queue, threadManager):
        super(FrameReader, self).__init__()
        self.cap = cap
        self.frame_queue = frame_queue
        self.threadManager = threadManager
        self.run_flag = True

    def run(self):
        count = 0
        while self.run_flag:
            # start_time = time.time()
            count += 1
            ret, frame = self.cap.read()
            if frame is None:
                logger.warning("Not getiing frames")
                self.stop()
                continue

            if self.frame_queue.full():
                # print ("frame queue size:",self.frame_queue.qsize())
                self.frame_queue.get()

            frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
            self.frame_queue.put(frame)
            # logger.info("Frame read: {} s".format(time.time() - start_time))

    def stop(self):
        self.run_flag = False


class PoseEstimator(threading.Thread):
    """ Thread to return pose estimation result(heatmap and part affinity field).
    Need to be stopped by calling stop() in IdentifyAndDisplayer."""

    def __init__(self, frame_queue, heatmap_queue, paf_queue, weight_path):
        super(PoseEstimator, self).__init__()
        self.frame_queue = frame_queue
        self.heatmap_queue = heatmap_queue
        self.paf_queue = paf_queue
        self.run_flag = True
        # pose estimation
        self.hp_generator = \
            demo_lib.HeatmapPafGenerator(weight_path)
        self.param_, self.model_ = self.hp_generator.get_configs()

    def put_results_to_queue(self, frame):
        heatmap_avg, paf_avg = self.hp_generator(frame)

        heatmap_results = {
            'param_': self.param_,
            'heatmap_avg': heatmap_avg
        }
        paf_results = {
            'frame': frame,
            'paf_avg': paf_avg
        }
        # print ("heatmap queue size",self.heatmap_queue.qsize())
        # print ("paf queue size",self.heatmap_queue.qsize())
        if self.heatmap_queue.full():
            self.heatmap_queue.get()
            self.paf_queue.get()

        self.heatmap_queue.put(heatmap_results)
        self.paf_queue.put(paf_results)
   
    def run(self):
        while self.run_flag:
            start_time = time.time()
            frame = self.frame_queue.get(timeout=QUEUE_WAIT_TIME)
            self.put_results_to_queue(frame)
            logger.info(
                "Pose Estimation: {} s".format(time.time() - start_time)
            )

    def stop(self):
        self.run_flag = False


class GaussianFilter(threading.Thread):
    def __init__(self, heatmap_queue, filter_queue):
        super(GaussianFilter, self).__init__()
        self.heatmap_queue = heatmap_queue
        self.filter_queue = filter_queue
        self.run_flag = True

    def put_filtering_result_to_queue(self, heatmap_results):
        heatmap_avg = heatmap_results['heatmap_avg']
        param_ = heatmap_results['param_']
        gaussian_heatmap = demo_lib.gaussian_filtering(heatmap_avg)
        filtering_results = {
            'heatmap_avg': heatmap_avg,
            'gaussian_heatmap': gaussian_heatmap,
            'param_': param_
        }
        self.filter_queue.put(filtering_results)

    def run(self):
        while self.run_flag:
            try:
                heatmap_results = \
                    self.heatmap_queue.get(timeout=QUEUE_WAIT_TIME)    
            except Empty:
                continue

            start_time = time.time()
            self.put_filtering_result_to_queue(heatmap_results)
            logger.info(
                "Gaussian Filtering: {} s".format(time.time() - start_time)
            )

    def stop(self):
        self.run_flag = False


class PeakCounter(threading.Thread):
    def __init__(self, filter_queue, all_peaks_queue):
        super(PeakCounter, self).__init__()
        self.filter_queue = filter_queue
        self.all_peaks_queue = all_peaks_queue
        self.run_flag = True
        # self.gfcp = GaussianFilterAndCountPeak()

    def put_peaks_to_queue(self, filter_results):
        param_ = filter_results['param_']
        heatmap_avg = filter_results['heatmap_avg']
        gaussian_heatmap = filter_results['gaussian_heatmap']
        # Call function in JH_pose_demo
        all_peaks = demo_lib.count_peak(heatmap_avg, gaussian_heatmap, param_)
        # all_peaks = self.gfcp(heatmap_avg, param_)
        peak_results = {
            'all_peaks': all_peaks,
            'param_': param_
        }
        self.all_peaks_queue.put(peak_results)

    def run(self):
        while self.run_flag:
            try:
                filter_results = self.filter_queue.get(timeout=QUEUE_WAIT_TIME)
            except Empty:
                continue

            start_time = time.time()
            self.put_peaks_to_queue(filter_results)
            logger.info("Peak Count: {} s".format(time.time() - start_time))

    def stop(self):
        self.run_flag = False


class connected_visualizer(threading.Thread):
    def __init__(self, all_peaks_queue, paf_queue, threadManager):
        super(connected_visualizer, self).__init__()
        self.all_peaks_queue = all_peaks_queue
        self.paf_queue = paf_queue
        self.CnV = demo_lib.ConnectAndVisualize()
        self.threadManager = threadManager
        self.run_flag = True

        if WRITE_VIDEO_OUTPUT:
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            self.video_out = cv2.VideoWriter(VIDEO_OUTPUT_FILENAME,
                                             fourcc, VIDEO_FPS,
                                             (VIDEO_W, VIDEO_H))

    def draw_canvas(self, peak_results, paf_results):
        all_peaks = peak_results['all_peaks']
        param_ = peak_results['param_']
        frame = paf_results['frame']
        paf_avg = paf_results['paf_avg']
        canvas = self.CnV(paf_avg, all_peaks, frame, param_)

        return canvas

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, VIDEO_W*2, VIDEO_H*2)

        while self.run_flag:
            start_time_fps = time.time()
            if cv2.waitKey(25) & 0xFF == ord('q'):
                self.threadManager.stop()

            try:
                peak_results = \
                    self.all_peaks_queue.get(timeout=QUEUE_WAIT_TIME)
                paf_results = self.paf_queue.get(timeout=QUEUE_WAIT_TIME) 
            except Empty:
                continue

            canvas = self.draw_canvas(peak_results, paf_results)
            fps = "FPS: {:.1f}".format(1.0 / (time.time()-start_time_fps))
            cv2.putText(canvas, fps, (30, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)
            cv2.imshow(WINDOW_NAME, canvas)

            if WRITE_VIDEO_OUTPUT:
                self.video_out.write(canvas)

        if WRITE_VIDEO_OUTPUT:
            logger.info(
                "Output: {} written.".format(VIDEO_OUTPUT_FILENAME)
            )
            self.video_out.release()

    def stop(self):
        self.run_flag = False


class PoseEstimationDemo():
    """ Provides main class for instance segmentation demo.
    The class has four threads, FrameReader, PoseEstimator, PeakCounter and
    connected_visualizer to go throught the whole process. """

    def __init__(self, video_source):
        self.frame_queue_maxsize = FRAME_QUEUE_MAX_SIZE
        self.cap = cv2.VideoCapture(video_source)
        print(self.cap.get(cv2.CAP_PROP_FPS))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_H)

        if video_source not in [0, 1]:
            self.frame_queue_maxsize = 0

        self.frame_queue = Queue(maxsize=self.frame_queue_maxsize)
        self.heatmap_queue = Queue(maxsize=self.frame_queue_maxsize)
        self.filter_queue = Queue()
        self.paf_queue = Queue(maxsize=self.frame_queue_maxsize)
        self.all_peaks_queue = Queue()
        # self.canvas_queue = Queue()

        self.frameReader = FrameReader(self.cap, self.frame_queue, self)
        self.pose_esimator = PoseEstimator(
            self.frame_queue, 
            self.heatmap_queue,
            self.paf_queue,
            weight_path=WEIGHT_PATH
        )
        self.gaussianfilter = GaussianFilter(
            self.heatmap_queue, self.filter_queue
        )
        self.peak_counter = PeakCounter(
            self.filter_queue, self.all_peaks_queue
        )
        self.connect_displayer = connected_visualizer(
            self.all_peaks_queue, self.paf_queue, self
        )

    def run(self):
        logger.debug("Threads started")
        self.frameReader.start()
        self.pose_esimator.start()
        self.gaussianfilter.start()
        self.peak_counter.start()
        self.connect_displayer.start()
        # self.displayer.start()

        self.connect_displayer.join()
        self.peak_counter.join()
        self.gaussianfilter.join()
        self.pose_esimator.join()
        self.frameReader.join()
        # self.displayer.join()
        logger.debug("Threads ended")

        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("End of video")

    def stop(self):
        self.connect_displayer.stop()
        self.peak_counter.stop()
        self.gaussianfilter.stop()
        self.pose_esimator.stop()
        self.frameReader.stop()


if __name__ == "__main__":
    logger.info("Welcome to UmboCV pose estimation demo!")
    logger.info("Press q to exit.")
    demo = PoseEstimationDemo(video_source=VIDEO_SOURCE)
    demo.run()


