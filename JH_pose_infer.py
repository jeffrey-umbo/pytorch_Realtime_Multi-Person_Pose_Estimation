from JH_pose_demo import *
import cv2
import torch
import numpy as np
import time

if __name__ == '__main__':
    torch.set_num_threads(torch.get_num_threads())
    hpg = HeatmapPafGenerator('./model/pose_model.pth.tar')
    param_, model_ = hpg.get_configs()
    # gfcp = GaussianFilterAndCountPeak()
    CnV = ConnectAndVisualize()



    print '==> Test on single image'
    test = np.ones((480, 640, 3))
    print("* Heatmap paf generator")
    heatmap_avg, paf_avg = hpg(test)
    gaussian_heatmap = gaussian_filtering(heatmap_avg)
    print("* Count peak")
    # all_peaks = gfcp(heatmap_avg, param_)
    all_peaks = count_peak(heatmap_avg, gaussian_heatmap, param_)
    # all_peaks = count_peak(heatmap_avg, param_)
    print("* CnD")
    canvas = CnV(paf_avg, all_peaks, test, param_)

    

    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if frame is not None:
            start_time_fps = time.time()
            print frame.shape

            s = time.time()
            heatmap_avg, paf_avg = hpg(frame)
            print("* GPU, {}".format(time.time()-s))
            gaussian_heatmap = gaussian_filtering(heatmap_avg)


            s = time.time()
            # all_peaks = gfcp(heatmap_avg, param_)
            all_peaks = count_peak(heatmap_avg, gaussian_heatmap, param_)
            # all_peaks = count_peak(heatmap_avg, param_)
            print("* Count peak, {}".format(time.time()-s))

            s = time.time()
            canvas = CnV(paf_avg, all_peaks, frame, param_)
            print("* CnV, {}".format(time.time()-s))

            fps = "FPS: {:.1f}".format(1.0 / (time.time() - start_time_fps))
            cv2.putText(canvas, fps, (30, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)
            print("==>"+fps)
            # Display the resulting frame
            cv2.imshow('Video', canvas)
        else:
            cv2.imshow('Video', canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
