import os
import cv2
import glob
import json
import numpy as np

from util import privacy, create_folder
from config import ActiveConfig as Config

boundingBox = []


class ActiveSample():

    def __init__(self, config):
        self.FNNUM = 0
        # Config
        self.config = config

        # Background sub
        self.knnbg = cv2.createBackgroundSubtractorKNN(
            history=500, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        # Segmentation
        self.segmentation = cv2.imread(self.config.seg_path)
        self.segmentation = cv2.resize(
            self.segmentation, (self.config.img_w, self.config.img_h))

        # Pre-process segmentation image
        self.segmask = self.segmentation.copy()
        for i in range(self.config.img_h):
            for j in range(self.config.img_w):
                if sum(self.segmentation[i][j]) == 0:
                    # region to ignore
                    self.segmask[i][j] = [0, 0, 0]
                else:
                    self.segmask[i][j] = [1, 1, 1]

        # privacy flag
        self.privacy_processing = False

        # active sample box result
        self.activesampleresult = []

        # visualization motion flag
        self.visual_flag = False

        # save image index
        self.save_img = True
        self.index = 0
        self.save_index_FP = 0
        self.save_index_FN = 0

        self.frame_idx = 0

        # optical threshold and BG threshold
        self.human_threshold = 0.3  # region thres for FN motion box
        self.human_threshold_FP = 0.2  # region thres for FP motion box
        self.opt_threshold = 0.001  # region thres for FN optical box
        self.iou_threshold = 0.4  # iou thres for FN motion box
        self.motion_box_th = 200  # response thres for motion box candidate
        self.convert_opt_to_binary_th = 30  # response thres for optical box candidate

        # function segmentation
        self.activesampledoor = False
        self.activesamplefloor = True
        self.function_seg_img = cv2.imread(self.config.function_seg)
        self.doormask = self.active_sample_door(self.function_seg_img.copy())

    def knn(self, frame):
        fgmask = self.knnbg.apply(frame)

        return fgmask

    def findfg(self, fgbgp, frame):
        all_bbox = []
        contours, hierarchy = cv2.findContours(
            fgbgp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_check = False
        for c in contours:
            if cv2.contourArea(c) < self.motion_box_th:
                continue
            motion_check = True
            (x, y, w, h) = cv2.boundingRect(c)
            all_bbox.append([x, y, x + w, y + h])

        if motion_check:
            motion_trigger = True
        else:
            motion_trigger = False

        return frame, motion_trigger, all_bbox

    def iou(self, bbox1, bbox2):
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])

        # print(f'x_left = {x_left}, y_top = {y_top}\nx_right = {x_right}, y_bottom  ={y_bottom}')

        if x_right < x_left or y_bottom < y_top:
            return 0

        intersection_area = abs(x_right - x_left) * abs(y_bottom - y_top)
        bb1_area = abs(bbox1[0] - bbox1[2]) * abs(bbox1[1] - bbox1[3])
        bb2_area = abs(bbox2[0] - bbox2[2]) * abs(bbox2[1] - bbox2[3])
        ioue = intersection_area / \
            float(bb1_area + bb2_area - intersection_area)

        return ioue

    def false_positive(self, vis, pred_box, thr_fp, human_threshold, FP_box, number, store_img):
        FP_cand = []

        # draw all preds on vis
        for people in range(len(pred_box)):
            box = pred_box[people]
            cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(
                box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(vis, "pred", (int(box[0]), int(
                box[1])), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

        for people in range(len(pred_box)):
            box = pred_box[people]
            region = thr_fp[int(box[1]):int(box[3]),
                            int(box[0]):int(box[2])] / 255
            motion_region = np.where(region == 1)

            # visualize
            if (len(motion_region[0]) / (region.shape[0] * region.shape[1]) < human_threshold):
                FP_cand.append([int(box[0]), int(box[1]),
                                int(box[2]), int(box[3])])
                cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(
                    box[2]), int(box[3])), (0, 0, 255), 2)
                cv2.putText(vis, "FP", (int(
                    box[0]) + 5, int(box[1] + 25)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

        FP_box.append(FP_cand)
        if self.save_img:
            if len(FP_cand) > 0:
                # print('save FP')
                # cv2.imwrite(os.path.join(self.config.img_path, "FP", str(
                #     self.save_index_FP).zfill(6) + ".png"), store_img)
                # cv2.imwrite(os.path.join(self.config.vis_path, "FP", str(
                #     self.save_index_FP).zfill(6) + ".png"), vis)
                self.save_index_FP += 1

        return FP_box

    def false_negative(self, vis, pred_box, motion_box, thr, mapf, human_threshold, FN_box, number, store_img):
        FN_cand = []
        box_too_large_thres = 40000

        # draw all preds on vis
        for people in range(len(pred_box)):
            box = pred_box[people]
            cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(
                box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(vis, "pred", (int(box[0]), int(
                box[1])), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

        for motion_b in motion_box:
            # filter out too large box
            if (motion_b[2] - motion_b[0]) * (motion_b[3] - motion_b[1]) > box_too_large_thres:
                continue
            # filter out non-floor region
            central_point = (motion_b[0] + (motion_b[2] - motion_b[0]) //
                             2, motion_b[1] + (motion_b[3] - motion_b[1]) // 2)
            if self.activesamplefloor and \
                    self.segmentation[central_point[1]][central_point[0]][0] != 0 or \
                    self.segmentation[central_point[1]][central_point[0]][1] != 100 or \
                    self.segmentation[central_point[1]][central_point[0]][2] != 0:
                continue

            # motion region
            region1 = thr[int(motion_b[1]):int(motion_b[3]),
                          int(motion_b[0]):int(motion_b[2])] / 255
            motion_region = np.where(region1 == 1)
            # optical region
            region2 = mapf[int(motion_b[1]):int(motion_b[3]),
                           int(motion_b[0]):int(motion_b[2])] / 255
            flow_region = np.where(region2 == 1)

            motion_ratio = len(motion_region[0]) / \
                (region1.shape[0] * region1.shape[1])
            flow_ratio = len(flow_region[0]) / \
                (region2.shape[0] * region2.shape[1])

            if motion_ratio > human_threshold and flow_ratio > self.opt_threshold:
                # calculate motion/optical box and pred box iou
                iou_check = False

                for people in range(len(pred_box)):
                    box = pred_box[people]
                    iou_result = self.iou(
                        motion_b, [int(box[0]), int(box[1]), int(box[2]), int(box[3])])
                    if iou_result > self.iou_threshold:
                        iou_check = True

                if not iou_check:
                    FN_cand.append([int(motion_b[0]), int(
                        motion_b[1]), int(motion_b[2]), int(motion_b[3])])
                    cv2.rectangle(vis, (int(motion_b[0]), int(motion_b[1])), (int(
                        motion_b[2]), int(motion_b[3])), (255, 0, 0), 2)
                    cv2.circle(vis, (int(central_point[0]), int(
                        central_point[1])), 1, (255, 0, 0), -1)
                    cv2.putText(vis, "FN", (int(motion_b[0]), int(
                        motion_b[1])), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                    # cv2.putText(vis, "FN ({:.3f}, {:.3f})".format(flow_ratio, motion_ratio), (int(motion_b[0]), int(motion_b[1])), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                else:
                    cv2.rectangle(vis, (int(motion_b[0]), int(motion_b[1])), (int(
                        motion_b[2]), int(motion_b[3])), (255, 0, 255), 2)
                    cv2.circle(vis, (int(central_point[0]), int(
                        central_point[1])), 1, (255, 0, 255), -1)
                    cv2.putText(vis, "fn", (int(motion_b[0]), int(
                        motion_b[1])), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 1, cv2.LINE_AA)
        FN_box.append(FN_cand)
        if self.save_img:
            if len(FN_cand) > 0:
                # print(FN_cand, self.save_index_FN)
                # self.FNNUM += 1
                print('save FN')
                for cand in FN_cand:
                    # boundingBox.append(str(self.save_index_FN).zfill(
                    #     6) + ".png" + ": " + str(cand) + '\n')
                    # print(str(self.save_index_FN).zfill(
                    #     6) + ".png" + ": " + str(cand) + '\n')

                    boundingBox.append(str(self.frame_idx).zfill(
                        6) + ".png" + ": " + str(cand) + '\n')
                    print(str(self.frame_idx).zfill(
                        6) + ".png" + ": " + str(cand) + '\n')

                # cv2.imwrite(os.path.join(self.config.img_path, "FN", str(
                #     self.save_index_FN).zfill(6) + ".png"), store_img)
                # cv2.imwrite(os.path.join(self.config.vis_path, "FN", str(
                #     self.save_index_FN).zfill(6) + ".png"), vis)
                cv2.imwrite(os.path.join(self.config.img_path, "FN", str(
                    self.frame_idx).zfill(6) + ".png"), store_img)
                cv2.imwrite(os.path.join(self.config.vis_path, "FN", str(
                    self.frame_idx).zfill(6) + ".png"), vis)

                self.save_index_FN += 1

        # print(FN_box)
        return FN_box

    def active_sample_door(self, doorseg):
        img = cv2.resize(doorseg, (800, 600))
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                if img[i, j, 0] == 100 and img[i, j, 1] == 100 and img[i, j, 2] == 100:
                    img[i, j, :] = [1, 1, 1]
                else:
                    img[i, j, :] = [0, 0, 0]

        return img

    def visualization(self, vis, thr, mapf):
        cv2.imshow("motion", thr)
        cv2.imshow("vis_result", vis)
        cv2.imshow("mapf", mapf)
        cv2.waitKey(100)

    def run(self, videolist, predlist):
        prediction = glob.glob(os.path.join(predlist, '*.json'))
        prediction.sort()

        for path in prediction:
            # prediction file
            video_path = os.path.join(
                videolist, os.path.basename(path).replace(".json", ".mp4"))
            pred_file = json.load(open(path))

            print(path)
            print(video_path)

            # if (path[-13:-11] not in ["14", "15", "16"]):
            # continue

            # read video
            cap = cv2.VideoCapture(video_path)

            # FP and FN cand
            FP_box = []
            FN_box = []

            # Start active sample
            self.index = 0  # for save images
            frame_index = 0  # for read video

            while(cap.isOpened()):
                # Read video image
                ret, img = cap.read()
                if ret:
                    img = cv2.resize(
                        img, (self.config.img_w, self.config.img_h))
                    if self.privacy_processing:
                        img = privacy(img)
                    store_img = img.copy()

                    if self.index == 0 and frame_index == 0:
                        frame1 = img.copy()
                        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                        hsv = np.zeros_like(frame1)
                        hsv[..., 1] = 255
                    else:

                        # segmentation pre-processing
                        img = img * self.segmask

                        # active sample door
                        if self.activesampledoor:
                            # TODO: not test yet
                            img = img * self.doormask

                        # BG sub
                        thr = self.knn(img)
                        thr_fp = cv2.dilate(thr, self.kernel)
                        vis, trigger, motion_box = self.findfg(thr, img)

                        # optical flow
                        next = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        flow = cv2.calcOpticalFlowFarneback(
                            prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        hsv[..., 0] = ang * 180 / np.pi / 2
                        hsv[..., 2] = cv2.normalize(
                            mag, None, 0, 255, cv2.NORM_MINMAX)
                        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                        # optical update
                        prvs = next

                        mapf = np.sum(rgb, axis=2)
                        mapf = np.array(
                            np.where(mapf > self.convert_opt_to_binary_th, 255, 0), dtype=np.uint8)

                        # False Positive box detection
                        FP_box = self.false_positive(
                            vis, pred_file['bbox'][self.index][frame_index], thr_fp, self.human_threshold_FP, FP_box, self.save_index_FP, store_img)

                        # False Negitave box detection
                        FN_box = self.false_negative(
                            vis, pred_file['bbox'][self.index][frame_index], motion_box, thr, mapf, self.human_threshold, FN_box, self.save_index_FN, store_img)

                        # append result
                        self.activesampleresult.append([FP_box, FN_box])

                        if self.visual_flag:
                            self.visualization(vis, thr, mapf)

                    # update video frame index
                    frame_index = (frame_index + 1) % 5

                    if frame_index == 0:
                        self.index += 1
                else:
                    # close cap
                    cap.release()

                # increment on frame index
                self.frame_idx += 1


if __name__ == "__main__":
    username = 'user02'
    date = '2021-02-27'
    config = Config()
    config.setup_info(username, date)
    activesample = ActiveSample(config)

    create_folder(config.img_path, config.vis_path)

    activesample.run(os.path.join(username, date),
                     os.path.join(username, date + '_prediction'))
    print(boundingBox)
    fh = open(f'{username}_FNbbox.txt', 'w')
    fh.writelines(boundingBox)
    fh.close()
