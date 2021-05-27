import json
import cv2
import os
import glob
import numpy as np


def visual_pose(img, points):
    edges = [[0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11],
            [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]
    ec = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
            (255, 0, 0), (0, 0, 255), (255, 0, 255),
            (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255),
            (255, 0, 0), (0, 0, 255), (255, 0, 255),
            (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)]
    colors_hp = [(255, 0, 255), (255, 0, 0), (0, 0, 255),
            (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
            (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
            (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
            (255, 0, 0), (0, 0, 255)]

    points = np.array(points, dtype=np.int32).reshape(17, 2)
    for j in range(17):
        cv2.circle(img, (points[j, 0], points[j, 1]), 3, colors_hp[j], -1)
    for j, e in enumerate(edges):
        if points[e].min() > 0:
            cv2.line(img, (points[e[0], 0], points[e[0], 1]),
                    (points[e[1], 0], points[e[1], 1]), ec[j], 2,
                    lineType=cv2.LINE_AA)


pred_path = 'user02-0227-1400-1600.json'
json_data = json.load(open(pred_path))

img_id = 0
annot_id = 0
for video_path in sorted(glob.glob('./2021-02-27/*.mp4'))[1400:1600]:
    cap = cv2.VideoCapture(video_path)
    while True:
        flag, img = cap.read()
        if not flag: break
        while int(json_data['annotations'][annot_id]['image_id']) == img_id:
            pts = []
            for k in range(0, 34, 2):
                pts.append(int(json_data['annotations'][annot_id]['keypoints'][k]))
                pts.append(int(json_data['annotations'][annot_id]['keypoints'][k + 1]))

            visual_pose(img, pts)

            cv2.rectangle(img,
                          (int(json_data['annotations'][annot_id]['bbox'][0]), int(json_data['annotations'][annot_id]['bbox'][1])),
                          (int(json_data['annotations'][annot_id]['bbox'][0] + json_data['annotations'][annot_id]['bbox'][2]), int(json_data['annotations'][annot_id]['bbox'][1] + json_data['annotations'][annot_id]['bbox'][3])),
                          (0, 255, 0), 2)
            annot_id += 1
        cv2.imshow("img", img)
        cv2.waitKey(1)
        print(img_id)
        img_id += 1

cap = cv2.VideoCapture()
