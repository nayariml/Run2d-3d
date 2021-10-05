""" This code contains the 2D tracker to OpenPose and AlphaPose.

    This code contains code developed by @akanazawa, projects Human dynamics and Motion Reconstruction

    @nayariml
    Update: 09/28/2021

"""

import os
import os.path as osp
import ipdb
import json
import math
import numpy as np
from glob import glob

MIN_KPS = 20 #min 80% of points from 25 2D keypoints

def read_json(json_path): #Openpose
    with open(json_path) as f1:
        data = json.load(f1)
    kps = []
    for people in data['people']:
        kp = np.array(people['pose_keypoints_2d']).reshape(-1, 3)
        kps.append(kp)
    return kps

def l2norm (kp):
    #Spin l2norm

    midhip = kp[8]; neck = kp[1]
    x1 = midhip[0]; y1 = midhip[1]
    x2 = neck[0]; y2 = neck[1]

    if 0 in midhip or 0 in neck:
        return 0

    dist = 0; dist_ = 0
    dist += pow((x1 - x2), 2)
    dist_ += pow((y1 - y2), 2)
    final = math.sqrt(dist + dist_)

    return final

def boxparams(vis, kp):

    vkp = kp[vis, :2]
    min_pt = np.min(vkp, axis=0)
    max_pt = np.max(vkp, axis=0)
    person_height = np.linalg.norm(max_pt - min_pt)
    if person_height == 0:
        #return None, None, None
        pass
    center = (min_pt + max_pt) / 2.
    scale = 150. / person_height

    radius = (224 / 2.) * (1 / scale)
    corner = center - radius # get x, y
    bbox = np.hstack([corner, radius * 2, radius * 2]) # x, y, height, width of the rectangle

    return center, scale, bbox

def ordering(cx, kps):

    scores = cx[:, 0]

    idcx = []; idkp = []

    # The Biggest score first
    idxs = np.argsort(scores)[::-1] #Get the positions ordered
    return cx[idxs],kps[idxs]

def IoU(box0, box1):
    #Compute the Intersection over Union (IoU) between two rectangle
    #box = [score, dtmid, cx, cy, scale, rect]
    #rect = x, y, height, width of the rectangle

    box1 = np.squeeze(box1)
    #Actual frame box
    boxA = box0[-4:]

    #Last frame box
    boxB = box1[-4:]

    #Compute the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])

    endA = boxA[:2] + boxA[2:]
    endB = boxB[:2] + boxB[2:]

    xB = min(endA[0], endB[0])
    yB = min(endA[1], endB[1])

    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]

    # Compute the area of intersection rectangle
    w = xB - xA + 1
    h = yB - yA + 1
    inter_area = float(w * h)

    # Compute the IoU
    iou = max(0, inter_area / (boxA_area + boxB_area - inter_area))

    return iou

def mtresults(last, cx, kps):
    #Measures how close to the previous frame it is
    #Return the new key positions
    #box = [score, dtmid, cx, cy, scale, rect]

    ndt = []; ndtmid = []; idx = []; nonecx = []; nonekps = []
    iou = []; iou_id = []; newbox = [];

    dtmid = cx[:,1] #actual frame/box

    ldt = [list(i) for i in last] #last frame/box
    ldt = np.vstack(ldt)
    ltmid = ldt[:,1]

    last_score = ldt[:,0]
    score = cx[:,0]
    #scores = np.hstack([last_score, score])
    ct = 0
    for pt in range (len(last)):
        for i in range(len(cx)):

            iouu = IoU(cx[i], last[pt])
            if iouu > 0.8:
                iou.append(iouu)
            else:
                iou.append(-1)

        iou_sort = np.argsort(iou)[::-1]#Sorted

        if pt == 0:
            iou_id.append(iou_sort[0])
        elif iou_sort[0] in iou_id:
            for p in iou_sort:
                if p not in iou_id:
                    iou_id.append(p)
                elif p == -1 and pt != 0:
                    iou_id.append(p)
                    break
        else:
            iou_id.append(iou_sort[0])
        del iou[:]

    return iou_id


def tracker_check_OP(frames,json_path):

    num_frames = len(frames)

    print('reading %s' % json_path)

    json_paths = sorted(glob(osp.join(json_path, "*.json")))

    if len(json_paths) != num_frames:
        print('Not all frames were detected.')
        return None

    all = []; logs = []
    for i, json_path in enumerate(json_paths):
        kps = read_json(json_path)
        all.append(kps)

    persons = {}
    boxes = []; params = []; keys = []; ctbox = []
    num_persons = []

    for i, frame in enumerate (all):
        npose = len(frame)
        num_persons.append(npose)
        if npose == 0:
            print('Not all frames have people detected in it.')
            id = 2
            return id, None

        for pose in frame:
            #Body part locations (x, y) and detection confidence (c)
            visible = pose[:, 2] > 0 #For all the positions (x, y) > 0 = True
            if np.sum(visible) >= MIN_KPS: #Count if the sum of points are more than min acceptable
                npose -= 1
                score = (np.sum(pose[visible, 2]) / np.sum(visible)) #Compute a score for each pose
                dtmid = l2norm(pose)
                center, scale, rect = boxparams(visible, pose)

                ctbox = np.hstack([score, dtmid, center, scale, rect])

                params.append(ctbox) #score and l2norm
                boxes.append(pose) #valid kps

        if npose != 0:
            log = [i, len(frame), npose]
            logs.append(log)

        if len(params) == 0:
            #print('Max persons found', max(num_persons))
            #print('\n')
            print("Not find persons in all frames")
            print('\n')
            id = 1
            ret = [num_frames, max(num_persons), i, logs]
            return id, ret

        prms = np.vstack(params) #vertically
        valid_kps = np.stack(boxes)
        #Only save the current frame params, so delete them
        del params[:]
        del boxes[:]

        cxx, vkpss = ordering(prms, valid_kps)

        #Set the number of persons in the beginning and the prior values
        if len(persons.keys()) == 0 and i == 0:
            for j, (pr, vlkps) in enumerate (zip (cxx, vkpss)):
                persons[j] = [(i, pr, vlkps)]
                keys.append(j)

        #IF not the first time, them:
        elif len(persons.keys()) == 0 and i != 0:
            print("Not find persons in all frames")
            print('\n')
            id = 1
            ret = [num_frames, max(num_persons), i, logs]
            return id, ret

        else:
            #Find matching persons
            mtscores = []
            lastbox = []
            for ks, boxs in iter(persons.items()):

                last_frame, last_box, last_kp = boxs[-1]
                lastbox.append(last_box)


            new_pos = mtresults(lastbox, cxx, vkpss) #last box, Current box, Current Kps

            k = 0
            for npos in new_pos:
                if npos == -1:
                    pass
                elif k < len(persons.keys()):
                    persons[k].append((i, cxx[npos], vkpss[npos]))
                    k+=1
                else:
                    pass

    if len(persons) != 1:
        survivors = {}
        nkey = 0
        for sch in range (len(persons)):
            if len(persons[sch]) < num_frames:
                pass
            else:
                survivors[nkey] = persons[sch]
                nkey += 1
        id = 0
        ret = [max(num_persons), survivors]

        return id, ret
    else:
        id = 0
        ret = [max(num_persons), persons]
        return id, ret


def read_json_AP(json_path): #AlphaPose
    with open(json_path) as f2:
        data = json.load(f2)
    kps = []
    for i in range(len(data)):
        kp = np.array(data[i]['keypoints']).reshape(-1, 3)
        kps.append(kp)
    return kps

def tracker_check_AP(frames,json_path):
    #Based on run_openpose.py https://github.com/akanazawa/human_dynamics/

    num_frames = len(frames)
    json_paths = osp.join(json_path, 'alphapose-results-forvis-tracked.json')#AlphaPose-Pose Flow
    json_path_woutflow = osp.join(json_path, 'alphapose-results.json')#AlphaPose

    print('reading %s' % json_paths)

    data_woutflow = read_json_AP(json_path_woutflow)


    with open(json_paths) as f:
        data = json.load(f)


    if len(data.keys()) != num_frames:
        print('Not all frames have people detected in it.')
        id = 2
        return id, None
    persons = {}; box = []; logs = []; keys = []
    kps_dict = {}
    kps_count = {}
    for i, key in enumerate(sorted(data.keys())):
        # People who are visible in this frame.
        track_ids = []
        for person in data[key]:
            kps = np.array(person['keypoints']).reshape(-1, 3)
            idx = int(person['idx'])
            if idx not in kps_dict.keys():
                # If this is the first time, fill up until now with None
                kps_dict[idx] = [None] * i
                kps_count[idx] = 0
            # Save these kps.
            kps_dict[idx].append(kps)
            track_ids.append(idx)
            kps_count[idx] += 1
        # If any person seen in the past is missing in this frame, add None.
        for idx in set(kps_dict.keys()).difference(track_ids):
            kps_dict[idx].append(None)

    kps_list = []
    counts_list = []
    for k in kps_dict:
        if kps_count[k] >= MIN_KPS:
            kps_list.append(kps_dict[k])
            counts_list.append(kps_count[k])

    # Sort it by the length so longest is first:
    sort_idx = np.argsort(counts_list)[::-1]
    kps_list_sorted = []
    for sort_id in sort_idx:
        kps_list_sorted.append(kps_list[sort_id])

    all_kps = kps_list_sorted
    print('Total number of PoseFlow tracks:', len(all_kps))
    nky = 0
    for nkey in range(len(all_kps)):

        track_id = nkey
        track_id = min(track_id, len(all_kps) - 1)
        #print('Processing track_id:', track_id)
        kps = all_kps[track_id]

        for k in range(len(kps)):

            if kps[k] is not None:
                box.append(kps[k])
            else:
                log = [k, len(all_kps), nkey] #total keys poseflow, frame error, key error
                logs.append(log)
            if len(box) == num_frames:
                persons[nky] = kps
                keys.append(nkey)
                nky += 1
        del box[:]

    if len(persons) == 0:
        print("Not find persons in all frames with PoseFLow")
        print("Let's try with AlphaPose's raw results")
        print('Reading:', json_path_woutflow)
        if len(data_woutflow) == num_frames:
            id = 4
            return id, [data_woutflow, num_frames, logs]
        else:
            print("It's not posssible to find persons in all frames in any way")
            ret = [num_frames, logs]
            id = 5
            return id, ret
    print('\n')
    print('Number of survivors:', len(persons))
    print('\n')
    id = 3

    ret = [len(all_kps), persons]

    return id, ret

if __name__ == '__main__':

    DATASET_PATH = "/home/nayari/projects/SoccerKicks/Rendered/"

    ACTION = "Freekick"#"Penalty"

    ACTION_ID = "6_freekick"

    DEFAULT = "hmmr_output"

    preds_file = "hmmr_output.pkl"

    print('\n')
    print("The action is:", ACTION + '\n')
    print("The action id is:", ACTION_ID + '\n')

    dir = DATASET_PATH + ACTION + '/' + ACTION_ID
    print("The dataset path", dir)

    res_dir = dir +'/AlphaPose_output'
    res_dir_openpose = dir + '/OpenPose_output/'

    img_paths = sorted(glob(osp.join(dir + '/video_frames/','*.png'))) #Original image

    im_paths = sorted(glob(osp.join(res_dir +'/vis/', '*.jpg'))) #Rendered images
    im_paths_open = sorted(glob(osp.join(res_dir_openpose, '*.jpg')))

    #Get the 2D kps
    id, tracker_al = tracker_check_OP(im_paths_open, res_dir_openpose)
