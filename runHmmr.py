###########
""" Run a single or a list of videos:

    -Hmmr modified

    Developed by @nayariml
    Update: 08/30/2021

    python 3.6
    TensorFlow 2.0
    Cuda version: 11.2 with GeForce 940MX
"""

import os
import os.path as osp
import subprocess
import time
import csv
import argparse
from glob import glob

from trackers import *

parser = argparse.ArgumentParser()

parser.add_argument(
    '--vid',
    action='store_true',
    help = 'Allow running all the content videos in a given directory.'
)

args = parser.parse_args()

hmmr_dir = 'human_dynamics/'

#hmmr_dir = '/home/nayari/projects/SoccerKicks/Fails/'

#hmmr_dir = '/home/nayari/projects/SoccerKicksPlus/'

vid_path = "human_dynamics/demo_data/" #Where the original videos are

#vid_path = '/home/nayari/projects/SoccerKicks/Fails/VideoClips/'

#vid_path = '/home/nayari/projects/SoccerKicksPlus/VideoClips/'

out_dir = "human_dynamics/demo_output/" #Where the results will be stored

#out_dir = '/home/nayari/projects/SoccerKicks/Fails/Rendered/'

out_dir = '/home/nayari/projects/SoccerKicksPlus/Rendered/'
############# HMMR model

model = 'models/hmmr_model.ckpt-1119816'

#############

def run_hmmr_AL(vid_path,video_out, trackid):

    print('Computing 3D per-frame results with HMMR - AlphaPose')
    al_time = time.perf_counter()

    curr_dir = os.getcwd()
    os.chdir(hmmr_dir)

    print(os.getcwd())

    cmd = [
        'python3', 'demo_video.py',
        '--vid_path', vid_path,
        '--track_id', trackid,
        '--load_path', model,
        '--out_dir', video_out,
        #'--vid_dir', If set, runs on all video in directory
    ]

    print('Running: {}'.format(' '.join(cmd)))

    ret = subprocess.call(cmd)
    if ret != 0:
        print('Issue running HMMR. Please make sure you have the alphapose installed in the correct directory.')
        exit(ret)

    os.chdir(curr_dir)
    AlEnd = (time.perf_counter() - al_time) / 60

    print('Hmmr- AlphaPose successfully ran!', AlEnd)
    print('----------')
    time.sleep(10)
    return AlEnd

def run_hmmr_OP(vid_path,video_out, trackid='0'):

    print('Computing 3D per-frame results with HMMR - OpenPose')
    op_time = time.perf_counter()

    curr_dir = os.getcwd()
    os.chdir(hmmr_dir)

    cmd = [
        'python3', 'demo_video_openpose.py',
        '--vid_path', vid_path,
        '--track_id', trackid,
        '--load_path', model,
        '--out_dir', video_out,
        #'--vid_dir', If set, runs on all video in directory
    ]

    print('Running: {}'.format(' '.join(cmd)))

    ret = subprocess.call(cmd)
    if ret != 0:
        print('Issue running HMMR. Please make sure you have the alphapose installed in the correct directory.')
        exit(ret)

    os.chdir(curr_dir)

    OpEnd = (time.perf_counter() - op_time) / 60
    print('Hmmr- OpenPose successfully ran!', OpEnd)
    print('----------')

    return OpEnd

def un_timeout(save, path_file):

    with open(path_file + '_time_out.txt', 'w') as f:
        f.write(str(save) + "\n")
    return
dic = {
        0 : 'OpenPose successfully ran!',
        1 : 'Not every frame has people detected on it - OpenPose',
        2 : 'Not every frame has people detected on it - AlphaPose',
        3 : 'AlphaPose successfully ran!',
        4 : 'Not find persons in all frames with AlphaPose-PoseFLow',
        5 : "It's not posssible to find persons in all frames in any way - AlphaPose",
        6 : 'OpenPose directory not found!',
        7 : 'AlphaPose directory not found!',
}
def check_results_path(hmmr_dir, path, vid_name):

    video = vid_path + vid_name
    video_out = out_dir + vid_name + '/'

    res_dir = osp.abspath(osp.join(path, 'AlphaPose_output'))
    res_dir_openpose =  osp.abspath(osp.join(path, 'OpenPose_output'))

    im_paths = sorted(glob(osp.join(res_dir +'/vis/', '*.jpg'))) #Rendered images
    im_paths_open = sorted(glob(osp.join(res_dir_openpose, '*.jpg')))

    print('\n')
    print("================================")
    print("================================" + '\n')

    if osp.exists(res_dir_openpose):
        print("HMMR_OP - OK!")
        print('Number of frames with OpenPose:', len(im_paths_open))
        print("================================")
        print("Searching for trackers")

        id, tracker = tracker_check_OP(im_paths_open, res_dir_openpose)
        if id == 0:
            for i in range(len(tracker[1])):
                print("Running HMMR_OP")
                OpEnd = run_hmmr_OP(video, video_out, str(i))
                Save = ("Time", OpEnd)
                path_out = hmmr_dir + save_path + 'HMMR_OP'
                un_timeout(Save, path_out)
                time.sleep(10)
            dt = [[vid_name, dic[id], len(tracker[1][0]), tracker[0], len(tracker[1])]]
            save_general(dt, hmmr_dir)
            path_op = path + 'hmmr_op_log.txt'
            save_log(id, tracker, path_op)
            print('Log saved')
        elif id == 1:
            #Video_name', 'id', 'num_frames','Max number of persons found', 'frame error, logs'
            data = [[vid_name, dic[id], tracker[0], tracker[1],  tracker[2],  tracker[3]]]
            save_unit(data, hmmr_dir)

            dt = [[vid_name, dic[id], tracker[0], None, None]]
            save_general(dt, hmmr_dir)

            path_op = path + 'hmmr_op_log_error.txt'
            save_log(id, data , path_op)
            print('Log saved')

    else:
        data = [[vid_name, dic[6], None, None,  None,  None]]
        save_unit(data, hmmr_dir)

        dt = [[vid_name, dic[6], None, None, None]]
        save_general(dt, hmmr_dir)

    print('\n')
    print("================================")
    print("================================" + '\n')

    if osp.exists(res_dir):
        print("HMMR_AP - OK!")
        print('Number of frames with AlphaPose:', len(im_paths))
        print("================================")
        print("Searching for trackers")

        id, tracker_al = tracker_check_AP(im_paths, res_dir)

        if id == 3:
            for i in range(len(tracker_al[1])):
                print("Running HMMR_AL", i)
                AlEnd = run_hmmr_AL(video, video_out, str(i))
                Save = ("Time", AlEnd)
                path_out =  hmmr_dir + save_path + 'HMMR_AL'
                un_timeout(Save, path_out)
                print('HMMR_AL finished')
            dt = [[vid_name, dic[id], len(tracker_al[1][0]), tracker_al[0], len(tracker_al[1])]]
            save_general(dt, hmmr_dir)
            path_ap = path + 'hmmr_al_log.txt'
            save_log(id, tracker_al, path_ap)
            print('Log saved')
        elif id == 4:
            data = [[vid_name, dic[id], tracker_al[1], 1,  None,  tracker_al[2]]]
            save_unit(data, hmmr_dir)

            dt = [[vid_name, dic[id], tracker_al[1], None, None]]
            save_general(dt, hmmr_dir)

            path_op = path + 'hmmr_al_log_error.txt'
            save_log(id, data, path_op)
            print('Log saved')
        elif id == 5:
            data = [[vid_name, dic[id], tracker_al[0], None,  None,  tracker_al[1]]]
            save_unit(data, hmmr_dir)

            dt = [[vid_name, dic[id], tracker_al[0], None, None]]
            save_general(dt, hmmr_dir)

            path_op = path + 'hmmr_al_log_error.txt'
            save_log(id, data, path_op)
            print('Log saved')
    else:
        data = [[vid_name, dic[7], None, None,  None,  None]]
        save_unit(data, hmmr_dir)

        dt = [[vid_name, dic[7], None, None, None]]
        save_general(dt, hmmr_dir)
    return

def save_log(id, data, path):

    if id == 0 or id == 3:
        with open(path, 'w') as f:
            f.write("Log:\t" + str(dic[id]) + "\n")
            f.write("Max number of persons found:\t" + str(data[0]) + "\n")
            f.write("Number of survivors: \t" + str(len(data[1])) + "\n")
            f.close()
    else:
        with open(path, 'w') as f:
            f.write("Log error:\t" + str(data[0][1]) + "\n")
            f.write("Number of frames:\t" + str(data[0][2]) + "\n")
            f.write("Max number of persons found:\t" + str(data[0][3]) + "\n")
            f.write("Frame error:\t" + str(data[0][4]) + "\n")
            f.write("logs [frame, persons_frame, person_error]:\t" + str(data[0][5]) + "\n")
            f.close()
    return

def save_unit(data, path):
    with open(path + '/logs_fails.csv', "a") as csvfile:
        file_is_empty = os.stat(path +'/logs_fails.csv').st_size == 0
        headers = ['Video_name', 'id', 'Num_frames','Max number of persons found', 'frame error', 'logs [frame, persons_frame, person_error]']
        writer = csv.writer(csvfile)
        if file_is_empty:
            writer.writerow(headers)
        writer.writerows(data)
    return

def save_general(data, path):
    with open(path + '/logs.csv', "a") as csvfile:
        file_is_empty = os.stat(path +'/logs.csv').st_size == 0
        headers = ['Video_name', 'id', 'Num_frames','Max number of persons found', 'Number of survivors']
        writer = csv.writer(csvfile)
        if file_is_empty:
            writer.writerow(headers)
        writer.writerows(data)
    return

if __name__ == '__main__':

    if args.vid:
        vid_paths = sorted(glob(vid_path + '/*.mp4'))
        print('Total number of videos path', len(vid_paths))
        for video in vid_paths:
            print('Working on the video path:', video)
            vid_name = osp.basename(video).split('.')[0]
            path = out_dir + vid_name + '/'
            vid= out_dir
            check_results_path(hmmr_dir, path, vid_name)
    else:

        name = "26_penalty1"#"4_penalty5"#"26_penalty"
        ext = '.mp4'

        video = vid_path + name + ext
        vid_name = osp.basename(video).split('.')[0]
        video_out = out_dir + vid_name + '/'
        vid= hmmr_dir + out_dir
        path = hmmr_dir + video_out

        check_results_path(vid,path, vid_name)
