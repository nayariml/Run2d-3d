###########
""" Run a single or a list of videos:

    Developed by @nayariml
    Update: 08/26/2021

    -OpenPose v1.7
    -AlphaPose v0.4.0

    python 3.6
    TensorFlow 2.0
    Cuda version: 11.2 with GeForce 940MX
"""

###########
import os
import os.path as osp
import subprocess
import time
import csv
import argparse
from glob import glob

parser = argparse.ArgumentParser()

parser.add_argument(
    '--vid',
    action='store_true',
    help = 'Allow running all the content videos in a given directory.'
)

args = parser.parse_args()

openpose_dir = '/home/nayari/projects/code/OpenPose'

alphapose_dir = 'AlphaPose/'

poseflow_dir = 'PoseFlow/'

vid_path = "/home/nayari/projects/SoccerKicksPlus/videoClip/" #Where the original videos are
out_dir = "/home/nayari/projects/SoccerKicksPlus/rendered/"#Where the results will be stored


############# AlphaPose pre-trained models specification
#Specification for Halpe 26 pre-trained models

cfg_dir = 'configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml'
checkpoint_dir = 'pretrained_models/halpe26_fast_res50_256x192.pth'


def run_openpose(vid_path, out_dir):

    if osp.exists(osp.join(out_dir, '*.json')):
        print('Per-frame detection OpenPose: done!')
        return

    if not osp.exists(vid_path): #If vid path not exists
        print('%s doesnt exist' % vid_path)
        import ipdb
        ipdb.set_trace()

    print('----------')
    print('Computing per-frame results with OpenPose')

    vid_name = osp.basename(vid_path).split('.')[0]

    out_dir_video = out_dir + '/' + vid_name + '.avi'

    start_time = time.perf_counter()

    cmd_base = '%s/build/examples/openpose/openpose.bin --video %%s --write_json %%s --scale_gap 0.25 --display 0 --write_images %%s --write_images_format jpg  --write_video %%s ' % (openpose_dir)

    #cmd_base += '--Display 0'
    curr_dir = os.getcwd() #Returns the current directory

    cmd = cmd_base % (vid_path, out_dir, out_dir, out_dir_video)

    print(cmd)

    os.chdir(openpose_dir) #Change the current directory to the given path

    print('OpenPose path:', os.getcwd())

    ret = os.system(cmd)

    if ret != 0:
        print('Issue running openpose. Please make sure you have the openpose installed in the correct directory.')
        exit(ret)

    os.chdir(curr_dir)
    print('OpenPose successfully ran!')
    print('----------')
    return start_time

def run_alphapose(vid_path, out_dir): #External

    if osp.exists(osp.join(out_dir, 'alphapose-results.json')):
        print('Per-frame detection AlphaPose: done!')
        return

    print('----------')
    print('Computing per-frame results with AlphaPose')

    start_time = time.perf_counter()

    cmd = [
        'python3', 'demo_inference.py',
        '--cfg', cfg_dir,
        '--checkpoint', checkpoint_dir,
        '--sp', #Use single process for pytorch
        '--gpus', '-1', #choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)
        '--video', vid_path,
        '--save_img','--save_video',
        '--outdir', out_dir,
    ]

    print('Running: {}'.format(' '.join(cmd)))

    curr_dir = os.getcwd()

    os.chdir(alphapose_dir) #Change the current directory to the given path

    print('AlphaPose path:', os.getcwd())

    ret = subprocess.call(cmd)


    if ret != 0:
        print('Issue running AlphaPose. Please make sure you have the alphapose installed in the correct directory.')
        exit(ret)

    os.chdir(curr_dir)
    print('AlphaPose successfully ran!')
    print('----------')
    return start_time


def run_poseflow(video, out_dir):

    img_dir = out_dir + '/vis/'
    print(img_dir)
    alphapose_json = osp.join(out_dir, 'alphapose-results.json')

    if not os.path.exists(alphapose_json):
        print('AlphaPose files not found')
        run_alphapose(video, out_dir)

    start_time = time.perf_counter()

    out_json = osp.join(out_dir, 'alphapose-results-forvis-tracked.json')

    if osp.exists(out_json):
        print('Tracking: done!')
        return out_json

    print('Computing tracking with PoseFlow')

    cmd = [
        'python3', 'tracker-general.py',
        '--imgdir', img_dir,
        '--in_json', alphapose_json,
        '--out_json', out_json,

    ]
    # '--visdir', out_dir,  # Uncomment this to visualize PoseFlow tracks.
    print('Running: {}'.format(' '.join(cmd)))
    curr_dir = os.getcwd()
    os.chdir(poseflow_dir)
    ret = subprocess.call(cmd)
    if ret != 0:
        print('Issue running PoseFlow. Please make sure you can run the above '
              'command from the commandline.')
        exit(ret)
    os.chdir(curr_dir)
    print('PoseFlow successfully ran!')
    print('----------')

    return start_time

def un_timeout(save, path_file):

    with open(path_file + '_time_out.txt', 'w') as f:
        f.write(str(save) + "\n")
    return

def saveTimes(save, video_out):
    save_csv = video_out + '_time_excution.csv'
    with open(save_csv, "a") as csvfile:
        file_is_empty = os.stat(save_csv).st_size == 0
        headers = ['OpTime','AlTime','PfTime']
        writer = csv.writer(csvfile)
        if file_is_empty:
            writer.writerow(headers)
        writer.writerows(save)
    return

def mkdir(dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)
    return

def main(video, video_out):

    mkdir(video_out)

    open_out = video_out + '/OpenPose_output'
    alpha_out = video_out + '/AlphaPose_output'

    mkdir(open_out)
    mkdir(alpha_out)

    ############################
    # Execute the following commands and give back the total time
    """
    OpTime = run_openpose(video, open_out)
    OpEnd = (time.perf_counter() - OpTime) / 60
    un_timeout(OpEnd, open_out)
    time.sleep(180)"""

    AlTime = run_alphapose(video, alpha_out)
    AlEnd = (time.perf_counter() - AlTime) / 60
    un_timeout(AlEnd, alpha_out)
    time.sleep(180)

    PfTime = run_poseflow(video, alpha_out)
    PfEnd = (time.perf_counter() - PfTime) / 60
    pf_fl = alpha_out + '/PoseFlow'
    un_timeout(PfEnd, pf_fl)
    time.sleep(180)

    #SaveEnds = [[OpEnd, AlEnd, PfEnd]]

    #saveTimes(SaveEnds, video_out)

    print("Everything done!")
    time.sleep(5)

if __name__ == '__main__':

    if args.vid:
        vid_paths = sorted(glob(vid_path + '/*.mp4'))
        for videos in vid_paths:
            print('Working on the video path:', videos)
            vidname = osp.basename(videos).split('.')[0]
            videoout = out_dir + vidname
            main(videos, videoout)

    else:
        name = "26_penalty2"
        ext = '.mp4'

        video = vid_path + name + ext
        vid_name = osp.basename(video).split('.')[0]
        video_out = out_dir + vid_name + '/'
        print('Current path:', os.getcwd())
        main(video, video_out)
