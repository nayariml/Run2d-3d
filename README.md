# Run2d-3d
This repository contains the scripts to automatically run the latest versions of [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose), [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and [Hmmr](https://github.com/nayariml/human_dynamics) systems from prompt.

Update the paths to your installed systems directories.

## Use cases

Run the 2D systems AlphaPose and OpenPose to a single video
*Update the file name in the script

python3 runOPAL.py

For a list of videos

python3 runOPAL.py --vid

Run the 3D system HMMR

python3 runHmmr.py 

2D tracker to OpenPose and AlphaPose with/without PoseFlow
*may have problems in crowded scenes

python3 trackers.py

## References

@akanazawa:

  [Human dynamics](https://github.com/akanazawa/human_dynamics)

  [Motion Reconstruction](https://github.com/akanazawa/motion_reconstruction)

