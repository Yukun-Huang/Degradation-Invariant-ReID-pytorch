# Degradation-Invariant-Re-ID-pytorch
This is the official code repository for *Learning Degradation-Invariant Representation for Robust Real-World Person Re-Identification*.


## Dataset
Download the MLR-CUHK03 dataset and reorganize the folders as follows:<br>
```
├── resolution-reid
│   ├── MLR-CUHK03
│       ├── train
│       ├── val
│       ├── gallery
│       ├── query
│           ├── 0020
│               ├── 00020_c0_00000.jpg
│               ├── ...
```

## Model
Trained model are provided. You may download it from [Google Drive](https://drive.google.com/drive/folders/1anHkFyEJaQWRsbkmVFjZX9y71zzb7rCs?usp=sharing), then move the `outputs` folder to your project's root directory.

## Dependencies
* Python 3.8
* PyTorch 1.8

## Usage
```
python3  test_reid.py
```

## Result
