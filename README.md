This project is based on [ultralytics/yolov3](https://github.com/ultralytics/yolov3).

LF-YOLO (Lighter and Faster YOLO) is used to detect defect of X-ray weld image. The related paper is available [here](http://arxiv.org/abs/2110.15045).

## Download

```download
$ git clone https://github.com/AlanLiangC/PCB-Defects-Detection.git
```

## Dateset Prepare
- Prepare VOC_PCB
- Make file tree like:
  ```
  - VOC_PCB
  |
  - convert_pcd_datasets.py
  ```

- do `python convert_pcd_datasets.py`

- The file tree become:
  ```
  - VOC_PCB
  |
  - convert_pcd_datasets.py
  |
  - VOC_PCB_train
    - images
    |
    - labels
  |
  - VOC_PCB_val
    - images
    |
    - labels
  ```

- Change the datasets' path in `./data/pcb.yaml` 

## Train
We provide multiple versions of LF-YOLO with different widths. 

```train
$ python train.py --data coco.yaml --cfg LF-YOLO.yaml      --weights '' --batch-size 1
                                         LF-YOLO-1.25.yaml                           1
                                         LF-YOLO-0.75.yaml                           1
                                         LF-YOLO-0.5.yaml                            1
```

## Requirements

Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov3/blob/master/requirements.txt) dependencies installed, including `torch>=1.7`. To install run:
```bash
$ pip install -r requirements.txt
```
## Inference
```bash
$ python detect.py --source data/images --weights LF-YOLO.pt --conf 0.25
```
