# careplus-active-sample

## Directory structure

```python
$root_path
    ├───── $username
    │       ├──── $date
    │       └──── $date + "_prediction"
    ├───── active_sample_output
    │               └──── $username
    │                       └──── $date
    │                               ├─── visualization
    │                               │       ├──── FP
    │                               │       └──── FN
    │                               └─── images
    │                                       ├──── FP
    │                                       └──── FN
    └────── seg
            ├── $username + "_seg.png"
            └── $username + "_fun_seg.png"
```

## Setup

- create folder **$root_path/active_sample_output** for the output of active sample
- create folder **$root_path/seg/** folder and put *binary segmentation map* and *functional segmentation map* in it
- specify your root path in **confiy.py**
    ```python
    img_w: "image width"
    img_h: "image height"

    root_path: "root path"
    ```
    > root_path should be the abs path to this repo
- (Optional) specify threshold for active sample in **activesample.py**
    ```python
    human_threshold: "region thres for FN motion box"
    human_threshold_FP: "region thres for FP motion box"
    opt_threshold: "region thres for FN optical box"
    iou_threshold: "iou thres for FN motion box"
    motion_box_th: "response thres for motion box candidate"
    convert_opt_to_binary_th: "response thres for optical box candidate"
    ```

## Data

- Google drive: [[here](https://drive.google.com/drive/folders/1c8dtQUZWjgXzx13lMWF_x3GPogIbsSfl?usp=sharing)]

## Run

- make sure the username and date are specified in **run.py**
    ```python
    username = 'user00'
    date = '2020-12-13'
    ```
- run
    ```python
    $ cd careplus-active-sample
    $ PYTHONPATH=./ python3 run.py
    ```
