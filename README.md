# VLAD-VSA  ACM MM 2021
VLAD-VSA: Cross-Domain Face Presentation Attack Detection with Vocabulary Separation and Adaptation. [PDF](https://dl.acm.org/doi/10.1145/3474085.3475284), [Googledrive](https://drive.google.com/file/d/1hcH2AHwInelY4nfJdWonN6cDdAIE4TlG), [BaiduDrive](https://pan.baidu.com/s/1TiR0KBRgt4lRr5ep8YzC7g), Fetch Code: st5g

- VLAD aggregation instead of global pooling.
- **Vocabulary adaptation**, a general trick to improve VLAD optimization.
- Vocabulary separation for domain generalization.

These codes are mainly based on the implementation of [SSDG](https://github.com/taylover-pei/SSDG-CVPR2020), specifically thanks Yunpei Jia
## Dependencies
python 3.6

pytorch 0.4

torchvision 0.2

## Preparation
1. Download the OULU-NPU, CASIA-FASD, Idiap Replay-Attack, and MSU-MFSD datasets. Download the [pretrained model](https://github.com/taylover-pei/SSDG-CVPR2020/tree/master/pretrained_model) and put it in the new-created `pretrained_model` package 

2. Detect, crop and resize faces to 256x256x3 using [MTCNN](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection)

   To be specific, we process every frame of each video and then utilize the `sample_frames` function in the `utils/utils.py` to sample frames during training.

   Put the processed frames in the path `root/data/dataset_name`.

3. Data Label Generation.

   Move to the `root/data_label` and generate the data label list:
   ```
   python generate_label.py
   ```
## Training
- VLAD aggregation
```python
python experiment/.../train_vlad_baseline.py
```
- VLAD + Vocabulary adaptation 
```python
python experiment/.../train_vlad_baseline2.py
```
- VLAD + Vocabulary adaptation + Vocabulary separation 
```python
python experiment/.../train_vlad_baseline3.py
```
The file `config.py` contains all the hype-parameters used during training. The parameters can be tuned and better performance may got.
## Testing
```
python experiment/.../dg_test.py
```
## Citation
Please cite this paper if the code is helpful to your research.
```
@inproceedings{wang2021vlad,
  title={VLAD-VSA: Cross-Domain Face Presentation Attack Detection with Vocabulary Separation and Adaptation},
  author={Wang, Jiong and Zhao, Zhou and Jin, Weike and Duan, Xinyu and Lei, Zhen and Huai, Baoxing and Wu, Yiling and He, Xiaofei},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={1497--1506},
  year={2021}
}
```
