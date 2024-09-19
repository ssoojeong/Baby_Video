# Baby_Video

### 각 실험별 환경설정 및ㄴ Inference 방법

## LaVie
```
conda env create -f environment.yml 
conda activate lavie
```

`cd base`

base/configs/sample.yaml에서 text prompt 수정 후
`python pipelines/sample.py --config configs/sample.yaml`


## StyleCrafter
```
conda create -n stylecrafter python=3.8.5
conda activate stylecrafter
pip install -r requirements.txt
```

eval_data/eval_video_gen.json에서 text 프롬프트 및 스타일 참조할 이미지 수정
`sh scripts/run_infer_video.sh`

(Optional) Infernce on your own data according to the [instructions](./eval_data/README.md)

## VGen
```
conda create -n vgen python=3.8
conda activate vgen
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
sudo apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
```

```
!pip install modelscope
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('iic/tf-t2v', cache_dir='models/')
```

`mv ./models/iic/tf-t2v/* ./models/`

data/videos/test_list_for_tft2v.txt 에서 프롬프트 수정
`python inference.py --cfg configs/tft2v_t2v_infer.yaml`