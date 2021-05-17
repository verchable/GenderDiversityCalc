# GenderDiversityCalc

GenderDiversityCalc is for understanding how much time women/men speaking and calculating a gender diversity ratio based on both total women and total men speaking time. In each video, we detect and track all people and we understand the "talking" action. Moreover, we also apply gender recognition. Finally, we have exact times of people talking and total speaking time of female and male speakers. 

The original problem was proposed by [VRT](https://www.vrt.be/en/) (national public-service broadcaster for Belgium) and it was one of the major challenges opened by [European Data Incubator](https://edincubator.eu/). 

## Installation

For [face recognition](https://github.com/deepinsight/insightface)

```bash
pip install mxnet-cu101mkl
pip install insightface
pip install --upgrade scikit-image
```
For [SlowFast](https://github.com/facebookresearch/SlowFast)
```bash
python -m pip install detectron2==0.2.1 -f \ https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.4/index.h
pip install pyyaml --ignore-installed
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install simplejson
conda install av -c conda-forge
python setup.py build develop
```
For [SceneDetection](https://github.com/Breakthrough/PySceneDetect/)
```bash
pip install scenedetect
```
Other installation
```bash
apt install ffmpeg
```
## Usage
Sample usage(inside pytorch_p36)
```python
python run_edi.py --video data/video_name.mp4
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
The code of GenderDiversityCalc is under Apache License 2.0
