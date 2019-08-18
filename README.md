# Korean Space Error Corrector

soyspacing 은 한국어 띄어쓰기 문제를 해결하기 위한 휴리스틱 알고리즘을 제공합니다. Conditional Random Field 와 비교하여 가벼운 모델 사이즈와 빠른 학습이 가능합니다. 

이 알고리즘은 [ScatterLab][scatter_url]의 [sunggu][sunggu_url]님, [Emily Yunha Shin][eyshin_url]님과 함께 작업하였습니다. 

- version = 0.1.23은 미완성된 CRF 기반 띄어쓰기 알고리즘을 포함하고 있었습니다. 
- version = 1.0.0부터 미완성된 CRF를 지우고 휴리스틱 기반 알고리즘만 제공합니다. 

현재 (1.0.4) 버전에서는 학습된 모델을 제공하지 않습니다. 띄어쓰기 교정은 이를 적용할 데이터셋의 단어 분포에 따라 적합한 모델이 다릅니다. 이러한 이유로 soyspacing 에서는 학습된 모델 대신, 학습이 가능한 패키지만을 제공합니다. 사용법은 아래의 usage 에, 더 자세한 설명은 [slides](https://raw.githubusercontent.com/lovit/soyspacing/master/tutorials/presentation.pdf) 를 참고하세요.

## Setup

```
pip install soyspacing
```

## Require

- Python >= 3.4 (not tested in Python 2)
- numpy >= 1.12.1

## Usage 

학습은 텍스트 파일 경로를 입력합니다. 

```python
from soyspacing.countbase import RuleDict, CountSpace

corpus_fname = '../demo_model/134963_norm.txt'
model = CountSpace()
model.train(corpus_fname)
```

학습된 모델의 저장을 위해서는 모델 파일 경로를 입력합니다. JSON 형식으로 모델을 저장할 수 있습니다. 저장된 파일 용량을 고려하며 JSON 형식이 아닐 때 save / load 가 좀 더 쉽습니다.

```python
model.save_model(model_fname, json_format=False)
```

학습된 모델을 불러올 수 있습니다. 

```python
model = CountSpace()
model.load_model(another_model_fname, json_format=False)
```

띄어쓰기 교정을 위한 패러메터는 네 가지가 있습니다. 이를 입력하지 않으면 default value 를 이용합니다. 

```python
verbose=False
mc = 10  # min_count
ft = 0.3 # force_abs_threshold
nt =-0.3 # nonspace_threshold
st = 0.3 # space_threshold

sent = '이건진짜좋은영화 라라랜드진짜좋은영화'

# with parameters
sent_corrected, tags = model.correct(
    doc=sent,
    verbose=verbose,
    force_abs_threshold=ft,
    nonspace_threshold=nt,
    space_threshold=st,
    min_count=mc)

# without parameters
sent_corrected, tags = model.correct(sent)
```

더 자세한 내용의 Jupyter notebook 형식 tutorial 파일이 ./tutorials/에 있습니다.

관련 연구 / 제안된 모델의 원리 / CRF 와의 성능 비교 / 그 외 활용 팁의 내용이 포함되어 있는 [presentation 파일][presentation]이 제공됩니다.  

## CRF based space error correction

pycrfsuite 를 이용하여 띄어쓰기를 교정하는 패키지입니다. pycrfsuite 에 데이터를 입력하기 편하도록 Template, Transformer 의 utils 를 함께 제공합니다. 

[링크][pycrfsuite_space]


[scatter_url]: http://www.scatterlab.co.kr/
[sunggu_url]: https://github.com/new21cccc
[eyshin_url]: https://github.com/eyshin05
[presentation]: /tutorials/presentation.pdf
[pycrfsuite_space]: https://github.com/lovit/pycrfsuite_spacing
