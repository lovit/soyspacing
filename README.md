# Korean Spacing Error Corrector

Soyspacing은 한국어 띄어쓰기 문제를 해결하기 위한 휴리스틱 알고리즘을 제공합니다. Conditional Random Field와 비교하여 가벼운 모델 사이즈와 빠른 학습이 가능합니다. 

이 알고리즘은 [ScatterLab][scatter_url]의 [sunggu][sunggu_url]님, [Emily Yunha Shin][eyshin_url]님과 함께 작업하였습니다. 

\* version = 0.1.23은 미완성된 CRF 기반 띄어쓰기 알고리즘을 포함하고 있었습니다. 

\* version = 1.0.0부터 미완성된 CRF를 지우고 휴리스틱 기반 알고리즘만 제공합니다. 

## Setup

	pip install soyspacing

## Require

- numpy >= 1.12.1


[scatter_url]: http://www.scatterlab.co.kr/
[sunggu_url]: https://github.com/new21cccc
[eyshin_url]: https://github.com/eyshin05
