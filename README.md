# Korean Spacing Error Corrector

There are two corrector algorithms. One is implemented from pycrfshuite and the other heuristic algorithm is implemented with pure Python code

soyspacing에는 두 가지 종류의 띄어쓰기 오류 교정 알고리즘이 포함되어 있습니다. 첫째는 카운팅 기반의 휴리스틱한 알고리즘이며, 둘째는 conditional random field 기반의 알고리즘입니다. 

휴리스틱 알고리즘은 [ScatterLab][http://www.scatterlab.co.kr/]의 [sunggu][https://github.com/new21cccc]님, [Emily Yunha Shin][https://github.com/eyshin05]님과 함께 작업하였습니다. 

## Setup

	pip install soyspacing

## Require

- python-crfsuite
