# State Farm VGG 최종 개선 패키지

최신 모델은 **운전자 5-fold CV + 전체/크롭 1:1 결합 + 얼굴 보정 0.5 + 클래스별 확률 보정**입니다.
전체 26명 OOF Log Loss **0.25035**, 정확도 **91.80%**입니다.
실제 최종 제출: **Public 0.23784 / Private 0.20226**. Private는 이전 0.21735보다 개선됐고 Public은 이전 0.22783보다 악화했습니다.
[전체 변경 과정과 붙여넣기용 표](RESULTS.md)를 확인하세요.

## 가장 쉬운 실행

1. `VGG_FINAL.ipynb`를 Kaggle에 업로드합니다.
2. State Farm 공식 대회 데이터를 연결하고 GPU와 Internet을 켭니다. 데이터 접근에는 본인 계정의 대회 규칙 동의가 필요합니다.
3. 위에서부터 실행합니다. 이전 개인 비공개 노트북이나 API 토큰은 코드에 필요하지 않습니다.
4. 결과: `/kaggle/working/vgg_final/final/submission.csv`.

전체 이미지/크롭을 각각 5개 모델로 학습하므로 전체 실행에는 시간이 걸립니다.
모든 코드 셀은 제공하지만 이 통합 포장 버전을 추가로 전체 재학습하지는 않았습니다.
동일한 개별 학습 단계는 실제 Kaggle GPU에서 실행했고, 최종 보정은 저장된 실제 예측으로 재검증했습니다.

## 단계를 나눠 실행

`train_full_cv.ipynb`, `train_crop_cv.ipynb`, `train_face_full.ipynb`, `calibrate_final.ipynb` 순서입니다.
별도 세션을 쓰면 앞 세 단계의 출력 폴더를 마지막 세션에 같은 구조로 연결하고 `WORK_ROOT`를 지정하세요.
한 세션에서 순서대로 실행하면 폴더 연결이 필요 없습니다.
Linux + CUDA 환경에서는 다음처럼 실행할 수도 있습니다.

```bash
pip install -r requirements.txt
python run_all.py --input /path/to/statefarm --output ./vgg_final_outputs
```

공식 CSV와 `imgs/train`, `imgs/test`를 원래 구조로 풀어둔 폴더가 필요합니다.
ImageNet 가중치만 외부 초기화로 사용하며, 테스트 이미지로 학습하거나 의사 라벨을 만들지 않습니다.
검증 운전자는 해당 학습 모델에서 제외하고, BN 통계도 학습 운전자에서만 계산합니다.
OOF는 개발용 추정값이며 완전한 nested CV 또는 독립 최종 점수는 아닙니다.

## 파일

- `VGG_FINAL.ipynb`: 최종 파이프라인 전체 실행 노트북.
- `src/`: 같은 단계의 Python 코드.
- `history/`: 최초 1에폭 기준 및 미세조정 노트북. 미세조정은 본인이 실행한 기준 노트북의 체크포인트를 입력으로 연결합니다.
- `results/`: 실제 제출 점수, 같은 조건에서의 검증 비교, 최종 설정.
- `GITHUB_UPLOAD.md`: 팀 저장소의 지정 브랜치로 옮기는 방법.

GPU 학습 데이터, 이미지, 가중치, 토큰, 큰 예측 파일은 이 GitHub 폴더에 넣지 않았습니다.
준비된 CSV는 RESULTS.md의 Kaggle 링크에서 받을 수 있습니다.
이 코드는 Kaggle에 자동 제출하거나 GitHub 다른 브랜치를 수정하지 않습니다.
