# 팀 저장소의 지정 브랜치로 옮기기

말한 구조는 팀 저장소의 공동 작업자(collaborator)로 참여하고, 지정된 작업 브랜치에 올리는 방식입니다.
현재 연결에서 aiast225/DeepLearning은 404여서 실제 브랜치 구조는 확인하지 못했습니다.
임시 보관 위치는 개인 공개 저장소의 `feat/vgg-statefarm-final-20261118` 브랜치입니다.
그 저장소의 main이나 기존 웹사이트 파일은 변경하지 않습니다.

## GitHub 웹 화면에서

1. 임시 브랜치의 `notebooks/vgg_final` 폴더를 내려받습니다.
2. 팀 저장소를 열고 **팀이 정해둔 본인 VGG 브랜치**를 선택합니다.
3. 기존 `notebooks` 폴더에서 Add file → Upload files를 선택하고 `vgg_final` 폴더를 올립니다.
4. 지정 브랜치에 커밋합니다. 파일별 구조가 따로 정해져 있다면 이 폴더 안의 파일만 해당 경로로 옮기세요.
5. 팀 규칙에 따라 기준 브랜치로 Pull Request를 만듭니다.

## Git 명령으로 최신 폴더 가져오기

```bash
git clone https://github.com/aiast225/DeepLearning.git
cd DeepLearning
git fetch origin
git switch <팀에서_정해준_VGG_브랜치>
git fetch https://github.com/Yunhu-Kwon/Survive-in-Korea.git feat/vgg-statefarm-final-20261118
git restore --source=FETCH_HEAD --worktree -- notebooks/vgg_final
git add notebooks/vgg_final
git commit -m "Add final VGG CV model and results"
git push origin HEAD
```

꺾쇠 부분은 실제 팀 VGG 브랜치로 바꿉니다.
이 방법은 최신 `notebooks/vgg_final` 폴더만 가져오므로 다른 프로젝트의 HTML 파일과 커밋 이력은 가져오지 않습니다.
이미 그 폴더를 수정했다면 웹 화면에서 필요한 파일을 비교해서 옮기세요.
정해진 팀 브랜치가 이미 있으면 새 이름을 임의로 만들 필요가 없습니다.
초대를 받았다면 GitHub 초대 수락 및 저장소 권한도 확인하세요.
