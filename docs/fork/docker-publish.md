# 포크 Docker 이미지 배포

이 저장소의 `Fork Reader - Publish Image` 워크플로는 기존 Dockerfile로 amd64와 arm64 이미지를 빌드해 `ghcr.io/graypawn/grimmory`에 배포한다. 별도 PAT나 Docker Hub 계정 없이 워크플로의 GITHUB_TOKEN을 사용한다. 리더 이벤트 회귀 테스트와 프런트엔드 타입 검사를 통과해야 배포한다. Android 실기기 검증을 대체하지는 않는다.

## 처음 설정

1. 독서 UI 변경과 이 워크플로를 기본 브랜치(develop)에 커밋·푸시한다. 기본 브랜치에 워크플로가 있어야 Actions의 수동 실행 메뉴를 사용할 수 있다.
2. 같은 커밋에서 배포용 `android-reader` 브랜치를 만들고 푸시한다. 이후에는 이 브랜치를 푸시할 때 자동 빌드한다. 수동으로는 Actions → Fork Reader - Publish Image → Run workflow에서 android-reader 브랜치를 선택한다. 선택한 브랜치의 코드가 빌드되므로 주의한다.
3. 포크에서 Actions가 비활성화되어 있으면 활성화한다. 저장소/조직 정책에서 사용된 Actions 및 packages:write 권한을 허용해야 한다.
4. 빌드 성공 후 GitHub 프로필 → Packages → grimmory → Package settings에서 공개 범위를 확인한다. 새 GHCR 패키지는 기본 비공개다. 로그인 없이 Docker에서 받고 싶으면 Public으로 변경한다. 비공개로 유지하면 서버에서 read:packages 권한의 classic PAT로 ghcr.io에 로그인한다.

기존 upstream 릴리스 워크플로는 수정하지 않았다. 배포 브랜치 이름은 원본의 main/develop 자동 릴리스와 분리했다. 포크에 불필요한 정기 실행 워크플로가 활성화되어 있으면 GitHub Actions에서 해당 워크플로만 비활성화할 수 있다.

## Docker에서 사용

기존 compose의 Grimmory 서비스에서 image만 바꾼다. DB, 환경변수, 포트, 볼륨은 기존 값을 유지한다.

```yaml
image: ghcr.io/graypawn/grimmory:latest
```

`latest`만 마지막 성공 빌드를 가리킨다. 특정 빌드로 되돌려야 할 때는 Actions 실행 요약에 표시되는 digest를 사용한다. 앱의 계정 메뉴에는 `graypawn-android-reader-<짧은 커밋 SHA>` 버전명이 표시된다.

기존 compose 디렉터리에서 실행한다. 서비스 이름이 grimmory가 아니면 실제 이름으로 바꾼다.

```sh
docker compose pull grimmory
docker compose up -d --no-deps grimmory
docker compose logs --tail=100 grimmory
```

새 UI가 보이지 않으면 브라우저/PWA를 완전히 닫고 다시 열어 이전 프런트엔드 캐시 사용 여부를 확인한다.

## 원본 업데이트

원본의 선택한 릴리스를 android-reader에 병합하고 변경 내용을 확인한 뒤 푸시한다. 빌드 성공 후 서버에서 위 명령으로 갱신한다. DB 마이그레이션이 포함될 수 있으므로 버전 업데이트 전 DB와 앱 데이터 백업을 보관한다. 이미지 태그를 되돌리는 것만으로 DB까지 되돌아가지는 않는다.

참고: https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images

GHCR 인증/공개 범위: https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry
