# Grimmory Android Reader Fork

이 저장소는 [Grimmory 원본 저장소](https://github.com/grimmory-tools/grimmory)를 기반으로 한 포크입니다.
Grimmory의 기본 기능과 원본의 저작권·라이선스 고지를 존중하며, Android 독서 화면의 입력 동작과 포크 배포에 필요한 변경을 유지합니다.

## 라이선스

이 프로젝트는 원본 프로젝트와 동일하게 [GNU Affero General Public License v3.0](LICENSE)을 따릅니다.

포크에서 수정한 소스도 같은 라이선스로 배포합니다. 이 저장소에서 제공하는 Docker 이미지는 이 저장소의 소스와 함께 제공되며, 네트워크로 서비스를 제공하는 경우에도 AGPL-3.0의 소스 제공 의무를 준수해야 합니다.

원본의 저작권 표시, `LICENSE`, `NOTICE` 및 기존 고지를 삭제하지 않았습니다. 이 포크의 변경사항은 이 저장소의 커밋과 소스에서 확인할 수 있습니다.

## 이 포크의 변경사항

### Android EPUB 독서 화면

- Grimmory의 상단·하단 독서 바: 화면 터치 시 지연 없이 표시되거나 숨겨집니다.
- Grimmory의 상단·하단 독서 바: 애니메이션 삭제 (Android 환경)
- Grimmory의 상단 독서 바에 Grimmory 컨텍스트 메뉴 활성화 버튼을 추가했습니다.
- 기본 상태에서는 Grimmory 컨텍스트 메뉴, 브라우저 컨텍스트 메뉴, 구글 검색 메뉴를 차단합니다.

### 포크 버전 표시

좌측 패널에서 계정명을 누르면 열리는 사용자 메뉴의 버전 표시에는 다음 형식의 포크 버전명이 표시됩니다.

```text
graypawn-android-reader-<짧은 커밋 SHA>
```

### Docker 이미지

GitHub Actions가 `android-reader` 브랜치의 변경을 검사한 뒤 amd64·arm64 이미지를 빌드하여 다음 GHCR 이미지로 게시합니다.

```text
ghcr.io/graypawn/grimmory:latest
```

## Docker로 실행

Docker와 Docker Compose가 필요합니다.

```yaml
services:
  grimmory:
    image: ghcr.io/graypawn/grimmory:latest
    container_name: grimmory
    environment:
      - USER_ID=1000
      - GROUP_ID=1000
      - TZ=Etc/UTC
      - DATABASE_URL=jdbc:mariadb://mariadb:3306/grimmory
      - DATABASE_USERNAME=grimmory
      - DATABASE_PASSWORD=your_secure_password
    depends_on:
      mariadb:
        condition: service_healthy
    ports:
      - "6060:6060"
    volumes:
      - ./data:/app/data
      - ./books:/books
      - ./bookdrop:/bookdrop
    restart: unless-stopped

  mariadb:
    image: lscr.io/linuxserver/mariadb:11.4.8
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=Etc/UTC
      - MYSQL_ROOT_PASSWORD=super_secure_password
      - MYSQL_DATABASE=grimmory
      - MYSQL_USER=grimmory
      - MYSQL_PASSWORD=your_secure_password
    volumes:
      - ./config:/config
    restart: unless-stopped
```

이미지를 갱신할 때는 Grimmory 서비스의 실제 이름에 맞춰 실행합니다.

```sh
docker compose pull grimmory
docker compose up -d --no-deps grimmory
```

GHCR 패키지가 비공개이면 서버에서 `read:packages` 권한이 있는 GitHub classic PAT로 로그인해야 합니다.

```sh
echo "$CR_PAT" | docker login ghcr.io -u graypawn --password-stdin
```

PWA 또는 브라우저에 이전 프런트엔드가 캐시되어 있으면 앱을 완전히 닫고 다시 연 뒤 변경사항을 확인합니다.

## 원본 프로젝트

- 원본 저장소: <https://github.com/grimmory-tools/grimmory>
- 원본 문서: <https://grimmory.org/docs>
- 이 포크의 Android 독서 변경 문서: [`docs/fork/android-reader.md`](docs/fork/android-reader.md)
- 이 포크의 Docker 배포 문서: [`docs/fork/docker-publish.md`](docs/fork/docker-publish.md)
