# Band Lyric Sync Tool

밴드 공연 영상을 입력하고, 붙여넣은 가사를 줄 단위로 싱크해 `ASS`, `SRT`, 자막 입힌 `MP4`를 만드는 로컬 웹앱입니다.

## 실행

PowerShell에서:

```powershell
cd band-lyric-sync-tool
.\run.ps1
```

브라우저에 표시되는 로컬 주소를 열면 됩니다.
이미 실행 중이면 [http://127.0.0.1:7860](http://127.0.0.1:7860)으로 접속하면 됩니다.

## Windows 배포본

개발 환경에서 설치본과 portable 압축 파일을 만들려면:

```powershell
.\build-release.ps1
```

- 설치본: `release/BandLyricSync-Setup-0.1.0-win64.exe`
- Portable: `release/BandLyricSync-0.1.0-portable-win64.zip`
- 설치본 빌드에는 Inno Setup 6가 필요합니다.
- Python, CUDA 런타임, FFmpeg는 배포본에 포함됩니다.
- Whisper와 Demucs 모델은 최초 사용 시 `%LOCALAPPDATA%\BandLyricSync\cache`에 다운로드됩니다.
- 작업 결과와 로그도 `%LOCALAPPDATA%\BandLyricSync`에 저장됩니다.

## 사용 흐름

1. 공연 영상 파일을 업로드합니다.
2. 아티스트/곡 제목을 입력합니다.
3. 가사를 붙여넣습니다.
   - 일반 가사: 자동으로 줄 단위 싱크를 시도합니다.
   - LRC 형식: `[01:23.45]가사` 타임스탬프를 그대로 사용합니다.
4. 폰트, 크기, 위치, 색상, 테두리 등을 지정합니다.
5. `Create subtitles`를 누르면 `ASS`, `SRT`, 자막 입힌 `MP4`가 생성됩니다.

## 정확도 메모

- `Reference audio DTW`는 빈 줄로 나눈 모든 입력 가사 블록을 각각 별도 자막으로 보존합니다.
- `Find LRCLIB match`는 오타와 괄호 표기를 보정하고, YouTube 공식 음원의 아티스트·곡 메타데이터까지 사용해 입력란을 자동완성합니다.
- LRCLIB에 동기화 가사가 정말 없으면 공식 음원의 분리된 보컬을 Whisper로 강제정렬해 자동으로 대체합니다.
- 입력한 YouTube 음원의 길이가 공연과 크게 다르면 아티스트/곡명으로 길이가 맞는 후보를 자동 선택합니다.
- 전체 가사가 실수로 두 번 붙여넣어진 경우 한 복사본만 자동으로 사용합니다.
- LRC의 내용 없는 타임스탬프도 절/간주 경계로 사용해 이전 가사가 오래 남지 않게 합니다.
- 반복 후렴을 잘못 연결하지 않도록 DTW의 진행 속도와 탐색 범위를 제한합니다.
- 전조를 12개 반음 후보에서 자동 탐색하고, 크로마와 피치 온셋을 결합한 다중 해상도 DTW를 사용합니다.
- 새 다중 해상도 정렬과 기존 정렬을 함께 채점해 더 안정적인 시간축을 자동 선택합니다.
- `Separate vocals first`를 켠 Reference DTW 모드는 공연 보컬을 겹치는 구간으로 두 번 강제정렬하고, 두 결과가 합의하는 가사 줄만 추가로 보정합니다.
- NVIDIA GPU가 있으면 Demucs는 CUDA를, 자막 영상 인코딩은 `h264_nvenc`를 자동으로 사용합니다.
- DTW 특징 추출과 시간 경로 계산은 CPU 작업이며, GPU 인코딩 실패 시 `libx264`로 자동 전환됩니다.
- 밴드 공연 음원은 악기/관객 소리 때문에 자동 싱크가 흔들릴 수 있습니다.
- `Separate vocals first`를 켜면 Demucs로 보컬을 분리한 뒤 정렬해서 정확도가 좋아질 수 있지만 시간이 더 걸립니다.
- 첫 사용 시 Whisper/Demucs 모델 파일을 추가로 다운로드할 수 있습니다.
- 가장 안정적인 입력은 사용자가 직접 확인한 정확한 가사입니다.
