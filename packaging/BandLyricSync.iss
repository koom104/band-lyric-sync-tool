#define AppName "Band Lyric Sync"
#define AppPublisher "koom104"
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef SourceRoot
  #define SourceRoot "..\build\BandLyricSync-portable"
#endif
#ifndef OutputRoot
  #define OutputRoot "..\release"
#endif

[Setup]
AppId={{65A5E8F3-891E-4BE4-A6FB-E1A74E8B5D09}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\BandLyricSync
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputRoot}
OutputBaseFilename=BandLyricSync-Setup-{#AppVersion}-win64
Compression=lzma2/fast
SolidCompression=yes
LZMAUseSeparateProcess=yes
WizardStyle=modern
UninstallDisplayIcon={app}\BandLyricSync.exe
CloseApplications=yes
RestartApplications=no

[Files]
Source: "{#SourceRoot}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\BandLyricSync.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\BandLyricSync.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "바탕 화면 바로가기 만들기"; GroupDescription: "추가 바로가기:"

[Run]
Filename: "{app}\BandLyricSync.exe"; Description: "{#AppName} 실행"; Flags: nowait postinstall skipifsilent
