# Assemble the anonymized ICLR2027 supplementary code zip.
# Re-runnable: stages to %TEMP%\iclr2027_supp_stage, verifies zero identity
# leaks, then writes paper/supplementary_iclr2027.zip.
# Contents per docs/iclr2027_submission_plan.md sec 1: specaccept/ + vjepa2
# calibrate + batch scripts + prereg docs (+ root code, configs, fig scripts).
# Never includes: main_v2.*, "paper (1).tex", kit zips, test_figs.*,
# __pycache__, vjepa2/upstream, data files, internal planning docs.

$ErrorActionPreference = 'Stop'
$repo  = Split-Path -Parent $PSScriptRoot
$stage = Join-Path $env:TEMP 'iclr2027_supp_stage'
$zip   = Join-Path $PSScriptRoot 'supplementary_iclr2027.zip'

if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage | Out-Null

# ---- file selection: (source relative path -> staged relative path) ----
$items = @()
foreach ($f in 'eval.py','jepa.py','module.py','train.py','utils.py') {
  $items += ,@("$f", "code/$f")
}
Get-ChildItem (Join-Path $repo 'config') -Recurse -File -Include *.yaml,*.yml | ForEach-Object {
  $rel = $_.FullName.Substring($repo.Length+1)
  $items += ,@($rel, "code/$($rel -replace '\\','/')")
}
Get-ChildItem (Join-Path $repo 'specaccept') -Recurse -File -Include *.py,*.md |
  Where-Object { $_.FullName -notmatch '__pycache__' } | ForEach-Object {
  $rel = $_.FullName.Substring($repo.Length+1)
  $items += ,@($rel, "code/$($rel -replace '\\','/')")
}
Get-ChildItem (Join-Path $repo 'vjepa2') -File -Include *.py,*.sbatch,*.md -Recurse -Depth 0 | ForEach-Object {
  $rel = $_.FullName.Substring($repo.Length+1)
  $items += ,@($rel, "code/$($rel -replace '\\','/')")
}
Get-ChildItem (Join-Path $repo 'vjepa2\specaccept_vjepa2') -File -Include *.py | ForEach-Object {
  $rel = $_.FullName.Substring($repo.Length+1)
  $items += ,@($rel, "code/$($rel -replace '\\','/')")
}
Get-ChildItem (Join-Path $repo 'batch') -Recurse -File -Include *.py,*.sbatch,*.sh,*.txt |
  Where-Object { $_.FullName -notmatch '__pycache__' } | ForEach-Object {
  $rel = $_.FullName.Substring($repo.Length+1)
  $items += ,@($rel, "code/$($rel -replace '\\','/')")
}
foreach ($d in 'certification_prereg.md','randreject_prereg.md','repro_smoke_prereg.md',
               'tworoom_paired_prereg.md','dinowm_prereg.md','spine_t125_completion.md',
               'taugrid_completion.md') {
  $items += ,@("docs\$d", "prereg/$d")
}
Get-ChildItem (Join-Path $repo 'paper\figs') -File -Filter make_*.py | ForEach-Object {
  $items += ,@("paper\figs\$($_.Name)", "figscripts/$($_.Name)")
}
$items += ,@('paper\figures\fig_tworoom_crossover.py', 'figscripts/fig_tworoom_crossover.py')

# ---- anonymizing transforms (contents AND path segments) ----
function Anonymize-Text([string]$s) {
  $s = $s -replace '(?i)--account=\S+', '--account=ANON'
  $s = $s -replace '(?i)research_project-?\S*', 'ANON_PROJECT'
  $s = $s -replace '(?i)deepmind', 'ANON_LAB'
  $s = $s -replace '/lustre/home/ha676', '$HOME'
  $s = $s -replace '/lustre/projects/[^\s"'']*', '$PROJECT'
  $s = $s -replace 'lustre', 'scratch'
  $s = $s -replace 'ha676', 'anon'
  $s = $s -replace 'hasaana2005@gmail\.com', 'anon@example.com'
  $s = $s -replace 'hasaa', 'anon'
  $s = $s -replace 'ssrhaso', 'anon'
  $s = $s -replace '(?i)isambard(-ai)?', 'clusterB'
  $s = $s -replace '(?i)(?<![A-Za-z])isca(?![A-Za-z])', 'clusterA'
  $s = $s -replace '(?i)u6ko', 'proj'
  $s = $s -replace '(?i)[a-z0-9.-]*\.ex\.ac\.uk', 'clusterA.example'
  $s = $s -replace '(?i)exeter', 'anon-inst'
  $s = $s -replace '(?i)bri(cs|stol)', 'anon-inst'
  return $s
}
function Anonymize-Path([string]$p) {
  $p = $p -replace '(?i)(?<![A-Za-z])isca(?![A-Za-z])', 'clusterA'
  $p = $p -replace '(?i)isambard', 'clusterB'
  return $p
}

$utf8 = New-Object System.Text.UTF8Encoding($false)
foreach ($pair in $items) {
  $src = Join-Path $repo $pair[0]
  if (-not (Test-Path $src)) { Write-Warning "missing: $($pair[0])"; continue }
  $dstRel = Anonymize-Path $pair[1]
  $dst = Join-Path $stage ($dstRel -replace '/','\')
  New-Item -ItemType Directory -Force (Split-Path -Parent $dst) | Out-Null
  $raw = [System.IO.File]::ReadAllText($src)
  [System.IO.File]::WriteAllText($dst, (Anonymize-Text $raw), $utf8)
}

# ---- bundle README ----
$readme = @'
Supplementary code and pre-registration documents (anonymized).

code/            evaluation + training code (root), configs, the spec-accept
                 package (specaccept/), the V-JEPA-2 transplant package
                 (vjepa2/specaccept_vjepa2/), and all cluster batch scripts
                 (batch/; clusterA = the A100 SLURM system, clusterB = the
                 GH200 system used early in the project).
prereg/          the frozen pre-registration documents: definitions, bars,
                 and banked verdicts for every claim in the paper (the
                 provenance referenced from the appendices).
figscripts/      scripts that generate the paper's figures from banked data.

Cluster usernames, hostnames, project codes, and institution names have been
replaced with neutral tokens; SLURM job IDs are retained as opaque provenance
identifiers. Data and checkpoints are not included for size reasons.
'@
[System.IO.File]::WriteAllText((Join-Path $stage 'README.md'), $readme, $utf8)

# ---- verification: zero identity leaks in contents or paths ----
$patterns = 'ha676','(?i)isambard','(?i)lustre','(?i)hasaa','(?i)u6ko','(?i)ssrhaso',
            '@gmail','(?i)ex\.ac\.uk','(?i)exeter','(?i)bristol',
            '(?i)deepmind','(?i)research_project',
            '(?i)(?<![A-Za-z])isca(?![A-Za-z])'
$leaks = @()
Get-ChildItem $stage -Recurse -File | ForEach-Object {
  $rel = $_.FullName.Substring($stage.Length+1)
  foreach ($pat in $patterns) {
    if ($rel -match $pat) { $leaks += "PATH  $rel  ~ $pat" }
  }
  $txt = [System.IO.File]::ReadAllText($_.FullName)
  foreach ($pat in $patterns) {
    if ($txt -match $pat) { $leaks += "TEXT  $rel  ~ $pat" }
  }
}
if ($leaks.Count -gt 0) {
  $leaks | ForEach-Object { Write-Output $_ }
  throw "identity leaks found: $($leaks.Count) - zip NOT written"
}

# zip with forward-slash entry names (Compress-Archive on PS5.1 writes
# backslashes, which flattens folders when unzipped on Linux)
if (Test-Path $zip) { Remove-Item -Force $zip }
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$za = [System.IO.Compression.ZipFile]::Open($zip, 'Create')
Get-ChildItem $stage -Recurse -File | ForEach-Object {
  $rel = $_.FullName.Substring($stage.Length+1).Replace('\','/')
  [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($za, $_.FullName, $rel) | Out-Null
}
$za.Dispose()
$n = (Get-ChildItem $stage -Recurse -File).Count
Write-Output "OK: $n files staged, zero leaks, wrote $zip"
