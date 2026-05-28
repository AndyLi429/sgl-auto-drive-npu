# install_skill.ps1
param([Parameter(Mandatory)][string]$GitHubUrl)

# 解析 owner/repo
$uri = [System.Uri]$GitHubUrl
$segments = $uri.AbsolutePath.Trim("/") -split "/"
$owner = $segments[0]
$repo  = $segments[1]

Write-Host "Repo: $owner/$repo"

# 用 GitHub API 列出 skills/ 目录
$apiBase = "https://api.github.com/repos/$owner/$repo/contents"
$rawBase = "https://raw.githubusercontent.com/$owner/$repo/main"

try {
    $root = Invoke-RestMethod -Uri $apiBase -Headers @{ "User-Agent" = "ps-skill-installer" }
} catch {
    Write-Host "FAIL: 无法访问 GitHub API - $($_.Exception.Message)"; exit 1
}

# 判断是否有 skills/ 目录
$skillsDir = $root | Where-Object { $_.name -eq "skills" -and $_.type -eq "dir" }

if ($skillsDir) {
    $skills = Invoke-RestMethod -Uri "$apiBase/skills" -Headers @{ "User-Agent" = "ps-skill-installer" }
    $skillList = $skills | Where-Object { $_.type -eq "dir" }
} else {
    # 没有 skills/ 子目录，当作单个 skill 处理
    $skillList = @([PSCustomObject]@{ name = $repo; path = "" })
}

foreach ($skill in $skillList) {
    $skillName = $skill.name
    $mdUrl = if ($skill.path) { "$rawBase/$($skill.path)/SKILL.md" } else { "$rawBase/SKILL.md" }

    Write-Host "`n[$skillName] $mdUrl"

    foreach ($base in @("C:\Users\$env:USERNAME\.claude\skills", "C:\Users\$env:USERNAME\.codex\skills")) {
        $dir = "$base\$skillName"
        New-Item -ItemType Directory -Force $dir | Out-Null
        try {
            Invoke-WebRequest -Uri $mdUrl -OutFile "$dir\SKILL.md" -ErrorAction Stop
            Write-Host "  OK $dir"
        } catch {
            Write-Host "  FAIL $dir - $($_.Exception.Message)"
        }
    }
}

Write-Host "`nDone."