# Push HEAD to the HF Space with a GIT_SHA stamp so GET /version reports the deployed commit.
# usage: .\deploy_hf.ps1   (needs the `hf-seudoe` git remote)
$ErrorActionPreference = "Stop"
$sha = git rev-parse HEAD
$url = git remote get-url hf-seudoe
$t = Join-Path ([IO.Path]::GetTempPath()) ("hfdeploy-" + [guid]::NewGuid())
New-Item -ItemType Directory $t | Out-Null
git archive HEAD --format=tar -o "$t.tar"
tar -xf "$t.tar" -C $t
Set-Content -NoNewline -Path (Join-Path $t "GIT_SHA") -Value $sha
Push-Location $t
git init -q -b main
git -c user.name=ci -c user.email=ci@users.noreply.github.com add -A
git -c user.name=ci -c user.email=ci@users.noreply.github.com commit -q -m "deploy $sha"
git push -q --force $url HEAD:main
Pop-Location
Remove-Item -Recurse -Force $t, "$t.tar"
Write-Host "deployed $($sha.Substring(0,7)) to HF Space"
