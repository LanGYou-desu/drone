# 一键编译《程序设计说明书》
# 用法：powershell -ExecutionPolicy Bypass -File .\build.ps1
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

Write-Host '[1/3] 第一遍编译（生成辅助文件）...'
xelatex -interaction=nonstopmode main.tex | Out-Null
Write-Host '[2/3] 第二遍编译（目录与交叉引用）...'
xelatex -interaction=nonstopmode main.tex | Out-Null
Write-Host '[3/3] 第三遍编译（稳定页码）...'
xelatex -interaction=nonstopmode main.tex | Out-Null

if (Test-Path main.pdf) {
    $size = [math]::Round((Get-Item main.pdf).Length / 1MB, 2)
    Write-Host "编译完成：main.pdf（$size MB）" -ForegroundColor Green
    $overfull = (Select-String -Path main.log -Pattern 'Overfull \hbox' | Measure-Object).Count
    Write-Host "警告统计：Overfull hbox=$overfull"
} else {
    Write-Error '编译失败，请查看 main.log'
}
