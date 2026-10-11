#Requires -RunAsAdministrator
[CmdletBinding(DefaultParameterSetName = 'Register')]
param(
    [Parameter(Mandatory = $true, ParameterSetName = 'Register')]
    [ValidateNotNullOrEmpty()]
    [string]$SshTarget,

    [Parameter(ParameterSetName = 'Register')]
    [ValidateNotNullOrEmpty()]
    [string]$Identity = "$HOME\.ssh\msynd_oci",

    [Parameter(ParameterSetName = 'Register')]
    [ValidateNotNullOrEmpty()]
    [string]$Log = "$env:LOCALAPPDATA\msynd\relay.log",

    [Parameter(Mandatory = $true, ParameterSetName = 'Unregister')]
    [switch]$Unregister
)

$ErrorActionPreference = 'Stop'
$taskName = 'msynd-relay'
if ($Unregister) {
    if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    }
    exit
}

$uv = (Get-Command uv.exe -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$Identity = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Identity)
$Log = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Log)
# cmd.exe expands these even inside quotes; reject them before building the action.
foreach ($value in @($PSScriptRoot, $uv, $SshTarget, $Identity, $Log)) {
    if ($value -match '["%!\r\n]') {
        throw 'Task arguments cannot contain double quotes, %, ! or line breaks.'
    }
}
New-Item -ItemType Directory -Path (Split-Path -Parent $Log) -Force | Out-Null

$command = 'set "PATH=C:\Windows\System32\OpenSSH;%PATH%" && "{0}" run python -m msynd.collect.relay --ssh "{1}" --identity "{2}" >> "{3}" 2>&1' -f $uv, $SshTarget, $Identity, $Log
$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\cmd.exe" `
    -Argument ('/d /s /c "' + $command + '"') -WorkingDirectory $PSScriptRoot
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType S4U
# ponytail: polling permits a 5-min gap plus the collector's 10-min breaker; use a Windows service if that matters.
$triggers = @(
    (New-ScheduledTaskTrigger -AtStartup),
    (New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5))
)
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
# ponytail: the log is never rotated; add rotation if disk usage matters.
Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal `
    -Trigger $triggers -Settings $settings -Force | Out-Null
