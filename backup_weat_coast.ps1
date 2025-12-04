# === SETTINGS ===
# Путь к ТВОЕЙ БД SQLite
$sourceDb = "C:\Users\administrator.WEST\PycharmProjects\WEAT_COAST\instance\inventory.db"

# Папка для бэкапов на внешнем диске F
$backupFolder = "F:\Backup"

# создать папку если нет
if (!(Test-Path $backupFolder)) {
    New-Item -ItemType Directory -Path $backupFolder | Out-Null
}

# timestamp в имени файла
$timestamp = (Get-Date -Format "yyyy-MM-dd_HH-mm-ss")
$backupFile = Join-Path $backupFolder "inventory_$timestamp.db"

# === CHECK IF DB EXISTS ===
if (!(Test-Path $sourceDb)) {
    Write-Output "ERROR: source DB not found: $sourceDb"
    exit 1
}

# === TRY COPY (HANDLE LOCKS) ===
try {
    Copy-Item -Path $sourceDb -Destination $backupFile -ErrorAction Stop
    Write-Output "Backup created: $backupFile"
}
catch {
    Write-Output "ERROR copying DB. Most likely DB file is locked. Error:"
    Write-Output $_
    exit 1
}

# === OPTIONAL: DELETE OLD BACKUPS (older than 30 days) ===
Get-ChildItem -Path $backupFolder -Filter "inventory_*.db" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force
