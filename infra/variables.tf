# ============================================================================
# 變數定義
#
# 有 default 者可直接使用；無 default 者必須於 terraform.tfvars 提供。
# ============================================================================

variable "aws_region" {
  description = "部署區域。東京為距離台灣最近且服務完整的成熟區域。"
  type        = string
  default     = "ap-northeast-1"
}

variable "project_name" {
  description = "資源名稱前綴，用於識別與批次清理。"
  type        = string
  default     = "nba-prediction"
}

variable "instance_type" {
  description = <<-EOT
    EC2 機型。t3.micro（2 vCPU）僅適合環境驗證；
    完整實驗於 16 執行緒環境需約 85 分鐘，於 2 vCPU 上將顯著拉長，
    實際執行時應改用運算最佳化機型（如 c7i.4xlarge）。
  EOT
  type        = string
  default     = "t3.micro"
}

variable "root_volume_size" {
  description = "根磁碟大小（GB）。需容納作業系統、Python 環境與 20MB 資料集。"
  type        = number
  default     = 20
}

variable "allowed_ssh_cidr" {
  description = <<-EOT
    允許 SSH 連入的來源網段，CIDR 格式。
    取得目前對外 IP：curl -s https://checkip.amazonaws.com
    再於其後加上 /32 表示單一位址，例如 "203.0.113.45/32"。
  EOT
  type        = string

  validation {
    condition     = can(cidrnetmask(var.allowed_ssh_cidr))
    error_message = "必須為有效的 CIDR 格式，例如 203.0.113.45/32。"
  }

  validation {
    condition     = var.allowed_ssh_cidr != "0.0.0.0/0"
    error_message = "禁止對全網際網路開放 SSH，請指定實際來源位址。"
  }
}

variable "public_key_path" {
  description = "本機 SSH 公鑰路徑。僅公鑰會上傳至 AWS。"
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}
