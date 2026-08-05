# ============================================================================
# NBA 預測實驗環境 — 基礎設施定義
#
# 建立一台可執行本專案實驗的 EC2 執行個體。
# 使用預設 VPC 以降低練習複雜度；正式環境應自建 VPC 與私有子網路。
# ============================================================================

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  # 統一標籤：所有由本專案建立的資源都會帶上，便於成本歸屬與批次清理
  default_tags {
    tags = {
      Project   = "nba-prediction"
      ManagedBy = "terraform"
    }
  }
}

# ----------------------------------------------------------------------------
# 資料來源（data source）：查詢既有資源，不建立任何東西
# ----------------------------------------------------------------------------

# 動態查詢最新的 Ubuntu 24.04 LTS AMI。
# 不寫死 AMI ID 的原因：AMI ID 隨區域而異，且會隨安全更新改版。
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical 官方帳號

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd*/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

data "aws_vpc" "default" {
  default = true
}

# ----------------------------------------------------------------------------
# 安全群組：雲端層級的防火牆
# ----------------------------------------------------------------------------

resource "aws_security_group" "experiment" {
  name        = "${var.project_name}-sg"
  description = "Allow SSH from a single trusted address"
  vpc_id      = data.aws_vpc.default.id

  # 僅允許來自指定位址的 SSH。
  # 切勿使用 0.0.0.0/0 —— 全球任何主機都能嘗試連線，
  # 開放的 SSH port 會在數分鐘內開始遭遇自動化密碼嘗試。
  ingress {
    description = "SSH from trusted address"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  # 對外連線全開：安裝套件、git clone 皆需要
  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-sg"
  }
}

# ----------------------------------------------------------------------------
# 金鑰對：沿用本機既有的 SSH 公鑰
# ----------------------------------------------------------------------------

# 僅上傳公鑰。私鑰始終留在本機，不經過 AWS，也不進入 state。
resource "aws_key_pair" "experiment" {
  key_name   = "${var.project_name}-key"
  public_key = file(var.public_key_path)

  tags = {
    Name = "${var.project_name}-key"
  }
}

# ----------------------------------------------------------------------------
# EC2 執行個體
# ----------------------------------------------------------------------------

resource "aws_instance" "experiment" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.instance_type
  key_name      = aws_key_pair.experiment.key_name

  vpc_security_group_ids = [aws_security_group.experiment.id]

  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    delete_on_termination = true # 隨執行個體一併刪除，避免遺留計費中的磁碟
  }

  tags = {
    Name = "${var.project_name}-experiment"
  }
}
