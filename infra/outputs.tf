# ============================================================================
# 輸出值
#
# terraform apply 完成後顯示，亦可隨時以 terraform output 查詢。
# ============================================================================

output "instance_id" {
  description = "EC2 執行個體 ID"
  value       = aws_instance.experiment.id
}

output "public_ip" {
  description = "對外 IP 位址"
  value       = aws_instance.experiment.public_ip
}

output "ami_id" {
  description = "實際使用的 AMI ID（隨區域與安全更新而異，記錄以利日後追溯）"
  value       = data.aws_ami.ubuntu.id
}

output "ssh_command" {
  description = "可直接複製執行的 SSH 連線指令"
  value       = "ssh ubuntu@${aws_instance.experiment.public_ip}"
}
