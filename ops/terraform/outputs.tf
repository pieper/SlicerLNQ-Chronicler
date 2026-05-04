output "doorman_public_ip" {
  description = "Floating IP attached to the doorman"
  value       = data.openstack_networking_floatingip_v2.doorman.address
}

output "doorman_internal_ip" {
  description = "Tenant network IP of the doorman"
  value       = openstack_compute_instance_v2.doorman.access_ip_v4
}

output "core_internal_ip" {
  description = "Tenant network IP of the core"
  value       = openstack_compute_instance_v2.core.access_ip_v4
}

output "domain" {
  value = var.domain_name
}
