provider "openstack" {
  cloud = var.cloud_name
}

data "openstack_networking_floatingip_v2" "doorman" {
  address = var.floating_ip_address
}

data "openstack_networking_network_v2" "tenant" {
  name = var.tenant_network_name
}

resource "openstack_compute_keypair_v2" "deploy" {
  name       = "lnq-chronicler-deploy"
  public_key = var.ssh_public_key
}
