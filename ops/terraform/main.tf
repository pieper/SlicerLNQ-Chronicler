provider "openstack" {
  # Authentication via OS_* environment variables.
  # Locally, set OS_CLOUD=<your-cloud-alias> from your clouds.yaml.
  # In CI, the workflow sets OS_AUTH_TYPE / OS_APPLICATION_CREDENTIAL_*.
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
