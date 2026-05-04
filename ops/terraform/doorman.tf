resource "openstack_compute_instance_v2" "doorman" {
  name      = "lnq-doorman"
  image_name = var.doorman_image
  flavor_name = var.doorman_flavor
  key_pair  = openstack_compute_keypair_v2.deploy.name

  security_groups = [openstack_networking_secgroup_v2.doorman.name]

  network {
    name = data.openstack_networking_network_v2.tenant.name
  }

  user_data = templatefile("${path.module}/../cloud-init/doorman.yml.tmpl", {
    domain_name       = var.domain_name
    letsencrypt_email = var.letsencrypt_email
    core_internal_ip  = openstack_compute_instance_v2.core.access_ip_v4
  })

  metadata = {
    role    = "doorman"
    project = "lnq"
  }

  depends_on = [openstack_compute_instance_v2.core]
}

resource "openstack_networking_floatingip_associate_v2" "doorman" {
  floating_ip = data.openstack_networking_floatingip_v2.doorman.address
  port_id     = openstack_compute_instance_v2.doorman.network[0].port
}
