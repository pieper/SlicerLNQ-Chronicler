resource "openstack_networking_port_v2" "doorman" {
  name               = "lnq-doorman-port"
  network_id         = data.openstack_networking_network_v2.tenant.id
  security_group_ids = [openstack_networking_secgroup_v2.doorman.id]
  admin_state_up     = true
}

resource "openstack_compute_instance_v2" "doorman" {
  name        = "lnq-doorman"
  image_name  = var.doorman_image
  flavor_name = var.doorman_flavor
  key_pair    = openstack_compute_keypair_v2.deploy.name

  # Security groups live on the port (above), not the instance, when using
  # an explicit pre-created port.

  network {
    port = openstack_networking_port_v2.doorman.id
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
  port_id     = openstack_networking_port_v2.doorman.id
}
