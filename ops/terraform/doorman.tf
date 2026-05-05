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
    domain_name              = var.domain_name
    letsencrypt_email        = var.letsencrypt_email
    core_internal_ip         = openstack_compute_instance_v2.core.access_ip_v4
    core_instance_name       = openstack_compute_instance_v2.core.name
    idle_timeout_sec         = var.idle_timeout_sec
    os_auth_url              = var.os_auth_url
    os_region_name           = var.os_region_name
    watchman_os_app_cred_id     = var.watchman_os_app_cred_id
    watchman_os_app_cred_secret = var.watchman_os_app_cred_secret
    watchman_index_js_b64       = base64encode(file("${path.module}/../watchman/index.js"))
    watchman_package_json_b64   = base64encode(file("${path.module}/../watchman/package.json"))
    # package-lock.json intentionally not embedded; embedding it pushes
    # user_data over OpenStack Nova's ~64KB limit. We use `npm install`
    # in cloud-init instead of `npm ci`. Lockfile remains in the repo
    # for local development reproducibility.
  })

  metadata = {
    role    = "doorman"
    project = "lnq"
  }

  # Serialize doorman creation: wait until both the core is up AND the FIP is
  # fully bound to the port. Otherwise Terraform fires the server-create in
  # parallel with the FIP-associate modify, and Nova rejects the bind because
  # the port is mid-mutation by Neutron.
  depends_on = [
    openstack_compute_instance_v2.core,
    openstack_networking_floatingip_associate_v2.doorman,
  ]
}

resource "openstack_networking_floatingip_associate_v2" "doorman" {
  floating_ip = data.openstack_networking_floatingip_v2.doorman.address
  port_id     = openstack_networking_port_v2.doorman.id
}
