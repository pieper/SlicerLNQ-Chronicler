resource "openstack_compute_instance_v2" "core" {
  name      = "lnq-core"
  image_name = var.core_image
  flavor_name = var.core_flavor
  key_pair  = openstack_compute_keypair_v2.deploy.name

  security_groups = [openstack_networking_secgroup_v2.core.name]

  network {
    name = data.openstack_networking_network_v2.tenant.name
  }

  user_data = templatefile("${path.module}/../cloud-init/core.yml.tmpl", {
    # Raw password for debconf-set-selections (CouchDB install).
    couchdb_admin_password = var.couchdb_admin_password
    # URL-encoded copy for embedding in DB_SERVER. Required because base64
    # passwords contain '/' and '=' which URL parsers misinterpret.
    couchdb_admin_password_url = urlencode(var.couchdb_admin_password)
    couchdb_version            = var.couchdb_version
    dicomweb_server_ref        = var.dicomweb_server_ref
  })

  metadata = {
    role    = "core"
    project = "lnq"
  }
}
