resource "openstack_networking_secgroup_v2" "doorman" {
  name        = "lnq-doorman"
  description = "Public-facing doorman: TLS + SSH"
}

resource "openstack_networking_secgroup_rule_v2" "doorman_https" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 443
  port_range_max    = 443
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.doorman.id
}

resource "openstack_networking_secgroup_rule_v2" "doorman_http" {
  # Needed for ACME HTTP-01 challenge during cert issuance/renewal
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 80
  port_range_max    = 80
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.doorman.id
}

resource "openstack_networking_secgroup_rule_v2" "doorman_ssh" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.doorman.id
}

resource "openstack_networking_secgroup_v2" "core" {
  name        = "lnq-core"
  description = "Core: reachable only from doorman and SSH"
}

resource "openstack_networking_secgroup_rule_v2" "core_couchdb_from_doorman" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 5984
  port_range_max    = 5984
  remote_group_id   = openstack_networking_secgroup_v2.doorman.id
  security_group_id = openstack_networking_secgroup_v2.core.id
}

resource "openstack_networking_secgroup_rule_v2" "core_dicomweb_from_doorman" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 5985
  port_range_max    = 5985
  remote_group_id   = openstack_networking_secgroup_v2.doorman.id
  security_group_id = openstack_networking_secgroup_v2.core.id
}

resource "openstack_networking_secgroup_rule_v2" "core_ssh" {
  # SSH from anywhere for now; tighten to a bastion or your IP in a later commit.
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.core.id
}
