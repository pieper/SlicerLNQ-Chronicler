variable "domain_name" {
  description = "Public hostname; A record must already point at floating_ip"
  type        = string
}

variable "floating_ip_address" {
  description = "Pre-reserved floating IP for the doorman"
  type        = string
}

variable "letsencrypt_email" {
  description = "Email Let's Encrypt uses for expiry notices"
  type        = string
}

variable "couchdb_admin_password" {
  description = "CouchDB admin password (sourced from a CI secret)"
  type        = string
  sensitive   = true
}

variable "doorman_image" {
  description = "Js2 image for the doorman (proxy only; minimal image is fine)"
  type        = string
  default     = "Featured-Ubuntu24"
}

variable "core_image" {
  description = "Js2 image for the core (use a featured image to preserve Exosphere web desktop)"
  type        = string
  default     = "Featured-Ubuntu24"
}

variable "doorman_flavor" {
  description = "Instance flavor for the doorman"
  type        = string
  default     = "m3.tiny"
}

variable "core_flavor" {
  description = "Instance flavor for the core"
  type        = string
  default     = "m3.medium"
}

variable "ssh_public_key" {
  description = "Public half of the deploy keypair, baked into cloud-init"
  type        = string
}

variable "tenant_network_name" {
  description = "Js2 tenant network for instance attachment"
  type        = string
  default     = "auto_allocated_network"
}

variable "dicomweb_server_ref" {
  description = "Git ref (branch / tag / commit SHA) of dcmjs-org/dicomweb-server to build at first boot"
  type        = string
  default     = "master"
}

variable "couchdb_version" {
  description = "CouchDB apt package version pin (matches Apache CouchDB Debian repo)"
  type        = string
  default     = "3.5.1"
}

# --- Watchman (activity-driven shelve/unshelve) -------------------------

variable "idle_timeout_sec" {
  description = "Seconds of no real activity before the watchman shelves the core"
  type        = number
  default     = 1200 # 20 minutes
}

variable "os_auth_url" {
  description = "OpenStack Keystone auth URL (used by the watchman to talk to Js2)"
  type        = string
}

variable "os_region_name" {
  description = "OpenStack region (used by the watchman)"
  type        = string
}

variable "watchman_os_app_cred_id" {
  description = "OpenStack application credential ID for the watchman (shelve/unshelve only)"
  type        = string
  sensitive   = true
}

variable "watchman_os_app_cred_secret" {
  description = "OpenStack application credential secret for the watchman"
  type        = string
  sensitive   = true
}
