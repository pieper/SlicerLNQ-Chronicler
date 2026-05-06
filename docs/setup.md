# One-time setup

Steps you do once before the first `chronicle.sh create`. Most of these you only repeat if you start fresh in a new Js2 project.

## 1. Js2 project + clouds.yaml

You need a Js2 (OpenStack) project with allocation, and `~/.config/openstack/clouds.yaml` set up so the `openstack` CLI can authenticate. See Js2's docs for the standard flow; the result is that `openstack --os-cloud <your-cloud-name> server list` works.

Note the cloud alias name; that goes in `chronicle.conf` as `OS_CLOUD`.

## 2. SSH keypair registered with Js2

```sh
ssh-keygen -t ed25519 -f ~/.ssh/lnq-chronicler-deploy -C "lnq-chronicler-deploy" -N ''
openstack --os-cloud <your-cloud> keypair create \
  --public-key ~/.ssh/lnq-chronicler-deploy.pub \
  lnq-chronicler-deploy
```

The keypair name (`lnq-chronicler-deploy`) goes in `chronicle.conf` as `KEY_NAME`. The private key path goes in `SSH_KEY`.

## 3. Reserve a floating IP

```sh
openstack --os-cloud <your-cloud> floating ip create public
```

Note the IPv4 address — goes in `chronicle.conf` as `FLOATING_IP`. Js2 reaps unattached floating IPs after some hours, so reserve this around the same time you're ready to create the instance, or be prepared to re-allocate.

## 4. DNS

Point `<DOMAIN_NAME>` (e.g. `lnq-chronicle.isomics.dev`) at the floating IP via an A record. TTL of 30 minutes (1800 s) is comfortable.

```sh
dig +short lnq-chronicle.isomics.dev
# Should print the floating IP from step 3.
```

Do this *before* `chronicle.sh create` — Caddy's automatic Let's Encrypt issuance fails if DNS isn't pointing at the instance when it boots, and it'll back off for a while before retrying.

## 5. Generate the CouchDB admin password

```sh
openssl rand -hex 32
```

Use **hex** (not `base64`) — base64 contains `/`, `+`, and `=`, which break URL parsing inside the `DB_SERVER=http://admin:PASSWORD@127.0.0.1` env var that dicomweb-server reads. Hex is URL-safe; saves you the URL-encoding dance.

The output goes in `chronicle.conf` as `COUCHDB_ADMIN_PASSWORD`. Save it in a password manager too — you'll need it to log into Fauxton or run direct curls against CouchDB.

## 6. Fill in chronicle.conf

```sh
cp chronicle.conf.example chronicle.conf
$EDITOR chronicle.conf
```

Required fields:

| Variable | Source |
|---|---|
| `OS_CLOUD` | step 1 |
| `INSTANCE_NAME` | your choice — `lnq-chronicle` is fine |
| `IMAGE` | `openstack image list --name 'Featured*' \| grep Ubuntu24` |
| `FLAVOR` | `m3.medium` (CouchDB + Caddy + dicomweb-server fits comfortably) |
| `NETWORK` | usually `auto_allocated_network` — confirm with `openstack network list` |
| `KEY_NAME` | step 2 |
| `SSH_KEY` | step 2 (private half) |
| `SSH_USER` | `ubuntu` for Js2 Featured-Ubuntu24 |
| `DOMAIN_NAME` | step 4 |
| `FLOATING_IP` | step 3 |
| `LETSENCRYPT_EMAIL` | your email for cert expiry notices |
| `COUCHDB_ADMIN_PASSWORD` | step 5 |
| `DICOMWEB_REF` | leave as `master` unless pinning a fork or commit SHA |

## 7. Run it

```sh
bin/chronicle.sh create
bin/chronicle.sh logs              # follow cloud-init progress
# In another terminal, wait ~5-10 min, then:
curl -i https://lnq-chronicle.isomics.dev/
```

If it works, you're done. If something fails, `bin/chronicle.sh ssh` and inspect:

- `sudo cloud-init status --long` — overall state
- `sudo tail -200 /var/log/cloud-init-output.log` — what each runcmd line did
- `sudo systemctl status caddy couchdb lnq-dicomweb` — service state
- `sudo journalctl -u caddy --no-pager -n 50` — Caddy log (ACME issues live here)

## Updating

When code changes (cloud-init, dicomweb-server ref), the cleanest path is a fresh instance:

```sh
bin/chronicle.sh destroy
bin/chronicle.sh create
```

For small in-place tweaks (Caddyfile, env), SSH in and edit. Document the deviation in `chronicle.conf` or the script so it survives a future destroy/create cycle.

## When the floating IP gets reaped

Js2 reclaims unattached FIPs after some hours. If your instance has been destroyed for a while, the FIP may be gone:

```sh
# Allocate a new one
openstack --os-cloud <your-cloud> floating ip create public

# Update chronicle.conf with the new IP
# Update DNS at your registrar to point at the new IP
# Wait for DNS to propagate (~5-30 min depending on TTL)
# Then:
bin/chronicle.sh create
```
