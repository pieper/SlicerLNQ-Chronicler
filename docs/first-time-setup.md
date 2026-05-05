# First-time setup

Run these once before the first `apply.yml`. Architectural context and the broader phase plan live in [SlicerLNQ/docs/architecture.md](https://github.com/pieper/SlicerLNQ/blob/main/docs/architecture.md).

## 1. Create the Swift container that holds Terraform state

The OpenTofu Swift backend needs the state container to exist before the first `tofu init`. From a shell where `OS_CLOUD` is set to your dev project:

```sh
openstack container create lnq-chronicler-tfstate
openstack container create lnq-chronicler-tfstate-archive
```

That's it — `versioning` for object recovery is enabled by default on Js2's Swift, and the archive container holds historical state files.

## 2. Confirm DNS and floating IP are wired up

```sh
dig +short lnq-chronicle.isomics.dev
# should return your reserved floating IP

openstack floating ip list --status DOWN
# your reserved IP should appear here (DOWN = not yet attached); the first
# apply will associate it with the doorman.
```

## 3. Confirm the GitHub Actions secrets are set

In the repo: Settings → Secrets and variables → Actions. The full set:

| Secret | Value |
|---|---|
| `OS_AUTH_URL` | `https://js2.jetstream-cloud.org:5000/v3/` |
| `OS_REGION_NAME` | `IU` |
| `OS_APP_CRED_ID` | `chronicler-actions` credential ID |
| `OS_APP_CRED_SECRET` | `chronicler-actions` credential secret |
| `CONSERVATOR_OS_APP_CRED_ID` | `chronicler-conservator` credential ID (used in a later commit) |
| `CONSERVATOR_OS_APP_CRED_SECRET` | `chronicler-conservator` credential secret (used in a later commit) |
| `SSH_PUBLIC_KEY` | contents of `~/.ssh/lnq-chronicler-deploy.pub` (one line) |
| `COUCHDB_ADMIN_PASSWORD` | output of `openssl rand -base64 32` |
| `LETSENCRYPT_EMAIL` | your email for Let's Encrypt notices |
| `FLOATING_IP` | your reserved floating IP, e.g. `149.165.152.66` |
| `DOMAIN_NAME` | `lnq-chronicle.isomics.dev` |

The conservator secrets aren't used yet but adding them now means the next commit doesn't need to wait on you.

The private SSH deploy key (`~/.ssh/lnq-chronicler-deploy`) is **not** yet wired into a workflow — it's used by future runbooks that SSH into the instances. Add it as `SSH_DEPLOY_KEY` when those workflows land.

## 4. Create a GitHub Environment for deploy gating (optional but recommended)

The `apply` and `destroy` workflows reference an environment named `production`. Settings → Environments → New environment → name `production`. Add a "Required reviewers" rule with yourself as reviewer. This means even with the typed-confirmation prompt, the workflow won't run until you click approve in the PR/Actions UI — a useful brake against fat-fingered destroys.

If you'd rather not use environments, delete the `environment: production` line from `apply.yml` and `destroy.yml`.

## 5. First apply

Actions → "tofu apply" → Run workflow → type `apply` in the confirm field → Run.

Expected timeline:
- Terraform: ~30 s to plan + apply (just creating instances and security groups, the FIP is pre-existing)
- Cloud-init on the **core**: ~3–5 min (CouchDB + Node + clone + npm ci on first boot)
- Cloud-init on the **doorman**: ~1–2 min (apt + Caddy + first cert issuance)
- Total wall-clock from "Run workflow" to first successful response on the public URL: ~5–8 min

## 6. Verify

```sh
# CouchDB welcome (proxied through the doorman)
curl https://lnq-chronicle.isomics.dev/

# Auth-required endpoint (will 401 without credentials)
curl https://lnq-chronicle.isomics.dev/_all_dbs

# With admin auth (replace with your COUCHDB_ADMIN_PASSWORD)
curl -u admin:PASSWORD https://lnq-chronicle.isomics.dev/_all_dbs

# dicomweb-server health (no studies yet, returns empty)
curl https://lnq-chronicle.isomics.dev/dicomweb/studies

# Fauxton in a browser:
open https://lnq-chronicle.isomics.dev/_utils/
```

If the doorman returns a TLS error in the first minute, that's Caddy still negotiating the Let's Encrypt cert — wait 30–60 s and retry.

## 7. SSH (for poking around)

```sh
ssh -i ~/.ssh/lnq-chronicler-deploy ubuntu@<doorman_public_ip>
ssh -i ~/.ssh/lnq-chronicler-deploy -J ubuntu@<doorman_public_ip> ubuntu@<core_internal_ip>
```

Or use Exosphere's web desktop / web shell on the core directly — that goes through the OpenStack control plane and bypasses the doorman entirely.

Useful spots once on the host:

- `/etc/caddy/Caddyfile` (doorman) — reverse proxy config
- `/var/log/caddy/access.log` (doorman) — JSON access log
- `journalctl -u caddy` (doorman)
- `journalctl -u couchdb` (core)
- `journalctl -u lnq-dicomweb` (core)
- `/etc/couchdb/local.d/10-lnq.ini` (core) — CouchDB config overlay
- `/opt/lnq/dicomweb-server/` (core) — server source tree

## 8. Tear-down

Actions → "tofu destroy" → Run workflow → type `destroy lnq-chronicle` → Run.

Floating IP, Swift state container, and DNS record stay — those are intentionally outside Terraform's blast radius so a clean `apply` can rebuild against the same external surface.
