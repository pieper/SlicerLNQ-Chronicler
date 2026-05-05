# Phase 1 Postmortem

The first end-to-end stand-up of SlicerLNQ-Chronicle (CouchDB + dicomweb-server behind Caddy on Jetstream2) hit eight bugs between the initial `tofu apply` and a working `https://lnq-chronicle.isomics.dev/dicomweb/studies → []`. Most were caught from cloud-init or systemd logs and fixed in single-line code changes. None were architectural; all were either ecosystem drift (upstream tool/package format changed) or my own mistakes about how cloud-init / dpkg / OpenStack interact.

This document is for future Steve, anyone forking Chronicler for another project, and any agent helping with this kind of infra work in 2026+.

## The bugs, in the order they surfaced

| # | Commit | Symptom | Root cause |
|---|---|---|---|
| 1 | `df93c64` | `tofu init` failed: "swift backend not supported in OpenTofu v1.3+" | The `swift` backend was deprecated and removed in Terraform 1.3; OpenTofu inherited that. Replaced with local backend + manual state sync to a Swift container via the `openstack` CLI. |
| 2 | `5ecbd9f` | "resource type `openstack_compute_floatingip_associate_v2` not supported" | `terraform-provider-openstack` v3 dropped the nova-network shims. Replaced with `openstack_networking_floatingip_associate_v2` against an explicitly-created port (the auto-port's ID isn't back-filled in v3). |
| 3 | `011b0bd` | `apt install caddy` failed: "Malformed entry in caddy-stable.list" | Cloudsmith updated their `debian.deb.txt` to already include `[signed-by=/usr/share/keyrings/...]`; my `sed` was injecting a *second* `signed-by=` directive. Stopped fighting it: place the key at `/usr/share/keyrings/`, consume the file verbatim. |
| 4 | `7fd8249` | dicomweb-server unit in restart loop with `200/CHDIR`; `/opt/lnq/dicomweb-server` not on disk | `git clone --branch main` printed "Cloning into..." then errored — dcmjs-org/dicomweb-server's default branch is `master`. Git removes the partial directory on failure; subsequent runcmd lines silently failed. Switched to clone-then-checkout for branch/tag/SHA flexibility. |
| 5 | `fb2957d` | dicomweb-server crash with `Cannot find module './production'` | `config/index.js` does `require('./' + NODE_ENV)`; only `config/development.js` exists upstream. Set `NODE_ENV=development` in the env file. |
| 6 | `4afe289` | dicomweb-server stuck in "Waiting for couchdb server" loop, returning 503 | The CouchDB admin password (from `openssl rand -base64 32`) contained `/` and `=`. Embedded raw in `DB_SERVER=http://admin:PASS@host`, the URL parser misread the `/` as a path separator. Fixed by passing `urlencode(var.couchdb_admin_password)` into the cloud-init template. |
| 7 | `49e24a1` | Same loop persisting after #6; deprecation warning in journal: "URL `http://...@127.0.0.1:5984:5984` is invalid" | dicomweb-server's config has a separate `dbPort` field (default 5984) that it appends to `dbServer`. Strip the port from `DB_SERVER`; let the server append its own. |
| 8 | `cbae517` | Fresh redeploy: Caddy unit fails `217/USER`; `caddy` user doesn't exist | `write_files` pre-placed `/etc/caddy/Caddyfile`. `apt install caddy` then hit dpkg's conffile-conflict mechanism, which prompted interactively. With no stdin in cloud-init, dpkg `--configure` aborted before postinst ran, so the `caddy` user was never created. `defer: true` on the write_files entry didn't change this — the file was on disk during the apt run anyway. Fix: don't use write_files for package-managed conffiles; install the package first, then overwrite its default config in `runcmd` via heredoc. |

## Meta-lessons

**Cloud-init `runcmd` does not halt on intermediate errors.** Each entry in `runcmd:` runs as a separate shell command, and only the *last* command's exit status determines whether cloud-init reports the module as failing. Bug #4 silently cascaded (failed `git clone` → failed `chown` → failed `npm ci`), and bug #2 in our debug history exited with `217/USER` only on the final `systemctl start`, not on whatever step actually broke the install. Future hardening: wrap groups of related commands in `bash -euo pipefail` heredocs so a failure aborts the group and surfaces in the cloud-init log.

**`write_files` interacts badly with package-managed config files.** If you pre-place a file that a package considers a conffile, dpkg's interactive conflict prompt fires during install. There's no stdin in cloud-init, so the install aborts mid-`--configure` and the postinst (which often creates users, sets up directories, registers systemd units) never runs. Always install the package first, then overwrite its defaults — either in `runcmd` via heredoc, or in a separate cloud-init-final stage script. `defer: true` does *not* reliably solve this in our experience; moved Caddyfile to runcmd entirely.

**Verify upstream sources, not memory.** Bugs #3, #4, #5, and #7 were all variants of "I assumed this worked the way it used to." Cloudsmith's deb.txt format changed; dcmjs-org/dicomweb-server's default branch is `master`; that repo only has `development.js`; that repo appends its own dbPort. Each could have been caught by reading the actual upstream files before writing the cloud-init. The cost of looking is ~30 seconds; the cost of finding out via a failed deploy is ~15 minutes plus context loss.

**Provider major versions move fast.** Three of the eight bugs were upstream-tooling moves:
- Swift backend removed in OpenTofu 1.3+
- `openstack_compute_floatingip_associate_v2` removed in provider v3
- `network[0].port` attribute on `openstack_compute_instance_v2` is input-only in v3

For long-lived infra: pin provider major versions in `versions.tf` and read the migration notes when bumping. We're pinned at `~> 3.0` here, which is correct.

**URL-embedded credentials are fragile.** Random base64 passwords (`openssl rand -base64`) contain `/`, `+`, `=`. URL parsers treat `/` as a path separator inside the userinfo section. Two ways to avoid this trap: generate URL-safe passwords (`openssl rand -hex`), or run the password through `urlencode` before embedding. We chose `urlencode` because the *raw* password is also needed for debconf preseeding; can't have both forms be the same.

## Things that worked despite our debugging chaos

- Js2's Swift state container as a backend substitute. Manual sync via the openstack CLI is uglier than a native backend but takes ~10 lines of workflow YAML and uses the credentials we already had.
- The "verify on instance, then commit fix, then redeploy" pattern. Iterating in place on the running instance let us test fixes in seconds, then bake them into the repo. Each redeploy was a clean validation that the committed code reproduced the manual state.
- Caddy's automatic TLS. Once DNS pointed at the right IP and the unit could actually start, ACME got us a real cert in <90 seconds with no further intervention.
- The split between `architecture.md` (intent) and the Chronicler repo (mechanism). When debugging, "what is this supposed to do?" was always answerable by reading docs in `SlicerLNQ/`, separately from the question of whether the implementation was buggy.

## Phase 2, commit 1 (watchman) — four follow-up fixes

Adding the watchman service in `6663cbc` introduced four regressions that took another half-day to walk through. None were architectural; all were variations on the same lessons phase 1 surfaced, which is the more useful diagnostic.

| Commit | Symptom | Root cause |
|---|---|---|
| `c2fa67d` + `019db02` | `npm ci` failed EUSAGE on every fresh apply | The lockfile had to land in *two* places: committed to the repo (commit 1), and added to the cloud-init's base64-embedded files (commit 2). I did the first and forgot the second. The repo had `package-lock.json` but the cloud-init didn't ship it to the instance. |
| `e76504a` | Caddy listening on :80 only, no auto-HTTPS, all curls hang | The phase-1 cloud-init ended with `systemctl reload caddy`. I dropped that line restructuring runcmd for watchman. The Caddy package's postinst starts caddy with the package default Caddyfile during `apt install`; when runcmd later overwrites the Caddyfile, caddy doesn't pick it up without a reload. |
| `2693af6` | Caddy fails to start on second restart with "permission denied" on access.log | The Caddy package's postinst pre-creates `/var/log/caddy/access.log` as `root:root`. Our `install -d -o caddy /var/log/caddy` sets ownership on the directory but doesn't recurse into pre-existing files. When Caddy tries to write the log it gets EACCES and exits. Need `chown -R` after `install -d`. |

## Future hardening (not phase 1 work)

1. **Wrap risky `runcmd` blocks in `bash -euo pipefail`** so silent cascades become loud failures in the cloud-init log. Bug #4 cost an hour because the subsequent commands swallowed the real error.
2. **Add a cloud-init `final` stage health check** that probes both `:5984/_up` and `:5985/studies` on the core, and `:443` on the doorman. If any fail, the cloud-init result is `error` instead of `done`, surfacing problems immediately rather than after the operator notices broken curls.
3. **Build a Js2 image with the heavy installs baked in** (CouchDB, Node 20, Caddy, dicomweb-server source) once the cloud-init is stable. First-boot time drops from ~7 min to ~1 min, and apt-source drift (bug #3) doesn't bite per-deploy.
4. **Add a runbook for Js2 floating-IP reaping.** Js2 reclaims unattached FIPs after some hours; ours got reaped during an overnight shelve. Document either keeping a no-op attachment or accepting the IP-and-DNS rotation cost.
5. **Document the `exouser` vs `ubuntu` SSH user variance.** Featured Js2 images use `ubuntu` for the cloud-image default; `exouser` is added by Exosphere when *Exosphere* provisions the instance. Terraform-provisioned instances only get `ubuntu`.
6. **Add `tofu validate` + `tofu plan -refresh-only` as a pre-merge gate** in the plan workflow. The lockfile bug (`019db02`) — base64-encoding a file that wasn't passed to `templatefile()` — is exactly what `tofu validate` catches. Cheap to add, real value.
7. **Replace manual base64-embedding of the watchman files** with a `for_each` over `fileset("${path.module}/../watchman", "*.{js,json}")` (or similar). The current pattern requires synchronizing two lists by hand: files in the repo and files in the templatefile call. That's a permanent footgun; whoever forks Chronicler will hit it.
8. **Always pair `apt install <package>` with `systemctl reload <service>` after writing custom configs.** The Caddy reload regression (`e76504a`) was the second time we got bitten by "package starts service with its default config; our config never takes effect." Make this an explicit pattern in the cloud-init style.
9. **Always pair `install -d <pkg-managed dir>` with `chown -R`.** Same package-pre-creates-files class of bug as the access.log issue. Belt-and-suspenders against postinst surprises.
