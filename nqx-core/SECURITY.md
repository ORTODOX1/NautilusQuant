# Security policy

## Reporting vulnerabilities

If you discover a security issue in NQX-Core, please **do not open a public
GitHub issue**.

Instead, email: <security@example.com> _(replace with maintainer's address)_.

Include:
- Affected component (e.g. `server/api.py`, `nqx/cpu.py`, RTL)
- Reproducer (Python snippet or curl command)
- Impact assessment (confidentiality / integrity / availability)
- Suggested mitigation if you have one

You should receive an acknowledgment within 72 hours. We will discuss
disclosure timeline with you (typically 30-90 days from confirmation).

## Threat model

NQX-Core is a **research artifact and pre-silicon emulator**, not a hardened
production service. The default threat model assumes:

| Trusted | Untrusted |
|---|---|
| Operator running the service | HTTP request bodies |
| Operator's filesystem (NQX config, ROM-LUT) | Vector data submitted to `/encode` |
| Local Python environment | Network requests from arbitrary clients |

If you deploy `server/` to a public network, also consider:
- **Rate limiting** (FastAPI `slowapi` or upstream nginx) — not built-in
- **Authentication** (FastAPI Depends + API keys) — not built-in
- **Body size limits** (FastAPI `max_request_size`) — set to 100 MB by default
- **HTTPS** (terminate at reverse proxy: nginx, Caddy, Cloudflare)

## Hardening checklist

Before public deploy:

- [ ] Set `NQX_BACKEND=cpu` if you don't trust torch/triton dependencies
- [ ] Enable `server/middleware.AccessLogMiddleware` (already on by default)
- [ ] Configure log rotation for `server/logs/access.jsonl`
- [ ] Mount a read-only filesystem for the Docker container (`--read-only`)
- [ ] Use a non-root user inside the container
- [ ] Configure `--memory` and `--cpus` limits on `docker run`
- [ ] Set up Prometheus alerting on `nqx_errors_total`
- [ ] Configure `/health/deep` for tamper detection (LUT hash check)

## Supported versions

| Version | Security patches |
|---|---|
| 0.1.x | ✅ Yes |
| < 0.1 | ❌ No |

## Cryptography note

NQX-Core does **not** perform cryptographic operations. It quantizes vectors
deterministically using a fixed lookup table (golden-angle cos/sin values).
This is a compression scheme, **not** an encryption scheme. Do not rely on
NQX-Core to keep KV-cache data confidential — anyone with the LUT (which is
public, derivable from φ and π) can decompress.

## Credentials handling

If you contribute via PR:
- **Never** put GitHub PATs, AWS keys, API tokens, or any other credentials
  in code, config, comments, commit messages, or PR descriptions.
- Use `gh auth login --web` (browser flow) instead of pasting tokens.
- Use SSH keys for git remote URLs when possible.
- If you accidentally commit a credential, **revoke it immediately** at the
  issuer (GitHub, AWS, etc.) and force-push a clean history. Notify the
  maintainer.
