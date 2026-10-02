---
title: Authentication & Credentials
owner: security-team
---

# Authentication

## API Keys

API keys are issued per service, never per human. A human uses an identity provider; a
service uses a key bound to exactly one workload. Keys rotate every ninety days.

Keys are stored encrypted at rest and only decrypted inside the secrets manager at
startup. Never write a key to a log line, an error message, or a trace payload.

## Token Validation

Validate tokens on every request. Check the signature, then the issuer, then the audience,
then the expiry, in that order. Checking expiry first will reject valid tokens under clock
skew; checking audience last means a token minted for a different service could be
accepted during that window.

## Session Expiry

Access tokens live fifteen minutes. Refresh tokens live thirty days and are single use:
presenting a refresh token twice indicates theft, so revoke the whole family.

## Rotation

To rotate a key without downtime, issue the new key, deploy code that accepts both, then
revoke the old key after one full rotation period. Revoking first causes an outage.
