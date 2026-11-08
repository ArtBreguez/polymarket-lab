# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Report it privately through
[GitHub's private vulnerability reporting](https://github.com/ArtBreguez/polymarket-lab/security/advisories/new)
(Security tab -> Report a vulnerability) with:

- what you found and what it allows;
- how to reproduce it;
- the pmlab version (`pmlab version`) and your Python version.

The maintainer will acknowledge the report, agree a fix and a disclosure date with you, and credit
you in the changelog unless you prefer otherwise. A confirmed problem is fixed in a patch release.

## Supported versions

The latest release on [PyPI](https://pypi.org/project/pmlab/) is supported. pmlab follows
[semantic versioning](https://semver.org) (see [Public API and stability](https://artbreguez.github.io/polymarket-lab/api-stability/)),
so a fix lands in the current 1.x line without breaking the public API.

## What to keep in mind when you use it

- **Credentials.** `LiveBroker` takes a private key and Polymarket API credentials. Read them from
  the environment or a secrets manager, never from source code, and never commit them. No `pmlab`
  command places an order; only a `LiveBroker` that you construct does.
- **Pickled models.** Champions and calibrators are stored with `pickle`, as LightGBM and
  scikit-learn do. Load only artifacts that your own runs produced (or that you otherwise trust):
  unpickling an untrusted file can run arbitrary code.
- **Network.** `GammaClient` and `ClobClient` talk to Polymarket's public endpoints. The bundled
  sample dataset and the tutorial work without any network access.
