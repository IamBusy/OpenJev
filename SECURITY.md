# Security

OpenJev is experimental local research software. Only the current development
line receives fixes; there is no guaranteed response time or supported production
service. The bundled HTTP server listens on localhost, has no authentication,
and must not be exposed to an untrusted network without deployment controls.

Models can produce incorrect or overconfident decisions, including when a state
contains malicious instructions. Output validation constrains shape, not truth
or permission to take an action. Treat model results as data in downstream code.

For a suspected vulnerability, use GitHub's private vulnerability reporting on
this repository when available. If unavailable, open an issue asking for a private
contact without exploit details, secrets or personal information. Do not post
live credentials. If a credential is exposed, revoke it through its provider.

Release downloads check SHA-256 hashes, reject unsafe archive paths and never
execute archive contents. Models use safetensors. Install dependencies from their
normal registries and keep local `.env` files outside version control.
