# Security

Use only trusted model, calibration and cache files. Some inherited research paths load PyTorch objects with pickle enabled; untrusted `.pt` files can execute code. Model weights belong in separate Hub repositories, not this source tree. Do not commit credentials, private prompts or machine-specific runtime files.

HTR native sources are a research integration, not a hardened multi-tenant service. File an issue for reproducible non-secret defects; report sensitive issues through GitHub private vulnerability reporting if enabled, rather than publishing secrets or exploitable deployments.
