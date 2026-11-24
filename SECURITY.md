# Security

**Reporting.** Please report a vulnerability privately through GitHub's "Report a vulnerability" (Security tab), not in a public issue. You will get an acknowledgement, and a fix or an explanation, before anything is disclosed.

**What the package does with your data.** It reads the files you pass, computes locally and writes only the outputs you ask for. It opens no network connections (see [docs/API.md](docs/API.md#offline-use)). Reports reduce file paths to file names and escape every string from the data.

**Supported versions.** Fixes go to the latest release.

**Verifying downloads.** Releases carry SHA-256 checksums, an SBOM and a Sigstore provenance attestation ([docs/API.md](docs/API.md#verifying-a-release)).
