<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |

## Reporting a vulnerability

Please report security problems privately. Do not open a public issue, pull request or discussion for them.

1. Open the repository's **Security** tab and click **Report a vulnerability**. This uses GitHub's private vulnerability reporting, so only the maintainers see your report.
2. Describe the problem, the affected version or commit, and the steps to reproduce it. Include a minimal config or proof of concept if you can.
3. Leave out real keys, real transcripts and personal data. Use dummy values instead.

What to expect:

- an acknowledgement within 7 days;
- an assessment, and a plan for a fix or a reason why it is not treated as a vulnerability;
- credit in the release notes when the fix ships, unless you ask not to be named.

Please give us a reasonable time to release a fix before you disclose the problem publicly.

## Scope

In scope: the `impulse_evals` package, its command line, the HTML and JSON reports it writes, and the examples in this repository. Examples are key or secret leaks past the redaction, unsafe handling of agent or judge output in the HTML report, and path traversal through run files.

Out of scope: problems in the language models, voice providers or agent platforms you connect Impulse to, and the behaviour of the agent under test. How Impulse handles keys and data is described in [Privacy and security](docs/privacy-and-security.md).
