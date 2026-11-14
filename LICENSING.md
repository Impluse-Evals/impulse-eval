<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Licensing

1. **Which files are under which licence.**

   | Part | Files | Licence |
   |---|---|---|
   | Core | everything in the repository except the plugin API and compliance pack files below (including `assertions/__init__.py`, `examples/`, `tests/`, `templates/`, `docs/`) | AGPL-3.0-only (`LICENSE`) |
   | Plugin API | `impulse_evals/plugin_types.py`, `impulse_evals/assertions/base.py`, `impulse_evals/metrics/base.py`, `impulse_evals/adapters/base.py`, `impulse_evals/judge/base.py` | Apache-2.0 (`LICENSE-APACHE`) |
   | Compliance packs | `impulse_evals/assertions/dpdp.py`, `rbi_kyc.py`, `trai_dlt.py`, `hipaa.py` | Elastic License 2.0 (`LICENSE-ELv2`) |

2. **Testing your own agents creates no obligation.** Running Impulse to test your own agents, inside your company or in CI, does not oblige you to publish your agent or your configs.

3. **When the AGPL applies.** The AGPL obligations apply when you distribute a modified Impulse, or when you let other people use a modified Impulse over a network.

4. **What the Elastic License 2.0 allows for the compliance packs.** Anyone may use, copy, modify, and redistribute the packs, including commercially. You may not provide them to third parties as a hosted or managed service, circumvent licence-key functionality, or remove licensing notices.

5. **The plugin API files are permissive.** The plugin API files (table above) are under the Apache License 2.0, a permissive licence: anyone may use, copy, modify, and redistribute them, including commercially, provided the licence and notices are kept.

6. This summary is not legal advice; the LICENSE files govern.
