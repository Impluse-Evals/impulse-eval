# SPDX-License-Identifier: Apache-2.0
"""``python -m impulse_evals``: the same CLI as the ``impulse`` command (SPEC.md section 17).

Help and usage texts name the program ``impulse``; exit codes and output are the console script's.
Importing this module runs nothing (the CLI is imported only when the module runs as ``__main__``).
"""

if __name__ == "__main__":
    from impulse_evals.cli import app

    app(prog_name="impulse")
